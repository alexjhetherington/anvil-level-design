"""Draw a white translucent overlay on faces marked as fixed hotspot.

The overlay rebuilds automatically on depsgraph updates (geometry changes,
fixed-hotspot toggling, undo, etc.) and is toggled via a scene-level bool
property.
"""

import bpy
import bmesh
import gpu
from bpy.types import Operator
from gpu_extras.batch import batch_for_shader

from ..utils import is_level_design_workspace, get_fixed_hotspot_layer, debug_log


# ---------------------------------------------------------------------------
#  Module state
# ---------------------------------------------------------------------------

_draw_handler = None
_overlay_tris = []
_last_object_name = None
_needs_rebuild = True


# ---------------------------------------------------------------------------
#  Overlay building
# ---------------------------------------------------------------------------

_COLOR = (0.0, 1.0, 0.0, 0.25)


def _triangulate_face_world(face, matrix_world):
    """Fan-triangulate a face and return a list of world-space vertex tuples."""
    verts = [matrix_world @ v.co for v in face.verts]
    tris = []
    for i in range(1, len(verts) - 1):
        tris.append(verts[0][:])
        tris.append(verts[i][:])
        tris.append(verts[i + 1][:])
    return tris


def _rebuild_overlay():
    """Rebuild the cached triangle list from the active edit-mesh."""
    global _overlay_tris, _last_object_name, _needs_rebuild
    _needs_rebuild = False
    _overlay_tris = []

    context = bpy.context
    obj = context.object
    if not obj or obj.type != 'MESH':
        _last_object_name = None
        return
    if context.mode != 'EDIT_MESH':
        _last_object_name = None
        return

    me = obj.data
    if me is None or not me.is_editmode:
        return

    try:
        bm = bmesh.from_edit_mesh(me)
    except (ReferenceError, RuntimeError):
        return
    if not bm.is_valid:
        return

    fixed_layer = bm.faces.layers.int.get("anvil_fixed_hotspot")
    if fixed_layer is None:
        _last_object_name = obj.name
        return

    bm.faces.ensure_lookup_table()
    matrix_world = obj.matrix_world
    for face in bm.faces:
        if face[fixed_layer] != 0:
            _overlay_tris.extend(_triangulate_face_world(face, matrix_world))

    _last_object_name = obj.name


def invalidate_overlay():
    """Mark the overlay as needing a rebuild on the next draw."""
    global _needs_rebuild
    _needs_rebuild = True


# ---------------------------------------------------------------------------
#  GPU drawing
# ---------------------------------------------------------------------------

def _draw_fixed_hotspot_overlay():
    """POST_VIEW draw callback."""
    if not is_level_design_workspace():
        return

    scene = bpy.context.scene
    if not hasattr(scene, 'level_design_props'):
        return
    props = scene.level_design_props
    if not props.show_fixed_hotspot_overlay:
        return

    if _needs_rebuild:
        _rebuild_overlay()

    if not _overlay_tris:
        return

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.uniform_float("color", _COLOR)
        batch = batch_for_shader(shader, 'TRIS', {"pos": _overlay_tris})
        batch.draw(shader)
    except Exception:
        pass
    finally:
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)


# ---------------------------------------------------------------------------
#  Draw handler management
# ---------------------------------------------------------------------------

def _register_draw_handler():
    global _draw_handler
    _unregister_draw_handler()
    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        _draw_fixed_hotspot_overlay, (), 'WINDOW', 'POST_VIEW'
    )


def _unregister_draw_handler():
    global _draw_handler
    if _draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        except Exception:
            pass
        _draw_handler = None


# ---------------------------------------------------------------------------
#  Toggle operator
# ---------------------------------------------------------------------------

class LEVELDESIGN_OT_toggle_fixed_hotspot_overlay(Operator):
    """Toggle overlay highlighting fixed hotspot faces"""
    bl_idname = "leveldesign.toggle_fixed_hotspot_overlay"
    bl_label = "Toggle Fixed Hotspot Overlay"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        props = context.scene.level_design_props
        props.show_fixed_hotspot_overlay = not props.show_fixed_hotspot_overlay
        invalidate_overlay()
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  Registration
# ---------------------------------------------------------------------------

classes = (
    LEVELDESIGN_OT_toggle_fixed_hotspot_overlay,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    _register_draw_handler()


def unregister():
    _unregister_draw_handler()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
