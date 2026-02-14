"""
Box Builder - Main Modal Operator

Thin subclass of ModalDrawBase that creates box geometry.
"""

import bpy
import bmesh
from mathutils import Vector

from . import geometry
from ..modal_draw.base_operator import ModalDrawBase
from ..modal_draw import snapping
from ..modal_draw import utils
from ...utils import is_level_design_workspace


def _get_selected_vertex_world_coords(context):
    """Return world-space positions of selected vertices in the active mesh.

    Respects the current selection mode (vertex/edge/face). Returns an empty
    set when nothing is selected or not in edit mode.
    """
    obj = context.active_object
    if obj is None or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
        return set()

    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.verts.ensure_lookup_table()
    world = obj.matrix_world

    coords = set()
    for v in bm.verts:
        if v.select:
            co = world @ v.co
            coords.add((co.x, co.y, co.z))

    return coords


class MESH_OT_box_builder(ModalDrawBase, bpy.types.Operator):
    """Create a box mesh using the 3-state draw workflow"""
    bl_idname = "leveldesign.box_builder"
    bl_label = "Box Builder"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not is_level_design_workspace():
            return False
        if context.mode == 'OBJECT':
            return True
        return (
            context.active_object is not None and
            context.active_object.type == 'MESH' and
            context.mode == 'EDIT_MESH'
        )

    def _is_valid_mode(self, context):
        return context.mode in ('EDIT_MESH', 'OBJECT')

    def _calculate_first_vertex_snap_3d(self, context, event):
        """Perspective snap with fallback to horizontal plane when no face is hit."""
        result = super()._calculate_first_vertex_snap_3d(context, event)
        snapped, face_normal, obj, was_clamped = result

        if snapped is not None:
            return result

        # No face hit - fall back to a horizontal plane at the lowest selected vertex Z
        coords = _get_selected_vertex_world_coords(context)
        if coords:
            z_height = min(co[2] for co in coords)
        else:
            z_height = 0.0

        plane_point = Vector((0, 0, z_height))
        plane_normal = Vector((0, 0, 1))

        point = utils.mouse_to_3d_on_plane(context, event, plane_point, plane_normal)
        if point is None:
            return (None, None, None, False)

        grid_size = utils.get_grid_size(context)
        if utils.is_snapping_enabled(context):
            snapped = snapping.snap_to_grid(point, grid_size)
            # Preserve the exact z_height (snap_to_grid snaps all 3 axes)
            snapped.z = z_height
        else:
            snapped = point

        return (snapped, Vector((0, 0, 1)), None, False)

    def _calculate_first_vertex_snap_2d(self, context, event):
        """Ortho snap that uses selected vertices to set the ambiguous axis value."""
        plane_data = utils.get_2d_view_plane(context)
        if plane_data is None:
            return (None, None)

        plane_point, plane_normal, axis1, axis2 = plane_data

        # Determine the best value for the ambiguous (depth) axis from selection
        coords = _get_selected_vertex_world_coords(context)
        view_type = utils.get_view_type(context)

        if coords:
            if view_type == 'TOP':
                plane_point = Vector((0, 0, max(co[2] for co in coords)))
            elif view_type == 'BOTTOM':
                plane_point = Vector((0, 0, min(co[2] for co in coords)))
            elif view_type == 'FRONT':
                plane_point = Vector((0, max(co[1] for co in coords), 0))
            elif view_type == 'BACK':
                plane_point = Vector((0, min(co[1] for co in coords), 0))
            elif view_type == 'RIGHT':
                plane_point = Vector((max(co[0] for co in coords), 0, 0))
            elif view_type == 'LEFT':
                plane_point = Vector((min(co[0] for co in coords), 0, 0))

        point = utils.mouse_to_3d_on_plane(context, event, plane_point, plane_normal)
        if point is None:
            return (None, None)

        grid_size = utils.get_grid_size(context)

        if utils.is_snapping_enabled(context):
            snapped = snapping.snap_to_grid(point, grid_size)
        else:
            snapped = point.copy()

        # Preserve the ambiguous axis value (don't snap it)
        if view_type in ('TOP', 'BOTTOM'):
            snapped.z = plane_point.z
        elif view_type in ('FRONT', 'BACK'):
            snapped.y = plane_point.y
        elif view_type in ('RIGHT', 'LEFT'):
            snapped.x = plane_point.x

        return (snapped, plane_normal)

    def _execute_action(self, context, first_vertex, second_vertex, depth,
                        local_x, local_y, local_z):
        rv3d = context.region_data
        view_forward = rv3d.view_rotation @ Vector((0, 0, -1))
        geometric_normal = local_x.cross(local_y)
        dot = geometric_normal.dot(view_forward)
        reverse_plane_normal = dot > 0

        props = context.scene.level_design_props
        ppm = props.pixels_per_meter

        if context.mode == 'OBJECT':
            return geometry.execute_box_builder_object_mode(
                first_vertex, second_vertex, depth,
                local_x, local_y, local_z,
                ppm, reverse_plane_normal
            )

        obj = context.active_object
        return geometry.execute_box_builder(
            first_vertex, second_vertex, depth,
            local_x, local_y, local_z,
            obj, ppm, reverse_plane_normal
        )

    def _get_tool_name(self):
        return "Box Builder"


def register():
    bpy.utils.register_class(MESH_OT_box_builder)


def unregister():
    bpy.utils.unregister_class(MESH_OT_box_builder)
