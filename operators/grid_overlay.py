"""Persistent grid overlay drawn on all visible mesh geometry.

Renders grid lines directly on mesh surfaces by extracting scene
triangles and drawing them with a grid-pattern fragment shader.
This avoids reading the viewport depth buffer, so Blender's floor
grid and other overlays are not affected.
"""

import bmesh
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from ..core.logging import debug_log
from ..core.workspace_check import is_level_design_workspace


# ---------------------------------------------------------------------------
#  Module state
# ---------------------------------------------------------------------------

_draw_handler = None
_shader = None


# ---------------------------------------------------------------------------
#  GLSL shader sources
#
#  Inputs/outputs/uniforms are declared via GPUShaderCreateInfo,
#  NOT in the shader source itself.
# ---------------------------------------------------------------------------

_VERTEX_SOURCE = (
    "void main()"
    "{"
    "  world_pos = position;"
    "  face_nrm = normal;"
    "  gl_Position = viewProjectionMatrix * vec4(position, 1.0);"
    "}"
)

_FRAGMENT_SOURCE = (
    "void main()"
    "{"
    "  vec3 dpdx = dFdx(world_pos);"
    "  vec3 dpdy = dFdy(world_pos);"
    ""
    "  vec3 grid_coord = world_pos / grid_size;"
    "  vec3 grid_fract = fract(grid_coord + 0.5) - 0.5;"
    "  vec3 grid_dist_world = abs(grid_fract) * grid_size;"
    ""
    "  float px_x = length(vec2(dpdx.x, dpdy.x));"
    "  float px_y = length(vec2(dpdx.y, dpdy.y));"
    "  float px_z = length(vec2(dpdx.z, dpdy.z));"
    ""
    "  float dist_x = grid_dist_world.x / max(px_x, 1e-8);"
    "  float dist_y = grid_dist_world.y / max(px_y, 1e-8);"
    "  float dist_z = grid_dist_world.z / max(px_z, 1e-8);"
    ""
    "  float half_w = line_width * 0.5;"
    "  float line_x = 1.0 - smoothstep(half_w - 0.5, half_w + 0.5, dist_x);"
    "  float line_y = 1.0 - smoothstep(half_w - 0.5, half_w + 0.5, dist_y);"
    "  float line_z = 1.0 - smoothstep(half_w - 0.5, half_w + 0.5, dist_z);"
    ""
    "  float px_thresh = 0.0001;"
    "  if (px_x < px_thresh) line_x = 0.0;"
    "  if (px_y < px_thresh) line_y = 0.0;"
    "  if (px_z < px_thresh) line_z = 0.0;"
    ""
    "  vec3 w = pow(abs(face_nrm), vec3(4.0));"
    "  float wsum = w.x + w.y + w.z;"
    "  w /= max(wsum, 1e-8);"
    ""
    "  float grid_yz = max(line_y, line_z);"
    "  float grid_xz = max(line_x, line_z);"
    "  float grid_xy = max(line_x, line_y);"
    "  float line_val = grid_yz * w.x + grid_xz * w.y + grid_xy * w.z;"
    "  float any_line = max(max(line_x, line_y), line_z);"
    "  if (any_line > 0.001) {"
    "    line_val = max(line_val, 0.15 * any_line);"
    "  }"
    ""
    "  if (line_val < 0.001) {"
    "    discard;"
    "  }"
    ""
    "  FragColor = vec4(0.7, 0.7, 0.7, line_val * opacity);"
    "}"
)


# ---------------------------------------------------------------------------
#  Shader creation
# ---------------------------------------------------------------------------

def _ensure_shader():
    global _shader

    if _shader is not None:
        return True

    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("grid_surface_interface")
        vert_out.smooth('VEC3', "world_pos")
        vert_out.flat('VEC3', "face_nrm")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', "viewProjectionMatrix")
        shader_info.push_constant('FLOAT', "grid_size")
        shader_info.push_constant('FLOAT', "line_width")
        shader_info.push_constant('FLOAT', "opacity")
        shader_info.vertex_in(0, 'VEC3', "position")
        shader_info.vertex_in(1, 'VEC3', "normal")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "FragColor")

        shader_info.vertex_source(_VERTEX_SOURCE)
        shader_info.fragment_source(_FRAGMENT_SOURCE)

        _shader = gpu.shader.create_from_info(shader_info)
        del vert_out
        del shader_info
    except Exception as e:
        print(f"Level Design Tools: Grid overlay shader failed: {e}")
        _shader = None
        return False

    return True


# ---------------------------------------------------------------------------
#  Mesh extraction
# ---------------------------------------------------------------------------

def _collect_mesh_batches(depsgraph, shader):
    """Build GPU triangle batches for every visible mesh object."""
    batches = []

    for inst in depsgraph.object_instances:
        obj = inst.object
        if obj.type != 'MESH':
            continue
        if not inst.show_self:
            continue

        # Use to_mesh() to get a snapshot that works in both object
        # and edit mode (obj.data is stale while in edit mode).
        try:
            mesh = obj.to_mesh()
        except RuntimeError:
            continue
        if mesh is None:
            continue

        vert_count = len(mesh.vertices)
        if vert_count == 0:
            obj.to_mesh_clear()
            continue

        mesh.calc_loop_triangles()
        tri_count = len(mesh.loop_triangles)
        if tri_count == 0:
            obj.to_mesh_clear()
            continue

        # Vertex positions (local space)
        cos = np.empty(vert_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', cos)
        cos = cos.reshape(-1, 3)

        # Transform to world space
        mat = np.array(inst.matrix_world, dtype=np.float32)
        ones = np.ones((vert_count, 1), dtype=np.float32)
        cos_h = np.hstack((cos, ones))
        world = (mat @ cos_h.T).T[:, :3]

        # Triangle indices
        tri_idx = np.empty(tri_count * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices', tri_idx)

        # Expand to per-triangle vertices (non-indexed) so each triangle
        # can carry its own flat face normal.
        tri_positions = world[tri_idx]  # (tri_count*3, 3)

        # Compute face normals from triangle edges
        tri_verts = tri_positions.reshape(tri_count, 3, 3)
        edge1 = tri_verts[:, 1] - tri_verts[:, 0]
        edge2 = tri_verts[:, 2] - tri_verts[:, 0]
        normals = np.cross(edge1, edge2)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(lengths, 1e-8)

        # Repeat each normal 3 times (once per vertex of the triangle)
        tri_normals = np.repeat(normals, 3, axis=0)  # (tri_count*3, 3)

        batch = batch_for_shader(
            shader, 'TRIS',
            {"position": tri_positions.tolist(),
             "normal": tri_normals.tolist()},
        )
        batches.append(batch)
        obj.to_mesh_clear()

    return batches


# ---------------------------------------------------------------------------
#  Edit-mode edge redraw
# ---------------------------------------------------------------------------

def _draw_edit_edges_on_top(context, depsgraph, region):
    """Redraw the active edit object's edges on top of the grid overlay.

    POST_VIEW callbacks run after Blender's edit-mode overlays, so the
    semi-transparent grid covers the wireframe edges.  Drawing the edges
    again here restores their visibility.
    """
    obj = context.edit_object
    if obj is None or obj.type != 'MESH':
        return

    eval_obj = obj.evaluated_get(depsgraph)
    try:
        mesh = eval_obj.to_mesh()
    except RuntimeError:
        return
    if mesh is None:
        return

    try:
        # Use bmesh to access selection state
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        if not bm.verts or not bm.edges:
            return

        mat = np.array(obj.matrix_world, dtype=np.float32)

        sel_positions = []
        unsel_positions = []

        for edge in bm.edges:
            v0 = np.array(edge.verts[0].co, dtype=np.float32)
            v1 = np.array(edge.verts[1].co, dtype=np.float32)
            # Transform to world space
            p0 = (mat @ np.append(v0, 1.0))[:3]
            p1 = (mat @ np.append(v1, 1.0))[:3]
            pair = [p0.tolist(), p1.tolist()]
            if edge.select:
                sel_positions.extend(pair)
            else:
                unsel_positions.extend(pair)

        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        shader.uniform_float("viewportSize", (region.width, region.height))

        # Draw unselected edges
        if unsel_positions:
            shader.uniform_float("lineWidth", 1.0)
            shader.uniform_float("color", (0.0, 0.0, 0.0, 1.0))
            batch = batch_for_shader(
                shader, 'LINES', {"pos": unsel_positions},
            )
            batch.draw(shader)

        # Draw selected edges in Blender's theme selection color
        if sel_positions:
            theme_3d = context.preferences.themes[0].view_3d
            sel_color = theme_3d.edge_mode_select
            shader.uniform_float("lineWidth", 1.0)
            shader.uniform_float("color",
                                 (sel_color[0], sel_color[1], sel_color[2], 1.0))
            batch = batch_for_shader(
                shader, 'LINES', {"pos": sel_positions},
            )
            batch.draw(shader)
    finally:
        eval_obj.to_mesh_clear()


# ---------------------------------------------------------------------------
#  Draw callback
# ---------------------------------------------------------------------------

def _draw_grid_overlay():
    context = bpy.context
    if not is_level_design_workspace():
        return

    scene = context.scene
    if not hasattr(scene, 'level_design_props'):
        return
    props = scene.level_design_props
    if not props.show_grid_overlay:
        return

    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return
    if not rv3d.is_perspective:
        return
    if region.width < 1 or region.height < 1:
        return

    if not _ensure_shader():
        return

    # Compute grid size (same logic as modal_draw/utils.py get_grid_size)
    anvil_scale = props.anvil_grid_scale
    if anvil_scale <= 0:
        anvil_scale = 1.0
    from .grid_tools import get_snap_unit_size
    us = scene.unit_settings
    grid_size = anvil_scale * get_snap_unit_size(us.system, us.length_unit, us.scale_length)

    if grid_size <= 0:
        return

    # Use the same matrices the GPU uses for built-in shaders (avoids
    # floating-point divergence that would cause z-fighting).
    view_proj = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()

    # Collect mesh batches
    depsgraph = context.evaluated_depsgraph_get()
    batches = _collect_mesh_batches(depsgraph, _shader)
    if not batches:
        return

    # GPU state: transparent overlay with depth testing against scene
    # geometry.  Matches the approach used by overlap_check.py.
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    gpu.state.face_culling_set('NONE')

    try:
        _shader.bind()
        _shader.uniform_float("viewProjectionMatrix", view_proj)
        _shader.uniform_float("grid_size", grid_size)
        _shader.uniform_float("line_width", 1.5)
        _shader.uniform_float("opacity", 0.3)
        for batch in batches:
            batch.draw(_shader)

        # In edit mode the grid is drawn after Blender's edit-mode edge
        # overlay (POST_VIEW), so the semi-transparent grid covers the
        # edges.  Redraw the active object's edges on top to restore them.
        if context.mode == 'EDIT_MESH':
            _draw_edit_edges_on_top(context, depsgraph, region)
    except Exception as e:
        print(f"Level Design Tools: Grid overlay draw error: {e}")
    finally:
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)
        gpu.state.face_culling_set('NONE')


# ---------------------------------------------------------------------------
#  Handler management
# ---------------------------------------------------------------------------

def _register_draw_handler():
    global _draw_handler
    _unregister_draw_handler()
    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        _draw_grid_overlay, (), 'WINDOW', 'POST_VIEW'
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

class LEVELDESIGN_OT_toggle_grid_overlay(bpy.types.Operator):
    """Toggle grid overlay on all visible geometry"""
    bl_idname = "leveldesign.toggle_grid_overlay"
    bl_label = "Toggle Grid Overlay"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        props = context.scene.level_design_props
        props.show_grid_overlay = not props.show_grid_overlay
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  Registration
# ---------------------------------------------------------------------------

classes = (
    LEVELDESIGN_OT_toggle_grid_overlay,
)

addon_keymaps = []

KEYMAPS_TO_REGISTER = [
    ("Object Mode", 'EMPTY'),
    ("Mesh", 'EMPTY'),
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    _register_draw_handler()

    # Hotkeys
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    for km_name, space_type in KEYMAPS_TO_REGISTER:
        km = kc.keymaps.new(name=km_name, space_type=space_type)

        kmi = km.keymap_items.new(
            "leveldesign.toggle_grid_overlay", 'G', 'PRESS',
            alt=True,
            head=True
        )
        addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    addon_keymaps.clear()

    _unregister_draw_handler()
    global _shader
    _shader = None
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
