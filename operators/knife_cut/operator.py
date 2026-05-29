"""
Knife Cut Tool - Main Modal Operator

Grid-snapped knife cut for level design workflows.
Snaps cut points to the Anvil grid by default. Hold Shift to go off-grid.
Hold Alt to snap to closest vertex (including previously placed cut points).
"""

import bpy
import bmesh
import gpu
import blf
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from bpy_extras.view3d_utils import location_3d_to_region_2d

from . import geometry
from ..modal_draw import snapping, utils
from ...core.workspace_check import is_level_design_workspace

# Visual constants
CUT_LINE_COLOR = (1.0, 0.6, 0.0, 0.9)
CUT_POINT_COLOR = (1.0, 0.8, 0.0, 1.0)
PREVIEW_LINE_COLOR = (1.0, 0.6, 0.0, 0.5)
CROSSHAIR_COLOR = (0.0, 1.0, 0.5, 0.8)
OFFGRID_CROSSHAIR_COLOR = (1.0, 0.3, 0.3, 0.8)
VERTEX_SNAP_CROSSHAIR_COLOR = (0.3, 0.8, 1.0, 1.0)


class MESH_OT_knife_cut(bpy.types.Operator):
    """Grid-snapped knife cut tool for level design"""
    bl_idname = "leveldesign.knife_cut"
    bl_label = "Knife Cut"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (
            is_level_design_workspace() and
            context.active_object is not None and
            context.active_object.type == 'MESH' and
            context.mode == 'EDIT_MESH'
        )

    def invoke(self, context, event):
        self._cut_points = []
        self._current_snap = None
        self._current_normal = None
        self._is_2d_view = utils.is_2d_view(context)
        self._is_off_grid = False
        self._is_vertex_snap = False

        # Register draw handlers
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback_3d, (context,), 'WINDOW', 'POST_VIEW'
        )
        self._draw_handler_px = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback_px, (context,), 'WINDOW', 'POST_PIXEL'
        )

        context.window.cursor_modal_set('KNIFE')
        context.window_manager.modal_handler_add(self)
        self._update_header(context)
        self._update_snap(context, event)
        utils.tag_redraw_all_3d_views()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        self._is_2d_view = utils.is_2d_view(context)

        # ESC to cancel
        if event.type == 'ESC' and event.value == 'PRESS':
            self._cleanup(context)
            return {'CANCELLED'}

        # Mouse move
        if event.type == 'MOUSEMOVE':
            self._update_snap(context, event)
            utils.tag_redraw_all_3d_views()
            return {'RUNNING_MODAL'}

        # Shift/Alt press/release — update snap to reflect mode toggle
        if event.type in ('LEFT_SHIFT', 'RIGHT_SHIFT', 'LEFT_ALT', 'RIGHT_ALT'):
            self._update_snap(context, event)
            utils.tag_redraw_all_3d_views()
            return {'PASS_THROUGH'}

        # Left click — add cut point
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._current_snap is not None:
                self._cut_points.append(
                    (self._current_snap.copy(), self._current_normal.copy())
                )
                self._update_header(context)
                utils.tag_redraw_all_3d_views()
            return {'RUNNING_MODAL'}

        # Enter/Space to confirm
        if event.type in ('RET', 'NUMPAD_ENTER', 'SPACE') and event.value == 'PRESS':
            return self._confirm(context)

        # Right click to confirm (if enough points) or cancel
        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            if len(self._cut_points) >= 2:
                return self._confirm(context)
            self._cleanup(context)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def _update_snap(self, context, event):
        """Update the snapped position under the mouse cursor."""
        alt_held = event.alt
        shift_held = event.shift
        self._is_off_grid = shift_held and not alt_held
        self._is_vertex_snap = alt_held

        if alt_held:
            self._update_snap_vertex(context, event)
        elif self._is_2d_view:
            self._update_snap_2d(context, event, shift_held)
        else:
            self._update_snap_3d(context, event, shift_held)

    def _update_snap_2d(self, context, event, off_grid):
        """Snap in 2D ortho view."""
        plane_data = utils.get_2d_view_plane(context)
        if plane_data is None:
            self._current_snap = None
            self._current_normal = None
            return

        plane_point, plane_normal, _, _ = plane_data
        point = utils.mouse_to_3d_on_plane(context, event, plane_point, plane_normal)
        if point is None:
            self._current_snap = None
            self._current_normal = None
            return

        if off_grid:
            self._current_snap = point
        else:
            grid_size = utils.get_grid_size(context)
            self._current_snap = snapping.snap_to_grid(point, grid_size)
        self._current_normal = plane_normal

    def _update_snap_3d(self, context, event, off_grid):
        """Snap in 3D perspective view."""
        if off_grid:
            hit, location, normal, _, _, _ = utils.raycast_scene(context, event)
            if hit and location is not None:
                self._current_snap = location
                self._current_normal = normal
            else:
                self._current_snap = None
                self._current_normal = None
        else:
            snapped, face_normal, obj, _ = snapping.calculate_first_vertex_snap_3d(
                context, event
            )
            if snapped is not None:
                self._current_snap = snapped
                self._current_normal = face_normal
            else:
                self._current_snap = None
                self._current_normal = None

    def _update_snap_vertex(self, context, event):
        """Snap to closest mesh vertex or previously placed cut point."""
        region = context.region
        rv3d = context.region_data
        if rv3d is None:
            self._current_snap = None
            self._current_normal = None
            return

        mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        best_pos = None
        best_normal = None
        best_dist_sq = float('inf')

        # Check mesh vertices
        obj = context.active_object
        if obj and obj.type == 'MESH' and context.mode == 'EDIT_MESH':
            bm = bmesh.from_edit_mesh(obj.data)
            world = obj.matrix_world
            for v in bm.verts:
                co_world = world @ v.co
                screen = location_3d_to_region_2d(region, rv3d, co_world)
                if screen is None:
                    continue
                d = (screen - mouse).length_squared
                if d < best_dist_sq:
                    best_dist_sq = d
                    best_pos = co_world
                    # Approximate normal from connected faces
                    if v.link_faces:
                        best_normal = v.link_faces[0].normal.copy()
                        best_normal = (world.to_3x3() @ best_normal).normalized()
                    else:
                        best_normal = Vector((0, 0, 1))

        # Check previously placed cut points
        for pt_world, pt_normal in self._cut_points:
            screen = location_3d_to_region_2d(region, rv3d, pt_world)
            if screen is None:
                continue
            d = (screen - mouse).length_squared
            if d < best_dist_sq:
                best_dist_sq = d
                best_pos = pt_world
                best_normal = pt_normal

        # Only snap if within a reasonable screen distance (50px)
        if best_pos is not None and best_dist_sq < 50 * 50:
            self._current_snap = best_pos.copy()
            self._current_normal = best_normal.copy()
        else:
            # Fall back to grid snap
            self._is_vertex_snap = False
            if self._is_2d_view:
                self._update_snap_2d(context, event, False)
            else:
                self._update_snap_3d(context, event, False)

    def _confirm(self, context):
        """Execute the knife cut."""
        if len(self._cut_points) < 2:
            self.report({'WARNING'}, "Need at least 2 points for a cut")
            self._cleanup(context)
            return {'CANCELLED'}

        obj = context.active_object
        success, message = geometry.execute_knife_cut(
            context, obj, self._cut_points
        )

        if success:
            self.report({'INFO'}, message)
        else:
            self.report({'ERROR'}, message)

        self._cleanup(context)
        return {'FINISHED'}

    def _cleanup(self, context):
        """Clean up resources."""
        if self._draw_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        if self._draw_handler_px is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler_px, 'WINDOW')
            self._draw_handler_px = None

        context.window.cursor_modal_restore()
        context.area.header_text_set(None)
        utils.tag_redraw_all_3d_views()

    def _update_header(self, context):
        """Update header text."""
        n = len(self._cut_points)
        if n == 0:
            text = "Knife Cut: Click to place first point | Shift=Off-Grid | ESC=Cancel"
        else:
            text = (f"Knife Cut: {n} point{'s' if n != 1 else ''} | "
                    f"Click=Add point | Enter/RMB=Confirm | Shift=Off-Grid | ESC=Cancel")
        context.area.header_text_set(text)

    def _draw_callback_3d(self, context):
        """Draw cut lines, points, and crosshair in 3D view."""
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        gpu.state.point_size_set(8.0)

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')

        # Confirmed cut segments
        if len(self._cut_points) >= 2:
            coords = [p[0] for p in self._cut_points]
            line_verts = []
            for i in range(len(coords) - 1):
                line_verts.extend([coords[i], coords[i + 1]])
            shader.uniform_float("color", CUT_LINE_COLOR)
            batch = batch_for_shader(shader, 'LINES', {"pos": line_verts})
            batch.draw(shader)

        # Confirmed points
        if self._cut_points:
            point_coords = [p[0] for p in self._cut_points]
            shader.uniform_float("color", CUT_POINT_COLOR)
            batch = batch_for_shader(shader, 'POINTS', {"pos": point_coords})
            batch.draw(shader)

        # Preview line from last point to current snap
        if self._cut_points and self._current_snap is not None:
            last_pt = self._cut_points[-1][0]
            shader.uniform_float("color", PREVIEW_LINE_COLOR)
            batch = batch_for_shader(shader, 'LINES', {
                "pos": [last_pt, self._current_snap]
            })
            batch.draw(shader)

        # Crosshair at current snap
        if self._current_snap is not None and self._current_normal is not None:
            grid_size = utils.get_grid_size(context)
            size = grid_size * 0.3

            if self._is_2d_view:
                t1, t2 = utils.get_2d_view_tangents(context)
            else:
                t1, t2 = utils.get_snap_aligned_tangents(self._current_normal)

            pos = self._current_snap
            cross_verts = [
                pos - t1 * size, pos + t1 * size,
                pos - t2 * size, pos + t2 * size,
            ]
            if self._is_vertex_snap:
                color = VERTEX_SNAP_CROSSHAIR_COLOR
            elif self._is_off_grid:
                color = OFFGRID_CROSSHAIR_COLOR
            else:
                color = CROSSHAIR_COLOR
            shader.uniform_float("color", color)
            batch = batch_for_shader(shader, 'LINES', {"pos": cross_verts})
            batch.draw(shader)

        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)

    def _draw_callback_px(self, context):
        """Draw 2D status indicator."""
        region = context.region
        n = len(self._cut_points)
        if n == 0:
            return

        font_id = 0
        label = f"Knife Cut: {n} point{'s' if n != 1 else ''}"
        blf.size(font_id, 14)
        blf.color(font_id, 1.0, 0.8, 0.0, 0.9)
        tw, th = blf.dimensions(font_id, label)
        blf.position(font_id, region.width / 2 - tw / 2, 30, 0)
        blf.draw(font_id, label)


def register():
    bpy.utils.register_class(MESH_OT_knife_cut)


def unregister():
    bpy.utils.unregister_class(MESH_OT_knife_cut)
