"""
3D Cursor to Grid Tool

Modal tool that places Blender's 3D cursor on the Anvil grid.
Hold Shift to place off-grid on the surface. Press ESC to cancel.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .modal_draw import snapping, utils
from ..core.workspace_check import is_level_design_workspace

# Visual constants
CROSSHAIR_COLOR = (0.0, 1.0, 0.5, 0.8)
OFFGRID_CROSSHAIR_COLOR = (1.0, 0.3, 0.3, 0.8)


class LEVELDESIGN_OT_cursor_to_grid(bpy.types.Operator):
    """Place 3D cursor on the Anvil grid. Shift=off-grid surface placement"""
    bl_idname = "leveldesign.cursor_to_grid"
    bl_label = "3D Cursor to Grid"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        self._current_snap = None
        self._current_normal = None
        self._is_2d_view = utils.is_2d_view(context)
        self._is_off_grid = False
        self._axis_lock_normal = None
        self._axis_lock_plane_point = None

        # Save original cursor position for ESC restore
        self._original_cursor = context.scene.cursor.location.copy()

        # Register draw handler
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback_3d, (context,), 'WINDOW', 'POST_VIEW'
        )

        context.window.cursor_modal_set('CROSSHAIR')
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

        # Shift/Ctrl press/release — update snap
        if event.type in ('LEFT_SHIFT', 'RIGHT_SHIFT',
                          'LEFT_CTRL', 'RIGHT_CTRL'):
            # Ctrl release clears axis lock
            if event.type in ('LEFT_CTRL', 'RIGHT_CTRL') and event.value == 'RELEASE':
                self._axis_lock_normal = None
                self._axis_lock_plane_point = None
            self._update_snap(context, event)
            utils.tag_redraw_all_3d_views()
            return {'PASS_THROUGH'}

        # Left click — place cursor
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._current_snap is not None:
                context.scene.cursor.location = self._current_snap.copy()
                self._cleanup(context)
                return {'FINISHED'}
            return {'RUNNING_MODAL'}

        # Right click to cancel
        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            self._cleanup(context)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def _update_snap(self, context, event):
        """Update the snapped position under the mouse cursor."""
        shift_held = event.shift
        ctrl_held = event.ctrl
        self._is_off_grid = shift_held and not ctrl_held

        if self._is_2d_view:
            self._update_snap_2d(context, event, shift_held)
        else:
            self._update_snap_3d(context, event, shift_held, ctrl_held)

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

    def _update_snap_3d(self, context, event, off_grid, ctrl_held=False):
        """Snap in 3D perspective view."""
        if ctrl_held and not off_grid:
            # Axis lock: capture the current face plane on first ctrl frame
            if self._axis_lock_normal is None:
                snapped, face_normal, _, _ = snapping.calculate_first_vertex_snap_3d(
                    context, event
                )
                if snapped is not None and face_normal is not None:
                    self._axis_lock_normal = face_normal.copy()
                    self._axis_lock_plane_point = snapped.copy()

            # Use locked plane if available
            if self._axis_lock_normal is not None:
                snapped, face_normal, _, _ = snapping.calculate_first_vertex_snap_3d_on_plane(
                    context, event,
                    self._axis_lock_plane_point, self._axis_lock_normal
                )
                if snapped is not None:
                    self._current_snap = snapped
                    self._current_normal = face_normal
                else:
                    self._current_snap = None
                    self._current_normal = None
                return

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

    def _cleanup(self, context):
        """Clean up resources."""
        if self._draw_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None

        context.window.cursor_modal_restore()
        context.area.header_text_set(None)
        utils.tag_redraw_all_3d_views()

    def _update_header(self, context):
        """Update header text."""
        context.area.header_text_set(
            "3D Cursor to Grid: Click to place | Shift=Off-Grid | Ctrl=Lock Plane | ESC=Cancel"
        )

    def _draw_callback_3d(self, context):
        """Draw crosshair at current snap point."""
        if self._current_snap is None or self._current_normal is None:
            return

        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
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
        color = OFFGRID_CROSSHAIR_COLOR if self._is_off_grid else CROSSHAIR_COLOR
        shader.uniform_float("color", color)
        batch = batch_for_shader(shader, 'LINES', {"pos": cross_verts})
        batch.draw(shader)

        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)


# Keymap items to track for cleanup
_addon_keymaps = []


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_cursor_to_grid)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='3D View', space_type='VIEW_3D')
        kmi = km.keymap_items.new(
            LEVELDESIGN_OT_cursor_to_grid.bl_idname,
            type='X',
            value='PRESS',
            shift=True,
        )
        _addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    bpy.utils.unregister_class(LEVELDESIGN_OT_cursor_to_grid)
