"""
3D Cursor to Grid tool.

Modal tool that places Blender's 3D cursor on the Anvil grid or surface.
"""

import bpy
from bpy.props import BoolProperty, FloatVectorProperty
from mathutils import Vector

from .modal_draw import preview, snapping, utils, view_context
from .modal_draw.default_grid_pivot import (
    calculate_default_grid_snap_2d,
    calculate_default_grid_snap_3d,
    selected_vertex_world_coords,
)
from ..core.workspace_check import is_level_design_workspace


class LEVELDESIGN_OT_cursor_to_grid(bpy.types.Operator):
    """Place Blender's 3D cursor on the Anvil grid"""
    bl_idname = "leveldesign.cursor_to_grid"
    bl_label = "3D Cursor to Grid"
    bl_options = {'REGISTER', 'UNDO'}

    action_cursor_location: FloatVectorProperty(
        size=3,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    action_has_cursor_location: BoolProperty(
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        self._active_view_target = None
        self._last_mouse_window_pos = None
        self._last_grid_size = None
        view_target, view_event, _view_changed = self._resolve_view_event(
            context, event
        )
        if view_target is None:
            self.report({'ERROR'}, "No 3D View available")
            return {'CANCELLED'}

        self._current_snap = None
        self._current_normal = None
        self._axis_lock_normal = None
        self._axis_lock_plane_point = None

        with context.temp_override(**view_target.override_kwargs()):
            self._is_2d_view = utils.is_2d_view(context)
            self._is_grid_snapped = utils.is_snapping_enabled(context)
            self._last_grid_size = utils.get_grid_size(context)

            self._preview = preview.get_preview()
            self._preview.register_handlers()
            self._preview.set_state('FIRST_VERTEX')

            context.window.cursor_modal_set('CROSSHAIR')
            context.window_manager.modal_handler_add(self)
            self._last_mouse_window_pos = (event.mouse_x, event.mouse_y)
            self._update_snap(context, view_event)
            self._update_header(context)
            utils.tag_redraw_all_3d_views()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if not is_level_design_workspace():
            self._cleanup(context)
            return {'CANCELLED'}

        view_target, view_event, view_changed = self._resolve_view_event(
            context, event
        )
        if view_target is None:
            self._cleanup(context)
            return {'CANCELLED'}

        with context.temp_override(**view_target.override_kwargs()):
            self._is_2d_view = utils.is_2d_view(context)

            if view_changed:
                self._update_header(context)

            current_grid_size = utils.get_grid_size(context)
            if current_grid_size != self._last_grid_size:
                self._last_grid_size = current_grid_size
                fake_event = self._synthetic_event_for_last_mouse(
                    self._axis_lock_normal is not None,
                    event.shift,
                    event.alt,
                )
                if fake_event is not None:
                    self._update_snap(context, fake_event)
                    utils.tag_redraw_all_3d_views()

            if event.type == 'ESC' and event.value == 'PRESS':
                self._cleanup(context)
                return {'CANCELLED'}

            if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
                self._cleanup(context)
                return {'CANCELLED'}

            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                self._last_mouse_window_pos = (event.mouse_x, event.mouse_y)
                self._update_snap(context, view_event)
                if self._current_snap is None:
                    return {'RUNNING_MODAL'}
                self.action_cursor_location = self._current_snap
                self.action_has_cursor_location = True
                result = self.execute(context)
                self._cleanup(context)
                return result

            if event.type == 'MOUSEMOVE':
                self._last_mouse_window_pos = (event.mouse_x, event.mouse_y)
                self._update_snap(context, view_event)
                self._update_header(context)
                utils.tag_redraw_all_3d_views()
                return {'RUNNING_MODAL'}

            if event.type in ('LEFT_CTRL', 'RIGHT_CTRL'):
                if event.value == 'RELEASE':
                    self._axis_lock_normal = None
                    self._axis_lock_plane_point = None
                self._update_snap(context, view_event)
                self._update_header(context)
                utils.tag_redraw_all_3d_views()
                return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        if not self.action_has_cursor_location:
            return {'CANCELLED'}
        context.scene.cursor.location = Vector(self.action_cursor_location)
        self.report({'INFO'}, "3D cursor placed")
        return {'FINISHED'}

    def _update_snap(self, context, event):
        self._is_grid_snapped = utils.is_snapping_enabled(context)

        if self._is_2d_view:
            self._update_snap_2d(context, event)
        else:
            self._update_snap_3d(context, event)

        self._update_preview(context)

    def _update_snap_2d(self, context, event):
        snapped, plane_normal = calculate_default_grid_snap_2d(
            context, event, self._selected_vertex_world_coords(context)
        )
        self._current_snap = snapped
        self._current_normal = plane_normal

    def _update_snap_3d(self, context, event):
        if not event.ctrl:
            self._axis_lock_normal = None
            self._axis_lock_plane_point = None

        if event.ctrl:
            if self._axis_lock_normal is None:
                snapped, face_normal, _, _ = self._calculate_first_vertex_snap_3d(
                    context, event
                )
                if snapped is not None and face_normal is not None:
                    self._axis_lock_normal = face_normal.copy()
                    self._axis_lock_plane_point = snapped.copy()

            if self._axis_lock_normal is not None:
                snapped, face_normal, _, _ = snapping.calculate_first_vertex_snap_3d_on_plane(
                    context,
                    event,
                    self._axis_lock_plane_point,
                    self._axis_lock_normal,
                )
                self._current_snap = snapped
                self._current_normal = face_normal
                return

        snapped, face_normal, _, _ = self._calculate_first_vertex_snap_3d(
            context, event
        )
        self._current_snap = snapped
        self._current_normal = face_normal

    def _selected_vertex_world_coords(self, context):
        return selected_vertex_world_coords(context.active_object, context.mode)

    def _calculate_first_vertex_snap_3d(self, context, event):
        result = snapping.calculate_first_vertex_snap_3d(context, event)
        snapped, _, _, _ = result
        if snapped is not None:
            return result

        return calculate_default_grid_snap_3d(
            context, event, self._selected_vertex_world_coords(context)
        )

    def _cleanup(self, context):
        preview.cleanup_preview()

        target = getattr(self, "_active_view_target", None)
        try:
            if target is not None:
                target.window.cursor_modal_restore()
            else:
                context.window.cursor_modal_restore()
        except ReferenceError:
            context.window.cursor_modal_restore()

        self._clear_header_for_target(target)
        try:
            if context.area is not None:
                context.area.header_text_set(None)
        except ReferenceError:
            pass
        self._active_view_target = None
        utils.tag_redraw_all_3d_views()

    def _clear_header_for_target(self, target):
        if target is None:
            return
        try:
            target.area.header_text_set(None)
        except ReferenceError:
            return

    def _set_active_view_target(self, target):
        previous = getattr(self, "_active_view_target", None)
        changed = previous is None or not target.matches(previous)
        if changed:
            self._clear_header_for_target(previous)
        self._active_view_target = target
        return changed

    def _resolve_view_event(self, context, event):
        target = view_context.view_target_under_mouse(
            context.window, event.mouse_x, event.mouse_y
        )
        if target is None:
            previous = getattr(self, "_active_view_target", None)
            if previous is not None and previous.is_live():
                target = previous
        if target is None:
            target = view_context.view_target_for_area(
                context.window,
                context.screen,
                context.area,
                context.region,
            )
        if target is None:
            return (None, None, False)

        changed = self._set_active_view_target(target)
        view_event = view_context.view_mouse_event_for_target(event, target)
        return (target, view_event, changed)

    def _synthetic_event_for_last_mouse(self, ctrl, shift, alt):
        target = getattr(self, "_active_view_target", None)
        last_mouse = getattr(self, "_last_mouse_window_pos", None)
        if target is None or last_mouse is None:
            return None

        return view_context.synthetic_mouse_event(
            last_mouse[0],
            last_mouse[1],
            target,
            ctrl,
            shift,
            alt,
        )

    def _update_header(self, context):
        snap_mode = "Grid Snap On" if self._is_grid_snapped else "Grid Snap Off"
        context.area.header_text_set(
            f"3D Cursor to Grid: Click to place | {snap_mode} | Ctrl=Lock Plane | ESC=Cancel"
        )

    def _update_preview(self, context):
        if self._current_snap is None or self._current_normal is None:
            self._preview.update_snap_point(None, None, None)
            self._preview.clear_face_grid()
            return

        if self._is_2d_view:
            tangent1, tangent2 = utils.get_2d_view_tangents(context)
            self._preview.update_snap_point(
                self._current_snap, tangent1, tangent2
            )
            self._preview.clear_face_grid()
        else:
            tangent1, tangent2 = utils.get_snap_aligned_tangents(
                self._current_normal
            )
            self._preview.update_snap_point(
                self._current_snap, tangent1, tangent2
            )
            grid_size = utils.get_grid_size(context)
            self._preview.update_face_grid(
                self._current_snap, self._current_normal, grid_size, False
            )


_addon_keymaps = []

KEYMAPS_TO_REGISTER = [
    ("Object Mode", 'EMPTY'),
    ("Mesh", 'EMPTY'),
]


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_cursor_to_grid)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    for km_name, space_type in KEYMAPS_TO_REGISTER:
        km = kc.keymaps.new(name=km_name, space_type=space_type)
        kmi = km.keymap_items.new(
            LEVELDESIGN_OT_cursor_to_grid.bl_idname,
            type='X',
            value='PRESS',
            shift=True,
            head=True,
        )
        _addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    _addon_keymaps.clear()

    preview.cleanup_preview()
    bpy.utils.unregister_class(LEVELDESIGN_OT_cursor_to_grid)
