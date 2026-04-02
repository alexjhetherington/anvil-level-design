"""Face orientation overlay toggle and correct-UV-slide memory hack."""

import ctypes

import bpy

from ..core.logging import debug_log
from ..core.workspace_check import is_level_design_workspace


# Memory offset of uvcalc_flag in ToolSettings struct (found experimentally for Blender 5.0)
_UVCALC_FLAG_OFFSET = 100
_UVCALC_TRANSFORM_CORRECT_SLIDE = 4  # 1 << 2


# HACK of the century. We need to turn off correct_uv slide but blender doesn't expose it
# So we do direct memory management. See /scripts for how to find the offset
# This is brittle as anything :)
def set_correct_uv_slide(enabled: bool):
    """Directly set the UVCALC_TRANSFORM_CORRECT_SLIDE flag in tool settings.

    This flag controls whether edge/vertex slide operations correct UVs.
    Blender's Python API doesn't expose this flag, so we modify it directly in memory.
    """
    try:
        ts = bpy.context.scene.tool_settings
        ts_ptr = ts.as_pointer()
        flag_ptr = ctypes.cast(ts_ptr + _UVCALC_FLAG_OFFSET, ctypes.POINTER(ctypes.c_ushort))

        if enabled:
            flag_ptr.contents.value |= _UVCALC_TRANSFORM_CORRECT_SLIDE
        else:
            flag_ptr.contents.value &= ~_UVCALC_TRANSFORM_CORRECT_SLIDE
    except Exception as e:
        print(f"Anvil Level Design: Failed to set correct_uv_slide — offset {_UVCALC_FLAG_OFFSET} may be wrong for this platform/Blender version: {e}", flush=True)


# Face orientation overlay tracking for vertex paint / sculpt modes
_last_tracked_mode = None
_FACE_ORIENTATION_MODES = {'PAINT_VERTEX', 'SCULPT'}
_face_orientation_msgbus_owner = object()


def _get_saved_face_orientation():
    """Read persisted face orientation state. Returns bool or None."""
    val = bpy.context.scene.level_design_props.saved_face_orientation
    if val == -1:
        return None
    return bool(val)


def _set_saved_face_orientation(value):
    """Write persisted face orientation state. Accepts bool or None."""
    if value is None:
        bpy.context.scene.level_design_props.saved_face_orientation = -1
    else:
        bpy.context.scene.level_design_props.saved_face_orientation = int(value)


def _on_object_mode_changed():
    """Msgbus callback when object mode changes. Toggles face orientation overlay."""
    global _last_tracked_mode
    try:
        if not is_level_design_workspace():
            return
        current_mode = bpy.context.mode
        if current_mode == _last_tracked_mode:
            return
        was_in = _last_tracked_mode in _FACE_ORIENTATION_MODES if _last_tracked_mode else False
        now_in = current_mode in _FACE_ORIENTATION_MODES
        _last_tracked_mode = current_mode
        debug_log(f"[FaceOrientation] Mode changed: was_in={was_in} now_in={now_in}")
        saved = _get_saved_face_orientation()
        if now_in and not was_in:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                if saved is None:
                                    _set_saved_face_orientation(space.overlay.show_face_orientation)
                                space.overlay.show_face_orientation = True
                        area.tag_redraw()
            debug_log(f"[FaceOrientation] Enabled (saved={_get_saved_face_orientation()})")
        elif was_in and not now_in:
            if saved is not None:
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D':
                            for space in area.spaces:
                                if space.type == 'VIEW_3D':
                                    space.overlay.show_face_orientation = saved
                            area.tag_redraw()
                debug_log(f"[FaceOrientation] Restored (was={saved})")
                _set_saved_face_orientation(None)
    except Exception:
        pass


def subscribe_object_mode():
    """Subscribe to object mode changes via msgbus."""
    try:
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.Object, "mode"),
            owner=_face_orientation_msgbus_owner,
            args=(),
            notify=_on_object_mode_changed,
            options={'PERSISTENT'},
        )
        debug_log("[FaceOrientation] Subscribed to object mode changes")
    except Exception as e:
        print(f"Anvil Level Design: Failed to subscribe to object mode changes: {e}", flush=True)


# Msgbus subscription owner for unit setting changes
_unit_msgbus_owner = object()


def _on_unit_settings_changed():
    """Msgbus callback: re-apply anvil grid scale when unit settings change."""
    from ..operators.grid_tools import apply_anvil_grid_scale

    try:
        scene = bpy.context.scene
        if not hasattr(scene, 'level_design_props'):
            return
        anvil_scale = scene.level_design_props.anvil_grid_scale
        if anvil_scale == 0.0:
            anvil_scale = 1.0
        unit_settings = scene.unit_settings
        apply_anvil_grid_scale(anvil_scale, unit_settings.system, unit_settings.length_unit)
    except Exception:
        pass


def subscribe_unit_settings():
    """Subscribe to unit setting changes via bpy.msgbus."""
    try:
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.UnitSettings, "system"),
            owner=_unit_msgbus_owner,
            args=(),
            notify=_on_unit_settings_changed,
        )
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.UnitSettings, "length_unit"),
            owner=_unit_msgbus_owner,
            args=(),
            notify=_on_unit_settings_changed,
        )
    except Exception:
        pass


def set_all_grid_scales_to_default():
    """Set anvil grid scale to 1 and apply with unit conversion on all 3D views."""
    from ..operators.grid_tools import apply_anvil_grid_scale

    scene = bpy.context.scene
    if hasattr(scene, 'level_design_props'):
        scene.level_design_props.anvil_grid_scale = 1.0
    unit_settings = scene.unit_settings
    apply_anvil_grid_scale(1.0, unit_settings.system, unit_settings.length_unit)


def disable_correct_uv_slide():
    """Disable the correct_uv flag for slide operations via direct memory access."""
    set_correct_uv_slide(False)


def unregister_msgbus():
    """Clear msgbus subscriptions and restore face orientation state."""
    global _last_tracked_mode

    bpy.msgbus.clear_by_owner(_face_orientation_msgbus_owner)
    saved = _get_saved_face_orientation()
    if saved is not None:
        try:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                space.overlay.show_face_orientation = saved
        except Exception:
            pass
        _set_saved_face_orientation(None)
    _last_tracked_mode = None

    bpy.msgbus.clear_by_owner(_unit_msgbus_owner)


def reset_mode_tracking():
    """Reset mode tracking state (called on file load)."""
    global _last_tracked_mode
    if _get_saved_face_orientation() is not None:
        _last_tracked_mode = bpy.context.mode
    else:
        _last_tracked_mode = None
