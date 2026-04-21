"""Save/load and undo/redo handlers."""

import bmesh
import bpy
from bpy.app.handlers import persistent

from ..core.logging import debug_log
from ..core.face_id import get_face_id_layer
from ..core.uv_layers import sync_uv_map_settings
from ..workspace import setup_addon_workspaces, subscribe_splash_watcher, reset_specialized_template_flag

from .face_cache import (
    face_data_cache, cache_face_data, update_ui_from_selection,
    snapshot_selection, reset as reset_face_cache,
)
from .active_image import (
    update_active_image_from_face, redraw_ui_panels,
    get_active_image, save_previous_image_state, restore_previous_image_state,
)
from .auto_hotspot import (
    get_undo_in_progress, set_undo_in_progress,
    set_auto_hotspot_pending,
)
from .mode_tracking import (
    set_all_grid_scales_to_default, disable_correct_uv_slide,
    subscribe_unit_settings, subscribe_object_mode, reset_mode_tracking,
)
from .file_browser import (
    start_file_browser_watcher, set_was_first_save, get_was_first_save,
    reset as reset_file_browser,
)


# Set True on file load to allow first depsgraph to sync active image from selected face
_file_loaded_into_edit_depsgraph = False


def get_file_loaded_into_edit_depsgraph():
    return _file_loaded_into_edit_depsgraph


def set_file_loaded_into_edit_depsgraph(value):
    global _file_loaded_into_edit_depsgraph
    _file_loaded_into_edit_depsgraph = value


def _clear_file_loaded_flag():
    """Clear the file loaded flag after timeout."""
    global _file_loaded_into_edit_depsgraph
    _file_loaded_into_edit_depsgraph = False


def _clear_undo_flag():
    """Clear the undo flag after depsgraph has processed."""
    set_undo_in_progress(False)


# Operators that abuse the undo stack as their redo-panel rollback mechanism.
# When one of these is the active operator, an undo_pre/undo_post firing is
# the operator adjusting itself, not a user-initiated undo. We must NOT set
# the undo-in-progress flag, or the depsgraph handler will skip UV reprojection
# and produce 0-area UVs.
_UNDO_ABUSING_OPERATORS = {
    "MESH_OT_spin",
}


def _active_operator_abuses_undo():
    op = bpy.context.active_operator
    return op is not None and op.bl_idname in _UNDO_ABUSING_OPERATORS


def _sync_weld_and_snapshot_selection():
    """Sync weld state from BMesh/Mesh to scene props after undo/redo.

    Also snapshots the current face selection so that the next depsgraph
    check_selection_changed call doesn't see a false change and incorrectly
    clear the restored weld state.
    """
    try:
        context = bpy.context
        if context.mode == 'EDIT_MESH':
            obj = context.active_object
            if obj and obj.type == 'MESH':
                bm = bmesh.from_edit_mesh(obj.data)
                from ..operators.weld import sync_weld_props
                sync_weld_props(context, bm)
                snapshot_selection(bm)
        else:
            from ..operators.weld import sync_weld_props
            sync_weld_props(context, None)
            from .face_cache import set_last_selected_face_indices, set_last_active_face_index
            set_last_selected_face_indices(set())
            set_last_active_face_index(-1)
    except Exception:
        from .face_cache import set_last_selected_face_indices, set_last_active_face_index
        set_last_selected_face_indices(set())
        set_last_active_face_index(-1)


def _migrate_legacy_uv_lock():
    """Migrate old per-object anvil_uv_lock boolean to per-UV-map settings."""
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        legacy_lock = obj.get("anvil_uv_lock", False)
        if not legacy_lock:
            continue

        me = obj.data
        if not me.uv_layers:
            continue

        sync_uv_map_settings(obj)
        for setting in obj.anvil_uv_map_settings:
            setting.locked = True

        if "anvil_uv_lock" in obj:
            del obj["anvil_uv_lock"]

        debug_log(f"[Migration] Migrated UV lock for '{obj.name}': locked all {len(me.uv_layers)} UV maps")


@persistent
def on_undo_pre(scene):
    """Handler called before an undo operation."""
    if _active_operator_abuses_undo():
        return
    set_undo_in_progress(True)
    set_auto_hotspot_pending(False)


@persistent
def on_undo_post(scene):
    """Handler called after an undo operation."""
    face_data_cache.clear()
    try:
        context = bpy.context
        if context.mode == 'EDIT_MESH':
            cache_face_data(context)
        else:
            from . import face_cache
            face_cache.last_face_count = 0
            face_cache.last_vertex_count = 0
    except Exception:
        from . import face_cache
        face_cache.last_face_count = 0
        face_cache.last_vertex_count = 0
    _sync_weld_and_snapshot_selection()
    bpy.app.timers.register(_clear_undo_flag, first_interval=0.05)
    from ..operators.fixed_hotspot_overlay import invalidate_overlay as _invalidate_fixed_overlay
    _invalidate_fixed_overlay()
    try:
        context = bpy.context
        saved_prev_name, saved_prev_path = save_previous_image_state()
        update_ui_from_selection(context)
        update_active_image_from_face(context)
        if get_active_image() is None:
            restore_previous_image_state(saved_prev_name, saved_prev_path)
        redraw_ui_panels(context)
    except Exception:
        pass


@persistent
def on_redo_pre(scene):
    """Handler called before a redo operation."""
    if _active_operator_abuses_undo():
        return
    set_undo_in_progress(True)
    set_auto_hotspot_pending(False)


@persistent
def on_redo_post(scene):
    """Handler called after a redo operation."""
    face_data_cache.clear()
    try:
        context = bpy.context
        if context.mode == 'EDIT_MESH':
            cache_face_data(context)
        else:
            from . import face_cache
            face_cache.last_face_count = 0
            face_cache.last_vertex_count = 0
    except Exception:
        from . import face_cache
        face_cache.last_face_count = 0
        face_cache.last_vertex_count = 0
    _sync_weld_and_snapshot_selection()
    bpy.app.timers.register(_clear_undo_flag, first_interval=0.05)
    from ..operators.fixed_hotspot_overlay import invalidate_overlay as _invalidate_fixed_overlay
    _invalidate_fixed_overlay()
    try:
        context = bpy.context
        saved_prev_name, saved_prev_path = save_previous_image_state()
        update_ui_from_selection(context)
        update_active_image_from_face(context)
        if get_active_image() is None:
            restore_previous_image_state(saved_prev_name, saved_prev_path)
        redraw_ui_panels(context)
    except Exception:
        pass


@persistent
def on_save_pre(dummy):
    """Handler called before saving a .blend file."""
    set_was_first_save(not bpy.data.filepath)

    if bpy.data.filepath:
        print("Anvil Level Design: Making all paths relative (pre save)", flush=True)
        bpy.ops.file.make_paths_relative()


@persistent
def on_save_post(dummy):
    """Handler called after saving a .blend file."""
    if get_was_first_save():
        set_was_first_save(False)
        print("Anvil Level Design: Making all paths relative (post save, for first save)", flush=True)
        bpy.ops.file.make_paths_relative()
        print("Anvil Level Design: Triggering second save to apply relative paths", flush=True)
        bpy.ops.wm.save_mainfile()


def _apply_addon_defaults_to_scene():
    """Copy addon preference defaults to the current scene's properties."""
    prefs = bpy.context.preferences.addons.get(__package__.rsplit('.', 1)[0])
    if not prefs:
        return
    pref = prefs.preferences
    props = bpy.context.scene.level_design_props

    props.pixels_per_meter = pref.pref_pixels_per_meter
    props.default_interpolation = pref.pref_default_interpolation
    props.default_texture_as_alpha = pref.pref_default_texture_as_alpha
    props.default_vertex_colors = pref.pref_default_vertex_colors
    props.default_roughness = pref.pref_default_roughness
    props.default_metallic = pref.pref_default_metallic
    props.default_emission_strength = pref.pref_default_emission_strength
    props.default_emission_color = pref.pref_default_emission_color[:]
    props.default_specular = pref.pref_default_specular


@persistent
def on_load_post(dummy):
    """Handler called after a .blend file is loaded."""
    global _file_loaded_into_edit_depsgraph

    reset_specialized_template_flag()
    if not bpy.data.filepath:
        setup_addon_workspaces()
        subscribe_splash_watcher()

    reset_face_cache()
    reset_file_browser()
    _file_loaded_into_edit_depsgraph = True
    reset_mode_tracking()

    # Apply addon preferences as defaults for new (unsaved) files
    if not bpy.data.filepath:
        _apply_addon_defaults_to_scene()

    bpy.app.timers.register(set_all_grid_scales_to_default, first_interval=0.1)
    bpy.app.timers.register(start_file_browser_watcher, first_interval=0.2)
    bpy.app.timers.register(disable_correct_uv_slide, first_interval=0.1)
    bpy.app.timers.register(subscribe_unit_settings, first_interval=0.1)
    bpy.app.timers.register(subscribe_object_mode, first_interval=0.1)
    bpy.app.timers.register(_clear_file_loaded_flag, first_interval=1.0)
    _migrate_legacy_uv_lock()
