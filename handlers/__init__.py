"""Handlers package — event system for the addon.

Re-exports public symbols so existing ``from .handlers import X`` still works.
"""

import bpy

from .face_cache import (  # noqa: F401
    face_data_cache,
    cache_face_data,
    cache_single_face,
    get_cached_layer_data,
    update_ui_from_selection,
    check_selection_changed,
    get_multi_face_mode,
    is_multi_face_unset_scale,
    is_multi_face_unset_rotation,
    is_multi_face_unset_offset,
    mark_multi_face_set_scale,
    mark_multi_face_set_rotation,
    mark_multi_face_set_offset,
    get_all_selected_hotspot,
    get_any_selected_hotspot,
    get_any_selected_fixed_hotspot,
)
from .active_image import (  # noqa: F401
    get_active_image,
    get_previous_image,
    set_active_image,
    set_previous_image,
    update_active_image_from_face,
    get_selected_faces_share_image,
    redraw_ui_panels,
)
from .auto_hotspot import apply_auto_hotspots  # noqa: F401
from .uv_world_scale import (  # noqa: F401
    apply_world_scale_uvs,
    apply_uv_lock,
)
from .file_browser import (  # noqa: F401
    apply_texture_from_file_browser,
    start_file_browser_watcher,
    consolidate_duplicate_materials,
)
from .mode_tracking import (  # noqa: F401
    set_correct_uv_slide,
    set_all_grid_scales_to_default,
    disable_correct_uv_slide,
)
from .lifecycle import (  # noqa: F401
    on_undo_pre,
    on_undo_post,
    on_redo_pre,
    on_redo_post,
    on_save_pre,
    on_save_post,
    on_load_post,
)
from .depsgraph import on_depsgraph_update  # noqa: F401
from . import file_browser as _file_browser
from . import mode_tracking as _mode_tracking


def register():
    _file_browser.register()

    _mode_tracking.subscribe_object_mode()

    if on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)
    if on_save_pre not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(on_save_pre)
    if on_save_post not in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.append(on_save_post)
    if on_undo_pre not in bpy.app.handlers.undo_pre:
        bpy.app.handlers.undo_pre.append(on_undo_pre)
    if on_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(on_undo_post)
    if on_redo_pre not in bpy.app.handlers.redo_pre:
        bpy.app.handlers.redo_pre.append(on_redo_pre)
    if on_redo_post not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(on_redo_post)

    bpy.app.timers.register(set_all_grid_scales_to_default, first_interval=0.1)
    bpy.app.timers.register(start_file_browser_watcher, first_interval=0.2)
    bpy.app.timers.register(disable_correct_uv_slide, first_interval=0.1)
    bpy.app.timers.register(_mode_tracking.subscribe_unit_settings, first_interval=0.1)


def unregister():
    _mode_tracking.unregister_msgbus()

    from .auto_hotspot import reset as reset_auto_hotspot
    reset_auto_hotspot()

    _file_browser.unregister()
    _file_browser.reset()

    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
    if on_save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(on_save_pre)
    if on_save_post in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.remove(on_save_post)
    if on_undo_pre in bpy.app.handlers.undo_pre:
        bpy.app.handlers.undo_pre.remove(on_undo_pre)
    if on_undo_post in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(on_undo_post)
    if on_redo_pre in bpy.app.handlers.redo_pre:
        bpy.app.handlers.redo_pre.remove(on_redo_pre)
    if on_redo_post in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.remove(on_redo_post)

    from .face_cache import reset as reset_face_cache
    reset_face_cache()
    from .active_image import reset as reset_active_image
    reset_active_image()
