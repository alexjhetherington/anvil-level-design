"""Public prefab operator re-exports."""

from ..core.modal_image_grid import (
    image_grid_grid_columns,
    image_grid_hit_test,
    image_grid_preview_icon_scale,
    image_grid_target_cell_width,
    image_grid_widget_unit,
)
from ..prefabs.assets import (  # noqa: F401
    iter_scene_prefab_assets,
    make_all_free_objects_assets as _make_all_free_objects_assets,
    refresh_library_objects as _refresh_library_objects,
    scan_library_prefab_assets as _scan_library_prefab_assets,
)
from ..prefabs.browser import (  # noqa: F401
    LEVELDESIGN_OT_prefab_browser,
    LEVELDESIGN_OT_prefab_browser_close,
    LEVELDESIGN_OT_prefab_browser_fix_layout,
    LEVELDESIGN_OT_prefab_browser_interaction,
    _PREFAB_BROWSER_GRID_SPEC,
    _prefab_browser_3d_view_context,
    _prefab_browser_display_items,
    _prefab_browser_display_items_for_filter,
    prefab_browser_modal,
)
from ..prefabs.operators import (  # noqa: F401
    LEVELDESIGN_OT_prefab_add_library,
    LEVELDESIGN_OT_prefab_clear_asset,
    LEVELDESIGN_OT_prefab_generate_previews,
    LEVELDESIGN_OT_prefab_instantiate,
    LEVELDESIGN_OT_prefab_make_free_objects_assets,
    LEVELDESIGN_OT_prefab_refresh_libraries,
    LEVELDESIGN_OT_prefab_remove_library,
    LEVELDESIGN_OT_prefab_reset_random_transform,
    LEVELDESIGN_OT_prefab_select_asset,
    LEVELDESIGN_OT_set_prefab_mode,
    classes,
    register,
    unregister,
)
from ..prefabs.previews import (  # noqa: F401
    capture_library_previews,
    cleanup_preview_cache,
    ensure_prefab_preview_data,
    invalidate_preview_cache,
    prefab_browser_preview_texture,
)


_prefab_browser_interaction = prefab_browser_modal.interaction
_PREFAB_BROWSER_SCREEN_KEY = prefab_browser_modal.screen_key


def _is_prefab_browser_window(window):
    return prefab_browser_modal.is_window(window)


def _prefab_browser_popup_windows(windows):
    return prefab_browser_modal.popup_windows(windows)


def _prefab_browser_layout_for_state(
        scene,
        search_text,
        region_width,
        region_height,
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        scroll_offset):
    return prefab_browser_modal.layout_for_state(
        scene,
        search_text,
        -1,
        region_width,
        region_height,
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        scroll_offset,
    )


def _prefab_browser_hit_test(rects, metrics, mouse_x, mouse_y):
    return image_grid_hit_test(rects, metrics, mouse_x, mouse_y)


def _prefab_browser_preview_icon_scale(preview_scale_percentage):
    return image_grid_preview_icon_scale(
        preview_scale_percentage,
        _PREFAB_BROWSER_GRID_SPEC,
    )


def _prefab_browser_widget_unit(ui_scale, pixel_size):
    return image_grid_widget_unit(ui_scale, pixel_size)


def _prefab_browser_target_cell_width(preview_scale_percentage, ui_scale, pixel_size):
    return image_grid_target_cell_width(
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        _PREFAB_BROWSER_GRID_SPEC,
    )


def _prefab_browser_grid_columns(
        region_width,
        preview_scale_percentage,
        ui_scale,
        pixel_size):
    return image_grid_grid_columns(
        region_width,
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        0,
        _PREFAB_BROWSER_GRID_SPEC,
    )


def _prefab_browser_short_label(text, preview_scale_percentage):
    max_chars = max(
        3,
        int(_prefab_browser_preview_icon_scale(preview_scale_percentage) * 2.8),
    )
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."
