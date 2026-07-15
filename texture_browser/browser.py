"""Floating texture browser hosted inside a temporary Preferences window."""

import os
import time

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator

from ..core.library import is_library_object
from ..core.modal_image_grid import (
    ImageGridSpec,
    PreferencesImageGridModal,
    draw_image_grid_rect,
    draw_image_grid_text,
    draw_image_grid_texture_display,
    image_grid_texture_bounds,
)
from ..core.logging import debug_log
from ..core.workspace_check import (
    HOTSPOT_MAPPING_WORKSPACE_NAME,
    LEVEL_DESIGN_WORKSPACE_NAME,
    is_hotspot_mapping_workspace,
    is_level_design_workspace,
)
from ..handlers.active_image import (
    redraw_ui_panels,
    set_active_image,
)
from .apply import apply_texture_path_to_selection
from .previews import (
    cleanup_texture_browser_preview_cache,
    invalidate_texture_browser_preview_cache,
    texture_browser_cached_preview_texture,
    texture_browser_preview_texture,
    texture_browser_preview_texture_needs_load,
)


_TEXTURE_BROWSER_SCREEN_KEY = "anvil_texture_browser_screen"
_TEXTURE_BROWSER_POPUP_KEY = "anvil_texture_browser_popup"
_TEXTURE_BROWSER_CONTENT_PANEL = "USERPREF_PT_addons"
_TEXTURE_BROWSER_GRID_SPEC = ImageGridSpec(4.5, 7.5, 2)
_TEXTURE_BROWSER_FOLDER_VIEW = -1
_TEXTURE_BROWSER_EMBEDDED_RECOVERY_INTERVAL_SECONDS = 0.25
_TEXTURE_BROWSER_COMMON_IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tga",
    ".tif",
    ".tiff",
    ".webp",
    ".exr",
    ".hdr",
    ".dds",
)
_TEXTURE_BROWSER_IMAGE_SUFFIXES = frozenset(_TEXTURE_BROWSER_COMMON_IMAGE_SUFFIXES)
_folder_scan_cache = {}
_collection_scan_cache = {}
_folder_scan_generation = 0
_popup_source_workspace_name = ""

_TEXTURE_BROWSER_LOCATE_HIGHLIGHT_SECONDS = 0.6
_TEXTURE_BROWSER_LOCATE_HIGHLIGHT_ALPHA = 1.0
_TEXTURE_BROWSER_APPLIED_FEEDBACK_SECONDS = 1.6
_TEXTURE_BROWSER_APPLIED_FEEDBACK_FADE_SECONDS = 0.35

_TEXTURE_BROWSER_WORKSPACE_NAMES = {
    LEVEL_DESIGN_WORKSPACE_NAME,
    HOTSPOT_MAPPING_WORKSPACE_NAME,
}


def _texture_browser_workspace_is_allowed():
    return is_level_design_workspace() or is_hotspot_mapping_workspace()


def _workspace_name_from_workspace(workspace):
    return getattr(workspace, "name", "") if workspace is not None else ""


def _current_workspace_name():
    return _workspace_name_from_workspace(getattr(bpy.context, "workspace", None))


def _set_popup_source_workspace_name(workspace_name):
    global _popup_source_workspace_name
    _popup_source_workspace_name = workspace_name


def _texture_browser_activation_workspace_name():
    if texture_browser_modal.is_popup_window(getattr(bpy.context, "window", None)):
        if _popup_source_workspace_name:
            return _popup_source_workspace_name
    return _current_workspace_name()


def _texture_browser_popup_source_is_level_design(window):
    return (
        texture_browser_modal.is_popup_window(window)
        and _popup_source_workspace_name == LEVEL_DESIGN_WORKSPACE_NAME
    )


def _addon_preferences():
    package_name = __package__.split(".", 1)[0]
    addon = bpy.context.preferences.addons.get(package_name)
    if addon is None:
        return None
    return addon.preferences


def _default_favorite_name(path):
    folder_path = os.path.normpath(_display_path(path))
    folder_names = []
    while folder_path and len(folder_names) < 2:
        folder_name = os.path.basename(folder_path)
        if folder_name:
            folder_names.append(folder_name)
        parent_path = os.path.dirname(folder_path)
        if parent_path == folder_path:
            break
        folder_path = parent_path
    return f".../{'/'.join(reversed(folder_names))}/"


def _ensure_texture_browser_preferences(prefs):
    if prefs is None:
        return
    for favorite in prefs.texture_browser_favorites:
        if not favorite.name.strip():
            favorite.name = _default_favorite_name(favorite.path)
    if prefs.texture_browser_filters_initialized:
        return
    prefs.texture_browser_include_suffixes.clear()
    for suffix in _TEXTURE_BROWSER_COMMON_IMAGE_SUFFIXES:
        item = prefs.texture_browser_include_suffixes.add()
        item.suffix = suffix
    prefs.texture_browser_active_include_index = 0
    prefs.texture_browser_active_exclude_index = 0
    prefs.texture_browser_filters_initialized = True


def _normal_path(path):
    if not path:
        return ""
    return os.path.normcase(os.path.abspath(bpy.path.abspath(path)))


def _display_path(path):
    return os.path.abspath(bpy.path.abspath(path)) if path else ""


def _folder_exists(path):
    return bool(path) and os.path.isdir(_display_path(path))


def _file_exists(path):
    return bool(path) and os.path.isfile(_display_path(path))


class _TextureBrowserAnimations:
    """Own transient visual feedback drawn inside the texture browser."""

    def __init__(self):
        self.locate_path = ""
        self.locate_started = 0.0
        self.applied_path = ""
        self.applied_face_count = 0
        self.applied_started = 0.0

    def start_locate(self, filepath):
        self.locate_path = _normal_path(filepath)
        self.locate_started = time.perf_counter()

    def start_applied(self, filepath, face_count):
        self.applied_path = _normal_path(filepath)
        self.applied_face_count = face_count
        self.applied_started = time.perf_counter()

    def locate_alpha(self, filepath):
        if _normal_path(filepath) != self.locate_path:
            return 0.0
        elapsed = time.perf_counter() - self.locate_started
        if elapsed >= _TEXTURE_BROWSER_LOCATE_HIGHLIGHT_SECONDS:
            return 0.0
        remaining = 1.0 - elapsed / _TEXTURE_BROWSER_LOCATE_HIGHLIGHT_SECONDS
        return _TEXTURE_BROWSER_LOCATE_HIGHLIGHT_ALPHA * remaining

    def applied_feedback(self, filepath):
        if _normal_path(filepath) != self.applied_path:
            return None
        elapsed = time.perf_counter() - self.applied_started
        if elapsed >= _TEXTURE_BROWSER_APPLIED_FEEDBACK_SECONDS:
            return None
        fade_started = (
            _TEXTURE_BROWSER_APPLIED_FEEDBACK_SECONDS
            - _TEXTURE_BROWSER_APPLIED_FEEDBACK_FADE_SECONDS
        )
        if elapsed <= fade_started:
            alpha = 1.0
        else:
            alpha = (
                _TEXTURE_BROWSER_APPLIED_FEEDBACK_SECONDS - elapsed
            ) / _TEXTURE_BROWSER_APPLIED_FEEDBACK_FADE_SECONDS
        return self.applied_face_count, alpha

    def tick(self):
        now = time.perf_counter()
        redraw = False
        if self.locate_path:
            redraw = True
            if now - self.locate_started >= _TEXTURE_BROWSER_LOCATE_HIGHLIGHT_SECONDS:
                self.locate_path = ""
        if self.applied_path:
            elapsed = now - self.applied_started
            fade_started = (
                _TEXTURE_BROWSER_APPLIED_FEEDBACK_SECONDS
                - _TEXTURE_BROWSER_APPLIED_FEEDBACK_FADE_SECONDS
            )
            if elapsed >= _TEXTURE_BROWSER_APPLIED_FEEDBACK_SECONDS:
                self.applied_path = ""
                self.applied_face_count = 0
                redraw = True
            elif elapsed >= fade_started:
                redraw = True
        return redraw


_texture_browser_animations = _TextureBrowserAnimations()


def _blend_home_folder():
    try:
        blend_filepath = bpy.data.filepath
    except AttributeError:
        blend_filepath = ""

    if blend_filepath:
        folder = os.path.dirname(bpy.path.abspath(blend_filepath))
        if folder:
            return os.path.abspath(folder)
    cwd = os.path.abspath(os.getcwd())
    drive, _tail = os.path.splitdrive(cwd)
    if drive:
        return drive + os.sep
    return os.path.abspath(os.sep)


def _preferred_texture_browser_folder(prefs):
    if prefs is not None and _folder_exists(prefs.texture_browser_last_folder_path):
        return os.path.abspath(bpy.path.abspath(prefs.texture_browser_last_folder_path))
    return _blend_home_folder()


def _remember_texture_browser_folder(folder_path):
    prefs = _addon_preferences()
    if prefs is None:
        return
    if not _folder_exists(folder_path):
        return
    prefs.texture_browser_last_folder_path = os.path.abspath(bpy.path.abspath(folder_path))


def _ensure_window_manager_folder(window_manager):
    if _folder_exists(window_manager.anvil_texture_browser_folder_path):
        _remember_texture_browser_folder(window_manager.anvil_texture_browser_folder_path)
        return
    window_manager.anvil_texture_browser_folder_path = _preferred_texture_browser_folder(
        _addon_preferences()
    )


def _load_texture_browser_persistent_settings(window_manager):
    prefs = _addon_preferences()
    _ensure_texture_browser_preferences(prefs)
    if prefs is not None:
        window_manager.anvil_texture_browser_preview_scale = prefs.texture_browser_preview_scale
    _ensure_window_manager_folder(window_manager)


def restore_texture_browser_settings_after_load():
    window_manager = bpy.context.window_manager
    if not hasattr(window_manager, "anvil_texture_browser_folder_path"):
        return
    if not hasattr(window_manager, "anvil_texture_browser_preview_scale"):
        return
    _load_texture_browser_persistent_settings(window_manager)
    texture_browser_modal.settings_update(
        window_manager,
        window_manager.windows,
        True,
    )


def restore_texture_browser_folder_after_load():
    restore_texture_browser_settings_after_load()


def _texture_browser_search_text(window_manager):
    return window_manager.anvil_texture_browser_search


def _texture_browser_collection_filter(window_manager):
    return window_manager.anvil_texture_browser_collection_index


def _texture_browser_preview_scale_value(window_manager):
    return window_manager.anvil_texture_browser_preview_scale


def _texture_browser_scroll_offset(window_manager):
    return window_manager.anvil_texture_browser_scroll_offset


def _set_texture_browser_scroll_offset(window_manager, scroll_offset):
    window_manager.anvil_texture_browser_scroll_offset = scroll_offset


def _suffix_set(collection):
    suffixes = set()
    for item in collection:
        suffix = item.suffix.strip().lower()
        if suffix:
            suffixes.add(suffix)
    return suffixes


def _texture_browser_filter_sets():
    prefs = _addon_preferences()
    _ensure_texture_browser_preferences(prefs)
    if prefs is None:
        return set(_TEXTURE_BROWSER_COMMON_IMAGE_SUFFIXES), set()
    return (
        _suffix_set(prefs.texture_browser_include_suffixes),
        _suffix_set(prefs.texture_browser_exclude_suffixes),
    )


def _suffix_allowed(filename, include_suffixes, exclude_suffixes):
    filename = filename.lower()
    if any(filename.endswith(suffix) for suffix in exclude_suffixes):
        return False
    if include_suffixes and not any(filename.endswith(suffix) for suffix in include_suffixes):
        return False
    return True


def _item_matches_search(filename, folder_label, search_text):
    text = search_text.strip().lower()
    if not text:
        return True
    return text in filename.lower() or text in folder_label.lower()


def _is_image_filepath(filepath):
    return os.path.splitext(filepath)[1].lower() in _TEXTURE_BROWSER_IMAGE_SUFFIXES


def _folder_scan_version(folder_path):
    try:
        mtime = os.path.getmtime(folder_path)
    except OSError:
        mtime = "missing"
    return f"{mtime}:{_folder_scan_generation}"


def _scan_folder_items(folder_path, include_suffixes, exclude_suffixes):
    folder_path = _display_path(folder_path)
    if not os.path.isdir(folder_path):
        return []

    key = (
        _normal_path(folder_path),
        tuple(sorted(include_suffixes)),
        tuple(sorted(exclude_suffixes)),
        _folder_scan_version(folder_path),
    )
    cached = _folder_scan_cache.get(key)
    if cached is not None:
        return cached

    folder_items = []
    file_items = []
    try:
        entries = list(os.scandir(folder_path))
    except OSError:
        entries = []

    for entry in entries:
        try:
            if entry.is_dir():
                filepath = os.path.abspath(entry.path)
                folder_items.append({
                    "filepath": filepath,
                    "filename": entry.name,
                    "folder_label": os.path.basename(folder_path) or folder_path,
                    "is_folder": True,
                    "is_image": False,
                    "suffix": "",
                    "source": 'DIRECTORY',
                })
                continue
            if not entry.is_file():
                continue
        except OSError:
            continue
        suffix = os.path.splitext(entry.name)[1].lower()
        if not _suffix_allowed(entry.name, include_suffixes, exclude_suffixes):
            continue
        filepath = os.path.abspath(entry.path)
        file_items.append({
            "filepath": filepath,
            "filename": entry.name,
            "folder_label": os.path.basename(folder_path) or folder_path,
            "is_folder": False,
            "is_image": _is_image_filepath(filepath),
            "suffix": suffix,
            "source": 'FOLDER',
        })

    folder_items.sort(key=lambda item: item["filename"].lower())
    file_items.sort(key=lambda item: item["filename"].lower())
    items = folder_items + file_items
    _folder_scan_cache[key] = items
    return items


def _collection_items(collection, include_suffixes, exclude_suffixes):
    key = (
        tuple(_normal_path(file_item.filepath) for file_item in collection.files),
        tuple(sorted(include_suffixes)),
        tuple(sorted(exclude_suffixes)),
        _folder_scan_generation,
    )
    cached = _collection_scan_cache.get(key)
    if cached is not None:
        return cached

    items = []
    seen_paths = set()
    for file_item in collection.files:
        filepath = _display_path(file_item.filepath)
        normal = _normal_path(filepath)
        if normal in seen_paths:
            continue
        seen_paths.add(normal)
        if not os.path.isfile(filepath):
            continue
        suffix = os.path.splitext(filepath)[1].lower()
        if not _suffix_allowed(os.path.basename(filepath), include_suffixes, exclude_suffixes):
            continue
        folder_path = os.path.dirname(filepath)
        items.append({
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "folder_label": os.path.basename(folder_path) or folder_path,
            "is_folder": False,
            "is_image": _is_image_filepath(filepath),
            "suffix": suffix,
            "source": 'COLLECTION',
        })
    items.sort(key=lambda item: (item["folder_label"].lower(), item["filename"].lower()))
    _collection_scan_cache[key] = items
    return items


def _texture_browser_display_items_for_filter(scene, search_text, collection_filter):
    include_suffixes, exclude_suffixes = _texture_browser_filter_sets()
    prefs = _addon_preferences()
    folder_view = False
    if prefs is None:
        base_items = []
    elif 0 <= collection_filter < len(prefs.texture_browser_collections):
        base_items = _collection_items(
            prefs.texture_browser_collections[collection_filter],
            include_suffixes,
            exclude_suffixes,
        )
    else:
        folder_view = True
        base_items = _scan_folder_items(
            bpy.context.window_manager.anvil_texture_browser_folder_path,
            include_suffixes,
            exclude_suffixes,
        )

    if folder_view:
        return [
            item for item in base_items
            if item["is_folder"] or _item_matches_search(
                item["filename"],
                item["folder_label"],
                search_text,
            )
        ]

    return [
        item for item in base_items
        if _item_matches_search(item["filename"], item["folder_label"], search_text)
    ]


def _texture_browser_display_items(scene, search_text):
    return _texture_browser_display_items_for_filter(
        scene,
        search_text,
        _TEXTURE_BROWSER_FOLDER_VIEW,
    )


def _texture_browser_rect_for_item(item, x, y, width, height):
    filepath = item["filepath"]
    return {
        "x": x,
        "y": y,
        "w": width,
        "h": height,
        "filepath": filepath,
        "filename": item["filename"],
        "folder_label": item["folder_label"],
        "is_folder": item["is_folder"],
        "is_image": item["is_image"],
        "suffix": item["suffix"],
        "source": item["source"],
        "key": f"{item['source']}:{_normal_path(filepath)}",
    }


def _current_collection_name(window_manager):
    collection = _current_collection(window_manager)
    if collection is not None:
        return collection.name
    return ""


def _current_collection(window_manager):
    prefs = _addon_preferences()
    if prefs is None:
        return None
    index = window_manager.anvil_texture_browser_collection_index
    if 0 <= index < len(prefs.texture_browser_collections):
        return prefs.texture_browser_collections[index]
    return None


def _current_favorite(window_manager):
    if window_manager.anvil_texture_browser_collection_index != _TEXTURE_BROWSER_FOLDER_VIEW:
        return None
    prefs = _addon_preferences()
    if prefs is None:
        return None
    current_path = _normal_path(window_manager.anvil_texture_browser_folder_path)
    for favorite in prefs.texture_browser_favorites:
        if _normal_path(favorite.path) == current_path:
            return favorite
    return None


def _texture_browser_home_label():
    if bpy.data.filepath:
        return "Go To .blend"
    return "Go To Root"


def _draw_texture_browser_navigation(layout, scene, window_manager):
    prefs = _addon_preferences()
    _ensure_texture_browser_preferences(prefs)

    layout.label(text="Texture Browser", icon='IMAGE_DATA')
    row = layout.row()
    row.operator(
        "leveldesign.texture_browser_home",
        text=_texture_browser_home_label(),
        icon='HOME',
    )

    layout.separator()
    header = layout.row(align=True)
    header.label(text="Favourite Folders")
    header.operator("leveldesign.texture_browser_add_favorite", text="", icon='ADD')

    if prefs is not None:
        if not prefs.texture_browser_favorites:
            disabled = layout.row()
            disabled.enabled = False
            disabled.label(text="No favourites", icon='INFO')
        for index, favorite in enumerate(prefs.texture_browser_favorites):
            row = layout.row(align=True)
            op = row.operator(
                "leveldesign.texture_browser_set_favorite",
                text=favorite.name,
                icon='FILE_FOLDER',
            )
            op.favorite_index = index
            remove_op = row.operator(
                "leveldesign.texture_browser_remove_favorite",
                text="",
                icon='REMOVE',
            )
            remove_op.favorite_index = index

    layout.separator()
    header = layout.row(align=True)
    header.label(text="Collections")
    header.operator("leveldesign.texture_browser_add_collection", text="", icon='ADD')

    if prefs is not None:
        if not prefs.texture_browser_collections:
            disabled = layout.row()
            disabled.enabled = False
            disabled.label(text="No collections", icon='INFO')
        for index, collection in enumerate(prefs.texture_browser_collections):
            row = layout.row(align=True)
            op = row.operator(
                "leveldesign.texture_browser_set_collection",
                text=collection.name or f"Collection {index + 1}",
                icon='OUTLINER_COLLECTION',
                depress=(window_manager.anvil_texture_browser_collection_index == index),
            )
            op.collection_index = index
            remove_op = row.operator(
                "leveldesign.texture_browser_remove_collection",
                text="",
                icon='REMOVE',
            )
            remove_op.collection_index = index


def _draw_texture_browser_header(
        layout,
        scene,
        window_manager,
        preferences,
        active_section_is_compatible):
    collection = _current_collection(window_manager)
    favorite = _current_favorite(window_manager)

    layout.row().template_header()
    row = layout.row(align=True)
    title_row = row.row(align=True)
    title_row.scale_x = 0.85
    title_row.label(text="Texture Browser", icon='IMAGE_DATA')
    row.separator(factor=2.0)
    search_row = row.row(align=True)
    search_row.scale_x = 2.0
    search_row.prop(window_manager, "anvil_texture_browser_search", text="", icon='VIEWZOOM')
    row.separator(factor=0.35)

    path_controls = row.row(align=True)
    path_controls.enabled = (window_manager.anvil_texture_browser_collection_index
                             == _TEXTURE_BROWSER_FOLDER_VIEW)
    path_controls.operator("leveldesign.texture_browser_parent_folder", text="", icon='TRIA_UP')
    folder_path_row = path_controls.row(align=True)
    folder_path_row.scale_x = 2.2
    folder_path_row.prop(window_manager, "anvil_texture_browser_folder_path", text="")
    path_controls.operator(
        "leveldesign.texture_browser_choose_folder",
        text="",
        icon='FILE_FOLDER',
    )

    if collection is not None:
        collection_row = row.row(align=True)
        collection_row.scale_x = 1.2
        collection_row.prop(collection, "name", text="", icon='OUTLINER_COLLECTION')
    elif favorite is not None:
        favorite_row = row.row(align=True)
        favorite_row.scale_x = 1.2
        favorite_row.prop(favorite, "name", text="", icon='FILE_FOLDER')

    row.separator(factor=0.5)
    row.prop(window_manager, "anvil_texture_browser_preview_scale", text="", slider=True)
    actions_row = row.row(align=True)
    actions_row.operator("leveldesign.texture_browser_refresh", text="", icon='FILE_REFRESH')
    actions_row.operator("leveldesign.texture_browser_edit_filters", text="", icon='FILTER')
    if not active_section_is_compatible:
        sub = row.row(align=True)
        sub.alert = True
        sub.operator("leveldesign.texture_browser_fix_layout", text="", icon='SHADERFX')


def _draw_texture_browser_content(layout, scene, window_manager, region_width, ui_scale, pixel_size):
    items = _texture_browser_display_items_for_filter(
        scene,
        window_manager.anvil_texture_browser_search,
        window_manager.anvil_texture_browser_collection_index,
    )
    collection_name = _current_collection_name(window_manager)

    if not collection_name and not _folder_exists(window_manager.anvil_texture_browser_folder_path):
        col = layout.column()
        col.separator()
        row = col.row()
        row.alignment = 'CENTER'
        row.label(text="Folder not found.", icon='ERROR')
        return

    if not items:
        col = layout.column()
        col.separator()
        row = col.row()
        row.alignment = 'CENTER'
        if collection_name:
            row.label(text="No files match the current filters.", icon='VIEWZOOM')
        else:
            row.label(text="No folders or files match the current filters.", icon='VIEWZOOM')
        return


def _texture_browser_collection_button_rect(rect, metrics):
    size = max(18, int(round(metrics["widget_unit"] * 0.9)))
    pad = max(5, int(round(metrics["widget_unit"] * 0.25)))
    return {
        "x": rect["x"] + rect["w"] - size - pad,
        "y": rect["y"] + rect["h"] - size - pad,
        "w": size,
        "h": size,
    }


def _point_in_rect(rect, x, y):
    return (
        rect["x"] <= x <= rect["x"] + rect["w"]
        and rect["y"] <= y <= rect["y"] + rect["h"]
    )


def _draw_texture_browser_file_icon(icon_x, icon_y, icon_w, icon_h, suffix, font_size):
    body_w = icon_w * 0.58
    body_h = icon_h * 0.72
    body_x = icon_x + (icon_w - body_w) / 2
    body_y = icon_y + (icon_h - body_h) / 2
    fold = min(body_w, body_h) * 0.22
    draw_image_grid_rect(body_x, body_y, body_w, body_h, (0.16, 0.18, 0.20, 0.88))
    draw_image_grid_rect(body_x + body_w - fold, body_y + body_h - fold, fold, fold, (0.34, 0.38, 0.43, 0.88))
    label = suffix.upper().lstrip(".") or "FILE"
    draw_image_grid_text(
        label,
        body_x + 3,
        body_y + body_h * 0.42,
        body_w - 6,
        font_size,
        (0.82, 0.86, 0.90, 0.95),
        'CENTER',
    )


def _draw_texture_browser_folder_icon(icon_x, icon_y, icon_w, icon_h):
    body_w = icon_w * 0.74
    body_h = icon_h * 0.54
    body_x = icon_x + (icon_w - body_w) / 2
    body_y = icon_y + (icon_h - body_h) / 2
    tab_w = body_w * 0.36
    tab_h = body_h * 0.24
    tab_x = body_x + body_w * 0.08
    tab_y = body_y + body_h
    draw_image_grid_rect(tab_x, tab_y, tab_w, tab_h, (0.84, 0.61, 0.25, 0.96))
    draw_image_grid_rect(body_x, body_y, body_w, body_h, (0.94, 0.72, 0.32, 0.96))
    draw_image_grid_rect(
        body_x,
        body_y + body_h * 0.58,
        body_w,
        body_h * 0.12,
        (1.0, 0.84, 0.46, 0.72),
    )


def _draw_texture_browser_locate_outline(
        x,
        y,
        width,
        height,
        metrics,
        alpha):
    thickness = max(1, int(round(metrics["widget_unit"] * 0.07)))
    color = (1.0, 1.0, 1.0, alpha)
    draw_image_grid_rect(
        x - thickness,
        y - thickness,
        width + thickness * 2,
        thickness,
        color,
    )
    draw_image_grid_rect(
        x - thickness,
        y + height,
        width + thickness * 2,
        thickness,
        color,
    )
    draw_image_grid_rect(x - thickness, y, thickness, height, color)
    draw_image_grid_rect(x + width, y, thickness, height, color)


def _draw_texture_browser_applied_feedback(
        x,
        y,
        width,
        height,
        metrics,
        face_count,
        alpha):
    inset = max(4, min(width, height) * 0.06)
    badge_width = min(
        width - inset * 2,
        max(metrics["widget_unit"] * 3.2, width * 0.68),
    )
    badge_height = min(
        height - inset * 2,
        max(metrics["widget_unit"] * 1.55, height * 0.24),
    )
    badge_x = x + (width - badge_width) / 2
    badge_y = y + (height - badge_height) / 2
    draw_image_grid_rect(
        badge_x,
        badge_y,
        badge_width,
        badge_height,
        (0.10, 0.10, 0.10, 0.90 * alpha),
    )
    count_font_size = max(11, int(round(badge_height * 0.43)))
    draw_image_grid_text(
        str(face_count),
        badge_x,
        badge_y + badge_height * 0.48,
        badge_width,
        count_font_size,
        (0.96, 0.96, 0.96, alpha),
        'CENTER',
    )
    face_label = "FACE TEXTURED" if face_count == 1 else "FACES TEXTURED"
    label_font_size = max(7, int(round(badge_height * 0.23)))
    draw_image_grid_text(
        face_label,
        badge_x,
        badge_y + badge_height * 0.12,
        badge_width,
        label_font_size,
        (0.72, 0.72, 0.72, alpha),
        'CENTER',
    )


def _draw_texture_browser_cell(rect, metrics):
    import gpu

    if rect["y"] + rect["h"] < metrics["grid_bottom"]:
        return
    if rect["y"] > metrics["grid_top"]:
        return

    x = rect["x"]
    y = rect["y"]
    width = rect["w"]
    height = rect["h"]
    is_hovered = "hovered_key" in metrics and rect["key"] == metrics["hovered_key"]
    padding = max(8, int(round(metrics["widget_unit"] * 0.45)))
    icon_space = min(metrics["icon_size"], int(width - padding * 2))
    label_y = y + padding + metrics["line_height"] + 4
    meta_y = y + padding + 2
    icon_y = label_y + metrics["line_height"] + 6
    icon_height = max(1, min(icon_space, int(y + height - padding - icon_y)))
    icon_x = x + (width - icon_space) / 2
    ui_scale = metrics["ui_scale"]
    meta_font_size = 11 * ui_scale
    no_preview_font_size = 12 * ui_scale
    label_font_size = 13 * ui_scale
    action_font_size = 15 * ui_scale

    gpu.state.blend_set('ALPHA')
    try:
        texture_info = None
        if is_hovered:
            draw_image_grid_rect(x, y, width, height, (0.24, 0.34, 0.46, 0.42))
        else:
            draw_image_grid_rect(x, y, width, height, (0.08, 0.09, 0.10, 0.20))

        if rect["is_folder"]:
            _draw_texture_browser_folder_icon(icon_x, icon_y, icon_space, icon_height)
        elif rect["is_image"]:
            texture_info = texture_browser_cached_preview_texture(rect["filepath"])
            if texture_info is not None:
                draw_image_grid_texture_display(
                    texture_info,
                    icon_x,
                    icon_y,
                    icon_space,
                    icon_height,
                )
            elif texture_browser_preview_texture_needs_load(rect["filepath"]):
                draw_image_grid_rect(icon_x, icon_y, icon_space, icon_height, (0.10, 0.10, 0.10, 0.35))
            else:
                draw_image_grid_rect(icon_x, icon_y, icon_space, icon_height, (0.10, 0.10, 0.10, 0.35))
                draw_image_grid_text(
                    "No preview",
                    icon_x,
                    icon_y + icon_height / 2 - 6,
                    icon_space,
                    no_preview_font_size,
                    (0.70, 0.70, 0.70, 0.90),
                    'CENTER',
                )
        else:
            _draw_texture_browser_file_icon(
                icon_x,
                icon_y,
                icon_space,
                icon_height,
                rect["suffix"],
                meta_font_size,
            )

        highlight_alpha = _texture_browser_animations.locate_alpha(rect["filepath"])
        if highlight_alpha > 0.0:
            highlight_x = icon_x
            highlight_y = icon_y
            highlight_width = icon_space
            highlight_height = icon_height
            if texture_info is not None:
                (
                    _texture,
                    highlight_x,
                    highlight_y,
                    highlight_width,
                    highlight_height,
                ) = image_grid_texture_bounds(
                    texture_info,
                    icon_x,
                    icon_y,
                    icon_space,
                    icon_height,
                )
            _draw_texture_browser_locate_outline(
                highlight_x,
                highlight_y,
                highlight_width,
                highlight_height,
                metrics,
                highlight_alpha,
            )

        applied_feedback = _texture_browser_animations.applied_feedback(rect["filepath"])
        if applied_feedback is not None:
            face_count, feedback_alpha = applied_feedback
            _draw_texture_browser_applied_feedback(
                icon_x,
                icon_y,
                icon_space,
                icon_height,
                metrics,
                face_count,
                feedback_alpha,
            )

        if not rect["is_folder"]:
            button_rect = _texture_browser_collection_button_rect(rect, metrics)
            draw_image_grid_rect(
                button_rect["x"],
                button_rect["y"],
                button_rect["w"],
                button_rect["h"],
                (0.05, 0.06, 0.07, 0.86),
            )
            draw_image_grid_text(
                "+",
                button_rect["x"],
                button_rect["y"] + 3,
                button_rect["w"],
                action_font_size,
                (0.95, 0.97, 1.0, 1.0),
                'CENTER',
            )

        draw_image_grid_text(
            rect["filename"],
            x + padding + 1,
            label_y - 1,
            width - padding * 2,
            label_font_size,
            (0.0, 0.0, 0.0, 0.75),
            'CENTER',
        )
        draw_image_grid_text(
            rect["filename"],
            x + padding,
            label_y,
            width - padding * 2,
            label_font_size,
            (0.96, 0.96, 0.96, 1.0),
            'CENTER',
        )
        draw_image_grid_text(
            rect["folder_label"],
            x + padding + 1,
            meta_y - 1,
            width - padding * 2,
            meta_font_size,
            (0.0, 0.0, 0.0, 0.70),
            'CENTER',
        )
        draw_image_grid_text(
            rect["folder_label"],
            x + padding,
            meta_y,
            width - padding * 2,
            meta_font_size,
            (0.68, 0.70, 0.73, 1.0),
            'CENTER',
        )
    finally:
        gpu.state.blend_set('NONE')


def _texture_browser_region_for_area(area, preferred_region):
    if preferred_region is not None:
        try:
            region_is_live = any(region == preferred_region for region in area.regions)
            if preferred_region.type == 'WINDOW' and region_is_live:
                return preferred_region
        except ReferenceError:
            pass

    try:
        regions = list(area.regions)
    except ReferenceError:
        return None
    return next((region for region in regions if region.type == 'WINDOW'), None)


def _texture_browser_3d_view_context_for_area(window, area, preferred_region):
    try:
        if area.type != 'VIEW_3D':
            return None
    except ReferenceError:
        return None

    region = _texture_browser_region_for_area(area, preferred_region)
    if region is None:
        return None

    try:
        space = area.spaces.active
    except ReferenceError:
        return None
    # Let Blender derive the screen from the target window. Explicitly
    # overriding it fails while the active modal context uses a temporary screen.
    return {
        "window": window,
        "area": area,
        "region": region,
        "space_data": space,
    }


def _texture_browser_3d_view_context_for_window_area(window, area, preferred_region):
    try:
        screen = getattr(window, "screen", None)
    except ReferenceError:
        return None
    if screen is None or area is None:
        return None
    try:
        area_is_live = any(candidate == area for candidate in screen.areas)
        if not area_is_live:
            return None
    except ReferenceError:
        return None
    return _texture_browser_3d_view_context_for_area(
        window,
        area,
        preferred_region,
    )


def _texture_browser_first_3d_view_context_for_window(window):
    try:
        screen = getattr(window, "screen", None)
    except ReferenceError:
        return None
    if screen is None:
        return None
    try:
        areas = list(screen.areas)
    except ReferenceError:
        return None
    for area in areas:
        view_context = _texture_browser_3d_view_context_for_area(
            window,
            area,
            None,
        )
        if view_context is not None:
            return view_context
    return None


def _texture_browser_source_3d_view_context(windows):
    source_window = texture_browser_modal.interaction.get("source_window")
    if source_window is None:
        return None
    if not texture_browser_modal.is_window_live(source_window, windows):
        return None
    if texture_browser_modal.is_popup_window(source_window):
        return None

    source_area = texture_browser_modal.interaction.get("source_area")
    source_region = texture_browser_modal.interaction.get("source_region")
    view_context = _texture_browser_3d_view_context_for_window_area(
        source_window,
        source_area,
        source_region,
    )
    if view_context is not None:
        return view_context
    return _texture_browser_first_3d_view_context_for_window(source_window)


def _texture_browser_3d_view_context(windows):
    source_view_context = _texture_browser_source_3d_view_context(windows)
    if source_view_context is not None:
        return source_view_context

    for window in windows:
        if texture_browser_modal.is_popup_window(window):
            continue
        view_context = _texture_browser_first_3d_view_context_for_window(window)
        if view_context is not None:
            return view_context
    return None


def _texture_browser_image_editor_context_for_area(window, area, preferred_region):
    try:
        if area.type != 'IMAGE_EDITOR':
            return None
    except ReferenceError:
        return None

    region = _texture_browser_region_for_area(area, preferred_region)
    if region is None:
        return None

    try:
        space = area.spaces.active
    except ReferenceError:
        return None
    return {
        "window": window,
        "area": area,
        "region": region,
        "space_data": space,
    }


def _texture_browser_image_editor_context_for_window_area(window, area, preferred_region):
    try:
        screen = getattr(window, "screen", None)
    except ReferenceError:
        return None
    if screen is None or area is None:
        return None
    try:
        area_is_live = any(candidate == area for candidate in screen.areas)
        if not area_is_live:
            return None
    except ReferenceError:
        return None
    return _texture_browser_image_editor_context_for_area(
        window,
        area,
        preferred_region,
    )


def _texture_browser_first_image_editor_context_for_window(window):
    try:
        screen = getattr(window, "screen", None)
    except ReferenceError:
        return None
    if screen is None:
        return None
    try:
        areas = list(screen.areas)
    except ReferenceError:
        return None
    for area in areas:
        view_context = _texture_browser_image_editor_context_for_area(
            window,
            area,
            None,
        )
        if view_context is not None:
            return view_context
    return None


def _texture_browser_source_image_editor_context(windows):
    source_window = texture_browser_modal.interaction.get("source_window")
    if source_window is None:
        return None
    if not texture_browser_modal.is_window_live(source_window, windows):
        return None
    if texture_browser_modal.is_popup_window(source_window):
        return None

    source_area = texture_browser_modal.interaction.get("source_area")
    source_region = texture_browser_modal.interaction.get("source_region")
    view_context = _texture_browser_image_editor_context_for_window_area(
        source_window,
        source_area,
        source_region,
    )
    if view_context is not None:
        return view_context
    return _texture_browser_first_image_editor_context_for_window(source_window)


def _texture_browser_image_editor_context(windows):
    source_view_context = _texture_browser_source_image_editor_context(windows)
    if source_view_context is not None:
        return source_view_context

    current_window = getattr(bpy.context, "window", None)
    if (current_window is not None
            and not texture_browser_modal.is_popup_window(current_window)):
        view_context = _texture_browser_first_image_editor_context_for_window(current_window)
        if view_context is not None:
            return view_context

    for window in windows:
        if texture_browser_modal.is_popup_window(window):
            continue
        if window == current_window:
            continue
        view_context = _texture_browser_first_image_editor_context_for_window(window)
        if view_context is not None:
            return view_context
    return None


def _load_texture_browser_image(filepath):
    if not filepath:
        return None
    if not os.path.isfile(filepath):
        return None
    try:
        return bpy.data.images.load(filepath, check_existing=True)
    except RuntimeError:
        return None


def _texture_browser_open_image_in_editor(filepath):
    image = _load_texture_browser_image(filepath)
    if image is None:
        print("Anvil Hotspots: Could not load image", flush=True)
        return {'RUNNING_MODAL'}

    view_context = _texture_browser_image_editor_context(bpy.context.window_manager.windows)
    if view_context is None:
        print("Anvil Hotspots: No Image Editor available for texture selection", flush=True)
        return {'RUNNING_MODAL'}

    from ..handlers.active_image import set_previous_image

    set_previous_image(image)
    space = view_context["space_data"]
    try:
        space.image = image
    except (AttributeError, TypeError, ReferenceError):
        print("Anvil Hotspots: Could not set the Image Editor texture", flush=True)
        return {'RUNNING_MODAL'}

    scene = getattr(bpy.context, "scene", None)
    if scene is not None and hasattr(scene, "hotspot_mapping_props"):
        scene.hotspot_mapping_props.active_texture = image.name

    area = view_context["area"]
    try:
        area.tag_redraw()
    except ReferenceError:
        pass
    return {'RUNNING_MODAL'}


def _texture_browser_activate_item(hit):
    if hit["is_folder"]:
        if not _set_folder(bpy.context.window_manager, hit["filepath"]):
            print(f"Anvil Level Design: Folder not found: {hit['filename']}", flush=True)
        return {'RUNNING_MODAL'}

    mouse_x = hit.get("mouse_x")
    mouse_y = hit.get("mouse_y")
    if mouse_x is not None and mouse_y is not None:
        region = texture_browser_modal.interaction.get("region")
        metrics = None
        if region is not None:
            preferences = bpy.context.preferences
            metrics, _rects = texture_browser_modal.layout_for_state(
                bpy.context.scene,
                bpy.context.window_manager.anvil_texture_browser_search,
                bpy.context.window_manager.anvil_texture_browser_collection_index,
                region.width,
                region.height,
                bpy.context.window_manager.anvil_texture_browser_preview_scale,
                preferences.system.ui_scale,
                preferences.system.pixel_size,
                texture_browser_modal.interaction["scroll_offset"],
            )
        if metrics is not None:
            button_rect = _texture_browser_collection_button_rect(hit, metrics)
            if _point_in_rect(button_rect, mouse_x, mouse_y):
                bpy.ops.leveldesign.texture_browser_add_file_to_collection(
                    'INVOKE_DEFAULT',
                    filepath=hit["filepath"],
                )
                return {'RUNNING_MODAL'}

    if not hit["is_image"]:
        print(f"Anvil Level Design: No image preview for {hit['filename']}", flush=True)
        return {'RUNNING_MODAL'}

    if _texture_browser_activation_workspace_name() == HOTSPOT_MAPPING_WORKSPACE_NAME:
        return _texture_browser_open_image_in_editor(hit["filepath"])

    view_context = _texture_browser_3d_view_context(bpy.context.window_manager.windows)
    if view_context is None:
        print("Anvil Level Design: No 3D View available for texture application", flush=True)
        return {'RUNNING_MODAL'}

    with bpy.context.temp_override(**view_context):
        bpy.ops.leveldesign.texture_browser_apply_file(filepath=hit["filepath"])
    return {'RUNNING_MODAL'}


def _texture_browser_warm_texture_item(item):
    if not item["is_image"]:
        return False
    if not texture_browser_preview_texture_needs_load(item["filepath"]):
        return False
    texture_browser_preview_texture(item["filepath"])
    return True


def _poll_scene(scene):
    return _texture_browser_workspace_is_allowed() and scene is not None


def _poll_scene_mode(context):
    return _poll_scene(getattr(context, "scene", None))


def _area_is_preferences(area):
    try:
        return area.type == 'PREFERENCES' or getattr(area, "ui_type", "") == 'PREFERENCES'
    except ReferenceError:
        return False


def _area_is_workspace_editor(area):
    try:
        if _area_is_preferences(area):
            return False
        return area.type not in {'TOPBAR', 'STATUSBAR'}
    except ReferenceError:
        return False


def _screen_has_docked_workspace_area(screen):
    if screen is None:
        return False
    try:
        areas = list(screen.areas)
    except ReferenceError:
        return False
    return any(_area_is_workspace_editor(area) for area in areas)


def _context_is_texture_browser_addon_preferences(workspace_name, active_section, area, screen):
    if workspace_name not in _TEXTURE_BROWSER_WORKSPACE_NAMES:
        return False
    if active_section != 'ADDONS':
        return False
    if area is None:
        return False
    if not _area_is_preferences(area):
        return False
    return _screen_has_docked_workspace_area(screen)


texture_browser_modal = PreferencesImageGridModal(
    "texture browser",
    _TEXTURE_BROWSER_SCREEN_KEY,
    _TEXTURE_BROWSER_POPUP_KEY,
    _TEXTURE_BROWSER_CONTENT_PANEL,
    'ADDONS',
    'ADDONS',
    "leveldesign.texture_browser_interaction",
    _poll_scene,
    _draw_texture_browser_navigation,
    _draw_texture_browser_header,
    _draw_texture_browser_content,
    _texture_browser_display_items_for_filter,
    _texture_browser_search_text,
    _texture_browser_collection_filter,
    _texture_browser_preview_scale_value,
    _texture_browser_scroll_offset,
    _set_texture_browser_scroll_offset,
    _texture_browser_rect_for_item,
    _draw_texture_browser_cell,
    _texture_browser_activate_item,
    _texture_browser_warm_texture_item,
    _TEXTURE_BROWSER_GRID_SPEC,
)
texture_browser_modal.context_allowed_func = _context_is_texture_browser_addon_preferences


def _screen_is_texture_browser_workspace_screen(screen):
    if screen is None:
        return False
    if getattr(screen, "name", "") in _TEXTURE_BROWSER_WORKSPACE_NAMES:
        return True
    for workspace_name in _TEXTURE_BROWSER_WORKSPACE_NAMES:
        workspace = bpy.data.workspaces.get(workspace_name)
        if workspace is None:
            continue
        try:
            if any(candidate == screen for candidate in workspace.screens):
                return True
        except (AttributeError, ReferenceError, TypeError):
            continue
    return False


def _window_is_texture_browser_embedded_host(window):
    if window is None:
        return False
    if texture_browser_modal.is_popup_window(window):
        return False
    try:
        screen = getattr(window, "screen", None)
        workspace_name = _workspace_name_from_workspace(getattr(window, "workspace", None))
    except (AttributeError, ReferenceError):
        return False
    if texture_browser_modal.preferences_area_for_window(window) is None:
        return False
    if (workspace_name not in _TEXTURE_BROWSER_WORKSPACE_NAMES
            and not _screen_is_texture_browser_workspace_screen(screen)):
        return False
    return _screen_has_docked_workspace_area(screen)


def _window_is_texture_browser_recovery_host(window):
    return (
        texture_browser_modal.is_window(window)
        or _window_is_texture_browser_embedded_host(window)
    )


def _open_texture_browser_region(window_manager):
    interaction_window = texture_browser_modal.interaction["window"]
    interaction_area = texture_browser_modal.interaction["area"]
    if (
            texture_browser_modal.is_window_live(
                interaction_window,
                window_manager.windows,
            )
            and texture_browser_modal.is_area_live(
                interaction_area,
                window_manager.windows,
            )):
        region = texture_browser_modal.region_for_area(interaction_area)
        if region is not None:
            return region

    for window in window_manager.windows:
        if not _window_is_texture_browser_recovery_host(window):
            continue
        area = texture_browser_modal.preferences_area_for_window(window)
        if area is None:
            continue
        region = texture_browser_modal.region_for_area(area)
        if region is not None:
            return region
    return None


def _scroll_texture_browser_to_filepath(
        scene,
        window_manager,
        preferences_system,
        region,
        filepath):
    items = _texture_browser_display_items_for_filter(
        scene,
        window_manager.anvil_texture_browser_search,
        _TEXTURE_BROWSER_FOLDER_VIEW,
    )
    target_path = _normal_path(filepath)
    target_index = next(
        (
            index for index, item in enumerate(items)
            if _normal_path(item["filepath"]) == target_path
        ),
        None,
    )
    if target_index is None:
        return False

    metrics, _rects = texture_browser_modal.layout_items(
        items,
        region.width,
        region.height,
        window_manager.anvil_texture_browser_preview_scale,
        preferences_system.ui_scale,
        preferences_system.pixel_size,
        0,
    )
    target_row = target_index // metrics["columns"]
    target_scroll = target_row * (metrics["cell_height"] + metrics["gap"])
    target_scroll = min(target_scroll, metrics["max_scroll"])
    texture_browser_modal.interaction["scroll_offset"] = target_scroll
    texture_browser_modal.remember_scroll_offset(window_manager, target_scroll)
    texture_browser_modal.interaction["warm_visible_items"] = True
    _texture_browser_animations.start_locate(filepath)
    texture_browser_modal.tag_preferences_areas(window_manager.windows)
    return True


def _recover_texture_browser_active_section():
    try:
        window_manager = getattr(bpy.context, "window_manager", None)
        preferences = getattr(bpy.context, "preferences", None)
        if window_manager is not None and preferences is not None:
            restored = texture_browser_modal.restore_active_section_if_browser_is_alone(
                preferences,
                window_manager.windows,
                _window_is_texture_browser_recovery_host,
            )
            if restored:
                debug_log("[TextureBrowser] Restored browser Preferences section")
    except Exception as exc:
        debug_log(f"[TextureBrowser] Preferences recovery failed: {exc}")
    return _TEXTURE_BROWSER_EMBEDDED_RECOVERY_INTERVAL_SECONDS


def texture_browser_search_update(window_manager, context):
    texture_browser_modal.settings_update(window_manager, context.window_manager.windows, True)


def texture_browser_collection_filter_update(window_manager, context):
    texture_browser_modal.settings_update(window_manager, context.window_manager.windows, True)


def texture_browser_preview_scale_update(window_manager, context):
    prefs = _addon_preferences()
    if prefs is not None:
        prefs.texture_browser_preview_scale = window_manager.anvil_texture_browser_preview_scale
    texture_browser_modal.settings_update(window_manager, context.window_manager.windows, False)


def texture_browser_folder_path_update(window_manager, context):
    if window_manager.anvil_texture_browser_collection_index != _TEXTURE_BROWSER_FOLDER_VIEW:
        window_manager.anvil_texture_browser_collection_index = _TEXTURE_BROWSER_FOLDER_VIEW
    _remember_texture_browser_folder(window_manager.anvil_texture_browser_folder_path)
    texture_browser_modal.settings_update(window_manager, context.window_manager.windows, True)


def _set_folder(window_manager, folder_path):
    if not _folder_exists(folder_path):
        return False
    folder_path = os.path.abspath(bpy.path.abspath(folder_path))
    window_manager.anvil_texture_browser_collection_index = _TEXTURE_BROWSER_FOLDER_VIEW
    window_manager.anvil_texture_browser_folder_path = folder_path
    _remember_texture_browser_folder(folder_path)
    return True


def _tag_texture_browser_changed(window_manager, reset_scroll):
    global _folder_scan_generation
    _folder_scan_generation += 1
    _folder_scan_cache.clear()
    _collection_scan_cache.clear()
    texture_browser_modal.settings_update(
        window_manager,
        window_manager.windows,
        reset_scroll,
    )


def _unique_collection_name(prefs):
    existing = {collection.name for collection in prefs.texture_browser_collections}
    index = 1
    while True:
        name = f"Collection {index}"
        if name not in existing:
            return name
        index += 1


def _collection_contains_path(collection, filepath):
    normal = _normal_path(filepath)
    for item in collection.files:
        if _normal_path(item.filepath) == normal:
            return True
    return False


def _add_filepath_to_collection(collection, filepath):
    if _collection_contains_path(collection, filepath):
        return False
    item = collection.files.add()
    item.filepath = os.path.abspath(bpy.path.abspath(filepath))
    collection.active_file_index = len(collection.files) - 1
    return True


def _texture_browser_collection_enum_items(operator, context):
    prefs = _addon_preferences()
    if prefs is None or not prefs.texture_browser_collections:
        return [("0", "Collection 1", "Collection 1")]
    items = []
    for index, collection in enumerate(prefs.texture_browser_collections):
        name = collection.name or f"Collection {index + 1}"
        items.append((str(index), name, name))
    return items


class LEVELDESIGN_OT_texture_browser(Operator):
    """Open a native Blender window showing texture folders and files"""
    bl_idname = "leveldesign.texture_browser"
    bl_label = "Texture Browser"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _poll_scene_mode(context)

    def execute(self, context):
        workspace_name = _workspace_name_from_workspace(getattr(context, "workspace", None))
        _set_popup_source_workspace_name(workspace_name)
        _load_texture_browser_persistent_settings(context.window_manager)
        return texture_browser_modal.open_popup(
            context.preferences,
            context.window_manager,
            context.window,
            context.area,
            context.region,
            context.temp_override,
            self.report,
        )


class LEVELDESIGN_OT_texture_browser_locate_file(Operator):
    """Show this image in the open texture browser"""
    bl_idname = "leveldesign.texture_browser_locate_file"
    bl_label = "Locate Texture in Browser"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        filepath = _display_path(self.filepath)
        if not _file_exists(filepath):
            self.report({'ERROR'}, "Texture file was not found")
            return {'CANCELLED'}

        region = _open_texture_browser_region(context.window_manager)
        if region is None:
            self.report({'WARNING'}, "Open the Texture Browser first")
            return {'CANCELLED'}

        folder_path = os.path.dirname(filepath)
        if not _set_folder(context.window_manager, folder_path):
            self.report({'ERROR'}, "Texture folder was not found")
            return {'CANCELLED'}
        context.window_manager.anvil_texture_browser_search = ""
        texture_browser_modal.settings_update(
            context.window_manager,
            context.window_manager.windows,
            True,
        )

        located = _scroll_texture_browser_to_filepath(
            context.scene,
            context.window_manager,
            context.preferences.system,
            region,
            filepath,
        )
        if not located:
            self.report({'WARNING'}, "Texture is hidden by the browser filters")
            return {'CANCELLED'}
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_apply_file(Operator):
    """Apply the selected texture browser image to selected faces"""
    bl_idname = "leveldesign.texture_browser_apply_file"
    bl_label = "Apply Texture Browser File"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() or _texture_browser_popup_source_is_level_design(context.window)

    def execute(self, context):
        set_active_image(None)

        obj = context.object
        original_mode = context.mode
        restore_object_mode = (
            original_mode == 'OBJECT'
            and obj is not None
            and obj.type == 'MESH'
            and obj.select_get()
            and not is_library_object(obj)
        )
        other_selected_objects = []

        if restore_object_mode:
            # Hotspot application uses Blender's edit-mode UV unwrap operator.
            # Isolate the active object so other selected meshes are untouched.
            other_selected_objects = [
                selected_obj
                for selected_obj in context.selected_objects
                if selected_obj != obj
            ]
            for selected_obj in other_selected_objects:
                selected_obj.select_set(False)

        try:
            if restore_object_mode:
                bpy.ops.object.mode_set(mode='EDIT')

            image, applied_face_count = apply_texture_path_to_selection(
                self.filepath,
                obj,
                original_mode,
                context.scene,
            )
        finally:
            if restore_object_mode and context.mode == 'EDIT_MESH':
                bpy.ops.object.mode_set(mode='OBJECT')
            for selected_obj in other_selected_objects:
                selected_obj.select_set(True)

        if image is None:
            redraw_ui_panels(context)
            self.report({'ERROR'}, "Could not load image")
            return {'CANCELLED'}

        from ..handlers.face_cache import update_ui_from_selection

        redraw_ui_panels(context)
        if applied_face_count > 0:
            update_ui_from_selection(context)
            set_active_image(image)
            redraw_ui_panels(context)
            bpy.ops.ed.undo_push(message="Apply Texture from Texture Browser")
            _texture_browser_animations.start_applied(
                self.filepath,
                applied_face_count,
            )
            texture_browser_modal.tag_preferences_areas(context.window_manager.windows)
            face_label = "face" if applied_face_count == 1 else "faces"
            self.report(
                {'INFO'},
                f"Texture applied to {applied_face_count} {face_label}",
            )
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_interaction(Operator):
    """Handle texture browser thumbnail clicks against the drawn cell geometry"""
    bl_idname = "leveldesign.texture_browser_interaction"
    bl_label = "Texture Browser Interaction"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return _poll_scene_mode(context) or texture_browser_modal.is_popup_window(context.window)

    def invoke(self, context, event):
        workspace = getattr(context, "workspace", None)
        workspace_name = getattr(workspace, "name", "")
        screen = getattr(context.window, "screen", None)
        return texture_browser_modal.invoke_interaction(
            self,
            context.window,
            screen,
            workspace_name,
            context.preferences.active_section,
            context.area,
            context.region,
            context.scene,
            context.window_manager,
        )

    def modal(self, context, event):
        timer_event = texture_browser_modal.event_is_timer(event)
        workspace_allowed = (
            _texture_browser_workspace_is_allowed()
            or texture_browser_modal.is_popup_window(context.window)
        )
        if (
                timer_event
                and workspace_allowed
                and _texture_browser_animations.tick()):
            area = texture_browser_modal.interaction["area"]
            if area is not None:
                texture_browser_modal.tag_area(area)
        return texture_browser_modal.modal(
            self,
            event,
            context.scene,
            context.window,
            context.window_manager,
            context.preferences,
        )


class LEVELDESIGN_OT_texture_browser_close(Operator):
    """Close the native texture browser window and restore Preferences drawing"""
    bl_idname = "leveldesign.texture_browser_close"
    bl_label = "Close Texture Browser"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        close_current_window = texture_browser_modal.is_popup_window(context.window)
        texture_browser_modal.restore_preferences(
            context.preferences,
            context.window_manager.windows,
            True,
            True,
        )
        if close_current_window:
            try:
                bpy.ops.wm.window_close()
            except RuntimeError:
                pass
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_fix_layout(Operator):
    """Return Preferences to the section used by the texture browser host"""
    bl_idname = "leveldesign.texture_browser_fix_layout"
    bl_label = "Fix Texture Browser Layout"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        texture_browser_modal.set_active_section(context.preferences, 'ADDONS')
        texture_browser_modal.tag_preferences_areas(context.window_manager.windows)
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_choose_folder(Operator):
    """Choose a texture browser folder"""
    bl_idname = "leveldesign.texture_browser_choose_folder"
    bl_label = "Choose Texture Folder"
    bl_options = {'REGISTER'}

    directory: StringProperty(subtype='DIR_PATH')

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def invoke(self, context, event):
        self.directory = context.window_manager.anvil_texture_browser_folder_path
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not _set_folder(context.window_manager, self.directory):
            self.report({'ERROR'}, "Choose an existing folder")
            return {'CANCELLED'}
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_parent_folder(Operator):
    """Go to the parent folder"""
    bl_idname = "leveldesign.texture_browser_parent_folder"
    bl_label = "Parent Folder"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        folder = _display_path(context.window_manager.anvil_texture_browser_folder_path)
        parent = os.path.dirname(os.path.normpath(folder))
        if not parent or parent == folder:
            return {'CANCELLED'}
        if not _set_folder(context.window_manager, parent):
            return {'CANCELLED'}
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_home(Operator):
    """Go to the texture browser home folder"""
    bl_idname = "leveldesign.texture_browser_home"
    bl_label = "Texture Browser Home"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        _set_folder(context.window_manager, _blend_home_folder())
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_refresh(Operator):
    """Refresh texture browser files and previews"""
    bl_idname = "leveldesign.texture_browser_refresh"
    bl_label = "Refresh Texture Browser"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        invalidate_texture_browser_preview_cache()
        _tag_texture_browser_changed(context.window_manager, False)
        self.report({'INFO'}, "Texture browser refreshed")
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_set_favorite(Operator):
    """Browse a favorite folder"""
    bl_idname = "leveldesign.texture_browser_set_favorite"
    bl_label = "Browse Favorite Folder"
    bl_options = {'REGISTER'}

    favorite_index: IntProperty()

    @classmethod
    def description(cls, context, properties):
        prefs = _addon_preferences()
        if prefs is not None and 0 <= properties.favorite_index < len(prefs.texture_browser_favorites):
            return _display_path(prefs.texture_browser_favorites[properties.favorite_index].path)
        return cls.bl_label

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None or not (0 <= self.favorite_index < len(prefs.texture_browser_favorites)):
            return {'CANCELLED'}
        favorite = prefs.texture_browser_favorites[self.favorite_index]
        if not _set_folder(context.window_manager, favorite.path):
            self.report({'ERROR'}, "Favorite folder not found")
            return {'CANCELLED'}
        prefs.texture_browser_active_favorite_index = self.favorite_index
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_add_favorite(Operator):
    """Add the current texture browser folder to favorites"""
    bl_idname = "leveldesign.texture_browser_add_favorite"
    bl_label = "Add Favorite Folder"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None:
            return {'CANCELLED'}
        folder = _display_path(context.window_manager.anvil_texture_browser_folder_path)
        if not os.path.isdir(folder):
            self.report({'ERROR'}, "Current folder does not exist")
            return {'CANCELLED'}
        normal = _normal_path(folder)
        for index, favorite in enumerate(prefs.texture_browser_favorites):
            if _normal_path(favorite.path) == normal:
                prefs.texture_browser_active_favorite_index = index
                return {'FINISHED'}
        item = prefs.texture_browser_favorites.add()
        item.path = os.path.abspath(folder)
        _ensure_texture_browser_preferences(prefs)
        prefs.texture_browser_active_favorite_index = len(prefs.texture_browser_favorites) - 1
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_remove_favorite(Operator):
    """Remove a favorite folder"""
    bl_idname = "leveldesign.texture_browser_remove_favorite"
    bl_label = "Remove Favorite Folder"
    bl_options = {'REGISTER'}

    favorite_index: IntProperty()

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None or not (0 <= self.favorite_index < len(prefs.texture_browser_favorites)):
            return {'CANCELLED'}
        prefs.texture_browser_favorites.remove(self.favorite_index)
        prefs.texture_browser_active_favorite_index = min(
            self.favorite_index,
            max(0, len(prefs.texture_browser_favorites) - 1),
        )
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_remove_active_favorite(Operator):
    """Remove the active favorite folder"""
    bl_idname = "leveldesign.texture_browser_remove_active_favorite"
    bl_label = "Remove Active Favorite Folder"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None:
            return {'CANCELLED'}
        index = prefs.texture_browser_active_favorite_index
        if not (0 <= index < len(prefs.texture_browser_favorites)):
            return {'CANCELLED'}
        prefs.texture_browser_favorites.remove(index)
        prefs.texture_browser_active_favorite_index = min(
            index,
            max(0, len(prefs.texture_browser_favorites) - 1),
        )
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_set_collection(Operator):
    """Browse a texture collection"""
    bl_idname = "leveldesign.texture_browser_set_collection"
    bl_label = "Browse Texture Collection"
    bl_options = {'REGISTER'}

    collection_index: IntProperty()

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None or not (0 <= self.collection_index < len(prefs.texture_browser_collections)):
            return {'CANCELLED'}
        prefs.texture_browser_active_collection_index = self.collection_index
        context.window_manager.anvil_texture_browser_collection_index = self.collection_index
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_add_collection(Operator):
    """Add a texture collection"""
    bl_idname = "leveldesign.texture_browser_add_collection"
    bl_label = "Add Texture Collection"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None:
            return {'CANCELLED'}
        collection = prefs.texture_browser_collections.add()
        collection.name = _unique_collection_name(prefs)
        prefs.texture_browser_active_collection_index = len(prefs.texture_browser_collections) - 1
        context.window_manager.anvil_texture_browser_collection_index = prefs.texture_browser_active_collection_index
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_remove_collection(Operator):
    """Remove a texture collection"""
    bl_idname = "leveldesign.texture_browser_remove_collection"
    bl_label = "Remove Texture Collection"
    bl_options = {'REGISTER'}

    collection_index: IntProperty()

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None or not (0 <= self.collection_index < len(prefs.texture_browser_collections)):
            return {'CANCELLED'}
        prefs.texture_browser_collections.remove(self.collection_index)
        prefs.texture_browser_active_collection_index = min(
            self.collection_index,
            max(0, len(prefs.texture_browser_collections) - 1),
        )
        if context.window_manager.anvil_texture_browser_collection_index == self.collection_index:
            context.window_manager.anvil_texture_browser_collection_index = _TEXTURE_BROWSER_FOLDER_VIEW
        elif context.window_manager.anvil_texture_browser_collection_index > self.collection_index:
            context.window_manager.anvil_texture_browser_collection_index -= 1
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_remove_active_collection(Operator):
    """Remove the active texture collection"""
    bl_idname = "leveldesign.texture_browser_remove_active_collection"
    bl_label = "Remove Active Texture Collection"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None:
            return {'CANCELLED'}
        index = prefs.texture_browser_active_collection_index
        if not (0 <= index < len(prefs.texture_browser_collections)):
            return {'CANCELLED'}
        bpy.ops.leveldesign.texture_browser_remove_collection(collection_index=index)
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_add_file_to_collection(Operator):
    """Add a file to a texture collection"""
    bl_idname = "leveldesign.texture_browser_add_file_to_collection"
    bl_label = "Add File To Collection"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')
    collection_key: EnumProperty(
        name="Collection",
        items=_texture_browser_collection_enum_items,
    )

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def invoke(self, context, event):
        prefs = _addon_preferences()
        if prefs is None:
            return {'CANCELLED'}
        if not prefs.texture_browser_collections:
            collection = prefs.texture_browser_collections.add()
            collection.name = _unique_collection_name(prefs)
            prefs.texture_browser_active_collection_index = 0
        active_index = prefs.texture_browser_active_collection_index
        if not (0 <= active_index < len(prefs.texture_browser_collections)):
            active_index = 0
        self.collection_key = str(active_index)
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None:
            return {'CANCELLED'}
        try:
            index = int(self.collection_key)
        except ValueError:
            return {'CANCELLED'}
        if not (0 <= index < len(prefs.texture_browser_collections)):
            return {'CANCELLED'}
        if not _file_exists(self.filepath):
            self.report({'ERROR'}, "File not found")
            return {'CANCELLED'}
        collection = prefs.texture_browser_collections[index]
        added = _add_filepath_to_collection(collection, self.filepath)
        prefs.texture_browser_active_collection_index = index
        if added:
            self.report({'INFO'}, f"Added to {collection.name}")
        else:
            self.report({'INFO'}, f"Already in {collection.name}")
        _tag_texture_browser_changed(context.window_manager, False)
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_edit_filters(Operator):
    """Edit texture browser suffix filters"""
    bl_idname = "leveldesign.texture_browser_edit_filters"
    bl_label = "Texture Browser Filters"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def invoke(self, context, event):
        prefs = _addon_preferences()
        _ensure_texture_browser_preferences(prefs)
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        prefs = _addon_preferences()
        if prefs is None:
            return

        layout = self.layout
        include_box = layout.box()
        include_row = include_box.row(align=True)
        include_row.label(text="Include suffixes")
        add_op = include_row.operator("leveldesign.texture_browser_add_suffix_filter", text="", icon='ADD')
        add_op.kind = 'INCLUDE'
        for index, item in enumerate(prefs.texture_browser_include_suffixes):
            row = include_box.row(align=True)
            row.prop(item, "suffix", text="")
            remove_op = row.operator("leveldesign.texture_browser_remove_suffix_filter", text="", icon='REMOVE')
            remove_op.kind = 'INCLUDE'
            remove_op.filter_index = index

        exclude_box = layout.box()
        exclude_row = exclude_box.row(align=True)
        exclude_row.label(text="Exclude suffixes")
        add_op = exclude_row.operator("leveldesign.texture_browser_add_suffix_filter", text="", icon='ADD')
        add_op.kind = 'EXCLUDE'
        for index, item in enumerate(prefs.texture_browser_exclude_suffixes):
            row = exclude_box.row(align=True)
            row.prop(item, "suffix", text="")
            remove_op = row.operator("leveldesign.texture_browser_remove_suffix_filter", text="", icon='REMOVE')
            remove_op.kind = 'EXCLUDE'
            remove_op.filter_index = index

        layout.operator(
            "leveldesign.texture_browser_restore_default_include_filters",
            text="Restore Image Defaults",
            icon='FILE_REFRESH',
        )

    def execute(self, context):
        _tag_texture_browser_changed(context.window_manager, True)
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_add_suffix_filter(Operator):
    """Add a texture browser suffix filter"""
    bl_idname = "leveldesign.texture_browser_add_suffix_filter"
    bl_label = "Add Texture Browser Suffix Filter"
    bl_options = {'REGISTER'}

    kind: StringProperty()

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None:
            return {'CANCELLED'}
        if self.kind == 'EXCLUDE':
            item = prefs.texture_browser_exclude_suffixes.add()
            item.suffix = ""
            prefs.texture_browser_active_exclude_index = len(prefs.texture_browser_exclude_suffixes) - 1
        else:
            item = prefs.texture_browser_include_suffixes.add()
            item.suffix = ""
            prefs.texture_browser_active_include_index = len(prefs.texture_browser_include_suffixes) - 1
        _tag_texture_browser_changed(context.window_manager, True)
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_remove_suffix_filter(Operator):
    """Remove a texture browser suffix filter"""
    bl_idname = "leveldesign.texture_browser_remove_suffix_filter"
    bl_label = "Remove Texture Browser Suffix Filter"
    bl_options = {'REGISTER'}

    kind: StringProperty()
    filter_index: IntProperty()

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None:
            return {'CANCELLED'}
        if self.kind == 'EXCLUDE':
            if not (0 <= self.filter_index < len(prefs.texture_browser_exclude_suffixes)):
                return {'CANCELLED'}
            prefs.texture_browser_exclude_suffixes.remove(self.filter_index)
            prefs.texture_browser_active_exclude_index = min(
                self.filter_index,
                max(0, len(prefs.texture_browser_exclude_suffixes) - 1),
            )
        else:
            if not (0 <= self.filter_index < len(prefs.texture_browser_include_suffixes)):
                return {'CANCELLED'}
            prefs.texture_browser_include_suffixes.remove(self.filter_index)
            prefs.texture_browser_active_include_index = min(
                self.filter_index,
                max(0, len(prefs.texture_browser_include_suffixes) - 1),
            )
        _tag_texture_browser_changed(context.window_manager, True)
        return {'FINISHED'}


class LEVELDESIGN_OT_texture_browser_restore_default_include_filters(Operator):
    """Restore default image suffix include filters"""
    bl_idname = "leveldesign.texture_browser_restore_default_include_filters"
    bl_label = "Restore Texture Browser Image Filters"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _texture_browser_workspace_is_allowed() or texture_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefs = _addon_preferences()
        if prefs is None:
            return {'CANCELLED'}
        prefs.texture_browser_include_suffixes.clear()
        for suffix in _TEXTURE_BROWSER_COMMON_IMAGE_SUFFIXES:
            item = prefs.texture_browser_include_suffixes.add()
            item.suffix = suffix
        prefs.texture_browser_active_include_index = 0
        _tag_texture_browser_changed(context.window_manager, True)
        return {'FINISHED'}


classes = (
    LEVELDESIGN_OT_texture_browser,
    LEVELDESIGN_OT_texture_browser_locate_file,
    LEVELDESIGN_OT_texture_browser_apply_file,
    LEVELDESIGN_OT_texture_browser_interaction,
    LEVELDESIGN_OT_texture_browser_close,
    LEVELDESIGN_OT_texture_browser_fix_layout,
    LEVELDESIGN_OT_texture_browser_choose_folder,
    LEVELDESIGN_OT_texture_browser_parent_folder,
    LEVELDESIGN_OT_texture_browser_home,
    LEVELDESIGN_OT_texture_browser_refresh,
    LEVELDESIGN_OT_texture_browser_set_favorite,
    LEVELDESIGN_OT_texture_browser_add_favorite,
    LEVELDESIGN_OT_texture_browser_remove_favorite,
    LEVELDESIGN_OT_texture_browser_remove_active_favorite,
    LEVELDESIGN_OT_texture_browser_set_collection,
    LEVELDESIGN_OT_texture_browser_add_collection,
    LEVELDESIGN_OT_texture_browser_remove_collection,
    LEVELDESIGN_OT_texture_browser_remove_active_collection,
    LEVELDESIGN_OT_texture_browser_add_file_to_collection,
    LEVELDESIGN_OT_texture_browser_edit_filters,
    LEVELDESIGN_OT_texture_browser_add_suffix_filter,
    LEVELDESIGN_OT_texture_browser_remove_suffix_filter,
    LEVELDESIGN_OT_texture_browser_restore_default_include_filters,
)

_addon_keymaps = []

KEYMAPS_TO_REGISTER = [
    ("Object Mode", 'EMPTY'),
    ("Mesh", 'EMPTY'),
    ("Image Generic", 'IMAGE_EDITOR'),
]


def _register_texture_browser_keymap(kc, km_name, space_type):
    km = kc.keymaps.new(name=km_name, space_type=space_type)
    kmi = km.keymap_items.new(
        "leveldesign.texture_browser",
        'TWO',
        'PRESS',
        shift=True,
        head=True,
    )
    _addon_keymaps.append((km, kmi))


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.WindowManager.anvil_texture_browser_search = StringProperty(
        name="Search",
        description="Filter files by name or folder",
        options={'TEXTEDIT_UPDATE'},
        update=texture_browser_search_update,
    )
    bpy.types.WindowManager.anvil_texture_browser_folder_path = StringProperty(
        name="Folder",
        description="Current texture browser folder",
        update=texture_browser_folder_path_update,
    )
    bpy.types.WindowManager.anvil_texture_browser_collection_index = IntProperty(
        name="Collection Filter",
        description="Selected texture collection, or folder view",
        default=_TEXTURE_BROWSER_FOLDER_VIEW,
        min=_TEXTURE_BROWSER_FOLDER_VIEW,
        update=texture_browser_collection_filter_update,
    )
    bpy.types.WindowManager.anvil_texture_browser_preview_scale = FloatProperty(
        name="Preview Scale",
        description="Texture browser preview thumbnail size",
        min=0.0,
        max=100.0,
        default=50.0,
        subtype='PERCENTAGE',
        update=texture_browser_preview_scale_update,
    )
    bpy.types.WindowManager.anvil_texture_browser_scroll_offset = IntProperty(
        name="Texture Browser Scroll Offset",
        description="Remembered texture browser scroll position",
        default=0,
        min=0,
        options={'HIDDEN'},
    )

    wm = bpy.context.window_manager
    _load_texture_browser_persistent_settings(wm)
    texture_browser_modal.persistent_draw_overrides = True
    texture_browser_modal.persistent_draw_handler = True
    texture_browser_modal.install_draw_overrides(wm.windows)
    texture_browser_modal.ensure_draw_handler()
    if not bpy.app.timers.is_registered(_recover_texture_browser_active_section):
        bpy.app.timers.register(
            _recover_texture_browser_active_section,
            first_interval=_TEXTURE_BROWSER_EMBEDDED_RECOVERY_INTERVAL_SECONDS,
            persistent=True,
        )

    kc = wm.keyconfigs.addon
    if kc:
        for km_name, space_type in KEYMAPS_TO_REGISTER:
            _register_texture_browser_keymap(kc, km_name, space_type)


def unregister():
    if bpy.app.timers.is_registered(_recover_texture_browser_active_section):
        bpy.app.timers.unregister(_recover_texture_browser_active_section)
    texture_browser_modal.persistent_draw_overrides = False
    texture_browser_modal.persistent_draw_handler = False
    texture_browser_modal.restore_preferences(
        bpy.context.preferences,
        bpy.context.window_manager.windows,
        False,
        False,
    )
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    _addon_keymaps.clear()

    if hasattr(bpy.types.WindowManager, "anvil_texture_browser_scroll_offset"):
        del bpy.types.WindowManager.anvil_texture_browser_scroll_offset
    if hasattr(bpy.types.WindowManager, "anvil_texture_browser_preview_scale"):
        del bpy.types.WindowManager.anvil_texture_browser_preview_scale
    if hasattr(bpy.types.WindowManager, "anvil_texture_browser_collection_index"):
        del bpy.types.WindowManager.anvil_texture_browser_collection_index
    if hasattr(bpy.types.WindowManager, "anvil_texture_browser_folder_path"):
        del bpy.types.WindowManager.anvil_texture_browser_folder_path
    if hasattr(bpy.types.WindowManager, "anvil_texture_browser_search"):
        del bpy.types.WindowManager.anvil_texture_browser_search

    cleanup_texture_browser_preview_cache()
    _folder_scan_cache.clear()
    _collection_scan_cache.clear()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
