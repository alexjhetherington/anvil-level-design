"""Floating prefab browser hosted inside a temporary Preferences window."""

import bpy
from bpy.props import IntProperty
from bpy.types import Operator

from ..core.modal_image_grid import (
    ImageGridSpec,
    PreferencesImageGridModal,
    draw_image_grid_rect,
    draw_image_grid_text,
    draw_image_grid_texture,
)
from ..core.workspace_check import is_level_design_workspace
from .previews import (
    prefab_browser_cached_preview_texture,
    prefab_browser_preview_texture,
    prefab_browser_preview_texture_needs_load,
)


_PREFAB_BROWSER_SCREEN_KEY = "anvil_prefab_browser_screen"
_PREFAB_BROWSER_POPUP_KEY = "anvil_prefab_browser_popup"
_PREFAB_BROWSER_CONTENT_PANEL = "USERPREF_PT_addons"
_PREFAB_BROWSER_GRID_SPEC = ImageGridSpec(5.0, 7.5, 2)


def _poll_scene(scene):
    return (is_level_design_workspace()
            and scene is not None
            and getattr(scene, "anvil_prefab_mode", 'SCENE') == 'SCENE')


def _poll_scene_mode(context):
    return _poll_scene(getattr(context, "scene", None))


def _prefab_browser_item_matches_search(asset_name, library_label, search_text):
    text = search_text.strip().lower()
    if not text:
        return True
    return text in asset_name.lower() or text in library_label.lower()


def _prefab_browser_library_object_count(lib_entry):
    return len([
        item for item in lib_entry.objects
        if (item.asset_type or 'OBJECT') == 'OBJECT'
    ])


def _prefab_browser_display_items_for_filter(scene, search_text, library_filter):
    items = []
    library_indices = range(len(scene.anvil_prefab_libraries))
    if 0 <= library_filter < len(scene.anvil_prefab_libraries):
        library_indices = (library_filter,)

    for lib_index in library_indices:
        lib_entry = scene.anvil_prefab_libraries[lib_index]
        library_label = bpy.path.basename(lib_entry.filepath) or "<empty>"
        for asset_item in lib_entry.objects:
            asset_type = asset_item.asset_type or 'OBJECT'
            if asset_type != 'OBJECT':
                continue
            if not _prefab_browser_item_matches_search(asset_item.name, library_label, search_text):
                continue
            items.append((lib_index, lib_entry.filepath, library_label, asset_type, asset_item.name))
    return items


def _prefab_browser_display_items(scene, search_text):
    return _prefab_browser_display_items_for_filter(scene, search_text, -1)


def _prefab_browser_search_text(window_manager):
    return window_manager.anvil_prefab_browser_search


def _prefab_browser_library_filter(window_manager):
    return window_manager.anvil_prefab_browser_library_filter


def _prefab_browser_preview_scale_value(window_manager):
    return window_manager.anvil_prefab_browser_preview_scale


def _prefab_browser_scroll_offset(window_manager):
    return window_manager.anvil_prefab_browser_scroll_offset


def _set_prefab_browser_scroll_offset(window_manager, scroll_offset):
    window_manager.anvil_prefab_browser_scroll_offset = scroll_offset


def _prefab_browser_rect_for_item(item, x, y, width, height):
    lib_index, filepath, library_label, asset_type, asset_name = item
    return {
        "x": x,
        "y": y,
        "w": width,
        "h": height,
        "library_index": lib_index,
        "filepath": filepath,
        "library_label": library_label,
        "asset_type": asset_type,
        "asset_name": asset_name,
        "key": f"{lib_index}:{asset_type}:{asset_name}",
    }


def _draw_prefab_browser_navigation(layout, scene, window_manager):
    layout.label(text="Anvil Prefabs", icon='ASSET_MANAGER')
    layout.separator()
    layout.label(text="Libraries")
    if not scene.anvil_prefab_libraries:
        row = layout.row()
        row.enabled = False
        row.label(text="No libraries", icon='INFO')
        return

    active_filter = _prefab_browser_library_filter(window_manager)
    if not (0 <= active_filter < len(scene.anvil_prefab_libraries)):
        active_filter = -1

    total_count = sum(
        _prefab_browser_library_object_count(lib_entry)
        for lib_entry in scene.anvil_prefab_libraries
    )
    row = layout.row()
    op = row.operator(
        "leveldesign.prefab_browser_set_library_filter",
        text=f"All ({total_count})",
        icon='ASSET_MANAGER',
        depress=(active_filter == -1),
    )
    op.library_index = -1
    layout.separator()

    for lib_index, lib_entry in enumerate(scene.anvil_prefab_libraries):
        object_count = _prefab_browser_library_object_count(lib_entry)
        label = bpy.path.basename(lib_entry.filepath) or "<empty>"
        row = layout.row()
        op = row.operator(
            "leveldesign.prefab_browser_set_library_filter",
            text=f"{label} ({object_count})",
            icon='FILE_BLEND',
            depress=(active_filter == lib_index),
        )
        op.library_index = lib_index


def _draw_prefab_browser_header(
        layout,
        scene,
        window_manager,
        preferences,
        active_section_is_compatible):
    layout.row().template_header()
    row = layout.row(align=True)
    title_row = row.row(align=True)
    title_row.scale_x = 0.85
    title_row.label(text="Prefab Browser", icon='ASSET_MANAGER')
    row.separator(factor=2.0)
    search_row = row.row(align=True)
    search_row.scale_x = 2.0
    search_row.prop(window_manager, "anvil_prefab_browser_search", text="", icon='VIEWZOOM')
    row.separator(factor=0.7)
    row.prop(window_manager, "anvil_prefab_browser_preview_scale", text="", slider=True)
    row.operator("leveldesign.prefab_refresh_libraries", text="", icon='FILE_REFRESH')
    if not active_section_is_compatible:
        sub = row.row(align=True)
        sub.alert = True
        sub.operator("leveldesign.prefab_browser_fix_layout", text="", icon='SHADERFX')


def _draw_prefab_browser_content(layout, scene, window_manager, region_width, ui_scale, pixel_size):
    search_text = window_manager.anvil_prefab_browser_search
    items = _prefab_browser_display_items_for_filter(
        scene,
        search_text,
        _prefab_browser_library_filter(window_manager),
    )

    if not scene.anvil_prefab_libraries:
        col = layout.column()
        col.separator()
        row = col.row()
        row.alignment = 'CENTER'
        row.label(text="No prefab libraries added.", icon='INFO')
        return

    if not items:
        col = layout.column()
        col.separator()
        row = col.row()
        row.alignment = 'CENTER'
        row.label(text="No prefabs match the current filters.", icon='VIEWZOOM')
        return


def _draw_prefab_browser_cell(rect, metrics):
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

    gpu.state.blend_set('ALPHA')
    try:
        if is_hovered:
            draw_image_grid_rect(x, y, width, height, (0.24, 0.34, 0.46, 0.42))
        else:
            draw_image_grid_rect(x, y, width, height, (0.08, 0.09, 0.10, 0.20))
        texture_info = prefab_browser_cached_preview_texture(
            rect["filepath"],
            rect["asset_type"],
            rect["asset_name"],
        )
        if texture_info is not None:
            draw_image_grid_texture(texture_info, icon_x, icon_y, icon_space, icon_height)
        elif prefab_browser_preview_texture_needs_load(
                rect["filepath"],
                rect["asset_type"],
                rect["asset_name"]):
            draw_image_grid_rect(
                icon_x,
                icon_y,
                icon_space,
                icon_height,
                (0.10, 0.10, 0.10, 0.35),
            )
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
        draw_image_grid_text(
            rect["asset_name"],
            x + padding + 1,
            label_y - 1,
            width - padding * 2,
            label_font_size,
            (0.0, 0.0, 0.0, 0.75),
            'CENTER',
        )
        draw_image_grid_text(
            rect["asset_name"],
            x + padding,
            label_y,
            width - padding * 2,
            label_font_size,
            (0.96, 0.96, 0.96, 1.0),
            'CENTER',
        )
        draw_image_grid_text(
            rect["library_label"],
            x + padding + 1,
            meta_y - 1,
            width - padding * 2,
            meta_font_size,
            (0.0, 0.0, 0.0, 0.70),
            'CENTER',
        )
        draw_image_grid_text(
            rect["library_label"],
            x + padding,
            meta_y,
            width - padding * 2,
            meta_font_size,
            (0.68, 0.70, 0.73, 1.0),
            'CENTER',
        )
    finally:
        gpu.state.blend_set('NONE')


def _prefab_browser_region_for_area(area, preferred_region):
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


def _prefab_browser_3d_view_context_for_area(window, area, preferred_region):
    try:
        if area.type != 'VIEW_3D':
            return None
    except ReferenceError:
        return None

    region = _prefab_browser_region_for_area(area, preferred_region)
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


def _prefab_browser_3d_view_context_for_window_area(window, area, preferred_region):
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
    return _prefab_browser_3d_view_context_for_area(
        window,
        area,
        preferred_region,
    )


def _prefab_browser_first_3d_view_context_for_window(window):
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
        view_context = _prefab_browser_3d_view_context_for_area(
            window,
            area,
            None,
        )
        if view_context is not None:
            return view_context
    return None


def _prefab_browser_source_3d_view_context(windows):
    source_window = prefab_browser_modal.interaction.get("source_window")
    if source_window is None:
        return None
    if not prefab_browser_modal.is_window_live(source_window, windows):
        return None
    if prefab_browser_modal.is_popup_window(source_window):
        return None

    source_area = prefab_browser_modal.interaction.get("source_area")
    source_region = prefab_browser_modal.interaction.get("source_region")
    view_context = _prefab_browser_3d_view_context_for_window_area(
        source_window,
        source_area,
        source_region,
    )
    if view_context is not None:
        return view_context
    return _prefab_browser_first_3d_view_context_for_window(source_window)


def _prefab_browser_3d_view_context(windows):
    source_view_context = _prefab_browser_source_3d_view_context(windows)
    if source_view_context is not None:
        return source_view_context

    for window in windows:
        if prefab_browser_modal.is_popup_window(window):
            continue
        view_context = _prefab_browser_first_3d_view_context_for_window(window)
        if view_context is not None:
            return view_context
    return None


def _prefab_browser_close_popup_window(window):
    if not prefab_browser_modal.is_popup_window(window):
        return
    prefab_browser_modal.restore_preferences(
        bpy.context.preferences,
        bpy.context.window_manager.windows,
        True,
        True,
    )
    try:
        with bpy.context.temp_override(window=window):
            bpy.ops.wm.window_close()
    except RuntimeError as exc:
        from ..core.logging import debug_log
        debug_log(f"[Prefabs] Could not close prefab browser window before placement: {exc}")


def _prefab_browser_activate_item(hit):
    view_context = _prefab_browser_3d_view_context(bpy.context.window_manager.windows)
    if view_context is None:
        print("Anvil Level Design: No 3D View available for prefab placement", flush=True)
        return {'CANCELLED'}

    _prefab_browser_close_popup_window(bpy.context.window)
    with bpy.context.temp_override(**view_context):
        return bpy.ops.leveldesign.prefab_instantiate(
            'INVOKE_DEFAULT',
            library_index=hit["library_index"],
            object_name=hit["asset_name"],
            asset_type=hit["asset_type"],
        )


def _prefab_browser_warm_texture_item(item):
    _lib_index, filepath, _library_label, asset_type, asset_name = item
    if not prefab_browser_preview_texture_needs_load(filepath, asset_type, asset_name):
        return False
    prefab_browser_preview_texture(filepath, asset_type, asset_name)
    return True


prefab_browser_modal = PreferencesImageGridModal(
    "prefab browser",
    _PREFAB_BROWSER_SCREEN_KEY,
    _PREFAB_BROWSER_POPUP_KEY,
    _PREFAB_BROWSER_CONTENT_PANEL,
    'ADDONS',
    'ADDONS',
    "leveldesign.prefab_browser_interaction",
    _poll_scene,
    _draw_prefab_browser_navigation,
    _draw_prefab_browser_header,
    _draw_prefab_browser_content,
    _prefab_browser_display_items_for_filter,
    _prefab_browser_search_text,
    _prefab_browser_library_filter,
    _prefab_browser_preview_scale_value,
    _prefab_browser_scroll_offset,
    _set_prefab_browser_scroll_offset,
    _prefab_browser_rect_for_item,
    _draw_prefab_browser_cell,
    _prefab_browser_activate_item,
    _prefab_browser_warm_texture_item,
    _PREFAB_BROWSER_GRID_SPEC,
)


def prefab_browser_search_update(window_manager, context):
    prefab_browser_modal.settings_update(window_manager, context.window_manager.windows, True)


def prefab_browser_library_filter_update(window_manager, context):
    prefab_browser_modal.settings_update(window_manager, context.window_manager.windows, True)


def prefab_browser_preview_scale_update(window_manager, context):
    prefab_browser_modal.settings_update(window_manager, context.window_manager.windows, False)


class LEVELDESIGN_OT_prefab_browser(Operator):
    """Open a native Blender window showing all prefabs in a grid"""
    bl_idname = "leveldesign.prefab_browser"
    bl_label = "Prefab Browser"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _poll_scene_mode(context)

    def execute(self, context):
        return prefab_browser_modal.open_popup(
            context.preferences,
            context.window_manager,
            context.window,
            context.area,
            context.region,
            context.temp_override,
            self.report,
        )


class LEVELDESIGN_OT_prefab_browser_set_library_filter(Operator):
    """Filter the prefab browser to one library, or show all libraries"""
    bl_idname = "leveldesign.prefab_browser_set_library_filter"
    bl_label = "Set Prefab Browser Library Filter"
    bl_options = {'INTERNAL'}

    library_index: IntProperty()

    @classmethod
    def poll(cls, context):
        return _poll_scene_mode(context) or prefab_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        library_index = self.library_index
        if library_index < -1 or library_index >= len(context.scene.anvil_prefab_libraries):
            library_index = -1
        context.window_manager.anvil_prefab_browser_library_filter = library_index
        return {'FINISHED'}


class LEVELDESIGN_OT_prefab_browser_interaction(Operator):
    """Handle prefab browser thumbnail clicks against the drawn cell geometry"""
    bl_idname = "leveldesign.prefab_browser_interaction"
    bl_label = "Prefab Browser Interaction"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return _poll_scene_mode(context) or prefab_browser_modal.is_popup_window(context.window)

    def invoke(self, context, event):
        workspace = getattr(context, "workspace", None)
        workspace_name = getattr(workspace, "name", "")
        screen = getattr(context.window, "screen", None)
        return prefab_browser_modal.invoke_interaction(
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
        return prefab_browser_modal.modal(
            self,
            event,
            context.scene,
            context.window,
            context.window_manager,
            context.preferences,
        )


class LEVELDESIGN_OT_prefab_browser_close(Operator):
    """Close the native prefab browser window and restore Preferences drawing"""
    bl_idname = "leveldesign.prefab_browser_close"
    bl_label = "Close Prefab Browser"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() or prefab_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        close_current_window = prefab_browser_modal.is_popup_window(context.window)
        prefab_browser_modal.restore_preferences(
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


class LEVELDESIGN_OT_prefab_browser_fix_layout(Operator):
    """Return Preferences to the section used by the prefab browser host"""
    bl_idname = "leveldesign.prefab_browser_fix_layout"
    bl_label = "Fix Prefab Browser Layout"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() or prefab_browser_modal.is_popup_window(context.window)

    def execute(self, context):
        prefab_browser_modal.set_active_section(context.preferences, 'ADDONS')
        prefab_browser_modal.tag_preferences_areas(context.window_manager.windows)
        return {'FINISHED'}


classes = (
    LEVELDESIGN_OT_prefab_browser,
    LEVELDESIGN_OT_prefab_browser_set_library_filter,
    LEVELDESIGN_OT_prefab_browser_interaction,
    LEVELDESIGN_OT_prefab_browser_close,
    LEVELDESIGN_OT_prefab_browser_fix_layout,
)
