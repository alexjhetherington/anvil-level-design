"""Reusable image-grid modals hosted inside a temporary Preferences window."""

import math
import time

import bpy


_IMAGE_GRID_SCROLL_DEFER_SECONDS = 0.05
_IMAGE_GRID_TIMER_STEP_SECONDS = 0.01
_IMAGE_GRID_WARM_TIME_BUDGET_SECONDS = 0.02
_IMAGE_GRID_WARM_PAGES_AHEAD = 2
_IMAGE_GRID_WARM_PAGES_BEHIND = 1
_IMAGE_GRID_TEXTURE_VERTEX_SOURCE = (
    "void main()"
    "{"
    "  uv_interp = texCoord;"
    "  gl_Position = ModelViewProjectionMatrix * vec4(pos.xy, 0.0, 1.0);"
    "}"
)
_IMAGE_GRID_DISPLAY_TEXTURE_FRAGMENT_SOURCE = (
    "float linear_to_srgb_channel(float value)"
    "{"
    "  if (value <= 0.0031308) {"
    "    return value * 12.92;"
    "  }"
    "  return 1.055 * pow(value, 1.0 / 2.4) - 0.055;"
    "}"
    "vec3 linear_to_srgb(vec3 value)"
    "{"
    "  vec3 clamped = clamp(value, vec3(0.0), vec3(1.0));"
    "  return vec3("
    "    linear_to_srgb_channel(clamped.r),"
    "    linear_to_srgb_channel(clamped.g),"
    "    linear_to_srgb_channel(clamped.b)"
    "  );"
    "}"
    "void main()"
    "{"
    "  vec4 tex_color = texture(image, uv_interp);"
    "  FragColor = vec4(linear_to_srgb(tex_color.rgb), tex_color.a);"
    "}"
)

_image_grid_display_texture_shader = None


class ImageGridSpec:
    def __init__(
            self,
            min_icon_scale,
            icon_scale_range,
            text_line_count):
        self.min_icon_scale = min_icon_scale
        self.icon_scale_range = icon_scale_range
        self.text_line_count = text_line_count


class PreferencesImageGridModal:
    def __init__(
            self,
            display_name,
            screen_key,
            popup_key,
            content_panel,
            active_section,
            compatible_section,
            interaction_operator_idname,
            poll_scene_func,
            draw_navigation_func,
            draw_header_func,
            draw_content_func,
            display_items_func,
            search_text_func,
            filter_value_func,
            preview_scale_func,
            scroll_offset_func,
            set_scroll_offset_func,
            item_rect_func,
            draw_cell_func,
            activate_item_func,
            warm_item_func,
            grid_spec):
        self.display_name = display_name
        self.screen_key = screen_key
        self.popup_key = popup_key
        self.content_panel = content_panel
        self.active_section = active_section
        self.compatible_section = compatible_section
        self.interaction_operator_idname = interaction_operator_idname
        self.poll_scene_func = poll_scene_func
        self.draw_navigation_func = draw_navigation_func
        self.draw_header_func = draw_header_func
        self.draw_content_func = draw_content_func
        self.display_items_func = display_items_func
        self.search_text_func = search_text_func
        self.filter_value_func = filter_value_func
        self.preview_scale_func = preview_scale_func
        self.scroll_offset_func = scroll_offset_func
        self.set_scroll_offset_func = set_scroll_offset_func
        self.item_rect_func = item_rect_func
        self.draw_cell_func = draw_cell_func
        self.activate_item_func = activate_item_func
        self.warm_item_func = warm_item_func
        self.grid_spec = grid_spec
        self.userpref_draws = {}
        self.area_types = {}
        self.context_allowed_func = None
        self.persistent_draw_overrides = False
        self.persistent_draw_handler = False
        self.previous_active_section = None
        self.draw_handler = None
        self.interaction = {
            "window": None,
            "area": None,
            "region": None,
            "source_window": None,
            "source_area": None,
            "source_region": None,
            "scroll_offset": 0,
            "hovered_key": None,
            "session_id": 0,
            "modal_id": 0,
            "scrollbar_dragging": False,
            "scrollbar_drag_y": 0,
            "scrollbar_drag_scroll_offset": 0,
            "timer": None,
            "defer_images_until": 0.0,
            "warm_visible_items": False,
            "warm_direction": 1,
            "docked_host": False,
            "interaction_start_pending": False,
            "pending_window": None,
            "pending_area": None,
            "pending_region": None,
        }

        def navigation_draw_override(panel, context):
            return self._navigation_draw_override(panel, context)

        def header_draw_override(panel, context):
            return self._header_draw_override(panel, context)

        def content_draw_override(panel, context):
            return self._content_draw_override(panel, context)

        def empty_draw_override(panel, context):
            return self._empty_draw_override(panel, context)

        self.navigation_draw_override = navigation_draw_override
        self.header_draw_override = header_draw_override
        self.content_draw_override = content_draw_override
        self.empty_draw_override = empty_draw_override

    def screen_flag(self, screen, key):
        if screen is None:
            return False
        return bool(screen.get(key, False))

    def is_window(self, window):
        if window is None:
            return False
        try:
            return self.screen_flag(window.screen, self.screen_key)
        except ReferenceError:
            return False

    def is_popup_window(self, window):
        if window is None:
            return False
        try:
            return self.screen_flag(window.screen, self.popup_key)
        except ReferenceError:
            return False

    def context_uses_browser(self, window, screen, workspace_name, active_section, area):
        try:
            if self.is_window(window):
                return True
            if self.context_allowed_func is None:
                return False
            return self.context_allowed_func(
                workspace_name,
                active_section,
                area,
                screen,
            )
        except (AttributeError, ReferenceError):
            return False

    def popup_windows(self, windows):
        return [window for window in windows if self.is_popup_window(window)]

    def is_window_live(self, window, windows):
        if window is None:
            return False
        try:
            return any(candidate == window for candidate in windows)
        except ReferenceError:
            return False

    def live_areas(self, windows):
        live_areas = set()
        for window in windows:
            try:
                screen = getattr(window, "screen", None)
            except ReferenceError:
                continue
            if screen is None:
                continue
            try:
                areas = list(screen.areas)
            except ReferenceError:
                continue
            for area in areas:
                live_areas.add(area)
        return live_areas

    def is_area_live(self, area, windows):
        if area is None:
            return False
        try:
            return area in self.live_areas(windows)
        except ReferenceError:
            return False

    def tag_area(self, area):
        try:
            area.tag_redraw()
        except (ReferenceError, RuntimeError):
            pass

    def tag_preferences_areas(self, windows):
        for window in windows:
            try:
                screen = getattr(window, "screen", None)
            except ReferenceError:
                continue
            if screen is None:
                continue
            try:
                areas = list(screen.areas)
            except ReferenceError:
                continue
            for area in areas:
                if area.type == 'PREFERENCES' or getattr(area, "ui_type", "") == 'PREFERENCES':
                    self.tag_area(area)

    def active_section_is_compatible(self, preferences):
        return preferences.active_section == self.compatible_section

    def scene_allowed_for_window(self, scene, window):
        if self.poll_scene_func(scene):
            return True
        return self.is_popup_window(window)

    def settings_update(self, window_manager, windows, reset_scroll):
        if reset_scroll:
            self.interaction["scroll_offset"] = 0
            self.remember_scroll_offset(window_manager, 0)
        self.interaction["hovered_key"] = None
        self.interaction["scrollbar_dragging"] = False
        self.interaction["defer_images_until"] = 0.0
        self.interaction["warm_visible_items"] = True
        self.interaction["warm_direction"] = 1
        self.tag_preferences_areas(windows)

    def display_items_for_window_manager(self, scene, window_manager):
        return self.display_items_func(
            scene,
            self.search_text_func(window_manager),
            self.filter_value_func(window_manager),
        )

    def remembered_scroll_offset(self, window_manager):
        return max(0, int(self.scroll_offset_func(window_manager)))

    def remember_scroll_offset(self, window_manager, scroll_offset):
        scroll_offset = max(0, int(scroll_offset))
        if int(self.scroll_offset_func(window_manager)) == scroll_offset:
            return
        self.set_scroll_offset_func(window_manager, scroll_offset)

    def set_active_section(self, preferences, section):
        try:
            preferences.active_section = section
        except TypeError:
            preferences.active_section = self.active_section

    def userpref_draw_types(self):
        draw_types = []
        for type_name in filter(lambda name: name.startswith("USERPREF_"), dir(bpy.types)):
            userpref_type = getattr(bpy.types, type_name)
            if hasattr(userpref_type, "draw"):
                draw_types.append((type_name, userpref_type))
        return draw_types

    def restore_userpref_draws(self):
        for userpref_type_name, draw_func in self.userpref_draws.items():
            if not hasattr(bpy.types, userpref_type_name):
                continue
            userpref_type = getattr(bpy.types, userpref_type_name)
            userpref_type.draw = draw_func
        self.userpref_draws.clear()

    def restore_areas(self, windows):
        live_areas = self.live_areas(windows)
        for area, ui_type in list(self.area_types.items()):
            if area in live_areas:
                area.ui_type = ui_type
        self.area_types.clear()

    def clear_screen_flags(self, windows):
        for window in windows:
            screen = getattr(window, "screen", None)
            if screen is None:
                continue
            if screen.get(self.screen_key) is not None:
                del screen[self.screen_key]
            if screen.get(self.popup_key) is not None:
                del screen[self.popup_key]

    def forget_window_areas(self, window):
        try:
            screen = getattr(window, "screen", None)
        except ReferenceError:
            return
        if screen is None:
            return
        try:
            areas = list(screen.areas)
        except ReferenceError:
            return
        for area in list(self.area_types.keys()):
            if area in areas:
                del self.area_types[area]

    def restore_preferences(self, preferences, windows, restore_active_section, tag_redraw):
        self.stop_interaction(tag_redraw, windows)
        if not self.persistent_draw_overrides:
            self.restore_userpref_draws()
        self.restore_areas(windows)
        self.clear_screen_flags(windows)
        if restore_active_section and self.previous_active_section is not None:
            self.set_active_section(preferences, self.previous_active_section)
        self.previous_active_section = None
        if tag_redraw:
            self.tag_preferences_areas(windows)

    def restore_if_closed(self, preferences, windows):
        window = self.interaction["window"]
        area = self.interaction["area"]
        if window is None:
            return False
        if self.interaction_host_is_live(window, area, windows):
            return False
        self.finish_invalid_interaction(preferences, windows)
        return True

    def interaction_host_is_live(self, window, area, windows):
        if not self.is_window_live(window, windows):
            return False
        if not self.is_area_live(area, windows):
            return False
        if self.is_window(window):
            return True
        return self.interaction.get("docked_host", False)

    def finish_invalid_interaction(self, preferences, windows):
        if self.interaction.get("docked_host", False):
            self.stop_interaction(True, windows)
            return
        self.restore_preferences(preferences, windows, True, True)

    def install_draw_overrides(self, windows):
        header_type = getattr(bpy.types, "USERPREF_HT_header", None)
        nav_type = getattr(bpy.types, "USERPREF_PT_navigation_bar", None)
        for userpref_type_name, userpref_type in self.userpref_draw_types():
            if userpref_type_name in self.userpref_draws:
                continue
            original_draw = userpref_type.draw
            draw_funcs = getattr(original_draw, "_draw_funcs", None)
            if userpref_type == header_type:
                replacement_draw = self.header_draw_override
            elif userpref_type == nav_type:
                replacement_draw = self.navigation_draw_override
            elif userpref_type_name == self.content_panel:
                replacement_draw = self.content_draw_override
            elif userpref_type_name.startswith("USERPREF_PT_"):
                replacement_draw = self.empty_draw_override
            else:
                continue
            self.userpref_draws[userpref_type_name] = original_draw
            userpref_type.draw = replacement_draw
            if draw_funcs is not None:
                userpref_type.draw._draw_funcs = draw_funcs
        self.tag_preferences_areas(windows)

    def install_preferences(self, preferences, windows):
        if self.previous_active_section is None:
            self.previous_active_section = preferences.active_section
        self.set_active_section(preferences, self.active_section)
        self.install_draw_overrides(windows)

    def mark_area(self, window, area, preferences, windows):
        self.area_types[area] = area.ui_type
        window.screen[self.screen_key] = True
        area.ui_type = 'PREFERENCES'
        self.install_preferences(preferences, windows)

    def region_for_area(self, area):
        if area is None:
            return None
        try:
            regions = list(area.regions)
        except ReferenceError:
            return None
        for region in regions:
            if region.type == 'WINDOW':
                return region
        return None

    def region_for_area_type(self, area, region_type):
        if area is None:
            return None
        try:
            regions = list(area.regions)
        except ReferenceError:
            return None
        for region in regions:
            try:
                if region.type == region_type:
                    return region
            except ReferenceError:
                continue
        return None

    def preferences_area_for_window(self, window):
        if window is None:
            return None
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
            try:
                if area.type == 'PREFERENCES' or getattr(area, "ui_type", "") == 'PREFERENCES':
                    return area
            except ReferenceError:
                continue
        return None

    def preferences_popup_window(self, windows, existing_windows, source_window):
        for candidate in windows:
            if candidate in existing_windows:
                continue
            if self.preferences_area_for_window(candidate) is not None:
                return candidate

        context_window = getattr(bpy.context, "window", None)
        if context_window is not None and context_window != source_window:
            if self.preferences_area_for_window(context_window) is not None:
                return context_window

        for candidate in reversed(list(windows)):
            if candidate == source_window:
                continue
            if self.preferences_area_for_window(candidate) is not None:
                return candidate
        return None

    def restore_active_section_if_browser_is_alone(self, preferences, windows, browser_window_func):
        if self.active_section_is_compatible(preferences):
            return False
        has_browser_window = False
        for window in windows:
            if self.preferences_area_for_window(window) is None:
                continue
            try:
                if browser_window_func(window):
                    has_browser_window = True
                    continue
            except (AttributeError, ReferenceError):
                pass
            return False
        if not has_browser_window:
            return False
        self.set_active_section(preferences, self.compatible_section)
        self.ensure_draw_handler()
        self.tag_preferences_areas(windows)
        return True

    def restore_popup_active_section_if_alone(self, preferences, windows):
        browser_window = self.interaction["window"]
        if not self.is_popup_window(browser_window):
            return False
        if not self.is_window_live(browser_window, windows):
            return False
        return self.restore_active_section_if_browser_is_alone(
            preferences,
            windows,
            self.is_window,
        )

    def show_area_header(self, window, area, temp_override_func):
        try:
            space = area.spaces.active
        except (AttributeError, ReferenceError):
            return
        if not hasattr(space, "show_region_header"):
            return
        try:
            space.show_region_header = True
        except (AttributeError, RuntimeError, TypeError):
            pass
        header_region = self.region_for_area_type(area, 'HEADER')
        if header_region is None:
            return
        try:
            header_is_bottom = header_region.alignment == 'BOTTOM'
        except (AttributeError, ReferenceError):
            return
        if not header_is_bottom:
            return
        try:
            with temp_override_func(window=window, area=area, region=header_region):
                bpy.ops.screen.region_flip()
        except RuntimeError:
            pass

    def remove_draw_handler(self):
        if self.persistent_draw_handler:
            return
        self.force_remove_draw_handler()

    def force_remove_draw_handler(self):
        if self.draw_handler is None:
            return
        try:
            bpy.types.SpacePreferences.draw_handler_remove(self.draw_handler, 'WINDOW')
        except Exception:
            pass
        self.draw_handler = None

    def ensure_draw_handler(self):
        if self.draw_handler is not None:
            return
        self.draw_handler = bpy.types.SpacePreferences.draw_handler_add(
            self.draw_canvas,
            (),
            'WINDOW',
            'POST_PIXEL',
        )

    def reset_interaction(self):
        self.interaction["session_id"] += 1
        self.interaction["modal_id"] += 1
        self.interaction["window"] = None
        self.interaction["area"] = None
        self.interaction["region"] = None
        self.interaction["source_window"] = None
        self.interaction["source_area"] = None
        self.interaction["source_region"] = None
        self.interaction["scroll_offset"] = 0
        self.interaction["hovered_key"] = None
        self.interaction["scrollbar_dragging"] = False
        self.interaction["scrollbar_drag_y"] = 0
        self.interaction["scrollbar_drag_scroll_offset"] = 0
        self.interaction["timer"] = None
        self.interaction["defer_images_until"] = 0.0
        self.interaction["warm_visible_items"] = False
        self.interaction["warm_direction"] = 1
        self.interaction["docked_host"] = False
        self.interaction["interaction_start_pending"] = False
        self.interaction["pending_window"] = None
        self.interaction["pending_area"] = None
        self.interaction["pending_region"] = None

    def claim_modal(self):
        self.interaction["modal_id"] += 1
        return (
            self.interaction["session_id"],
            self.interaction["modal_id"],
        )

    def stop_interaction(self, tag_redraw, windows):
        self.remove_timer()
        self.remove_draw_handler()
        area = self.interaction["area"]
        self.reset_interaction()
        if tag_redraw and self.is_area_live(area, windows):
            self.tag_area(area)

    def start_interaction(self, window, area, region, window_manager, windows, docked_host):
        self.stop_interaction(True, windows)
        self.interaction["window"] = window
        self.interaction["area"] = area
        self.interaction["region"] = region
        self.interaction["scroll_offset"] = self.remembered_scroll_offset(window_manager)
        self.interaction["warm_visible_items"] = True
        self.interaction["warm_direction"] = 1
        self.interaction["docked_host"] = docked_host
        self.defer_image_loads()
        self.ensure_draw_handler()
        self.tag_area(area)

    def interaction_targets(self, window, area):
        try:
            return self.interaction["window"] == window and self.interaction["area"] == area
        except ReferenceError:
            return False

    def interaction_modal_is_running_for(self, window, area):
        if not self.interaction_targets(window, area):
            return False
        return self.interaction["timer"] is not None

    def request_interaction_start(self, window, area, region):
        self.interaction["pending_window"] = window
        self.interaction["pending_area"] = area
        self.interaction["pending_region"] = region
        if self.interaction["interaction_start_pending"]:
            return
        self.interaction["interaction_start_pending"] = True
        try:
            bpy.app.timers.register(self.start_pending_interaction, first_interval=0.0)
        except ValueError:
            self.interaction["interaction_start_pending"] = False

    def start_pending_interaction(self):
        self.interaction["interaction_start_pending"] = False
        window = self.interaction["pending_window"]
        area = self.interaction["pending_area"]
        region = self.interaction["pending_region"]
        self.interaction["pending_window"] = None
        self.interaction["pending_area"] = None
        self.interaction["pending_region"] = None

        window_manager = getattr(bpy.context, "window_manager", None)
        if window_manager is None:
            return None
        windows = window_manager.windows
        if not self.is_window_live(window, windows):
            return None
        if not self.is_area_live(area, windows):
            return None
        if self.interaction_modal_is_running_for(window, area):
            return None

        try:
            if region is None or region.type != 'WINDOW':
                region = self.region_for_area(area)
        except ReferenceError:
            region = self.region_for_area(area)
        if region is None:
            return None

        try:
            with bpy.context.temp_override(window=window, area=area, region=region):
                workspace = getattr(bpy.context, "workspace", None)
                workspace_name = getattr(workspace, "name", "")
                screen = getattr(window, "screen", None)
                active_section = bpy.context.preferences.active_section
                if active_section != self.compatible_section:
                    return None
                if not self.context_uses_browser(
                        window,
                        screen,
                        workspace_name,
                        active_section,
                        area):
                    return None
                if not self.scene_allowed_for_window(bpy.context.scene, window):
                    return None
                if not self.active_section_is_compatible(bpy.context.preferences):
                    return None
                op_namespace, op_name = self.interaction_operator_idname.split(".", 1)
                getattr(getattr(bpy.ops, op_namespace), op_name)('INVOKE_DEFAULT')
        except (AttributeError, ReferenceError, RuntimeError):
            return None
        return None

    def ensure_interaction_for_context(
            self,
            window,
            screen,
            workspace_name,
            active_section,
            area,
            region,
            scene,
            window_manager):
        if self.interaction_modal_is_running_for(window, area):
            return
        if not self.context_uses_browser(
                window,
                screen,
                workspace_name,
                active_section,
                area):
            return
        if not self.scene_allowed_for_window(scene, window):
            return
        if not self.is_window_live(window, window_manager.windows):
            return
        if not self.is_area_live(area, window_manager.windows):
            return
        if region is None:
            return
        try:
            if region.type != 'WINDOW':
                return
        except ReferenceError:
            return
        self.request_interaction_start(window, area, region)

    def remember_source_context(self, window, area, region, windows):
        if not self.is_window_live(window, windows):
            return
        if self.is_popup_window(window):
            return
        self.interaction["source_window"] = window
        self.interaction["source_area"] = area
        self.interaction["source_region"] = region

    def add_timer(self, window_manager, window):
        self.remove_timer()
        try:
            timer = window_manager.event_timer_add(
                _IMAGE_GRID_TIMER_STEP_SECONDS,
                window=window,
            )
        except RuntimeError:
            timer = None
        self.interaction["timer"] = timer

    def remove_timer(self):
        timer = self.interaction.get("timer")
        if timer is None:
            return
        window_manager = getattr(bpy.context, "window_manager", None)
        if window_manager is not None:
            try:
                window_manager.event_timer_remove(timer)
            except (ReferenceError, RuntimeError):
                pass
        self.interaction["timer"] = None

    def open_popup(
            self,
            preferences,
            window_manager,
            source_window,
            source_area,
            source_region,
            temp_override_func,
            report_func):
        windows = window_manager.windows
        self.stop_interaction(True, windows)
        for window in self.popup_windows(windows):
            self.forget_window_areas(window)
            with temp_override_func(window=window):
                bpy.ops.wm.window_close()

        existing_windows = list(windows)
        try:
            result = bpy.ops.screen.userpref_show(section=self.active_section)
        except RuntimeError as exc:
            report_func({'ERROR'}, f"Could not open {self.display_name} window: {exc}")
            return {'CANCELLED'}
        if 'FINISHED' not in result:
            report_func({'ERROR'}, f"Could not open {self.display_name} window: {result}")
            return {'CANCELLED'}

        new_window = self.preferences_popup_window(
            windows,
            existing_windows,
            source_window,
        )
        if new_window is None:
            report_func(
                {'ERROR'},
                f"Could not open {self.display_name} as a separate Preferences window",
            )
            return {'CANCELLED'}

        if not new_window.screen.areas:
            report_func({'ERROR'}, f"Could not find an area for the {self.display_name}")
            return {'CANCELLED'}

        new_window.screen[self.popup_key] = True
        area = self.preferences_area_for_window(new_window)
        if area is None:
            report_func({'ERROR'}, f"Could not find a Preferences area for the {self.display_name}")
            return {'CANCELLED'}
        self.show_area_header(new_window, area, temp_override_func)
        self.mark_area(
            new_window,
            area,
            preferences,
            windows,
        )
        region = self.region_for_area(area)
        if region is not None:
            self.start_interaction(
                new_window,
                area,
                region,
                window_manager,
                windows,
                False,
            )
            self.remember_source_context(source_window, source_area, source_region, windows)
            with temp_override_func(window=new_window, area=area, region=region):
                op_namespace, op_name = self.interaction_operator_idname.split(".", 1)
                getattr(getattr(bpy.ops, op_namespace), op_name)('INVOKE_DEFAULT')
        return {'FINISHED'}

    def layout_items(
            self,
            items,
            region_width,
            region_height,
            preview_scale_percentage,
            ui_scale,
            pixel_size,
            scroll_offset):
        return image_grid_layout_items(
            items,
            region_width,
            region_height,
            preview_scale_percentage,
            ui_scale,
            pixel_size,
            scroll_offset,
            self.item_rect_func,
            self.grid_spec,
        )

    def layout_for_state(
            self,
            scene,
            search_text,
            display_filter,
            region_width,
            region_height,
            preview_scale_percentage,
            ui_scale,
            pixel_size,
            scroll_offset):
        items = self.display_items_func(scene, search_text, display_filter)
        return self.layout_items(
            items,
            region_width,
            region_height,
            preview_scale_percentage,
            ui_scale,
            pixel_size,
            scroll_offset,
        )

    def draw_canvas(self):
        context = bpy.context
        window = context.window
        screen = getattr(window, "screen", None)
        workspace = getattr(context, "workspace", None)
        workspace_name = getattr(workspace, "name", "")
        if not self.context_uses_browser(
                window,
                screen,
                workspace_name,
                context.preferences.active_section,
                context.area):
            return
        if not self.scene_allowed_for_window(context.scene, context.window):
            return
        if self.interaction["area"] is not None and context.area != self.interaction["area"]:
            if self.is_area_live(self.interaction["area"], context.window_manager.windows):
                return
            self.stop_interaction(False, context.window_manager.windows)

        region = context.region
        if region is None or region.width <= 0 or region.height <= 0:
            return
        if not self.active_section_is_compatible(context.preferences):
            return

        self.ensure_interaction_for_context(
            window,
            screen,
            workspace_name,
            context.preferences.active_section,
            context.area,
            region,
            context.scene,
            context.window_manager,
        )

        scene = context.scene
        window_manager = context.window_manager
        preferences_system = context.preferences.system
        items = self.display_items_for_window_manager(scene, window_manager)
        metrics, rects = self.layout_items(
            items,
            region.width,
            region.height,
            self.preview_scale_func(window_manager),
            preferences_system.ui_scale,
            preferences_system.pixel_size,
            self.interaction["scroll_offset"],
        )
        self.interaction["scroll_offset"] = metrics["scroll_offset"]
        self.remember_scroll_offset(window_manager, metrics["scroll_offset"])
        metrics["hovered_key"] = self.interaction["hovered_key"]
        metrics["scrollbar_dragging"] = self.interaction["scrollbar_dragging"]
        metrics["defer_image_loads"] = (
            self.defer_images_active()
            or self.interaction["scrollbar_dragging"]
        )
        metrics["defer_images"] = (
            metrics["defer_image_loads"]
            or self.interaction["warm_visible_items"]
        )
        metrics["warm_direction"] = self.interaction["warm_direction"]
        if self.interaction["timer"] is None:
            if metrics["defer_image_loads"]:
                self.interaction["warm_visible_items"] = True
            else:
                warm_deadline = time.perf_counter() + _IMAGE_GRID_WARM_TIME_BUDGET_SECONDS
                warmed_count, visible_pending = self.warm_cache_for_draw(
                    items,
                    metrics,
                    warm_deadline,
                )
                self.interaction["warm_visible_items"] = visible_pending
                if warmed_count > 0 or visible_pending:
                    self.tag_area(context.area)

        import gpu

        gpu.state.blend_set('ALPHA')
        try:
            for rect in rects:
                self.draw_cell_func(rect, metrics)
            draw_image_grid_scrollbar(metrics)
        finally:
            gpu.state.blend_set('NONE')

    def _navigation_draw_override(self, panel, context):
        window = context.window
        screen = getattr(window, "screen", None)
        workspace = getattr(context, "workspace", None)
        workspace_name = getattr(workspace, "name", "")
        if not self.context_uses_browser(
                window,
                screen,
                workspace_name,
                context.preferences.active_section,
                context.area):
            draw_func = self.userpref_draws.get(type(panel).__name__)
            if draw_func is not None:
                return draw_func(panel, context)
            return None
        self.draw_navigation_func(
            panel.layout,
            context.scene,
            context.window_manager,
        )
        return None

    def _header_draw_override(self, panel, context):
        window = context.window
        screen = getattr(window, "screen", None)
        workspace = getattr(context, "workspace", None)
        workspace_name = getattr(workspace, "name", "")
        if not self.context_uses_browser(
                window,
                screen,
                workspace_name,
                context.preferences.active_section,
                context.area):
            draw_func = self.userpref_draws.get(type(panel).__name__)
            if draw_func is not None:
                return draw_func(panel, context)
            return None
        self.draw_header_func(
            panel.layout,
            context.scene,
            context.window_manager,
            context.preferences,
            self.active_section_is_compatible(context.preferences),
        )
        return None

    def _content_draw_override(self, panel, context):
        window = context.window
        screen = getattr(window, "screen", None)
        workspace = getattr(context, "workspace", None)
        workspace_name = getattr(workspace, "name", "")
        if not self.context_uses_browser(
                window,
                screen,
                workspace_name,
                context.preferences.active_section,
                context.area):
            draw_func = self.userpref_draws.get(type(panel).__name__)
            if draw_func is not None:
                return draw_func(panel, context)
            return None
        self.draw_content_func(
            panel.layout,
            context.scene,
            context.window_manager,
            context.region.width,
            context.preferences.system.ui_scale,
            context.preferences.system.pixel_size,
        )
        return None

    def _empty_draw_override(self, panel, context):
        window = context.window
        screen = getattr(window, "screen", None)
        workspace = getattr(context, "workspace", None)
        workspace_name = getattr(workspace, "name", "")
        if not self.context_uses_browser(
                window,
                screen,
                workspace_name,
                context.preferences.active_section,
                context.area):
            draw_func = self.userpref_draws.get(type(panel).__name__)
            if draw_func is not None:
                return draw_func(panel, context)
        return None

    def invoke_interaction(
            self,
            operator,
            window,
            screen,
            workspace_name,
            active_section,
            area,
            region,
            scene,
            window_manager):
        windows = window_manager.windows
        if not self.context_uses_browser(
                window,
                screen,
                workspace_name,
                active_section,
                area):
            return {'CANCELLED'}
        if not self.scene_allowed_for_window(scene, window):
            return {'CANCELLED'}
        if active_section != self.compatible_section:
            return {'CANCELLED'}
        if window is None or area is None:
            return {'CANCELLED'}
        if not self.is_window_live(window, windows):
            return {'CANCELLED'}
        if not self.is_area_live(area, windows):
            return {'CANCELLED'}
        if self.interaction_modal_is_running_for(window, area):
            return {'CANCELLED'}
        try:
            if region is None or region.type != 'WINDOW':
                region = self.region_for_area(area)
        except ReferenceError:
            region = self.region_for_area(area)
        if region is None:
            return {'CANCELLED'}
        if not self.interaction_targets(window, area):
            self.start_interaction(
                window,
                area,
                region,
                window_manager,
                windows,
                not self.is_window(window),
            )
        self.interaction["region"] = region
        operator._session_id, operator._modal_id = self.claim_modal()
        self.add_timer(window_manager, window)
        window_manager.modal_handler_add(operator)
        return {'RUNNING_MODAL'}

    def is_current_modal(self, operator):
        return (
            hasattr(operator, "_session_id")
            and hasattr(operator, "_modal_id")
            and operator._session_id == self.interaction["session_id"]
            and operator._modal_id == self.interaction["modal_id"]
        )

    def update_hover(self, area, hit):
        hovered_key = None
        if hit is not None:
            hovered_key = hit["key"]
        if hovered_key == self.interaction["hovered_key"]:
            return
        self.interaction["hovered_key"] = hovered_key
        self.tag_area(area)

    def defer_image_loads(self):
        self.interaction["defer_images_until"] = (
            time.perf_counter() + _IMAGE_GRID_SCROLL_DEFER_SECONDS
        )

    def defer_images_active(self):
        return time.perf_counter() < self.interaction["defer_images_until"]

    def modal_timer(self, scene, window_manager, preferences, region, area):
        preferences_system = preferences.system
        items = self.display_items_for_window_manager(scene, window_manager)
        metrics, _rects = self.layout_items(
            items,
            region.width,
            region.height,
            self.preview_scale_func(window_manager),
            preferences_system.ui_scale,
            preferences_system.pixel_size,
            self.interaction["scroll_offset"],
        )
        self.interaction["scroll_offset"] = metrics["scroll_offset"]
        self.remember_scroll_offset(window_manager, metrics["scroll_offset"])
        metrics["defer_image_loads"] = self.defer_images_active()
        metrics["warm_direction"] = self.interaction["warm_direction"]

        warm_deadline = time.perf_counter() + _IMAGE_GRID_WARM_TIME_BUDGET_SECONDS
        warmed_count, visible_pending = self.warm_cache_for_draw(
            items,
            metrics,
            warm_deadline,
        )
        self.interaction["warm_visible_items"] = visible_pending
        if warmed_count > 0 or visible_pending:
            self.tag_area(area)

    def warm_cache_for_draw(self, items, metrics, warm_deadline):
        if len(items) <= 0 or metrics["visible_item_count"] <= 0:
            return 0, False
        if metrics.get("defer_image_loads", False):
            return 0, True

        # Keep the priority order obvious: visible page first, then nearby pages.
        warmed_count = 0
        visible_complete = True
        visible_start = metrics["visible_start_index"]
        visible_end = metrics["visible_end_index"]

        warmed_count, visible_complete = self.warm_item_range(
            items,
            visible_start,
            visible_end,
            1,
            warm_deadline,
            warmed_count,
        )
        if not visible_complete:
            return warmed_count, True

        warmed_count = self.warm_nearby_items(
            items,
            metrics,
            warm_deadline,
            warmed_count,
        )

        return warmed_count, False

    def warm_nearby_items(self, items, metrics, warm_deadline, warmed_count):
        visible_start = metrics["visible_start_index"]
        visible_end = metrics["visible_end_index"]
        visible_rows = max(
            1,
            metrics["visible_last_row"] - metrics["visible_first_row"] + 1,
        )
        page_count = metrics["columns"] * visible_rows
        ahead_count = page_count * _IMAGE_GRID_WARM_PAGES_AHEAD
        behind_count = page_count * _IMAGE_GRID_WARM_PAGES_BEHIND

        if metrics.get("warm_direction", 1) < 0:
            ranges = (
                (visible_start - 1, max(-1, visible_start - ahead_count - 1), -1),
                (visible_end, min(len(items), visible_end + behind_count), 1),
            )
        else:
            ranges = (
                (visible_end, min(len(items), visible_end + ahead_count), 1),
                (visible_start - 1, max(-1, visible_start - behind_count - 1), -1),
            )

        for start_index, stop_index, step in ranges:
            if time.perf_counter() >= warm_deadline:
                break
            warmed_count, _range_complete = self.warm_item_range(
                items,
                start_index,
                stop_index,
                step,
                warm_deadline,
                warmed_count,
            )
        return warmed_count

    def warm_item_range(
            self,
            items,
            start_index,
            stop_index,
            step,
            warm_deadline,
            warmed_count):
        index = start_index
        while index != stop_index:
            if time.perf_counter() >= warm_deadline:
                return warmed_count, False
            if 0 <= index < len(items):
                if self.warm_item_func(items[index]):
                    warmed_count += 1
            index += step
        return warmed_count, True

    def scroll(self, window_manager, area, metrics, direction):
        previous_scroll_offset = self.interaction["scroll_offset"]
        step = metrics["cell_height"] + metrics["gap"]
        scroll_offset = self.interaction["scroll_offset"] + direction * step
        self.interaction["scroll_offset"] = min(
            max(0, scroll_offset),
            metrics["max_scroll"],
        )
        if self.interaction["scroll_offset"] != previous_scroll_offset:
            self.interaction["warm_direction"] = direction
            self.interaction["warm_visible_items"] = True
            self.remember_scroll_offset(window_manager, self.interaction["scroll_offset"])
        self.defer_image_loads()
        self.tag_area(area)

    def scroll_page_to_mouse(self, window_manager, area, metrics, mouse_y):
        previous_scroll_offset = self.interaction["scroll_offset"]
        scroll_offset = image_grid_scrollbar_page_scroll(metrics, mouse_y)
        self.interaction["scroll_offset"] = scroll_offset
        if scroll_offset > previous_scroll_offset:
            self.interaction["warm_direction"] = 1
            self.interaction["warm_visible_items"] = True
        elif scroll_offset < previous_scroll_offset:
            self.interaction["warm_direction"] = -1
            self.interaction["warm_visible_items"] = True
        if scroll_offset != previous_scroll_offset:
            self.remember_scroll_offset(window_manager, scroll_offset)
        self.defer_image_loads()
        self.tag_area(area)

    def start_scrollbar_drag(self, area, metrics, mouse_y):
        self.interaction["scrollbar_dragging"] = True
        self.interaction["scrollbar_drag_y"] = mouse_y
        self.interaction["scrollbar_drag_scroll_offset"] = metrics["scroll_offset"]
        self.interaction["warm_visible_items"] = True
        self.defer_image_loads()
        self.tag_area(area)

    def update_scrollbar_drag(self, window_manager, area, metrics, mouse_y):
        previous_scroll_offset = self.interaction["scroll_offset"]
        scroll_offset = image_grid_scrollbar_drag_scroll(
            metrics,
            self.interaction["scrollbar_drag_y"],
            mouse_y,
            self.interaction["scrollbar_drag_scroll_offset"],
        )
        self.interaction["scroll_offset"] = scroll_offset
        if scroll_offset > previous_scroll_offset:
            self.interaction["warm_direction"] = 1
            self.interaction["warm_visible_items"] = True
        elif scroll_offset < previous_scroll_offset:
            self.interaction["warm_direction"] = -1
            self.interaction["warm_visible_items"] = True
        if scroll_offset != previous_scroll_offset:
            self.remember_scroll_offset(window_manager, scroll_offset)
        self.defer_image_loads()
        self.tag_area(area)

    def stop_scrollbar_drag(self, area):
        self.interaction["scrollbar_dragging"] = False
        self.interaction["defer_images_until"] = 0.0
        self.tag_area(area)

    def event_is_timer(self, event):
        if event.type != 'TIMER':
            return False
        event_timer = getattr(event, "timer", None)
        if event_timer is None:
            return True
        return event_timer == self.interaction.get("timer")

    def modal(self, operator, event, scene, active_window, window_manager, preferences):
        if not self.is_current_modal(operator):
            return {'FINISHED'}

        windows = window_manager.windows
        if not self.scene_allowed_for_window(scene, active_window):
            self.finish_invalid_interaction(preferences, windows)
            return {'FINISHED'}
        if self.restore_if_closed(preferences, windows):
            return {'FINISHED'}

        browser_window = self.interaction["window"]
        area = self.interaction["area"]
        if area is None or browser_window is None:
            self.finish_invalid_interaction(preferences, windows)
            return {'FINISHED'}
        if (self.interaction.get("docked_host", False)
                and not self.active_section_is_compatible(preferences)):
            self.finish_invalid_interaction(preferences, windows)
            return {'FINISHED'}
        if (not self.is_window_live(browser_window, windows)
                or not self.is_area_live(area, windows)):
            self.finish_invalid_interaction(preferences, windows)
            return {'FINISHED'}
        region = self.region_for_area(area)
        if region is None:
            self.finish_invalid_interaction(preferences, windows)
            return {'FINISHED'}
        self.interaction["region"] = region
        if self.event_is_timer(event):
            self.restore_popup_active_section_if_alone(preferences, windows)
            if self.active_section_is_compatible(preferences):
                self.modal_timer(scene, window_manager, preferences, region, area)
            return {'RUNNING_MODAL'}
        if active_window != browser_window:
            return {'PASS_THROUGH'}
        if region.width <= 0 or region.height <= 0:
            return {'PASS_THROUGH'}

        preferences_system = preferences.system
        metrics, rects = self.layout_for_state(
            scene,
            self.search_text_func(window_manager),
            self.filter_value_func(window_manager),
            region.width,
            region.height,
            self.preview_scale_func(window_manager),
            preferences_system.ui_scale,
            preferences_system.pixel_size,
            self.interaction["scroll_offset"],
        )
        self.interaction["scroll_offset"] = metrics["scroll_offset"]
        self.remember_scroll_offset(window_manager, metrics["scroll_offset"])
        mouse_x, mouse_y = image_grid_mouse_region_coords(region, event.mouse_x, event.mouse_y)
        mouse_in_region = image_grid_mouse_in_region(region, mouse_x, mouse_y)
        if not mouse_in_region:
            if event.type == 'MOUSEMOVE':
                if self.interaction["scrollbar_dragging"]:
                    self.update_scrollbar_drag(window_manager, area, metrics, mouse_y)
                    return {'RUNNING_MODAL'}
                self.update_hover(area, None)
                return {'PASS_THROUGH'}
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                if self.interaction["scrollbar_dragging"]:
                    self.stop_scrollbar_drag(area)
                    return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        scrollbar_hit = image_grid_scrollbar_hit_test(metrics, mouse_x, mouse_y)
        hit = None
        if scrollbar_hit is None:
            hit = image_grid_hit_test(rects, metrics, mouse_x, mouse_y)

        if event.type == 'MOUSEMOVE':
            if self.interaction["scrollbar_dragging"]:
                self.update_scrollbar_drag(window_manager, area, metrics, mouse_y)
                return {'RUNNING_MODAL'}
            self.update_hover(area, hit)
            return {'PASS_THROUGH'}

        if (event.type == 'LEFTMOUSE'
                and event.value == 'PRESS'
                and scrollbar_hit == 'THUMB'):
            self.start_scrollbar_drag(area, metrics, mouse_y)
            return {'RUNNING_MODAL'}

        if (event.type == 'LEFTMOUSE'
                and event.value == 'PRESS'
                and scrollbar_hit == 'TRACK'):
            self.scroll_page_to_mouse(window_manager, area, metrics, mouse_y)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self.interaction["scrollbar_dragging"]:
                self.stop_scrollbar_drag(area)
                return {'RUNNING_MODAL'}

        if (event.type == 'WHEELDOWNMOUSE'
                and event.value == 'PRESS'
                and image_grid_mouse_in_grid(metrics, mouse_y)):
            self.scroll(window_manager, area, metrics, 1)
            return {'RUNNING_MODAL'}

        if (event.type == 'WHEELUPMOUSE'
                and event.value == 'PRESS'
                and image_grid_mouse_in_grid(metrics, mouse_y)):
            self.scroll(window_manager, area, metrics, -1)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and hit is not None:
            hit["mouse_x"] = mouse_x
            hit["mouse_y"] = mouse_y
            result = self.activate_item_func(hit)
            if 'FINISHED' in result:
                return {'FINISHED'}
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}


def image_grid_preview_icon_scale(preview_scale_percentage, grid_spec):
    return (
        preview_scale_percentage / 100.0 * grid_spec.icon_scale_range
        + grid_spec.min_icon_scale
    )


def image_grid_widget_unit(ui_scale, pixel_size):
    # Mirrors Blender's U.widget_unit calculation from source/blender/windowmanager/intern/wm_window.cc.
    return int(math.floor(18.0 * ui_scale + 0.5)) + (2 * int(pixel_size))


def image_grid_cell_padding(widget_unit):
    return widget_unit


def image_grid_item_gap():
    return 0


def image_grid_target_cell_width(preview_scale_percentage, ui_scale, pixel_size, grid_spec):
    icon_scale = image_grid_preview_icon_scale(preview_scale_percentage, grid_spec)
    widget_unit = image_grid_widget_unit(ui_scale, pixel_size)
    return int(math.ceil(icon_scale * widget_unit + image_grid_cell_padding(widget_unit)))


def image_grid_grid_columns(
        region_width,
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        reserved_width,
        grid_spec):
    widget_unit = image_grid_widget_unit(ui_scale, pixel_size)
    target_cell_width = image_grid_target_cell_width(
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        grid_spec,
    )
    margin = max(12, int(round(widget_unit * 0.6)))
    gap = image_grid_item_gap()
    available_width = region_width - margin * 2 - reserved_width
    if available_width <= target_cell_width:
        return 1
    return max(1, int((available_width + gap) // (target_cell_width + gap)))


def image_grid_canvas_metrics(
        region_width,
        region_height,
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        item_count,
        grid_spec):
    widget_unit = image_grid_widget_unit(ui_scale, pixel_size)
    scrollbar_width = max(14, int(round(widget_unit * 0.7)))
    scrollbar_gap = max(6, int(round(widget_unit * 0.3)))
    scrollbar_edge_margin = max(4, int(round(widget_unit * 0.25)))
    reserved_width = 0
    if item_count > 0:
        reserved_width = scrollbar_width + scrollbar_gap + scrollbar_edge_margin
    metrics = _image_grid_canvas_metrics_with_reserved_width(
        region_width,
        region_height,
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        item_count,
        reserved_width,
        scrollbar_width,
        scrollbar_gap,
        grid_spec,
    )
    metrics["scrollbar_visible"] = item_count > 0
    metrics["scrollbar_edge_margin"] = scrollbar_edge_margin
    return metrics


def _image_grid_canvas_metrics_with_reserved_width(
        region_width,
        region_height,
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        item_count,
        reserved_width,
        scrollbar_width,
        scrollbar_gap,
        grid_spec):
    widget_unit = image_grid_widget_unit(ui_scale, pixel_size)
    icon_size = int(round(image_grid_preview_icon_scale(preview_scale_percentage, grid_spec) * widget_unit))
    margin = max(12, int(round(widget_unit * 0.6)))
    gap = image_grid_item_gap()
    line_height = max(18, int(round(widget_unit * 0.85)))
    label_space = line_height * grid_spec.text_line_count + int(round(widget_unit * 0.45))
    cell_height = icon_size + label_space + margin
    target_cell_width = image_grid_target_cell_width(
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        grid_spec,
    )
    columns = image_grid_grid_columns(
        region_width,
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        reserved_width,
        grid_spec,
    )
    cell_width = target_cell_width
    grid_top = region_height - margin
    grid_bottom = margin
    grid_right = region_width - margin - reserved_width
    content_height = image_grid_content_height_for_columns(item_count, columns, cell_height, gap)
    viewport_height = max(1, grid_top - grid_bottom)
    max_scroll = max(0, content_height - viewport_height)
    return {
        "widget_unit": widget_unit,
        "region_width": region_width,
        "icon_size": icon_size,
        "margin": margin,
        "gap": gap,
        "line_height": line_height,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "columns": columns,
        "grid_top": grid_top,
        "grid_bottom": grid_bottom,
        "grid_right": grid_right,
        "content_height": content_height,
        "max_scroll": max_scroll,
        "scrollbar_width": scrollbar_width,
        "scrollbar_gap": scrollbar_gap,
        "scrollbar_reserved_width": reserved_width,
    }


def image_grid_grid_row_count(item_count, columns):
    if item_count <= 0:
        return 0
    return int(math.ceil(item_count / columns))


def image_grid_content_height_for_columns(item_count, columns, cell_height, gap):
    rows = image_grid_grid_row_count(item_count, columns)
    if rows <= 0:
        return 0
    return rows * cell_height + (rows - 1) * gap


def image_grid_content_height(item_count, metrics):
    return image_grid_content_height_for_columns(
        item_count,
        metrics["columns"],
        metrics["cell_height"],
        metrics["gap"],
    )


def image_grid_max_scroll(item_count, metrics):
    viewport_height = max(1, metrics["grid_top"] - metrics["grid_bottom"])
    content_height = image_grid_content_height(item_count, metrics)
    return max(0, content_height - viewport_height)


def image_grid_clamped_scroll(scroll_offset, item_count, metrics):
    return min(max(0, int(scroll_offset)), image_grid_max_scroll(item_count, metrics))


def image_grid_layout_items(
        items,
        region_width,
        region_height,
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        scroll_offset,
        item_rect_func,
        grid_spec):
    metrics = image_grid_canvas_metrics(
        region_width,
        region_height,
        preview_scale_percentage,
        ui_scale,
        pixel_size,
        len(items),
        grid_spec,
    )
    item_count = len(items)
    columns = metrics["columns"]
    scroll_offset = min(max(0, int(scroll_offset)), metrics["max_scroll"])
    metrics["scroll_offset"] = scroll_offset
    metrics["item_count"] = item_count

    rects = []
    first_row, last_row = image_grid_visible_row_range(
        item_count,
        columns,
        metrics,
        scroll_offset,
    )
    first_index = min(item_count, first_row * columns)
    last_index = min(item_count, (last_row + 1) * columns)
    metrics["visible_first_row"] = first_row
    metrics["visible_last_row"] = last_row
    metrics["visible_start_index"] = first_index
    metrics["visible_end_index"] = last_index
    metrics["visible_item_count"] = max(0, last_index - first_index)

    for index in range(first_index, last_index):
        item = items[index]
        row = index // columns
        column = index % columns
        x = metrics["margin"] + column * (metrics["cell_width"] + metrics["gap"])
        y = (
            metrics["grid_top"]
            - metrics["cell_height"]
            - row * (metrics["cell_height"] + metrics["gap"])
            + scroll_offset
        )
        rect = item_rect_func(item, x, y, metrics["cell_width"], metrics["cell_height"])
        rect["item_index"] = index
        rects.append(rect)
    return metrics, rects


def image_grid_visible_row_range(item_count, columns, metrics, scroll_offset):
    row_count = image_grid_grid_row_count(item_count, columns)
    if row_count <= 0:
        return 0, -1

    row_stride = metrics["cell_height"] + metrics["gap"]
    viewport_height = max(1, metrics["grid_top"] - metrics["grid_bottom"])
    first_row = int(math.ceil((scroll_offset - metrics["cell_height"]) / row_stride))
    last_row = int(math.floor((scroll_offset + viewport_height) / row_stride))
    first_row = max(0, first_row)
    last_row = min(row_count - 1, last_row)
    if last_row < first_row:
        return first_row, first_row - 1
    return first_row, last_row


def image_grid_rect_contains(rect, mouse_x, mouse_y):
    return (
        rect["x"] <= mouse_x <= rect["x"] + rect["w"]
        and rect["y"] <= mouse_y <= rect["y"] + rect["h"]
    )


def image_grid_hit_test(rects, metrics, mouse_x, mouse_y):
    if not image_grid_mouse_in_grid(metrics, mouse_y):
        return None
    item_index = image_grid_item_index_at_mouse(metrics, mouse_x, mouse_y)
    if item_index is not None:
        visible_start_index = metrics.get("visible_start_index")
        if visible_start_index is not None:
            rect_offset = item_index - visible_start_index
            if 0 <= rect_offset < len(rects):
                rect = rects[rect_offset]
                if (
                        rect.get("item_index") == item_index
                        and image_grid_rect_contains(rect, mouse_x, mouse_y)):
                    return rect
        elif item_index < len(rects):
            rect = rects[item_index]
            if image_grid_rect_contains(rect, mouse_x, mouse_y):
                return rect
    for rect in rects:
        if image_grid_rect_contains(rect, mouse_x, mouse_y):
            return rect
    return None


def image_grid_item_index_at_mouse(metrics, mouse_x, mouse_y):
    item_count = metrics.get("item_count")
    if item_count is None or item_count <= 0:
        return None

    x_offset = mouse_x - metrics["margin"]
    if x_offset < 0:
        return None
    column_stride = metrics["cell_width"] + metrics["gap"]
    column = int(x_offset // column_stride)
    if column < 0 or column >= metrics["columns"]:
        return None
    if x_offset - column * column_stride > metrics["cell_width"]:
        return None

    y_offset = metrics["grid_top"] + metrics["scroll_offset"] - mouse_y
    if y_offset < 0:
        return None
    row_stride = metrics["cell_height"] + metrics["gap"]
    row = int(y_offset // row_stride)
    if y_offset - row * row_stride > metrics["cell_height"]:
        return None

    item_index = row * metrics["columns"] + column
    if item_index >= item_count:
        return None
    return item_index


def image_grid_mouse_in_grid(metrics, mouse_y):
    return metrics["grid_bottom"] <= mouse_y <= metrics["grid_top"]


def image_grid_mouse_region_coords(region, mouse_x, mouse_y):
    return mouse_x - region.x, mouse_y - region.y


def image_grid_mouse_in_region(region, mouse_x, mouse_y):
    return 0 <= mouse_x <= region.width and 0 <= mouse_y <= region.height


def image_grid_scrollbar_rect(metrics):
    if not metrics.get("scrollbar_visible", False):
        return None
    track_height = max(1, metrics["grid_top"] - metrics["grid_bottom"])
    content_height = max(track_height, metrics["content_height"])
    thumb_range = 0
    if metrics["max_scroll"] > 0:
        thumb_height = max(
            metrics["widget_unit"],
            int(round(track_height * track_height / content_height)),
        )
        thumb_height = min(track_height, thumb_height)
        thumb_range = max(1, track_height - thumb_height)
        scroll_fraction = metrics["scroll_offset"] / metrics["max_scroll"]
        thumb_y = metrics["grid_bottom"] + thumb_range * (1.0 - scroll_fraction)
    else:
        thumb_height = track_height
        thumb_y = metrics["grid_bottom"]
    track_x = (
        metrics["region_width"]
        - metrics["scrollbar_width"]
        - metrics["scrollbar_edge_margin"]
    )
    return {
        "track_x": track_x,
        "track_y": metrics["grid_bottom"],
        "track_w": metrics["scrollbar_width"],
        "track_h": track_height,
        "thumb_x": track_x,
        "thumb_y": thumb_y,
        "thumb_w": metrics["scrollbar_width"],
        "thumb_h": thumb_height,
        "thumb_range": thumb_range,
    }


def image_grid_scrollbar_hit_test(metrics, mouse_x, mouse_y):
    rect = image_grid_scrollbar_rect(metrics)
    if rect is None:
        return None
    if metrics["max_scroll"] <= 0:
        return None
    thumb = {
        "x": rect["thumb_x"],
        "y": rect["thumb_y"],
        "w": rect["thumb_w"],
        "h": rect["thumb_h"],
    }
    track = {
        "x": rect["track_x"],
        "y": rect["track_y"],
        "w": rect["track_w"],
        "h": rect["track_h"],
    }
    if image_grid_rect_contains(thumb, mouse_x, mouse_y):
        return 'THUMB'
    if image_grid_rect_contains(track, mouse_x, mouse_y):
        return 'TRACK'
    return None


def image_grid_scrollbar_drag_scroll(metrics, start_mouse_y, mouse_y, start_scroll_offset):
    rect = image_grid_scrollbar_rect(metrics)
    if rect is None:
        return 0
    if rect["thumb_range"] <= 0:
        return 0
    delta_y = mouse_y - start_mouse_y
    scroll_delta = -delta_y / rect["thumb_range"] * metrics["max_scroll"]
    return min(max(0, int(round(start_scroll_offset + scroll_delta))), metrics["max_scroll"])


def image_grid_scrollbar_page_scroll(metrics, mouse_y):
    rect = image_grid_scrollbar_rect(metrics)
    if rect is None:
        return 0
    viewport_height = max(1, metrics["grid_top"] - metrics["grid_bottom"])
    thumb_top = rect["thumb_y"] + rect["thumb_h"]
    if mouse_y > thumb_top:
        return max(0, metrics["scroll_offset"] - viewport_height)
    if mouse_y < rect["thumb_y"]:
        return min(metrics["max_scroll"], metrics["scroll_offset"] + viewport_height)
    return metrics["scroll_offset"]


def draw_image_grid_rect(x, y, width, height, color):
    import gpu
    from gpu_extras.batch import batch_for_shader

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.uniform_float("color", color)
    verts = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
    batch.draw(shader)


def image_grid_text_for_width(text, max_width, font_size):
    import blf

    font_id = 0
    blf.size(font_id, font_size)
    width, _height = blf.dimensions(font_id, text)
    if width <= max_width:
        return text
    suffix = "..."
    available = max_width - blf.dimensions(font_id, suffix)[0]
    if available <= 0:
        return suffix
    trimmed = text
    while trimmed:
        candidate = trimmed + suffix
        width, _height = blf.dimensions(font_id, candidate)
        if width <= max_width:
            return candidate
        trimmed = trimmed[:-1]
    return suffix


def draw_image_grid_text(text, x, y, max_width, font_size, color, align):
    import blf

    font_id = 0
    blf.size(font_id, font_size)
    display_text = image_grid_text_for_width(text, max_width, font_size)
    text_width, _text_height = blf.dimensions(font_id, display_text)
    draw_x = x
    if align == 'CENTER':
        draw_x = x + (max_width - text_width) / 2
    elif align == 'RIGHT':
        draw_x = x + max_width - text_width
    blf.color(font_id, color[0], color[1], color[2], color[3])
    blf.position(font_id, draw_x, y, 0)
    blf.draw(font_id, display_text)


def _image_grid_texture_bounds(texture_info, x, y, width, height):
    texture, texture_width, texture_height = texture_info
    aspect = texture_width / texture_height
    draw_width = width
    draw_height = height
    if aspect > 1.0:
        draw_height = width / aspect
    else:
        draw_width = height * aspect
    draw_x = x + (width - draw_width) / 2
    draw_y = y + (height - draw_height) / 2
    return texture, draw_x, draw_y, draw_width, draw_height


def _ensure_image_grid_display_texture_shader():
    import gpu

    global _image_grid_display_texture_shader
    if _image_grid_display_texture_shader is not None:
        return _image_grid_display_texture_shader

    vert_info = gpu.types.GPUStageInterfaceInfo("image_grid_display_texture_iface")
    vert_info.smooth('VEC2', "uv_interp")

    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant('MAT4', "ModelViewProjectionMatrix")
    shader_info.sampler(0, 'FLOAT_2D', "image")
    shader_info.vertex_in(0, 'VEC2', "pos")
    shader_info.vertex_in(1, 'VEC2', "texCoord")
    shader_info.vertex_out(vert_info)
    shader_info.fragment_out(0, 'VEC4', "FragColor")
    shader_info.vertex_source(_IMAGE_GRID_TEXTURE_VERTEX_SOURCE)
    shader_info.fragment_source(_IMAGE_GRID_DISPLAY_TEXTURE_FRAGMENT_SOURCE)

    try:
        _image_grid_display_texture_shader = gpu.shader.create_from_info(shader_info)
    except Exception:
        _image_grid_display_texture_shader = None
    return _image_grid_display_texture_shader


def draw_image_grid_texture(texture_info, x, y, width, height):
    import gpu
    from gpu_extras.batch import batch_for_shader

    texture, draw_x, draw_y, draw_width, draw_height = _image_grid_texture_bounds(
        texture_info,
        x,
        y,
        width,
        height,
    )

    shader = gpu.shader.from_builtin('IMAGE')
    shader.uniform_sampler("image", texture)
    verts = [
        (draw_x, draw_y),
        (draw_x + draw_width, draw_y),
        (draw_x + draw_width, draw_y + draw_height),
        (draw_x, draw_y + draw_height),
    ]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": uvs})
    batch.draw(shader)


def draw_image_grid_texture_display(texture_info, x, y, width, height):
    import gpu
    from gpu_extras.batch import batch_for_shader

    shader = _ensure_image_grid_display_texture_shader()
    if shader is None:
        draw_image_grid_texture(texture_info, x, y, width, height)
        return

    texture, draw_x, draw_y, draw_width, draw_height = _image_grid_texture_bounds(
        texture_info,
        x,
        y,
        width,
        height,
    )
    verts = [
        (draw_x, draw_y),
        (draw_x + draw_width, draw_y),
        (draw_x + draw_width, draw_y + draw_height),
        (draw_x, draw_y + draw_height),
    ]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": uvs})

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_sampler("image", texture)
    batch.draw(shader)


def draw_image_grid_scrollbar(metrics):
    rect = image_grid_scrollbar_rect(metrics)
    if rect is None:
        return
    track_color = (0.06, 0.07, 0.08, 0.82)
    thumb_color = (0.66, 0.72, 0.80, 0.94)
    if metrics["max_scroll"] <= 0:
        thumb_color = (0.38, 0.42, 0.48, 0.55)
    if metrics.get("scrollbar_dragging", False):
        thumb_color = (0.78, 0.84, 0.92, 1.0)
    draw_image_grid_rect(
        rect["track_x"],
        rect["track_y"],
        rect["track_w"],
        rect["track_h"],
        track_color,
    )
    draw_image_grid_rect(
        rect["thumb_x"],
        rect["thumb_y"],
        rect["thumb_w"],
        rect["thumb_h"],
        thumb_color,
    )
