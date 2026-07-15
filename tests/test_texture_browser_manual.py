"""Targeted manual texture browser visual and performance checks.

These methods are intentionally named with ``manual_`` so unittest discovery
does not run them as part of the normal suite. Run them by full test name when
you want a visible browser window to inspect.
"""

import os
import struct
import time

import bpy

from . import base_test
from .base_test import AnvilTestCase
from .test_texture_browser_host import (
    _restore_texture_browser_preferences,
    _snapshot_texture_browser_preferences,
)
from ..core.modal_image_grid import image_grid_hit_test
from ..texture_browser import browser as texture_browser
from ..texture_browser import persistence as texture_browser_persistence


VISUAL_TEXTURE_FILE_COUNT = 3000
VISUAL_SECONDARY_TEXTURE_FILE_COUNT = 600
VISUAL_NON_IMAGE_FILE_COUNT = 300
VISUAL_COLLECTION_FILE_COUNT = 1000
VISUAL_TEXTURE_SIZE = 16
VISUAL_LAYOUT_SAMPLE_COUNT = 5
VISUAL_HIT_TEST_SAMPLE_COUNT = 200


def _test_output_root():
    output_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "test_outputs")
    )
    os.makedirs(output_root, exist_ok=True)
    return output_root


def _clear_folder(folder):
    if os.path.isdir(folder):
        for root, dirs, files in os.walk(folder, topdown=False):
            for filename in files:
                os.remove(os.path.join(root, filename))
            for dirname in dirs:
                os.rmdir(os.path.join(root, dirname))
    os.makedirs(folder, exist_ok=True)


def _remove_folder(folder):
    if not os.path.isdir(folder):
        return
    for root, dirs, files in os.walk(folder, topdown=False):
        for filename in files:
            os.remove(os.path.join(root, filename))
        for dirname in dirs:
            os.rmdir(os.path.join(root, dirname))
    os.rmdir(folder)


def _remove_visual_texture_folders():
    output_root = _test_output_root()
    _remove_folder(os.path.join(output_root, "visual_texture_browser_primary"))
    _remove_folder(os.path.join(output_root, "visual_texture_browser_secondary"))


def _bmp_color(texture_index, x, y):
    hue = (texture_index * 37 + x * 11 + y * 19) % 255
    red = (80 + hue) % 255
    green = (130 + hue * 2) % 255
    blue = (210 + hue * 3) % 255
    edge = x == 0 or y == 0 or x == VISUAL_TEXTURE_SIZE - 1 or y == VISUAL_TEXTURE_SIZE - 1
    if edge:
        return 12, 12, 14
    shade = 0.70 + (x + y) / (VISUAL_TEXTURE_SIZE * 2.0) * 0.30
    return int(blue * shade), int(green * shade), int(red * shade)


def _write_visual_bmp(filepath, texture_index):
    width = VISUAL_TEXTURE_SIZE
    height = VISUAL_TEXTURE_SIZE
    row_stride = ((width * 3 + 3) // 4) * 4
    pixel_bytes = bytearray()
    for y in reversed(range(height)):
        row = bytearray()
        for x in range(width):
            row.extend(_bmp_color(texture_index, x, y))
        row.extend(b"\0" * (row_stride - width * 3))
        pixel_bytes.extend(row)

    file_size = 54 + len(pixel_bytes)
    header = bytearray()
    header.extend(b"BM")
    header.extend(struct.pack("<IHHI", file_size, 0, 0, 54))
    header.extend(struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        height,
        1,
        24,
        0,
        len(pixel_bytes),
        2835,
        2835,
        0,
        0,
    ))

    with open(filepath, "wb") as file:
        file.write(header)
        file.write(pixel_bytes)


def _seed_texture_folder(folder, prefix, image_count, non_image_count):
    _clear_folder(folder)
    filepaths = []
    for texture_index in range(image_count):
        filepath = os.path.join(folder, f"{prefix}_{texture_index + 1:04d}.bmp")
        _write_visual_bmp(filepath, texture_index)
        filepaths.append(filepath)

    for file_index in range(non_image_count):
        filepath = os.path.join(folder, f"{prefix}_notes_{file_index + 1:04d}.txt")
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(f"Texture browser non-image file {file_index + 1}\n")

    return filepaths


def _configure_texture_browser_preferences(primary_folder, collection_filepaths):
    prefs = texture_browser._addon_preferences()
    texture_browser._ensure_texture_browser_preferences(prefs)
    prefs.texture_browser_preview_scale = 0.0
    prefs.texture_browser_last_folder_path = primary_folder
    prefs.texture_browser_include_suffixes.clear()
    item = prefs.texture_browser_include_suffixes.add()
    item.suffix = ".bmp"
    prefs.texture_browser_active_include_index = 0
    prefs.texture_browser_exclude_suffixes.clear()
    prefs.texture_browser_active_exclude_index = 0
    prefs.texture_browser_filters_initialized = True

    prefs.texture_browser_collections.clear()
    collection = prefs.texture_browser_collections.add()
    collection.name = "Visual Collection"
    for filepath in collection_filepaths:
        item = collection.files.add()
        item.filepath = filepath
    collection.active_file_index = 0
    prefs.texture_browser_active_collection_index = 0


def _texture_browser_popup_window():
    popup_windows = texture_browser.texture_browser_modal.popup_windows(
        bpy.context.window_manager.windows
    )
    if not popup_windows:
        return None
    return popup_windows[0]


def _texture_browser_area_and_region(window):
    area = window.screen.areas[0]
    region = next(region for region in area.regions if region.type == 'WINDOW')
    return area, region


def _redraw_texture_browser_window(window):
    with bpy.context.temp_override(window=window):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)


def _print_texture_browser_perf(name, value, unit):
    print(
        f"Anvil Level Design: Texture browser perf | {name}={value:.3f}{unit}",
        flush=True,
    )


def _print_texture_browser_count(name, value):
    print(
        f"Anvil Level Design: Texture browser perf | {name}={value}",
        flush=True,
    )


def _measure_call(callable_to_measure):
    start = time.perf_counter()
    result = callable_to_measure()
    return result, (time.perf_counter() - start) * 1000.0


def _print_timing_samples(name, samples):
    _print_texture_browser_perf(f"{name}_avg_ms", sum(samples) / len(samples), "")
    _print_texture_browser_perf(f"{name}_min_ms", min(samples), "")
    _print_texture_browser_perf(f"{name}_max_ms", max(samples), "")


def _visible_rects(rects, metrics):
    visible = []
    for index, rect in enumerate(rects):
        if rect["y"] + rect["h"] < metrics["grid_bottom"]:
            continue
        if rect["y"] > metrics["grid_top"]:
            continue
        visible.append((index, rect))
    return visible


def _layout_for_scroll_fraction(scene, region, display_filter, fraction):
    window_manager = bpy.context.window_manager
    preferences_system = bpy.context.preferences.system
    base_metrics, _base_rects = texture_browser.texture_browser_modal.layout_for_state(
        scene,
        window_manager.anvil_texture_browser_search,
        display_filter,
        region.width,
        region.height,
        window_manager.anvil_texture_browser_preview_scale,
        preferences_system.ui_scale,
        preferences_system.pixel_size,
        0,
    )
    scroll_offset = int(base_metrics["max_scroll"] * fraction)
    return texture_browser.texture_browser_modal.layout_for_state(
        scene,
        window_manager.anvil_texture_browser_search,
        display_filter,
        region.width,
        region.height,
        window_manager.anvil_texture_browser_preview_scale,
        preferences_system.ui_scale,
        preferences_system.pixel_size,
        scroll_offset,
    )


def _measure_layout(scene, region, display_filter, fraction, name):
    samples = []
    metrics = None
    rects = None
    for _sample_index in range(VISUAL_LAYOUT_SAMPLE_COUNT):
        (metrics, rects), elapsed_ms = _measure_call(
            lambda: _layout_for_scroll_fraction(scene, region, display_filter, fraction)
        )
        samples.append(elapsed_ms)
    visible = _visible_rects(rects, metrics)
    _print_timing_samples(f"layout_{name}", samples)
    _print_texture_browser_count(f"layout_{name}_rect_count", len(rects))
    _print_texture_browser_count(f"layout_{name}_visible_rect_count", len(visible))
    return metrics, rects, visible


def _measure_hit_test(rects, metrics, visible, name):
    if not visible:
        return
    visible_index = None
    visible_rect = None
    for candidate_index, candidate_rect in visible:
        center_y = candidate_rect["y"] + candidate_rect["h"] / 2
        if metrics["grid_bottom"] <= center_y <= metrics["grid_top"]:
            visible_index = candidate_index
            visible_rect = candidate_rect
            break
    if visible_rect is None:
        return
    mouse_x = visible_rect["x"] + visible_rect["w"] / 2
    mouse_y = visible_rect["y"] + visible_rect["h"] / 2
    samples = []
    hit = None
    for _sample_index in range(VISUAL_HIT_TEST_SAMPLE_COUNT):
        hit, elapsed_ms = _measure_call(
            lambda: image_grid_hit_test(rects, metrics, mouse_x, mouse_y)
        )
        samples.append(elapsed_ms)
    if hit is None or hit["key"] != visible_rect["key"]:
        raise AssertionError(f"Hit test did not find visible rect for {name}")
    _print_timing_samples(f"hit_test_{name}_visible", samples)
    _print_texture_browser_count(f"hit_test_{name}_visible_rect_index", visible_index)


def _warm_visible_texture_previews(visible, name):
    samples = []
    warmed_count = 0
    for _index, rect in visible:
        _result, elapsed_ms = _measure_call(
            lambda: texture_browser._texture_browser_warm_texture_item(rect)
        )
        samples.append(elapsed_ms)
        if _result:
            warmed_count += 1
    if samples:
        _print_timing_samples(f"warm_visible_{name}", samples)
    _print_texture_browser_count(f"warm_visible_{name}_count", warmed_count)


def _measure_texture_browser_model(scene, region):
    _print_texture_browser_count("folder_texture_file_count", VISUAL_TEXTURE_FILE_COUNT)
    _print_texture_browser_count("secondary_texture_file_count", VISUAL_SECONDARY_TEXTURE_FILE_COUNT)
    _print_texture_browser_count("collection_file_count", VISUAL_COLLECTION_FILE_COUNT)
    _print_texture_browser_count("non_image_file_count", VISUAL_NON_IMAGE_FILE_COUNT)
    _print_texture_browser_count("region_width", region.width)
    _print_texture_browser_count("region_height", region.height)

    folder_items, display_folder_ms = _measure_call(
        lambda: texture_browser._texture_browser_display_items_for_filter(
            scene,
            "",
            texture_browser._TEXTURE_BROWSER_FOLDER_VIEW,
        )
    )
    _print_texture_browser_count("display_folder_count", len(folder_items))
    _print_texture_browser_perf("display_folder_ms", display_folder_ms, "")

    folder_matches, display_folder_search_ms = _measure_call(
        lambda: texture_browser._texture_browser_display_items_for_filter(
            scene,
            "visual_texture_0250",
            texture_browser._TEXTURE_BROWSER_FOLDER_VIEW,
        )
    )
    _print_texture_browser_count("display_folder_search_count", len(folder_matches))
    _print_texture_browser_perf("display_folder_search_ms", display_folder_search_ms, "")

    collection_items, display_collection_ms = _measure_call(
        lambda: texture_browser._texture_browser_display_items_for_filter(scene, "", 0)
    )
    _print_texture_browser_count("display_collection_count", len(collection_items))
    _print_texture_browser_perf("display_collection_ms", display_collection_ms, "")

    collection_matches, display_collection_search_ms = _measure_call(
        lambda: texture_browser._texture_browser_display_items_for_filter(
            scene,
            "secondary_texture_0500",
            0,
        )
    )
    _print_texture_browser_count("display_collection_search_count", len(collection_matches))
    _print_texture_browser_perf("display_collection_search_ms", display_collection_search_ms, "")

    top_metrics, top_rects, top_visible = _measure_layout(
        scene,
        region,
        texture_browser._TEXTURE_BROWSER_FOLDER_VIEW,
        0.0,
        "folder_top",
    )
    middle_metrics, middle_rects, middle_visible = _measure_layout(
        scene,
        region,
        texture_browser._TEXTURE_BROWSER_FOLDER_VIEW,
        0.5,
        "folder_middle",
    )
    end_metrics, end_rects, end_visible = _measure_layout(
        scene,
        region,
        texture_browser._TEXTURE_BROWSER_FOLDER_VIEW,
        1.0,
        "folder_end",
    )
    collection_metrics, collection_rects, collection_visible = _measure_layout(
        scene,
        region,
        0,
        0.5,
        "collection_middle",
    )

    _print_texture_browser_count("grid_columns", top_metrics["columns"])
    _measure_hit_test(top_rects, top_metrics, top_visible, "folder_top")
    _measure_hit_test(middle_rects, middle_metrics, middle_visible, "folder_middle")
    _measure_hit_test(end_rects, end_metrics, end_visible, "folder_end")
    _measure_hit_test(
        collection_rects,
        collection_metrics,
        collection_visible,
        "collection_middle",
    )
    _warm_visible_texture_previews(top_visible, "folder_top")


def _set_texture_browser_scroll_fraction(scene, region, display_filter, fraction):
    metrics, _rects = _layout_for_scroll_fraction(scene, region, display_filter, fraction)
    texture_browser.texture_browser_modal.interaction["scroll_offset"] = metrics["scroll_offset"]


def _hold_visible_texture_browser(window, area, label, metric_name, seconds):
    print(f"Anvil Level Design: Texture browser visual check - {label}", flush=True)
    redraw_samples = []
    for _step in range(int(seconds * 10)):
        start = time.perf_counter()
        _redraw_texture_browser_window(window)
        redraw_samples.append((time.perf_counter() - start) * 1000.0)
        area.tag_redraw()
        yield 0.1
    _print_texture_browser_perf(
        f"redraw_{metric_name}_first_ms",
        redraw_samples[0],
        "",
    )
    _print_timing_samples(f"redraw_{metric_name}", redraw_samples)


def _close_texture_browser_popups():
    for window in texture_browser.texture_browser_modal.popup_windows(
            bpy.context.window_manager.windows):
        with bpy.context.temp_override(window=window):
            bpy.ops.leveldesign.texture_browser_close()


class TextureBrowserManualVisualTest(AnvilTestCase):

    def setUp(self):
        self._texture_browser_saves_were_suspended = (
            texture_browser_persistence.texture_browser_saves_suspended()
        )
        texture_browser_persistence.set_texture_browser_saves_suspended(True)
        self._preferences_snapshot = _snapshot_texture_browser_preferences()

    def tearDown(self):
        _close_texture_browser_popups()
        _restore_texture_browser_preferences(self._preferences_snapshot)
        texture_browser_persistence.set_texture_browser_saves_suspended(
            self._texture_browser_saves_were_suspended
        )
        texture_browser._folder_scan_cache.clear()
        texture_browser._collection_scan_cache.clear()
        if not base_test.save_outputs:
            _remove_visual_texture_folders()
        super().tearDown()

    def manual_texture_browser_thousands_of_files_visible_check(self):
        scene = bpy.context.scene
        output_root = _test_output_root()
        primary_folder = os.path.join(output_root, "visual_texture_browser_primary")
        secondary_folder = os.path.join(output_root, "visual_texture_browser_secondary")

        _print_texture_browser_count("target_texture_file_count", VISUAL_TEXTURE_FILE_COUNT)
        primary_paths, seed_primary_ms = _measure_call(
            lambda: _seed_texture_folder(
                primary_folder,
                "visual_texture",
                VISUAL_TEXTURE_FILE_COUNT,
                VISUAL_NON_IMAGE_FILE_COUNT,
            )
        )
        _print_texture_browser_perf("seed_primary_folder_ms", seed_primary_ms, "")
        secondary_paths, seed_secondary_ms = _measure_call(
            lambda: _seed_texture_folder(
                secondary_folder,
                "secondary_texture",
                VISUAL_SECONDARY_TEXTURE_FILE_COUNT,
                0,
            )
        )
        _print_texture_browser_perf("seed_secondary_folder_ms", seed_secondary_ms, "")

        collection_paths = primary_paths[:400] + secondary_paths[:600]
        _configure_texture_browser_preferences(primary_folder, collection_paths)

        window_manager = bpy.context.window_manager
        texture_browser._set_folder(window_manager, primary_folder)
        window_manager.anvil_texture_browser_collection_index = texture_browser._TEXTURE_BROWSER_FOLDER_VIEW
        window_manager.anvil_texture_browser_search = ""
        window_manager.anvil_texture_browser_preview_scale = 0.0

        result = bpy.ops.leveldesign.texture_browser()
        self.assertEqual(result, {'FINISHED'})
        popup_window = _texture_browser_popup_window()
        self.assertIsNotNone(popup_window)
        area, region = _texture_browser_area_and_region(popup_window)

        items = texture_browser._texture_browser_display_items_for_filter(
            scene,
            "",
            texture_browser._TEXTURE_BROWSER_FOLDER_VIEW,
        )
        self.assertEqual(len(items), VISUAL_TEXTURE_FILE_COUNT)

        _measure_texture_browser_model(scene, region)

        _set_texture_browser_scroll_fraction(
            scene,
            region,
            texture_browser._TEXTURE_BROWSER_FOLDER_VIEW,
            0.0,
        )
        yield from _hold_visible_texture_browser(
            popup_window,
            area,
            "top of 3,000 texture files",
            "folder_top",
            4,
        )

        _set_texture_browser_scroll_fraction(
            scene,
            region,
            texture_browser._TEXTURE_BROWSER_FOLDER_VIEW,
            0.5,
        )
        yield from _hold_visible_texture_browser(
            popup_window,
            area,
            "middle of 3,000 texture files",
            "folder_middle",
            4,
        )

        window_manager.anvil_texture_browser_collection_index = 0
        _set_texture_browser_scroll_fraction(scene, region, 0, 0.5)
        yield from _hold_visible_texture_browser(
            popup_window,
            area,
            "middle of 1,000-file collection",
            "collection_middle",
            4,
        )
