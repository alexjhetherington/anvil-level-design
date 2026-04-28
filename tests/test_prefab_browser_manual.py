"""Targeted manual prefab browser visual checks.

These methods are intentionally named with ``manual_`` so unittest discovery
does not run them as part of the normal suite. Run them by full test name when
you want a visible browser window to inspect.
"""

import os
import time

import bpy

from .base_test import AnvilTestCase
from ..operators import prefab_ops
from ..prefabs.previews import ensure_prefab_preview_data


VISUAL_PREFAB_LIBRARY_COUNT = 6
VISUAL_PREFABS_PER_LIBRARY = 500
VISUAL_PREFAB_COUNT = VISUAL_PREFAB_LIBRARY_COUNT * VISUAL_PREFABS_PER_LIBRARY
VISUAL_PREVIEW_SIZE = 16
VISUAL_LAYOUT_SAMPLE_COUNT = 5
VISUAL_HIT_TEST_SAMPLE_COUNT = 200


def _test_output_root():
    output_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "test_outputs")
    )
    os.makedirs(output_root, exist_ok=True)
    return output_root


def _preview_pixels(library_index, prefab_index):
    hue = (library_index * 43 + prefab_index * 7) % 255
    red = (80 + hue) % 255 / 255.0
    green = (150 + hue * 2) % 255 / 255.0
    blue = (220 + hue * 3) % 255 / 255.0
    pixels = []
    for y in range(VISUAL_PREVIEW_SIZE):
        for x in range(VISUAL_PREVIEW_SIZE):
            edge = x == 0 or y == 0 or x == VISUAL_PREVIEW_SIZE - 1 or y == VISUAL_PREVIEW_SIZE - 1
            if edge:
                pixels.extend((0.05, 0.05, 0.06, 1.0))
            else:
                shade = 0.72 + (x + y) / (VISUAL_PREVIEW_SIZE * 2.0) * 0.28
                pixels.extend((red * shade, green * shade, blue * shade, 1.0))
    return pixels


def _create_visual_prefab_mesh(name):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [
            (-0.5, -0.5, -0.5),
            (0.5, -0.5, -0.5),
            (0.5, 0.5, -0.5),
            (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5),
        ],
        [],
        [
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ],
    )
    mesh.update()
    return mesh


def _create_visual_prefab_library(filepath, library_index, prefab_count):
    if os.path.isfile(filepath):
        os.remove(filepath)

    mesh = _create_visual_prefab_mesh(
        f"visual_prefab_library_{library_index + 1:02d}_mesh"
    )
    asset_objects = []
    try:
        for prefab_index in range(prefab_count):
            obj = bpy.data.objects.new(
                (
                    f"Visual_{library_index + 1:02d}_"
                    f"Prefab_{prefab_index + 1:04d}"
                ),
                mesh,
            )
            obj.asset_mark()
            obj.preview_ensure()
            obj.preview.image_size = (VISUAL_PREVIEW_SIZE, VISUAL_PREVIEW_SIZE)
            obj.preview.image_pixels_float = _preview_pixels(library_index, prefab_index)
            asset_objects.append(obj)
        bpy.data.libraries.write(filepath, set(asset_objects))
    finally:
        for obj in asset_objects:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.name in bpy.data.meshes:
            bpy.data.meshes.remove(mesh)


def _seed_prefab_libraries(scene, library_count, prefabs_per_library):
    scene.anvil_prefab_libraries.clear()
    output_root = _test_output_root()
    for library_index in range(library_count):
        filepath = os.path.join(
            output_root,
            f"visual_prefab_library_{library_index + 1:02d}.blend",
        )
        _create_visual_prefab_library(filepath, library_index, prefabs_per_library)
        lib_entry = scene.anvil_prefab_libraries.add()
        lib_entry.filepath = filepath
        refreshed = prefab_ops._refresh_library_objects(lib_entry)
        if not refreshed:
            raise AssertionError(f"Could not read generated prefab library: {filepath}")


def _preview_data_for_first_prefab(scene):
    lib_entry = scene.anvil_prefab_libraries[0]
    asset_item = lib_entry.objects[0]
    return ensure_prefab_preview_data(
        lib_entry.filepath,
        asset_item.asset_type,
        asset_item.name,
    )


def _assert_generated_preview_available(scene):
    preview = _preview_data_for_first_prefab(scene)
    if preview is None:
        raise AssertionError("Generated prefab library did not provide preview data")


def _prefab_browser_popup_window():
    popup_windows = prefab_ops._prefab_browser_popup_windows(
        bpy.context.window_manager.windows
    )
    if not popup_windows:
        return None
    return popup_windows[0]


def _prefab_browser_area_and_region(window):
    area = window.screen.areas[0]
    region = next(region for region in area.regions if region.type == 'WINDOW')
    return area, region


def _redraw_prefab_browser_window(window):
    with bpy.context.temp_override(window=window):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)


def _set_prefab_browser_scroll_fraction(scene, region, fraction):
    window_manager = bpy.context.window_manager
    preferences_system = bpy.context.preferences.system
    metrics, _rects = prefab_ops._prefab_browser_layout_for_state(
        scene,
        window_manager.anvil_prefab_browser_search,
        region.width,
        region.height,
        window_manager.anvil_prefab_browser_preview_scale,
        preferences_system.ui_scale,
        preferences_system.pixel_size,
        0,
    )
    prefab_ops._prefab_browser_interaction["scroll_offset"] = int(
        metrics["max_scroll"] * fraction
    )


def _print_prefab_browser_perf(name, value, unit):
    print(
        f"Anvil Level Design: Prefab browser perf | {name}={value:.3f}{unit}",
        flush=True,
    )


def _print_prefab_browser_count(name, value):
    print(
        f"Anvil Level Design: Prefab browser perf | {name}={value}",
        flush=True,
    )


def _measure_call(callable_to_measure):
    start = time.perf_counter()
    result = callable_to_measure()
    return result, (time.perf_counter() - start) * 1000.0


def _print_timing_samples(name, samples):
    _print_prefab_browser_perf(f"{name}_avg_ms", sum(samples) / len(samples), "")
    _print_prefab_browser_perf(f"{name}_min_ms", min(samples), "")
    _print_prefab_browser_perf(f"{name}_max_ms", max(samples), "")


def _visible_rects(rects, metrics):
    visible = []
    for index, rect in enumerate(rects):
        if rect["y"] + rect["h"] < metrics["grid_bottom"]:
            continue
        if rect["y"] > metrics["grid_top"]:
            continue
        visible.append((index, rect))
    return visible


def _layout_for_scroll_fraction(scene, region, fraction):
    window_manager = bpy.context.window_manager
    preferences_system = bpy.context.preferences.system
    base_metrics, _base_rects = prefab_ops._prefab_browser_layout_for_state(
        scene,
        window_manager.anvil_prefab_browser_search,
        region.width,
        region.height,
        window_manager.anvil_prefab_browser_preview_scale,
        preferences_system.ui_scale,
        preferences_system.pixel_size,
        0,
    )
    scroll_offset = int(base_metrics["max_scroll"] * fraction)
    return prefab_ops._prefab_browser_layout_for_state(
        scene,
        window_manager.anvil_prefab_browser_search,
        region.width,
        region.height,
        window_manager.anvil_prefab_browser_preview_scale,
        preferences_system.ui_scale,
        preferences_system.pixel_size,
        scroll_offset,
    )


def _measure_layout(scene, region, fraction, name):
    samples = []
    metrics = None
    rects = None
    for _sample_index in range(VISUAL_LAYOUT_SAMPLE_COUNT):
        (metrics, rects), elapsed_ms = _measure_call(
            lambda: _layout_for_scroll_fraction(scene, region, fraction)
        )
        samples.append(elapsed_ms)
    visible = _visible_rects(rects, metrics)
    _print_timing_samples(f"layout_{name}", samples)
    _print_prefab_browser_count(f"layout_{name}_rect_count", len(rects))
    _print_prefab_browser_count(f"layout_{name}_visible_rect_count", len(visible))
    return metrics, rects, visible


def _measure_hit_test(rects, metrics, visible, name):
    if not visible:
        return
    visible_index, visible_rect = visible[0]
    mouse_x = visible_rect["x"] + visible_rect["w"] / 2
    mouse_y = visible_rect["y"] + visible_rect["h"] / 2
    samples = []
    hit = None
    for _sample_index in range(VISUAL_HIT_TEST_SAMPLE_COUNT):
        hit, elapsed_ms = _measure_call(
            lambda: prefab_ops._prefab_browser_hit_test(rects, metrics, mouse_x, mouse_y)
        )
        samples.append(elapsed_ms)
    self_check = hit is not None and hit["key"] == visible_rect["key"]
    if not self_check:
        raise AssertionError(f"Hit test did not find visible rect for {name}")
    _print_timing_samples(f"hit_test_{name}_visible", samples)
    _print_prefab_browser_count(f"hit_test_{name}_visible_rect_index", visible_index)


def _measure_prefab_browser_model(scene, region):
    _print_prefab_browser_count("library_count", len(scene.anvil_prefab_libraries))
    _print_prefab_browser_count("expected_prefab_count", VISUAL_PREFAB_COUNT)
    _print_prefab_browser_count("region_width", region.width)
    _print_prefab_browser_count("region_height", region.height)

    items, display_all_ms = _measure_call(
        lambda: prefab_ops._prefab_browser_display_items(scene, "")
    )
    _print_prefab_browser_count("display_all_count", len(items))
    _print_prefab_browser_perf("display_all_ms", display_all_ms, "")

    library_matches, display_library_search_ms = _measure_call(
        lambda: prefab_ops._prefab_browser_display_items(scene, "visual_03")
    )
    _print_prefab_browser_count("display_library_search_count", len(library_matches))
    _print_prefab_browser_perf(
        "display_library_search_ms",
        display_library_search_ms,
        "",
    )

    prefab_matches, display_prefab_search_ms = _measure_call(
        lambda: prefab_ops._prefab_browser_display_items(scene, "prefab_0500")
    )
    _print_prefab_browser_count("display_prefab_search_count", len(prefab_matches))
    _print_prefab_browser_perf(
        "display_prefab_search_ms",
        display_prefab_search_ms,
        "",
    )

    top_metrics, top_rects, top_visible = _measure_layout(scene, region, 0.0, "top")
    middle_metrics, middle_rects, middle_visible = _measure_layout(scene, region, 0.5, "middle")
    end_metrics, end_rects, end_visible = _measure_layout(scene, region, 1.0, "end")

    _print_prefab_browser_count("grid_columns", top_metrics["columns"])
    _measure_hit_test(top_rects, top_metrics, top_visible, "top")
    _measure_hit_test(middle_rects, middle_metrics, middle_visible, "middle")
    _measure_hit_test(end_rects, end_metrics, end_visible, "end")


def _hold_visible_prefab_browser(window, area, label, metric_name, seconds):
    print(f"Anvil Level Design: Prefab browser visual check - {label}", flush=True)
    redraw_samples = []
    for _step in range(int(seconds * 10)):
        start = time.perf_counter()
        _redraw_prefab_browser_window(window)
        redraw_samples.append((time.perf_counter() - start) * 1000.0)
        area.tag_redraw()
        yield 0.1
    _print_prefab_browser_perf(
        f"redraw_{metric_name}_first_ms",
        redraw_samples[0],
        "",
    )
    _print_timing_samples(f"redraw_{metric_name}", redraw_samples)


def _close_prefab_browser_popups():
    for window in prefab_ops._prefab_browser_popup_windows(
            bpy.context.window_manager.windows):
        with bpy.context.temp_override(window=window):
            bpy.ops.leveldesign.prefab_browser_close()


class PrefabBrowserManualVisualTest(AnvilTestCase):

    def tearDown(self):
        _close_prefab_browser_popups()
        super().tearDown()

    def manual_prefab_browser_thousands_of_prefabs_visible_check(self):
        scene = bpy.context.scene
        scene.anvil_prefab_mode = 'SCENE'
        _print_prefab_browser_count("target_prefab_count", VISUAL_PREFAB_COUNT)
        _seed_result, seed_elapsed_ms = _measure_call(
            lambda: _seed_prefab_libraries(
                scene,
                VISUAL_PREFAB_LIBRARY_COUNT,
                VISUAL_PREFABS_PER_LIBRARY,
            )
        )
        _print_prefab_browser_perf("seed_libraries_ms", seed_elapsed_ms, "")

        preview, preview_probe_ms = _measure_call(
            lambda: _preview_data_for_first_prefab(scene)
        )
        _print_prefab_browser_perf("first_preview_probe_ms", preview_probe_ms, "")
        if preview is None:
            raise AssertionError("Generated prefab library did not provide preview data")

        bpy.context.window_manager.anvil_prefab_browser_search = ""
        bpy.context.window_manager.anvil_prefab_browser_preview_scale = 0.0

        items = prefab_ops._prefab_browser_display_items(scene, "")
        self.assertEqual(len(items), VISUAL_PREFAB_COUNT)

        result = bpy.ops.leveldesign.prefab_browser()
        self.assertEqual(result, {'FINISHED'})
        popup_window = _prefab_browser_popup_window()
        self.assertIsNotNone(popup_window)
        area, region = _prefab_browser_area_and_region(popup_window)

        _measure_prefab_browser_model(scene, region)

        _set_prefab_browser_scroll_fraction(scene, region, 0.0)
        yield from _hold_visible_prefab_browser(
            popup_window,
            area,
            "top of 3,000 prefabs",
            "top",
            4,
        )

        _set_prefab_browser_scroll_fraction(scene, region, 0.5)
        yield from _hold_visible_prefab_browser(
            popup_window,
            area,
            "middle of 3,000 prefabs",
            "middle",
            4,
        )

        _set_prefab_browser_scroll_fraction(scene, region, 1.0)
        yield from _hold_visible_prefab_browser(
            popup_window,
            area,
            "end of 3,000 prefabs",
            "end",
            4,
        )
