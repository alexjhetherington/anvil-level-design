"""Manual performance measurements for scenes containing many prefabs.

The method is intentionally named with ``manual_`` so normal unittest
discovery skips it. Run it by exact name when comparing prefab performance.
It reports timings rather than enforcing machine-dependent thresholds.
"""

import os
import statistics
import time

import bpy

from .base_test import AnvilTestCase, _get_window
from ..operators import library_object_overlay
from ..prefabs.operators import _next_prefab_object_name


PERF_PREFAB_COUNT = 750
PERF_MESH_GRID_SIZE = 20
PERF_REDRAW_SAMPLE_COUNT = 30
PERF_NAME_SAMPLE_COUNT = 100


def _test_output_root():
    output_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "test_outputs")
    )
    os.makedirs(output_root, exist_ok=True)
    return output_root


def _create_grid_mesh(name, grid_size):
    mesh = bpy.data.meshes.new(name)
    vertices = []
    faces = []
    for y in range(grid_size + 1):
        for x in range(grid_size + 1):
            vertices.append((float(x), float(y), 0.0))

    row_width = grid_size + 1
    for y in range(grid_size):
        for x in range(grid_size):
            first = y * row_width + x
            faces.append((first, first + 1, first + row_width + 1, first + row_width))

    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def _create_linked_perf_mesh(filepath):
    mesh_name = "PerfPrefabMesh"
    mesh = _create_grid_mesh(mesh_name, PERF_MESH_GRID_SIZE)
    try:
        bpy.data.libraries.write(filepath, {mesh})
    finally:
        bpy.data.meshes.remove(mesh)

    with bpy.data.libraries.load(filepath, link=True) as (_data_from, data_to):
        data_to.meshes = [mesh_name]
    if not data_to.meshes:
        raise AssertionError("Could not link the generated prefab performance mesh")
    return data_to.meshes[0]


def _seed_prefab_scene(scene, linked_mesh):
    columns = 30
    spacing = float(PERF_MESH_GRID_SIZE + 2)
    prefabs = []
    for index in range(PERF_PREFAB_COUNT):
        obj = bpy.data.objects.new("PerfPrefab", linked_mesh)
        obj.location = (
            float(index % columns) * spacing,
            float(index // columns) * spacing,
            0.0,
        )
        scene.collection.objects.link(obj)
        prefabs.append(obj)
    return prefabs


def _create_edit_target(scene):
    mesh = bpy.data.meshes.new("PerfEditTargetMesh")
    mesh.from_pydata(
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    obj = bpy.data.objects.new("PerfEditTarget", mesh)
    scene.collection.objects.link(obj)
    return obj


def _measure_ms(callable_to_measure):
    start = time.perf_counter()
    result = callable_to_measure()
    return result, (time.perf_counter() - start) * 1000.0


def _print_count(name, value):
    print(
        f"Anvil Level Design: Prefab scene perf | {name}={value}",
        flush=True,
    )


def _print_timing(name, value):
    print(
        f"Anvil Level Design: Prefab scene perf | {name}={value:.3f}ms",
        flush=True,
    )


def _print_samples(name, samples):
    ordered = sorted(samples)
    percentile_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    _print_timing(f"{name}_avg", sum(samples) / len(samples))
    _print_timing(f"{name}_median", statistics.median(samples))
    _print_timing(f"{name}_p95", ordered[percentile_index])
    _print_timing(f"{name}_min", min(samples))
    _print_timing(f"{name}_max", max(samples))


def _redraw_viewport():
    window = _get_window()
    with bpy.context.temp_override(window=window):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)


class PrefabScenePerformanceManualTest(AnvilTestCase):

    def manual_many_prefabs_overlay_and_placement_performance_stats(self):
        scene = bpy.context.scene
        filepath = os.path.join(_test_output_root(), "prefab_scene_performance_library.blend")
        if os.path.isfile(filepath):
            os.remove(filepath)

        linked_mesh = _create_linked_perf_mesh(filepath)
        prefabs = _seed_prefab_scene(scene, linked_mesh)
        edit_target = _create_edit_target(scene)
        bpy.context.view_layer.update()

        _print_count("prefab_count", len(prefabs))
        _print_count("mesh_polygon_count", len(linked_mesh.polygons))
        _print_count("overlay_triangle_count", len(prefabs) * len(linked_mesh.polygons) * 2)

        library_object_overlay._object_cache.clear()
        library_object_overlay._invalidated_object_keys.clear()
        library_objects, collect_ms = _measure_ms(
            lambda: library_object_overlay._collect_library_mesh_objects(
                bpy.context.view_layer.objects,
            )
        )
        _print_timing("collect_library_objects", collect_ms)

        _cold_batches, cold_cache_ms = _measure_ms(
            lambda: library_object_overlay._visible_cached_batches(library_objects)
        )
        warm_batches, warm_cache_ms = _measure_ms(
            lambda: library_object_overlay._visible_cached_batches(library_objects)
        )
        _print_timing("overlay_cache_cold", cold_cache_ms)
        _print_timing("overlay_cache_warm", warm_cache_ms)
        _print_count("overlay_object_cache_entries", len(library_object_overlay._object_cache))
        _print_count("overlay_visible_batches", len(warm_batches))
        _print_count("overlay_unique_batches", len({id(batch) for batch in warm_batches}))

        name_samples = []
        next_name = ""
        for _sample_index in range(PERF_NAME_SAMPLE_COUNT):
            next_name, elapsed_ms = _measure_ms(
                lambda: _next_prefab_object_name("PerfPrefab", "", None)
            )
            name_samples.append(elapsed_ms)
        _print_samples("next_prefab_name", name_samples)
        _print_count("next_prefab_name_result", next_name)

        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        edit_target.select_set(True)
        bpy.context.view_layer.objects.active = edit_target
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.context.view_layer.update()

        redraw_samples = []
        for _sample_index in range(PERF_REDRAW_SAMPLE_COUNT):
            _result, elapsed_ms = _measure_ms(_redraw_viewport)
            redraw_samples.append(elapsed_ms)
            yield 0.05
        _print_samples("viewport_redraw", redraw_samples)
