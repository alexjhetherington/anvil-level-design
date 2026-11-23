"""Manual performance measurements for scenes containing many prefabs.

The method is intentionally named with ``manual_`` so normal unittest
discovery skips it. Run it by exact name when comparing prefab performance.
It reports timings rather than enforcing machine-dependent thresholds.
"""

import os
import statistics
import time
from array import array

import bpy
from mathutils import Quaternion, Vector

from .base_test import AnvilTestCase, _get_window
from ..core.viewport_overlay import (
    SCREEN_STRIPE_INSTANCE_CHUNK_SIZE,
    screen_stripe_instance_draw_status,
    screen_striped_matrix_values,
)
from ..operators import library_object_overlay
from ..prefabs.operators import (
    _local_prefab_object_names,
    _next_prefab_object_name,
)


PERF_PREFAB_COUNT = 7500
PERF_MESH_GRID_SIZE = 20
PERF_REDRAW_SAMPLE_COUNT = 30
PERF_REDRAW_WARMUP_COUNT = 5
PERF_TRANSFORM_SAMPLE_COUNT = 30
PERF_CHUNK_SWEEP_SAMPLE_COUNT = 10
PERF_CHUNK_SWEEP_SIZES = (256, 512, 1024, 2048, 4096, 8192)
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
    columns = 100
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


def _print_value(name, value):
    print(
        f"Anvil Level Design: Prefab scene perf | {name}={value:.6f}",
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


def _all_3d_view_contexts():
    window = _get_window()
    contexts = []
    for area in window.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        region = next(
            (candidate for candidate in area.regions if candidate.type == 'WINDOW'),
            None,
        )
        if region is None:
            continue
        space = area.spaces.active
        contexts.append({
            "window": window,
            "screen": window.screen,
            "area": area,
            "region": region,
            "space_data": space,
            "region_data": space.region_3d,
        })
    return contexts


def _frame_prefab_grid(prefabs):
    view_contexts = _all_3d_view_contexts()
    if len(view_contexts) < 2:
        raise AssertionError(
            "Prefab performance workspace needs orthographic and perspective viewports"
        )
    perspective_context = max(
        view_contexts,
        key=lambda item: item["region"].width * item["region"].height,
    )
    minimum_x = min(obj.location.x for obj in prefabs)
    maximum_x = max(obj.location.x + PERF_MESH_GRID_SIZE for obj in prefabs)
    minimum_y = min(obj.location.y for obj in prefabs)
    maximum_y = max(obj.location.y + PERF_MESH_GRID_SIZE for obj in prefabs)
    for view_context in view_contexts:
        region_3d = view_context["region_data"]
        view_context["space_data"].clip_end = max(
            maximum_x - minimum_x,
            maximum_y - minimum_y,
        ) * 10.0
        region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
        region_3d.view_location = Vector((
            (minimum_x + maximum_x) * 0.5,
            (minimum_y + maximum_y) * 0.5,
            0.0,
        ))
        region_3d.view_distance = max(
            maximum_x - minimum_x,
            maximum_y - minimum_y,
        )
        if view_context is perspective_context:
            region_3d.view_perspective = 'PERSP'
        else:
            region_3d.view_perspective = 'ORTHO'

    for _fit_step in range(3):
        for view_context in view_contexts:
            view_context["area"].tag_redraw()
        _redraw_viewport()
        for view_context in view_contexts:
            region_3d = view_context["region_data"]
            projected_bounds = _projected_prefab_grid_bounds(
                prefabs,
                region_3d.perspective_matrix.copy(),
            )
            extent = max(
                abs(projected_bounds["x_min"]),
                abs(projected_bounds["x_max"]),
                abs(projected_bounds["y_min"]),
                abs(projected_bounds["y_max"]),
            )
            region_3d.view_distance *= extent / 0.9

    for view_context in view_contexts:
        view_context["area"].tag_redraw()
    _redraw_viewport()
    results = []
    for view_context in view_contexts:
        view_projection_matrix = view_context[
            "region_data"
        ].perspective_matrix.copy()
        results.append((
            _projected_prefab_grid_bounds(prefabs, view_projection_matrix),
            view_context,
        ))
    return results


def _projected_prefab_grid_bounds(prefabs, view_projection_matrix):
    x_values = []
    y_values = []
    z_values = []
    local_corners = (
        Vector((0.0, 0.0, 0.0, 1.0)),
        Vector((float(PERF_MESH_GRID_SIZE), 0.0, 0.0, 1.0)),
        Vector((0.0, float(PERF_MESH_GRID_SIZE), 0.0, 1.0)),
        Vector((
            float(PERF_MESH_GRID_SIZE),
            float(PERF_MESH_GRID_SIZE),
            0.0,
            1.0,
        )),
    )
    for obj in prefabs:
        object_projection = view_projection_matrix @ obj.matrix_world
        for local_corner in local_corners:
            clip_corner = object_projection @ local_corner
            if clip_corner.w <= 0.0:
                raise AssertionError(
                    f"Prefab '{obj.name}' projected behind the viewport"
                )
            inverse_w = 1.0 / clip_corner.w
            x_values.append(clip_corner.x * inverse_w)
            y_values.append(clip_corner.y * inverse_w)
            z_values.append(clip_corner.z * inverse_w)
    return {
        "x_min": min(x_values),
        "x_max": max(x_values),
        "y_min": min(y_values),
        "y_max": max(y_values),
        "z_min": min(z_values),
        "z_max": max(z_values),
    }


def _matrix_signature(matrix):
    return tuple(float(value) for row in matrix for value in row)


def _collect_redraw_samples(sample_count):
    samples = []
    for _sample_index in range(sample_count):
        _result, elapsed_ms = _measure_ms(_redraw_viewport)
        samples.append(elapsed_ms)
        yield 0.05
    return samples


def _collect_transform_redraw_samples(prefabs, sample_count, chunk_size):
    samples = []
    for sample_index in range(sample_count):
        prefab_index = (sample_index * chunk_size) % len(prefabs)
        prefab = prefabs[prefab_index]
        start = time.perf_counter()
        prefab.location.x += 0.01
        bpy.context.view_layer.update()
        object_pointer = prefab.as_pointer()
        if object_pointer not in library_object_overlay._pending_transform_matrices:
            raise AssertionError(
                f"Depsgraph did not queue transformed prefab '{prefab.name}'"
            )
        _redraw_viewport()
        if library_object_overlay._pending_transform_matrices:
            raise AssertionError("Draw did not consume queued prefab transforms")
        samples.append((time.perf_counter() - start) * 1000.0)
        yield 0.05
    return samples


def _capture_window_pixels(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)
    window = _get_window()
    with bpy.context.temp_override(window=window):
        result = bpy.ops.screen.screenshot(
            filepath=filepath,
            check_existing=False,
        )
    if result != {'FINISHED'}:
        raise AssertionError(f"Could not capture benchmark screenshot: {result}")

    image = bpy.data.images.load(filepath, check_existing=False)
    try:
        pixels = array('f', [0.0]) * len(image.pixels)
        image.pixels.foreach_get(pixels)
        return {
            "width": image.size[0],
            "height": image.size[1],
            "pixels": pixels,
        }
    finally:
        bpy.data.images.remove(image)


def _pixel_difference_stats(overlay_capture, base_capture, region):
    width = overlay_capture["width"]
    height = overlay_capture["height"]
    overlay_pixels = overlay_capture["pixels"]
    base_pixels = base_capture["pixels"]
    minimum_x = max(0, region.x)
    maximum_x = min(width, region.x + region.width)
    minimum_y = max(0, region.y)
    maximum_y = min(height, region.y + region.height)
    changed_pixel_count = 0
    total_difference = 0.0
    maximum_difference = 0.0
    pixel_count = 0
    for y in range(minimum_y, maximum_y):
        for x in range(minimum_x, maximum_x):
            index = (y * width + x) * 4
            difference = max(
                abs(overlay_pixels[index + channel] - base_pixels[index + channel])
                for channel in range(3)
            )
            total_difference += difference
            maximum_difference = max(maximum_difference, difference)
            if difference > 0.01:
                changed_pixel_count += 1
            pixel_count += 1
    return {
        "changed": changed_pixel_count,
        "fraction": changed_pixel_count / pixel_count,
        "mean": total_difference / pixel_count,
        "maximum": maximum_difference,
    }


def _assert_overlay_changes_viewport(test_case, framed_viewports):
    output_root = _test_output_root()
    overlay_path = os.path.join(output_root, "prefab_gpu_overlay_enabled.png")
    base_path = os.path.join(output_root, "prefab_gpu_overlay_disabled.png")

    _redraw_viewport()
    overlay_capture = _capture_window_pixels(overlay_path)
    draw_count, drawn_instance_count, draw_error = (
        screen_stripe_instance_draw_status()
    )
    test_case.assertIsNone(draw_error)
    test_case.assertGreater(draw_count, 0)
    test_case.assertEqual(drawn_instance_count, PERF_PREFAB_COUNT)

    library_object_overlay._unregister_draw_handler()
    try:
        _redraw_viewport()
        base_capture = _capture_window_pixels(base_path)
    finally:
        library_object_overlay._register_draw_handler()
        _redraw_viewport()

    test_case.assertEqual(overlay_capture["width"], base_capture["width"])
    test_case.assertEqual(overlay_capture["height"], base_capture["height"])
    for viewport_index, (_bounds, view_context) in enumerate(
            framed_viewports, start=1):
        stats = _pixel_difference_stats(
            overlay_capture,
            base_capture,
            view_context["region"],
        )
        prefix = f"viewport_{viewport_index}_overlay_visual"
        _print_count(f"{prefix}_changed_pixel_count", stats["changed"])
        _print_value(f"{prefix}_changed_pixel_fraction", stats["fraction"])
        _print_value(f"{prefix}_mean_pixel_difference", stats["mean"])
        _print_value(f"{prefix}_max_pixel_difference", stats["maximum"])
        if view_context["region_data"].is_perspective:
            test_case.assertGreater(stats["changed"], 1000)
            test_case.assertGreater(stats["maximum"], 0.05)
        else:
            test_case.assertLess(stats["changed"], 100)
            test_case.assertLess(stats["fraction"], 0.001)
            test_case.assertLess(stats["maximum"], 0.05)
    _print_count("overlay_visual_draw_call_count", draw_count)
    _print_count("overlay_visual_drawn_instance_count", drawn_instance_count)


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
        framed_viewports = _frame_prefab_grid(prefabs)
        perspective_viewports = [
            item
            for item in framed_viewports
            if item[1]["region_data"].is_perspective
        ]
        orthographic_viewports = [
            item
            for item in framed_viewports
            if not item[1]["region_data"].is_perspective
        ]
        self.assertEqual(len(perspective_viewports), 1)
        self.assertGreaterEqual(len(orthographic_viewports), 1)
        projected_bounds, view_context = perspective_viewports[0]

        _print_count("prefab_count", len(prefabs))
        _print_count("mesh_polygon_count", len(linked_mesh.polygons))
        _print_count("overlay_triangle_count", len(prefabs) * len(linked_mesh.polygons) * 2)
        viewport_count = len(framed_viewports)
        _print_count("viewport_3d_area_count", viewport_count)
        for viewport_index, (bounds, framed_context) in enumerate(
                framed_viewports, start=1):
            for axis in ("x", "y", "z"):
                _print_value(
                    f"viewport_{viewport_index}_ndc_{axis}_min",
                    bounds[f"{axis}_min"],
                )
                _print_value(
                    f"viewport_{viewport_index}_ndc_{axis}_max",
                    bounds[f"{axis}_max"],
                )
                self.assertGreaterEqual(bounds[f"{axis}_min"], -1.0)
                self.assertLessEqual(bounds[f"{axis}_max"], 1.0)
            _print_count(
                f"viewport_{viewport_index}_is_perspective",
                int(framed_context["region_data"].is_perspective),
            )
            projected_extent = max(
                abs(bounds["x_min"]),
                abs(bounds["x_max"]),
                abs(bounds["y_min"]),
                abs(bounds["y_max"]),
            )
            self.assertGreaterEqual(projected_extent, 0.88)
            self.assertLessEqual(projected_extent, 0.92)
            region = framed_context["region"]
            _print_value(
                f"viewport_{viewport_index}_projected_width_pixels",
                (bounds["x_max"] - bounds["x_min"])
                * region.width * 0.5,
            )
            _print_value(
                f"viewport_{viewport_index}_projected_height_pixels",
                (bounds["y_max"] - bounds["y_min"])
                * region.height * 0.5,
            )

        library_object_overlay._mesh_cache.clear()
        library_object_overlay._surface_cache.clear()
        library_object_overlay._invalidated_mesh_keys.clear()
        library_object_overlay._pending_transform_matrices.clear()
        library_objects, collect_ms = _measure_ms(
            lambda: library_object_overlay._collect_library_mesh_objects(
                bpy.context.view_layer.objects,
            )
        )
        _print_timing("collect_library_objects", collect_ms)
        _cold_batches, cold_cache_ms = _measure_ms(
            lambda: library_object_overlay._cached_surface_batches(
                scene,
                bpy.context.view_layer,
                view_context["space_data"],
            )
        )
        warm_batches, warm_cache_ms = _measure_ms(
            lambda: library_object_overlay._cached_surface_batches(
                scene,
                bpy.context.view_layer,
                view_context["space_data"],
            )
        )
        _print_timing("overlay_cache_cold", cold_cache_ms)
        _print_timing("overlay_cache_warm", warm_cache_ms)
        _print_count("overlay_mesh_cache_entries", len(library_object_overlay._mesh_cache))
        _print_count("overlay_draw_groups", len(warm_batches))
        _print_count(
            "overlay_unique_batches",
            len({id(batch) for batch, _instance_buffers in warm_batches}),
        )
        _print_count(
            "overlay_instance_buffer_count",
            sum(len(instance_buffers) for _batch, instance_buffers in warm_batches),
        )
        expected_buffer_count = (
            PERF_PREFAB_COUNT + SCREEN_STRIPE_INSTANCE_CHUNK_SIZE - 1
        ) // SCREEN_STRIPE_INSTANCE_CHUNK_SIZE
        self.assertEqual(
            sum(len(instance_buffers) for _batch, instance_buffers in warm_batches),
            expected_buffer_count,
        )
        _print_count(
            "overlay_visible_instance_count",
            sum(
                instance_count
                for _batch, instance_buffers in warm_batches
                for _uniform_buffer, instance_count in instance_buffers
            ),
        )
        visible_instance_count = sum(
            instance_count
            for _batch, instance_buffers in warm_batches
            for _uniform_buffer, instance_count in instance_buffers
        )
        self.assertEqual(visible_instance_count, PERF_PREFAB_COUNT)
        cached_triangle_count = sum(
            entry.triangle_count
            for entry in library_object_overlay._mesh_cache.values()
        )
        _print_count("overlay_cached_mesh_triangle_count", cached_triangle_count)
        _print_count(
            "overlay_cached_triangle_instance_count",
            cached_triangle_count * visible_instance_count,
        )
        expected_triangle_count = PERF_MESH_GRID_SIZE * PERF_MESH_GRID_SIZE * 2
        self.assertEqual(cached_triangle_count, expected_triangle_count)

        previous_instance_buffers = tuple(
            instance_buffer
            for _batch, instance_buffers in warm_batches
            for instance_buffer, _instance_count in instance_buffers
        )
        previous_instance_buffer_ids = tuple(
            id(instance_buffer)
            for instance_buffer in previous_instance_buffers
        )
        old_matrix_values = screen_striped_matrix_values(prefabs[0].matrix_world)
        prefabs[0].location.x += 1.0
        bpy.context.view_layer.update()
        self.assertIn(
            prefabs[0].as_pointer(),
            library_object_overlay._pending_transform_matrices,
        )
        self.assertNotEqual(
            old_matrix_values,
            screen_striped_matrix_values(prefabs[0].matrix_world),
        )
        rebuilt_batches, invalidated_cache_ms = _measure_ms(
            lambda: library_object_overlay._cached_surface_batches(
                scene,
                bpy.context.view_layer,
                view_context["space_data"],
            )
        )
        rebuilt_instance_buffer_ids = tuple(
            id(instance_buffer)
            for _batch, instance_buffers in rebuilt_batches
            for instance_buffer, _instance_count in instance_buffers
        )
        self.assertNotEqual(
            rebuilt_instance_buffer_ids,
            previous_instance_buffer_ids,
        )
        changed_buffer_count = sum(
            previous_id != rebuilt_id
            for previous_id, rebuilt_id in zip(
                previous_instance_buffer_ids,
                rebuilt_instance_buffer_ids,
            )
        )
        _print_count("overlay_transform_changed_buffer_count", changed_buffer_count)
        self.assertEqual(changed_buffer_count, 1)
        self.assertFalse(library_object_overlay._pending_transform_matrices)
        surface_key = library_object_overlay._surface_cache_key(
            scene,
            bpy.context.view_layer,
            view_context["space_data"],
        )
        surface = library_object_overlay._surface_cache[surface_key]
        mesh_instances, chunk_index, instance_index = (
            surface.object_locations[prefabs[0].as_pointer()]
        )
        chunk = mesh_instances.chunks[chunk_index]
        value_index = instance_index * 16
        self.assertEqual(
            chunk.matrix_values[value_index:value_index + 16],
            screen_striped_matrix_values(prefabs[0].matrix_world),
        )
        _print_timing("overlay_cache_after_transform", invalidated_cache_ms)

        transform_samples = []
        for sample_index in range(PERF_TRANSFORM_SAMPLE_COUNT):
            prefab_index = (
                sample_index * SCREEN_STRIPE_INSTANCE_CHUNK_SIZE
            ) % PERF_PREFAB_COUNT
            prefab = prefabs[prefab_index]
            prefab.location.y += 0.01
            bpy.context.view_layer.update()
            self.assertIn(
                prefab.as_pointer(),
                library_object_overlay._pending_transform_matrices,
            )
            _rebuilt_batches, elapsed_ms = _measure_ms(
                lambda: library_object_overlay._cached_surface_batches(
                    scene,
                    bpy.context.view_layer,
                    view_context["space_data"],
                )
            )
            self.assertFalse(library_object_overlay._pending_transform_matrices)
            transform_samples.append(elapsed_ms)
        _print_samples("overlay_chunk_transform", transform_samples)

        prefabs[-1].hide_viewport = True
        bpy.context.view_layer.update()
        self.assertFalse(library_object_overlay._surface_cache)
        hidden_batches = library_object_overlay._cached_surface_batches(
            scene,
            bpy.context.view_layer,
            view_context["space_data"],
        )
        hidden_instance_count = sum(
            instance_count
            for _batch, instance_buffers in hidden_batches
            for _texture, instance_count in instance_buffers
        )
        self.assertEqual(hidden_instance_count, PERF_PREFAB_COUNT - 1)
        prefabs[-1].hide_viewport = False
        bpy.context.view_layer.update()
        self.assertFalse(library_object_overlay._surface_cache)
        restored_batches = library_object_overlay._cached_surface_batches(
            scene,
            bpy.context.view_layer,
            view_context["space_data"],
        )
        restored_instance_count = sum(
            instance_count
            for _batch, instance_buffers in restored_batches
            for _texture, instance_count in instance_buffers
        )
        self.assertEqual(restored_instance_count, PERF_PREFAB_COUNT)

        used_object_names, name_scan_ms = _measure_ms(_local_prefab_object_names)
        _print_timing("local_prefab_name_scan", name_scan_ms)
        next_name_indices = {}
        name_samples = []
        next_name = ""
        for _sample_index in range(PERF_NAME_SAMPLE_COUNT):
            next_name, elapsed_ms = _measure_ms(
                lambda: _next_prefab_object_name(
                    "PerfPrefab",
                    "",
                    used_object_names,
                    next_name_indices,
                )
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

        _assert_overlay_changes_viewport(self, framed_viewports)

        production_chunk_size = (
            library_object_overlay.SCREEN_STRIPE_INSTANCE_CHUNK_SIZE
        )
        try:
            for chunk_size in PERF_CHUNK_SWEEP_SIZES:
                library_object_overlay.SCREEN_STRIPE_INSTANCE_CHUNK_SIZE = (
                    chunk_size
                )
                library_object_overlay.invalidate_overlay_inventory()
                yield from _collect_redraw_samples(3)
                chunk_redraw_samples = yield from _collect_redraw_samples(
                    PERF_CHUNK_SWEEP_SAMPLE_COUNT
                )
                _print_samples(
                    f"chunk_{chunk_size}_viewport_redraw",
                    chunk_redraw_samples,
                )
                transform_redraw_samples = yield from (
                    _collect_transform_redraw_samples(
                        prefabs,
                        PERF_CHUNK_SWEEP_SAMPLE_COUNT,
                        chunk_size,
                    )
                )
                _print_samples(
                    f"chunk_{chunk_size}_transform_redraw",
                    transform_redraw_samples,
                )
        finally:
            library_object_overlay.SCREEN_STRIPE_INSTANCE_CHUNK_SIZE = (
                production_chunk_size
            )
            library_object_overlay.invalidate_overlay_inventory()
            _redraw_viewport()

        current_cached_surface_batches = (
            library_object_overlay._cached_surface_batches
        )

        def previous_gpu_cache_batches(scene, view_layer, viewport):
            uncached_library_objects = (
                library_object_overlay._collect_library_mesh_objects(
                    view_layer.objects
                )
            )
            tuple(
                (
                    obj.as_pointer(),
                    _matrix_signature(obj.matrix_world),
                )
                for obj in uncached_library_objects
            )
            return current_cached_surface_batches(
                scene,
                view_layer,
                viewport,
            )

        library_object_overlay._cached_surface_batches = previous_gpu_cache_batches
        try:
            yield from _collect_redraw_samples(PERF_REDRAW_WARMUP_COUNT)
            previous_redraw_samples = yield from _collect_redraw_samples(
                PERF_REDRAW_SAMPLE_COUNT
            )
        finally:
            library_object_overlay._cached_surface_batches = (
                current_cached_surface_batches
            )
        _print_samples(
            "viewport_redraw_previous_gpu_cache",
            previous_redraw_samples,
        )

        library_object_overlay._surface_cache.clear()
        yield from _collect_redraw_samples(PERF_REDRAW_WARMUP_COUNT)
        current_redraw_samples = yield from _collect_redraw_samples(
            PERF_REDRAW_SAMPLE_COUNT
        )
        _print_samples(
            "viewport_redraw_current_gpu_cache",
            current_redraw_samples,
        )

        library_object_overlay._unregister_draw_handler()
        try:
            yield from _collect_redraw_samples(PERF_REDRAW_WARMUP_COUNT)
            base_redraw_samples = yield from _collect_redraw_samples(
                PERF_REDRAW_SAMPLE_COUNT
            )
        finally:
            library_object_overlay._register_draw_handler()
        _print_samples("viewport_redraw_without_overlay", base_redraw_samples)
        _print_timing(
            "viewport_redraw_current_overlay_median_cost",
            statistics.median(current_redraw_samples)
            - statistics.median(base_redraw_samples),
        )
        _print_timing(
            "viewport_redraw_cache_median_improvement",
            statistics.median(previous_redraw_samples)
            - statistics.median(current_redraw_samples),
        )
