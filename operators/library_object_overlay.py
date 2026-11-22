"""Draw a neutral overlay on visible library objects while editing local mesh."""

import bpy

from ..core.library import is_library_object
from ..core.workspace_check import is_level_design_workspace
from ..core.viewport_overlay import (
    SCREEN_STRIPE_INSTANCE_CHUNK_SIZE,
    build_screen_striped_instance_batch,
    build_screen_striped_matrix_texture,
    draw_screen_striped_instances,
    indexed_mesh_triangles_local,
    screen_striped_matrix_values,
)


_draw_handler = None
_mesh_cache = {}
_surface_cache = {}
_invalidated_mesh_keys = set()
_pending_transform_matrices = {}
_SURFACE_CACHE_LIMIT = 8
_LIGHT_COLOR = (0.09, 0.09, 0.09, 0.62)
_DARK_COLOR = (0.08, 0.08, 0.08, 0.62)
_STRIPE_WIDTH = 9.0


class _LibraryMeshOverlayEntry:
    def __init__(self, signature, batch, triangle_count):
        self.signature = signature
        self.batch = batch
        self.triangle_count = triangle_count


class _SurfaceInstanceChunk:
    def __init__(self, objects):
        self.object_pointers = tuple(obj.as_pointer() for obj in objects)
        self.instance_count = len(objects)
        self.matrix_values = []
        for obj in objects:
            self.matrix_values.extend(
                screen_striped_matrix_values(obj.matrix_world)
            )
        self.texture = build_screen_striped_matrix_texture(
            self.matrix_values,
            self.instance_count,
        )

    def rebuild(self, matrix_updates):
        new_values = self.matrix_values.copy()
        for instance_index, matrix in matrix_updates.items():
            value_index = instance_index * 16
            new_values[value_index:value_index + 16] = (
                screen_striped_matrix_values(matrix)
            )
        new_texture = build_screen_striped_matrix_texture(
            new_values,
            self.instance_count,
        )
        self.matrix_values = new_values
        self.texture = new_texture


class _SurfaceMeshInstances:
    def __init__(self, batch, objects):
        self.batch = batch
        self.chunks = []
        self.instance_buffers = []
        for start in range(0, len(objects), SCREEN_STRIPE_INSTANCE_CHUNK_SIZE):
            chunk = _SurfaceInstanceChunk(
                objects[start:start + SCREEN_STRIPE_INSTANCE_CHUNK_SIZE]
            )
            self.chunks.append(chunk)
            self.instance_buffers.append((chunk.texture, chunk.instance_count))

    def rebuild_chunk(self, chunk_index, matrix_updates):
        chunk = self.chunks[chunk_index]
        chunk.rebuild(matrix_updates)
        self.instance_buffers[chunk_index] = (
            chunk.texture,
            chunk.instance_count,
        )


class _SurfaceOverlayEntry:
    def __init__(self):
        self.mesh_instances = []
        self.object_locations = {}
        self.draw_groups = []

    def add_mesh_instances(self, batch, objects):
        mesh_instances = _SurfaceMeshInstances(batch, objects)
        self.mesh_instances.append(mesh_instances)
        self.draw_groups.append((batch, mesh_instances.instance_buffers))
        for chunk_index, chunk in enumerate(mesh_instances.chunks):
            for instance_index, object_pointer in enumerate(
                    chunk.object_pointers):
                self.object_locations[object_pointer] = (
                    mesh_instances,
                    chunk_index,
                    instance_index,
                )

    def apply_transform_updates(self, pending_matrices):
        updates_by_chunk = {}
        applied_pointers = set()
        for object_pointer, matrix in pending_matrices.items():
            location = self.object_locations.get(object_pointer)
            if location is None:
                continue
            mesh_instances, chunk_index, instance_index = location
            updates_by_chunk.setdefault(
                (mesh_instances, chunk_index),
                {},
            )[instance_index] = matrix
            applied_pointers.add(object_pointer)
        for (mesh_instances, chunk_index), matrix_updates in (
                updates_by_chunk.items()):
            mesh_instances.rebuild_chunk(chunk_index, matrix_updates)
        return applied_pointers


def _collect_library_mesh_objects(objects):
    result = []
    for obj in objects:
        if obj.type != 'MESH':
            continue
        if not is_library_object(obj):
            continue
        if obj.data is None:
            continue
        result.append(obj)
    return result


def _mesh_cache_key(mesh):
    try:
        return mesh.as_pointer()
    except ReferenceError:
        return None


def _library_mesh_signature(mesh):
    return (
        len(mesh.vertices),
        len(mesh.polygons),
    )


def _build_mesh_cache_entry(mesh, signature):
    vertices, triangle_indices = indexed_mesh_triangles_local(mesh)
    return _LibraryMeshOverlayEntry(
        signature,
        build_screen_striped_instance_batch(vertices, triangle_indices),
        len(triangle_indices),
    )


def _clear_surface_cache():
    _surface_cache.clear()


def invalidate_overlay():
    """Invalidate every cached prefab overlay mesh."""
    _mesh_cache.clear()
    _invalidated_mesh_keys.clear()
    _clear_surface_cache()
    _pending_transform_matrices.clear()


def invalidate_overlay_inventory():
    """Invalidate cached object membership without rebuilding mesh geometry."""
    _clear_surface_cache()
    _pending_transform_matrices.clear()


def queue_overlay_object_transform(obj):
    """Queue one prefab matrix for chunk-local upload in draw context."""
    if obj is None or obj.type != 'MESH':
        return
    source_obj = getattr(obj, "original", obj)
    if not is_library_object(source_obj):
        return
    try:
        object_pointer = source_obj.as_pointer()
    except ReferenceError:
        return
    if not any(
            object_pointer in surface.object_locations
            for surface in _surface_cache.values()):
        _clear_surface_cache()
        _pending_transform_matrices.clear()
        return
    _pending_transform_matrices[object_pointer] = obj.matrix_world.copy()


def invalidate_overlay_object(obj):
    """Queue changed prefab instance data for backwards-compatible callers."""
    queue_overlay_object_transform(obj)


def invalidate_overlay_mesh(mesh):
    """Invalidate changed linked mesh overlay geometry."""
    mesh = getattr(mesh, "original", mesh)
    if mesh is None or mesh.library is None:
        return
    key = _mesh_cache_key(mesh)
    if key is not None:
        _invalidated_mesh_keys.add(key)
        _clear_surface_cache()
        _pending_transform_matrices.clear()


def _build_surface_entry(library_objects, view_layer, viewport):
    objects_by_mesh_key = {}
    meshes_by_key = {}

    for obj in library_objects:
        mesh = obj.data
        key = _mesh_cache_key(mesh)
        if key is None:
            continue
        meshes_by_key[key] = mesh

        if not obj.visible_get(view_layer=view_layer, viewport=viewport):
            continue
        objects_by_mesh_key.setdefault(key, []).append(obj)

    surface = _SurfaceOverlayEntry()
    for key, objects in objects_by_mesh_key.items():
        mesh = meshes_by_key[key]
        signature = _library_mesh_signature(mesh)
        entry = _mesh_cache.get(key)
        if (
            entry is None
            or key in _invalidated_mesh_keys
            or entry.signature != signature
        ):
            entry = _build_mesh_cache_entry(mesh, signature)
            _mesh_cache[key] = entry
            _invalidated_mesh_keys.discard(key)

        if entry.batch is not None:
            surface.add_mesh_instances(entry.batch, objects)

    live_keys = set(meshes_by_key)
    stale_keys = set(_mesh_cache) - live_keys
    for stale_key in stale_keys:
        del _mesh_cache[stale_key]
    _invalidated_mesh_keys.intersection_update(live_keys)

    return surface


def _surface_cache_key(scene, view_layer, viewport):
    try:
        local_view = viewport.local_view
        local_view_pointer = (
            local_view.as_pointer()
            if local_view is not None
            else 0
        )
        return (
            scene.as_pointer(),
            view_layer.as_pointer(),
            viewport.as_pointer(),
            local_view_pointer,
            bool(viewport.use_local_collections),
        )
    except ReferenceError:
        return None


def _prune_surface_cache(key):
    if key is None:
        return
    viewport_prefix = key[:3]
    stale_keys = [
        cached_key
        for cached_key in _surface_cache
        if cached_key[:3] == viewport_prefix and cached_key != key
    ]
    for stale_key in stale_keys:
        del _surface_cache[stale_key]
    while key not in _surface_cache and len(_surface_cache) >= _SURFACE_CACHE_LIMIT:
        oldest_key = next(iter(_surface_cache))
        del _surface_cache[oldest_key]


def _apply_pending_transform_updates():
    if not _pending_transform_matrices:
        return

    applied_counts = {
        object_pointer: 0
        for object_pointer in _pending_transform_matrices
    }
    for surface in _surface_cache.values():
        applied_pointers = surface.apply_transform_updates(
            _pending_transform_matrices
        )
        for object_pointer in applied_pointers:
            applied_counts[object_pointer] += 1

    if any(count == 0 for count in applied_counts.values()):
        _clear_surface_cache()
    _pending_transform_matrices.clear()


def _cached_surface_batches(scene, view_layer, viewport):
    _apply_pending_transform_updates()
    key = _surface_cache_key(scene, view_layer, viewport)
    _prune_surface_cache(key)
    if key is not None:
        cached_surface = _surface_cache.get(key)
        if cached_surface is not None:
            return cached_surface.draw_groups

    library_objects = _collect_library_mesh_objects(view_layer.objects)
    surface = _build_surface_entry(
        library_objects,
        view_layer,
        viewport,
    )
    if key is not None:
        _surface_cache[key] = surface
    return surface.draw_groups


def _draw_library_object_overlay():
    if not is_level_design_workspace():
        return

    context = bpy.context
    if context.mode != 'EDIT_MESH':
        return
    region_data = context.region_data
    if region_data is None or not region_data.is_perspective:
        return

    active_obj = context.object
    if active_obj is None or active_obj.type != 'MESH':
        return
    if is_library_object(active_obj):
        return

    draw_screen_striped_instances(
        _cached_surface_batches(
            context.scene,
            context.view_layer,
            context.space_data,
        ),
        _LIGHT_COLOR,
        _DARK_COLOR,
        _STRIPE_WIDTH,
    )


def _register_draw_handler():
    global _draw_handler
    _unregister_draw_handler()
    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        _draw_library_object_overlay, (), 'WINDOW', 'POST_VIEW'
    )


def _unregister_draw_handler():
    global _draw_handler
    if _draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        except Exception:
            pass
        _draw_handler = None


def register():
    _register_draw_handler()


def unregister():
    _unregister_draw_handler()
    _mesh_cache.clear()
    _surface_cache.clear()
    _invalidated_mesh_keys.clear()
    _pending_transform_matrices.clear()
