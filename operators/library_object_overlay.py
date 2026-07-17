"""Draw a neutral overlay on visible library objects while editing local mesh."""

import bpy

from ..core.library import is_library_object
from ..core.workspace_check import is_level_design_workspace
from ..core.viewport_overlay import (
    build_screen_striped_batch,
    draw_screen_striped_batches,
    triangulate_mesh_world,
)


_draw_handler = None
_object_cache = {}
_invalidated_object_keys = set()
_LIGHT_COLOR = (0.09, 0.09, 0.09, 0.62)
_DARK_COLOR = (0.08, 0.08, 0.08, 0.62)
_STRIPE_WIDTH = 9.0


class _LibraryObjectOverlayEntry:
    def __init__(self, signature, data_pointer, batch):
        self.signature = signature
        self.data_pointer = data_pointer
        self.batch = batch


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


def _matrix_signature(matrix):
    values = []
    for row in matrix:
        for value in row:
            values.append(float(value))
    return tuple(values)


def _object_cache_key(obj):
    try:
        return obj.as_pointer()
    except ReferenceError:
        return None


def _library_object_signature(obj):
    data = obj.data
    return (
        data.as_pointer(),
        len(data.vertices),
        len(data.polygons),
        _matrix_signature(obj.matrix_world),
    )


def _build_object_cache_entry(obj, signature):
    data = obj.data
    tris = triangulate_mesh_world(data, obj.matrix_world)
    return _LibraryObjectOverlayEntry(
        signature,
        data.as_pointer(),
        build_screen_striped_batch(tris),
    )


def invalidate_overlay():
    """Invalidate every cached prefab overlay object."""
    _invalidated_object_keys.update(_object_cache.keys())


def invalidate_overlay_object(obj):
    """Invalidate one linked prefab object's cached overlay geometry."""
    if obj is None or obj.type != 'MESH' or not is_library_object(obj):
        return
    key = _object_cache_key(obj)
    if key is not None:
        _invalidated_object_keys.add(key)


def invalidate_overlay_mesh(mesh):
    """Invalidate cached prefab objects that share changed linked mesh data."""
    if mesh is None or mesh.library is None:
        return
    try:
        data_pointer = mesh.as_pointer()
    except ReferenceError:
        return
    for key, entry in _object_cache.items():
        if entry.data_pointer == data_pointer:
            _invalidated_object_keys.add(key)


def _visible_cached_batches(library_objects):
    live_keys = set()
    batches = []

    for obj in library_objects:
        key = _object_cache_key(obj)
        if key is None:
            continue
        live_keys.add(key)

        if not obj.visible_get():
            continue

        signature = _library_object_signature(obj)
        entry = _object_cache.get(key)
        if (
            entry is None
            or key in _invalidated_object_keys
            or entry.signature != signature
        ):
            entry = _build_object_cache_entry(obj, signature)
            _object_cache[key] = entry
            _invalidated_object_keys.discard(key)

        if entry.batch is not None:
            batches.append(entry.batch)

    stale_keys = set(_object_cache.keys()) - live_keys
    for key in stale_keys:
        del _object_cache[key]
    _invalidated_object_keys.intersection_update(live_keys)

    return batches


def _draw_library_object_overlay():
    if not is_level_design_workspace():
        return

    context = bpy.context
    if context.mode != 'EDIT_MESH':
        return

    active_obj = context.object
    if active_obj is None or active_obj.type != 'MESH':
        return
    if is_library_object(active_obj):
        return

    library_objects = _collect_library_mesh_objects(context.view_layer.objects)
    draw_screen_striped_batches(
        _visible_cached_batches(library_objects),
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
    _object_cache.clear()
    _invalidated_object_keys.clear()
