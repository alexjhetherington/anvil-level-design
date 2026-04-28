"""Draw a neutral overlay on visible library objects while editing local mesh."""

import bpy

from ..core.library import is_library_object
from ..core.workspace_check import is_level_design_workspace
from ..core.viewport_overlay import (
    TriangleOverlayCache,
    draw_screen_striped_tris,
    triangulate_mesh_world,
)


_draw_handler = None
_objects_for_rebuild = []
_last_signature = None
_LIGHT_COLOR = (0.09, 0.09, 0.09, 0.62)
_DARK_COLOR = (0.08, 0.08, 0.08, 0.62)
_STRIPE_WIDTH = 9.0


def _collect_visible_library_mesh_objects(objects):
    result = []
    for obj in objects:
        if obj.type != 'MESH':
            continue
        if not obj.visible_get():
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
            values.append(round(value, 5))
    return tuple(values)


def _library_object_signature(objects):
    result = []
    for obj in objects:
        data = obj.data
        result.append((
            obj.name,
            data.name,
            len(data.vertices),
            len(data.polygons),
            _matrix_signature(obj.matrix_world),
        ))
    return tuple(result)


def _rebuild_overlay():
    tris = []
    for obj in _objects_for_rebuild:
        if obj.data is None:
            continue
        tris.extend(triangulate_mesh_world(obj.data, obj.matrix_world))
    return tris


_overlay_cache = TriangleOverlayCache(_rebuild_overlay)


def invalidate_overlay():
    global _last_signature
    _last_signature = None
    _overlay_cache.invalidate()


def _draw_library_object_overlay():
    global _objects_for_rebuild, _last_signature

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

    library_objects = _collect_visible_library_mesh_objects(context.view_layer.objects)
    signature = _library_object_signature(library_objects)
    if signature != _last_signature:
        _objects_for_rebuild = library_objects
        _last_signature = signature
        _overlay_cache.invalidate()

    draw_screen_striped_tris(
        _overlay_cache.get_tris(),
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
    global _objects_for_rebuild, _last_signature
    _unregister_draw_handler()
    _objects_for_rebuild = []
    _last_signature = None
    _overlay_cache.clear()
