"""Prefab thumbnail cache and library preview capture."""

import math
import os
import re
import shutil
import tempfile

import bpy
import bpy.utils.previews
from mathutils import Euler

from .assets import find_existing_linked_object, normalize_path


_preview_collection = None
_prefab_browser_texture_cache = {}
_prefab_browser_texture_missing_cache = set()
_preview_cache_generations = {}
_preview_cache_versions = {}


def _preview_cache_version(abs_path):
    generation = _preview_cache_generations.get(abs_path, 0)
    cached = _preview_cache_versions.get(abs_path)
    if cached is not None and cached[0] == generation:
        return cached[1]
    try:
        mtime = str(os.path.getmtime(abs_path))
    except OSError:
        mtime = "missing"
    version = f"{mtime}:{generation}"
    _preview_cache_versions[abs_path] = (generation, version)
    return version


def _preview_key(abs_path, asset_type, asset_name):
    return f"{abs_path}::{_preview_cache_version(abs_path)}::{asset_type}::{asset_name}"


def _preview_path_prefix(abs_path):
    return f"{abs_path}::"


def _preview_cache_key(filepath, asset_type, asset_name):
    abs_path = normalize_path(filepath)
    if not abs_path:
        return None
    return _preview_key(abs_path, asset_type, asset_name)


def _get_preview_collection():
    global _preview_collection
    if _preview_collection is None:
        _preview_collection = bpy.utils.previews.new()
    return _preview_collection


def _remove_preview_collection():
    global _preview_collection
    if _preview_collection is not None:
        bpy.utils.previews.remove(_preview_collection)
        _preview_collection = None


def _invalidate_preview_collection(abs_path):
    if _preview_collection is None:
        return
    prefix = _preview_path_prefix(abs_path)
    try:
        keys = [key for key in _preview_collection.keys() if key.startswith(prefix)]
    except Exception:
        _remove_preview_collection()
        return
    for key in keys:
        try:
            del _preview_collection[key]
        except Exception:
            _remove_preview_collection()
            return


def invalidate_preview_cache(filepath):
    abs_path = normalize_path(filepath)
    if not abs_path:
        return
    _preview_cache_generations[abs_path] = _preview_cache_generations.get(abs_path, 0) + 1
    if abs_path in _preview_cache_versions:
        del _preview_cache_versions[abs_path]
    prefix = _preview_path_prefix(abs_path)

    for key in list(_prefab_browser_texture_cache.keys()):
        if key.startswith(prefix):
            del _prefab_browser_texture_cache[key]
    for key in list(_prefab_browser_texture_missing_cache):
        if key.startswith(prefix):
            _prefab_browser_texture_missing_cache.remove(key)
    _invalidate_preview_collection(abs_path)


def cleanup_preview_cache():
    _remove_preview_collection()
    _prefab_browser_texture_cache.clear()
    _prefab_browser_texture_missing_cache.clear()
    _preview_cache_generations.clear()
    _preview_cache_versions.clear()


def _remove_preview_only_object(obj):
    if obj is None:
        return
    try:
        bpy.data.objects.remove(obj, do_unlink=True)
    except (ReferenceError, RuntimeError):
        pass


def _copy_preview_to_collection(pcoll, key, asset):
    if asset is None:
        return None

    preview = asset.preview
    if preview is None:
        return None

    width, height = preview.image_size
    if width <= 0 or height <= 0:
        return None

    pixels = list(preview.image_pixels_float)
    if not pixels:
        return None

    new_preview = pcoll.new(key)
    new_preview.image_size = (width, height)
    new_preview.image_pixels_float = pixels
    return new_preview


def ensure_prefab_preview_data(filepath, asset_type, asset_name):
    """Return preview data for the prefab thumbnail, or None if unavailable.

    Reads the embedded ID preview from the source .blend into a custom preview
    collection so it can be reused by custom GPU drawing.
    """
    if asset_type != 'OBJECT':
        return None
    abs_path = normalize_path(filepath)
    if not abs_path or not os.path.isfile(abs_path):
        return None

    pcoll = _get_preview_collection()
    key = _preview_key(abs_path, asset_type, asset_name)
    if key in pcoll:
        return pcoll[key]

    asset = find_existing_linked_object(abs_path, asset_name)
    preview_only_asset = asset is not None and asset.users == 0

    if asset is None:
        with bpy.data.libraries.load(abs_path, link=True, assets_only=True) as (data_from, data_to):
            if asset_name not in data_from.objects:
                return None
            data_to.objects = [asset_name]
        if not data_to.objects:
            return None
        asset = data_to.objects[0]
        preview_only_asset = True

    try:
        return _copy_preview_to_collection(pcoll, key, asset)
    finally:
        if preview_only_asset:
            _remove_preview_only_object(asset)


def prefab_browser_preview_texture(filepath, asset_type, asset_name):
    import gpu

    key = _preview_cache_key(filepath, asset_type, asset_name)
    if key is None:
        return None
    if key in _prefab_browser_texture_cache:
        return _prefab_browser_texture_cache[key]
    if key in _prefab_browser_texture_missing_cache:
        return None

    preview = ensure_prefab_preview_data(filepath, asset_type, asset_name)
    if preview is None:
        _prefab_browser_texture_missing_cache.add(key)
        return None
    width, height = preview.image_size
    if width <= 0 or height <= 0:
        _prefab_browser_texture_missing_cache.add(key)
        return None
    pixels = list(preview.image_pixels_float)
    if not pixels:
        _prefab_browser_texture_missing_cache.add(key)
        return None
    try:
        buffer = gpu.types.Buffer('FLOAT', width * height * 4, pixels)
        texture = gpu.types.GPUTexture((width, height), format='RGBA32F', data=buffer)
    except Exception:
        _prefab_browser_texture_missing_cache.add(key)
        return None
    texture_info = (texture, width, height)
    _prefab_browser_texture_cache[key] = texture_info
    return texture_info


def prefab_browser_cached_preview_texture(filepath, asset_type, asset_name):
    key = _preview_cache_key(filepath, asset_type, asset_name)
    if key is None:
        return None
    return _prefab_browser_texture_cache.get(key)


def prefab_browser_preview_texture_needs_load(filepath, asset_type, asset_name):
    key = _preview_cache_key(filepath, asset_type, asset_name)
    if key is None:
        return False
    return (
        key not in _prefab_browser_texture_cache
        and key not in _prefab_browser_texture_missing_cache
    )


def _find_3d_viewport():
    """Return (window, area, region) of a 3D viewport, preferring Material Preview."""
    wm = bpy.context.window_manager
    if wm is None:
        return None, None, None
    fallback = (None, None, None)
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region is None:
                continue
            if area.spaces.active.shading.type == 'MATERIAL':
                return window, area, region
            if fallback[0] is None:
                fallback = (window, area, region)
    return fallback


def _preview_filename(asset_type, asset_name):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", asset_name).strip("._")
    if not safe_name:
        safe_name = "asset"
    return f"preview_{asset_type}_{safe_name}.png"


def capture_library_previews(scene):
    """Capture viewport-shaded previews for object assets.

    Uses bpy.ops.render.opengl(view_context=True) so the user's chosen
    viewport shading determines the preview look - bypassing Blender's
    built-in dark workbench preview path. Forces a 3/4 perspective angle
    and disables overlays (grid, axes, gizmos) for clean thumbnails.

    Only the active scene can have its preview captured per save, since
    rendering needs a 3D viewport currently displaying that scene. This
    function will no-op if no 3D viewport is available.
    """
    window, area, region = _find_3d_viewport()
    if area is None:
        print("Anvil Level Design: No 3D viewport found; skipping library preview capture")
        return

    space = area.spaces.active
    region_3d = space.region_3d
    overlay = space.overlay

    if not scene.view_layers:
        return
    view_layer = scene.view_layers[0]

    target_assets = []
    for obj in scene.collection.all_objects:
        if obj.asset_data is not None:
            target_assets.append(('OBJECT', obj.name, obj, [obj]))
    if not target_assets:
        return

    saved_active = view_layer.objects.active
    saved_selected = [o for o in view_layer.objects if o.select_get()]
    saved_hidden = {o.name: o.hide_get() for o in view_layer.objects}
    saved_resolution = (scene.render.resolution_x, scene.render.resolution_y)
    saved_filepath = scene.render.filepath
    saved_format = scene.render.image_settings.file_format
    saved_view_rotation = region_3d.view_rotation.copy()
    saved_view_location = region_3d.view_location.copy()
    saved_view_distance = region_3d.view_distance
    saved_view_perspective = region_3d.view_perspective
    saved_show_overlays = overlay.show_overlays
    saved_shading_type = space.shading.type

    tmpdir = tempfile.mkdtemp(prefix="anvil_prefab_preview_")

    print(f"Anvil Level Design: Capturing previews for {len(target_assets)} prefab asset(s) in scene '{scene.name}'", flush=True)

    # 3/4 perspective angle (similar to Blender's default startup view).
    preview_rotation = Euler(
        (math.radians(63), 0.0, math.radians(46)), 'XYZ'
    ).to_quaternion()

    try:
        scene.render.resolution_x = 256
        scene.render.resolution_y = 256
        scene.render.image_settings.file_format = 'PNG'
        overlay.show_overlays = False
        region_3d.view_perspective = 'PERSP'
        space.shading.type = 'MATERIAL'

        for asset_type, asset_name, asset, visible_objects in target_assets:
            try:
                for o in list(view_layer.objects):
                    o.hide_set(o not in visible_objects)
                view_layer.update()

                bpy.ops.object.select_all(action='DESELECT')
                active_obj = None
                for obj in visible_objects:
                    if next((o for o in view_layer.objects if o.name == obj.name), None) is None:
                        continue
                    obj.select_set(True)
                    if active_obj is None:
                        active_obj = obj
                if active_obj is not None:
                    view_layer.objects.active = active_obj

                region_3d.view_rotation = preview_rotation

                png_path = os.path.join(tmpdir, _preview_filename(asset_type, asset_name))
                scene.render.filepath = png_path

                with bpy.context.temp_override(window=window, area=area, region=region, space_data=space):
                    if active_obj is not None:
                        bpy.ops.view3d.view_selected()
                    region_3d.view_distance *= 0.8
                    bpy.ops.render.opengl(view_context=True, write_still=True)

                if not os.path.isfile(png_path):
                    print(f"Anvil Level Design: Preview PNG missing for '{asset_name}'", flush=True)
                    continue

                img = bpy.data.images.load(png_path, check_existing=False)
                try:
                    asset.preview_ensure()
                    if asset.preview is not None:
                        w, h = img.size
                        asset.preview.image_size = (w, h)
                        asset.preview.image_pixels_float = list(img.pixels)
                finally:
                    bpy.data.images.remove(img)
            except Exception as exc:
                print(f"Anvil Level Design: Failed to capture preview for '{asset_name}': {exc}", flush=True)
    finally:
        scene.render.resolution_x, scene.render.resolution_y = saved_resolution
        scene.render.filepath = saved_filepath
        scene.render.image_settings.file_format = saved_format
        overlay.show_overlays = saved_show_overlays
        space.shading.type = saved_shading_type
        region_3d.view_rotation = saved_view_rotation
        region_3d.view_location = saved_view_location
        region_3d.view_distance = saved_view_distance
        region_3d.view_perspective = saved_view_perspective
        for o in list(view_layer.objects):
            if o.name in saved_hidden:
                o.hide_set(saved_hidden[o.name])
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except RuntimeError:
            pass
        for o in saved_selected:
            try:
                o.select_set(True)
            except RuntimeError:
                pass
        if saved_active is not None:
            view_layer.objects.active = saved_active
        try:
            shutil.rmtree(tmpdir)
        except OSError:
            pass
        invalidate_preview_cache(bpy.data.filepath)
