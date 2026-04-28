"""Lazy GPU preview cache for texture browser files."""

import os

import bpy


_texture_browser_texture_cache = {}
_texture_browser_texture_missing_cache = set()
_texture_browser_loaded_versions = {}
_texture_browser_cache_generation = 0


def _texture_browser_abs_path(filepath):
    if not filepath:
        return ""
    return os.path.normcase(os.path.abspath(bpy.path.abspath(filepath)))


def _texture_browser_cache_version(abs_path):
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        mtime = "missing"
    return f"{mtime}:{_texture_browser_cache_generation}"


def _texture_browser_preview_key(filepath):
    abs_path = _texture_browser_abs_path(filepath)
    if not abs_path:
        return None
    return f"{abs_path}::{_texture_browser_cache_version(abs_path)}"


def invalidate_texture_browser_preview_cache():
    global _texture_browser_cache_generation
    _texture_browser_cache_generation += 1
    _texture_browser_texture_cache.clear()
    _texture_browser_texture_missing_cache.clear()


def cleanup_texture_browser_preview_cache():
    _texture_browser_texture_cache.clear()
    _texture_browser_texture_missing_cache.clear()
    _texture_browser_loaded_versions.clear()


def texture_browser_preview_texture(filepath):
    import gpu

    key = _texture_browser_preview_key(filepath)
    if key is None:
        return None
    if key in _texture_browser_texture_cache:
        return _texture_browser_texture_cache[key]
    if key in _texture_browser_texture_missing_cache:
        return None

    abs_path = _texture_browser_abs_path(filepath)
    if not os.path.isfile(abs_path):
        _texture_browser_texture_missing_cache.add(key)
        return None

    try:
        image = bpy.data.images.load(abs_path, check_existing=True)
    except RuntimeError:
        _texture_browser_texture_missing_cache.add(key)
        return None

    version = _texture_browser_cache_version(abs_path)
    if _texture_browser_loaded_versions.get(abs_path) != version:
        try:
            image.reload()
        except RuntimeError:
            pass
        _texture_browser_loaded_versions[abs_path] = version

    width = int(image.size[0])
    height = int(image.size[1])
    if width <= 0 or height <= 0:
        _texture_browser_texture_missing_cache.add(key)
        return None

    try:
        texture = gpu.texture.from_image(image)
    except Exception:
        _texture_browser_texture_missing_cache.add(key)
        return None

    texture_info = (texture, width, height)
    _texture_browser_texture_cache[key] = texture_info
    return texture_info


def texture_browser_cached_preview_texture(filepath):
    key = _texture_browser_preview_key(filepath)
    if key is None:
        return None
    return _texture_browser_texture_cache.get(key)


def texture_browser_preview_texture_needs_load(filepath):
    key = _texture_browser_preview_key(filepath)
    if key is None:
        return False
    return (
        key not in _texture_browser_texture_cache
        and key not in _texture_browser_texture_missing_cache
    )
