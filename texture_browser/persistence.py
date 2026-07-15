"""External persistence for texture browser favourites and collections."""

import json
import os

import bpy

from ..core.logging import debug_log


_TEXTURE_BROWSER_DATA_VERSION = 1
_TEXTURE_BROWSER_DATA_DIRECTORY = "anvil_level_design"
_TEXTURE_BROWSER_DATA_FILENAME = "texture_browser.json"
_loading_texture_browser_data = False
_texture_browser_data_save_blocked = False
_texture_browser_saves_suspended = False


def texture_browser_data_filepath():
    config_directory = bpy.utils.user_resource('CONFIG')
    if not config_directory:
        return ""
    return os.path.join(
        config_directory,
        _TEXTURE_BROWSER_DATA_DIRECTORY,
        _TEXTURE_BROWSER_DATA_FILENAME,
    )


def texture_browser_saves_suspended():
    return _texture_browser_saves_suspended


def set_texture_browser_saves_suspended(suspended):
    global _texture_browser_saves_suspended
    _texture_browser_saves_suspended = suspended


def _addon_preferences():
    package_name = __package__.split(".", 1)[0]
    addon = bpy.context.preferences.addons.get(package_name)
    if addon is None:
        return None
    return addon.preferences


def _texture_browser_properties_available(preferences):
    return (
        preferences is not None
        and hasattr(preferences, "texture_browser_favorites")
        and hasattr(preferences, "texture_browser_collections")
    )


def _texture_browser_data(preferences):
    return {
        "version": _TEXTURE_BROWSER_DATA_VERSION,
        "favorites": [
            {
                "name": favorite.name,
                "path": favorite.path,
            }
            for favorite in preferences.texture_browser_favorites
        ],
        "active_favorite_index": preferences.texture_browser_active_favorite_index,
        "collections": [
            {
                "name": collection.name,
                "files": [item.filepath for item in collection.files],
                "active_file_index": collection.active_file_index,
            }
            for collection in preferences.texture_browser_collections
        ],
        "active_collection_index": preferences.texture_browser_active_collection_index,
    }


def save_texture_browser_data():
    if (
            _loading_texture_browser_data
            or _texture_browser_data_save_blocked
            or _texture_browser_saves_suspended):
        return False

    preferences = _addon_preferences()
    if not _texture_browser_properties_available(preferences):
        return False

    filepath = texture_browser_data_filepath()
    if not filepath:
        print(
            "Anvil Level Design: Error saving texture browser data: "
            "Blender user config directory is unavailable",
            flush=True,
        )
        return False

    directory = os.path.dirname(filepath)
    temporary_filepath = filepath + ".tmp"
    try:
        os.makedirs(directory, exist_ok=True)
        with open(temporary_filepath, "w", encoding="utf-8") as file:
            json.dump(_texture_browser_data(preferences), file, indent=2)
            file.write("\n")
        os.replace(temporary_filepath, filepath)
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"Anvil Level Design: Error saving texture browser data: {exc}",
            flush=True,
        )
        return False

    debug_log(f"[TextureBrowser] Saved user data to {filepath}")
    return True


def _string_value(value):
    return value if isinstance(value, str) else ""


def _integer_value(value):
    return value if isinstance(value, int) else 0


def _populate_texture_browser_data(preferences, data):
    favorites = data.get("favorites", [])
    collections = data.get("collections", [])
    if not isinstance(favorites, list) or not isinstance(collections, list):
        raise ValueError("favorites and collections must be lists")

    preferences.texture_browser_favorites.clear()
    for favorite_data in favorites:
        if not isinstance(favorite_data, dict):
            continue
        favorite = preferences.texture_browser_favorites.add()
        favorite.name = _string_value(favorite_data.get("name"))
        favorite.path = _string_value(favorite_data.get("path"))

    preferences.texture_browser_active_favorite_index = _integer_value(
        data.get("active_favorite_index")
    )

    preferences.texture_browser_collections.clear()
    for collection_data in collections:
        if not isinstance(collection_data, dict):
            continue
        collection = preferences.texture_browser_collections.add()
        collection.name = _string_value(collection_data.get("name"))
        filepaths = collection_data.get("files", [])
        if isinstance(filepaths, list):
            for filepath in filepaths:
                if not isinstance(filepath, str):
                    continue
                item = collection.files.add()
                item.filepath = filepath
        collection.active_file_index = _integer_value(collection_data.get("active_file_index"))

    preferences.texture_browser_active_collection_index = _integer_value(
        data.get("active_collection_index")
    )


def load_texture_browser_data():
    global _loading_texture_browser_data, _texture_browser_data_save_blocked

    preferences = _addon_preferences()
    if not _texture_browser_properties_available(preferences):
        return False

    filepath = texture_browser_data_filepath()
    if not filepath or not os.path.isfile(filepath):
        _texture_browser_data_save_blocked = False
        return save_texture_browser_data()

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("root value must be an object")
        if data.get("version") != _TEXTURE_BROWSER_DATA_VERSION:
            raise ValueError(f"unsupported version {data.get('version')!r}")

        _loading_texture_browser_data = True
        _populate_texture_browser_data(preferences, data)
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
        _texture_browser_data_save_blocked = True
        print(
            f"Anvil Level Design: Error loading texture browser data: {exc}",
            flush=True,
        )
        return False
    finally:
        _loading_texture_browser_data = False

    _texture_browser_data_save_blocked = False
    debug_log(f"[TextureBrowser] Loaded user data from {filepath}")
    return True
