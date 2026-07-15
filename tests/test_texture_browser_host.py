import os

import bpy

from .base_test import AnvilTestCase
from .helpers import TEXTURE_PATH, create_vertical_plane
from ..core.materials import get_image_from_material
from ..handlers import get_active_image
from ..texture_browser import browser as texture_browser
from ..texture_browser import persistence as texture_browser_persistence


HOST_TEXTURE_BROWSER_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "test_outputs", "texture_browser_host")
)
HOTSPOT_TEXTURE_PATH = os.path.join(os.path.dirname(TEXTURE_PATH), "dev_hotspot.png")


def _normal_path(path):
    return os.path.normcase(os.path.abspath(path))


def _ensure_output_dir():
    os.makedirs(HOST_TEXTURE_BROWSER_OUTPUT_DIR, exist_ok=True)
    return HOST_TEXTURE_BROWSER_OUTPUT_DIR


def _test_folder(name):
    folder = os.path.join(_ensure_output_dir(), name)
    os.makedirs(folder, exist_ok=True)
    return folder


def _touch_file(filepath):
    with open(filepath, "wb") as file:
        file.write(b"texture-browser-test")


def _set_suffix_filters(prefs, include_suffixes, exclude_suffixes):
    prefs.texture_browser_include_suffixes.clear()
    for suffix in include_suffixes:
        item = prefs.texture_browser_include_suffixes.add()
        item.suffix = suffix
    prefs.texture_browser_exclude_suffixes.clear()
    for suffix in exclude_suffixes:
        item = prefs.texture_browser_exclude_suffixes.add()
        item.suffix = suffix
    prefs.texture_browser_filters_initialized = True


def _snapshot_texture_browser_preferences():
    prefs = texture_browser._addon_preferences()
    if prefs is None:
        return None

    collections = []
    for collection in prefs.texture_browser_collections:
        collections.append((
            collection.name,
            [item.filepath for item in collection.files],
            collection.active_file_index,
        ))

    return {
        "preview_scale": prefs.texture_browser_preview_scale,
        "filters_initialized": prefs.texture_browser_filters_initialized,
        "last_folder_path": prefs.texture_browser_last_folder_path,
        "favorites": [
            (favorite.path, favorite.name)
            for favorite in prefs.texture_browser_favorites
        ],
        "active_favorite_index": prefs.texture_browser_active_favorite_index,
        "collections": collections,
        "active_collection_index": prefs.texture_browser_active_collection_index,
        "include_suffixes": [item.suffix for item in prefs.texture_browser_include_suffixes],
        "active_include_index": prefs.texture_browser_active_include_index,
        "exclude_suffixes": [item.suffix for item in prefs.texture_browser_exclude_suffixes],
        "active_exclude_index": prefs.texture_browser_active_exclude_index,
    }


def _restore_texture_browser_preferences(snapshot):
    if snapshot is None:
        return
    prefs = texture_browser._addon_preferences()
    if prefs is None:
        return

    prefs.texture_browser_preview_scale = snapshot["preview_scale"]
    prefs.texture_browser_filters_initialized = snapshot["filters_initialized"]
    prefs.texture_browser_last_folder_path = snapshot["last_folder_path"]

    prefs.texture_browser_favorites.clear()
    for path, name in snapshot["favorites"]:
        item = prefs.texture_browser_favorites.add()
        item.path = path
        item.name = name
    prefs.texture_browser_active_favorite_index = snapshot["active_favorite_index"]

    prefs.texture_browser_collections.clear()
    for name, filepaths, active_file_index in snapshot["collections"]:
        collection = prefs.texture_browser_collections.add()
        collection.name = name
        for filepath in filepaths:
            item = collection.files.add()
            item.filepath = filepath
        collection.active_file_index = active_file_index
    prefs.texture_browser_active_collection_index = snapshot["active_collection_index"]

    prefs.texture_browser_include_suffixes.clear()
    for suffix in snapshot["include_suffixes"]:
        item = prefs.texture_browser_include_suffixes.add()
        item.suffix = suffix
    prefs.texture_browser_active_include_index = snapshot["active_include_index"]

    prefs.texture_browser_exclude_suffixes.clear()
    for suffix in snapshot["exclude_suffixes"]:
        item = prefs.texture_browser_exclude_suffixes.add()
        item.suffix = suffix
    prefs.texture_browser_active_exclude_index = snapshot["active_exclude_index"]


def _close_texture_browser_popups():
    for window in texture_browser.texture_browser_modal.popup_windows(
            bpy.context.window_manager.windows):
        with bpy.context.temp_override(window=window):
            bpy.ops.leveldesign.texture_browser_close()


def _remove_output_files():
    if not os.path.isdir(HOST_TEXTURE_BROWSER_OUTPUT_DIR):
        return
    for root, dirs, files in os.walk(HOST_TEXTURE_BROWSER_OUTPUT_DIR, topdown=False):
        for filename in files:
            os.remove(os.path.join(root, filename))
        for dirname in dirs:
            os.rmdir(os.path.join(root, dirname))
    os.rmdir(HOST_TEXTURE_BROWSER_OUTPUT_DIR)


class TextureBrowserHostTest(AnvilTestCase):

    def setUp(self):
        self._texture_browser_saves_were_suspended = (
            texture_browser_persistence.texture_browser_saves_suspended()
        )
        texture_browser_persistence.set_texture_browser_saves_suspended(True)
        self._preferences_snapshot = _snapshot_texture_browser_preferences()
        prefs = texture_browser._addon_preferences()
        texture_browser._ensure_texture_browser_preferences(prefs)
        texture_browser._folder_scan_cache.clear()
        texture_browser._collection_scan_cache.clear()

    def tearDown(self):
        _close_texture_browser_popups()
        _restore_texture_browser_preferences(self._preferences_snapshot)
        texture_browser_persistence.set_texture_browser_saves_suspended(
            self._texture_browser_saves_were_suspended
        )
        texture_browser._folder_scan_cache.clear()
        texture_browser._collection_scan_cache.clear()
        _remove_output_files()
        super().tearDown()

    def test_texture_browser_last_folder_path_uses_saved_existing_folder_and_falls_back_when_missing(self):
        prefs = texture_browser._addon_preferences()
        window_manager = bpy.context.window_manager
        existing_folder = _test_folder("existing")
        missing_folder = os.path.join(_ensure_output_dir(), "missing")

        prefs.texture_browser_last_folder_path = existing_folder
        window_manager.anvil_texture_browser_folder_path = ""
        texture_browser._ensure_window_manager_folder(window_manager)

        self.assertEqual(
            _normal_path(window_manager.anvil_texture_browser_folder_path),
            _normal_path(existing_folder),
        )

        prefs.texture_browser_last_folder_path = missing_folder
        window_manager.anvil_texture_browser_folder_path = ""
        texture_browser._ensure_window_manager_folder(window_manager)

        self.assertEqual(
            _normal_path(window_manager.anvil_texture_browser_folder_path),
            _normal_path(texture_browser._blend_home_folder()),
        )

    def test_texture_browser_folder_and_collection_search_are_scoped_to_active_view(self):
        prefs = texture_browser._addon_preferences()
        window_manager = bpy.context.window_manager
        folder = _test_folder("folder")
        other_folder = _test_folder("other_folder")
        stone_path = os.path.join(folder, "stone.png")
        wood_path = os.path.join(folder, "wood.png")
        metal_path = os.path.join(other_folder, "metal.png")
        _touch_file(stone_path)
        _touch_file(wood_path)
        _touch_file(metal_path)
        _set_suffix_filters(prefs, [".png"], [])

        prefs.texture_browser_collections.clear()
        collection = prefs.texture_browser_collections.add()
        collection.name = "Mixed"
        item = collection.files.add()
        item.filepath = metal_path

        texture_browser._set_folder(window_manager, folder)
        folder_stone_matches = texture_browser._texture_browser_display_items_for_filter(
            bpy.context.scene,
            "stone",
            texture_browser._TEXTURE_BROWSER_FOLDER_VIEW,
        )
        folder_metal_matches = texture_browser._texture_browser_display_items_for_filter(
            bpy.context.scene,
            "metal",
            texture_browser._TEXTURE_BROWSER_FOLDER_VIEW,
        )
        collection_metal_matches = texture_browser._texture_browser_display_items_for_filter(
            bpy.context.scene,
            "metal",
            0,
        )
        collection_stone_matches = texture_browser._texture_browser_display_items_for_filter(
            bpy.context.scene,
            "stone",
            0,
        )

        self.assertEqual([item["filename"] for item in folder_stone_matches], ["stone.png"])
        self.assertEqual(folder_metal_matches, [])
        self.assertEqual([item["filename"] for item in collection_metal_matches], ["metal.png"])
        self.assertEqual(collection_stone_matches, [])

    def test_texture_browser_empty_include_suffixes_allow_non_image_files_with_no_preview_item(self):
        prefs = texture_browser._addon_preferences()
        window_manager = bpy.context.window_manager
        folder = _test_folder("mixed_files")
        notes_path = os.path.join(folder, "notes.txt")
        _touch_file(notes_path)
        _set_suffix_filters(prefs, [], [])

        texture_browser._set_folder(window_manager, folder)
        items = texture_browser._texture_browser_display_items_for_filter(
            bpy.context.scene,
            "",
            texture_browser._TEXTURE_BROWSER_FOLDER_VIEW,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["filename"], "notes.txt")
        self.assertFalse(items[0]["is_image"])
        self.assertEqual(items[0]["suffix"], ".txt")

    def test_texture_browser_apply_file_operator_applies_image_and_sets_active_image(self):
        obj = create_vertical_plane("TextureBrowserApplyPlane")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        result = bpy.ops.leveldesign.texture_browser_apply_file(
            filepath=HOTSPOT_TEXTURE_PATH,
        )

        self.assertEqual(result, {'FINISHED'})
        material = obj.data.materials[obj.data.polygons[0].material_index]
        image = get_image_from_material(material)
        self.assertIsNotNone(image)
        self.assertEqual(
            _normal_path(bpy.path.abspath(image.filepath)),
            _normal_path(HOTSPOT_TEXTURE_PATH),
        )
        self.assertEqual(get_active_image(), image)
