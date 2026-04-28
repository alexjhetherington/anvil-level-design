import os

import bpy

from .base_test import AnvilTestCase
from ..operators import prefab_ops


def _new_asset_object(name):
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.asset_mark()
    return obj


class _FakeScreen(dict):

    def __init__(self):
        super().__init__()
        self.areas = _FakeBlenderCollection()


class _FakeBlenderCollection(list):

    def __contains__(self, _item):
        raise TypeError("bpy_prop_collection.__contains__ expects a data-block name")


class _FakeWindow:

    def __init__(self):
        self.screen = _FakeScreen()


class _FakeRegion:

    def __init__(self, region_type):
        self.type = region_type


class _FakeSpaces:

    def __init__(self, active):
        self.active = active


class _FakeArea:

    def __init__(self, area_type):
        self.type = area_type
        self.regions = _FakeBlenderCollection([_FakeRegion('WINDOW')])
        self.spaces = _FakeSpaces(object())


class PrefabBrowserHostTest(AnvilTestCase):

    def tearDown(self):
        bpy.context.scene.anvil_prefab_libraries.clear()
        super().tearDown()

    def test_prefab_browser_window_marker_only_matches_marked_screens(self):
        marked_window = _FakeWindow()
        unmarked_window = _FakeWindow()
        marked_window.screen[prefab_ops._PREFAB_BROWSER_SCREEN_KEY] = True

        self.assertTrue(prefab_ops._is_prefab_browser_window(marked_window))
        self.assertFalse(prefab_ops._is_prefab_browser_window(unmarked_window))
        self.assertFalse(prefab_ops._is_prefab_browser_window(None))

    def test_prefab_browser_operator_opens_marked_preferences_popup_window(self):
        scene = bpy.context.scene
        scene.anvil_prefab_mode = 'SCENE'
        source_window = next(
            window for window in bpy.context.window_manager.windows
            if not prefab_ops._is_prefab_browser_window(window)
        )
        source_area = next(area for area in source_window.screen.areas if area.type == 'VIEW_3D')
        source_region = next(region for region in source_area.regions if region.type == 'WINDOW')

        try:
            with bpy.context.temp_override(
                    window=source_window,
                    area=source_area,
                    region=source_region):
                result = bpy.ops.leveldesign.prefab_browser()
            self.assertEqual(result, {'FINISHED'})

            popup_windows = prefab_ops._prefab_browser_popup_windows(
                bpy.context.window_manager.windows
            )
            self.assertEqual(len(popup_windows), 1)
            opened_window = popup_windows[0]

            self.assertTrue(prefab_ops._is_prefab_browser_window(opened_window))
            self.assertEqual(opened_window.screen.areas[0].ui_type, 'PREFERENCES')
            self.assertEqual(
                prefab_ops._prefab_browser_interaction["source_window"],
                source_window,
            )
        finally:
            for window in prefab_ops._prefab_browser_popup_windows(
                    bpy.context.window_manager.windows):
                with bpy.context.temp_override(window=window):
                    bpy.ops.leveldesign.prefab_browser_close()

    def test_prefab_browser_prefab_placement_uses_source_window_3d_view_context(self):
        first_window = _FakeWindow()
        first_window.screen.areas.append(_FakeArea('VIEW_3D'))
        source_window = _FakeWindow()
        source_area = _FakeArea('VIEW_3D')
        source_window.screen.areas.append(source_area)
        source_region = source_area.regions[0]

        interaction = prefab_ops._prefab_browser_interaction
        interaction["source_window"] = source_window
        interaction["source_area"] = source_area
        interaction["source_region"] = source_region

        try:
            view_context = prefab_ops._prefab_browser_3d_view_context(
                [first_window, source_window]
            )

            self.assertIs(view_context["window"], source_window)
            self.assertIs(view_context["area"], source_area)
            self.assertIs(view_context["region"], source_region)
        finally:
            interaction["source_window"] = None
            interaction["source_area"] = None
            interaction["source_region"] = None

    def test_prefab_browser_instantiate_from_popup_closes_popup_window(self):
        scene = bpy.context.scene
        scene.anvil_prefab_mode = 'SCENE'
        asset_obj = _new_asset_object("PopupBox")
        test_output_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "test_outputs")
        )
        os.makedirs(test_output_root, exist_ok=True)
        filepath = os.path.join(test_output_root, "prefab_browser_popup_close.blend")

        try:
            bpy.data.libraries.write(filepath, {asset_obj})
            bpy.data.objects.remove(asset_obj, do_unlink=True)

            lib_entry = scene.anvil_prefab_libraries.add()
            lib_entry.filepath = filepath
            item = lib_entry.objects.add()
            item.name = "PopupBox"
            item.asset_type = 'OBJECT'

            open_result = bpy.ops.leveldesign.prefab_browser()
            self.assertEqual(open_result, {'FINISHED'})
            popup_window = prefab_ops._prefab_browser_popup_windows(
                bpy.context.window_manager.windows
            )[0]
            area = popup_window.screen.areas[0]
            region = next(region for region in area.regions if region.type == 'WINDOW')

            with bpy.context.temp_override(window=popup_window, area=area, region=region):
                add_result = bpy.ops.leveldesign.prefab_instantiate(
                    library_index=0,
                    object_name="PopupBox",
                    asset_type='OBJECT',
                )

            self.assertEqual(add_result, {'FINISHED'})
            self.assertEqual(
                prefab_ops._prefab_browser_popup_windows(bpy.context.window_manager.windows),
                [],
            )
            self.assertIsNotNone(bpy.context.view_layer.objects.active)
            self.assertEqual(bpy.context.view_layer.objects.active.name, "PopupBox")
        finally:
            for window in prefab_ops._prefab_browser_popup_windows(
                    bpy.context.window_manager.windows):
                with bpy.context.temp_override(window=window):
                    bpy.ops.leveldesign.prefab_browser_close()
            if os.path.isfile(filepath):
                os.remove(filepath)

    def test_prefab_browser_display_items_flattens_libraries_with_library_labels(self):
        scene = bpy.context.scene
        scene.anvil_prefab_libraries.clear()
        test_output_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "test_outputs")
        )
        castle_path = os.path.join(test_output_root, "castle.blend")
        props_path = os.path.join(test_output_root, "props.blend")

        castle_library = scene.anvil_prefab_libraries.add()
        castle_library.filepath = castle_path
        door = castle_library.objects.add()
        door.name = "Door"
        door.asset_type = 'OBJECT'
        hidden_collection = castle_library.objects.add()
        hidden_collection.name = "Castle Collection"
        hidden_collection.asset_type = 'COLLECTION'

        props_library = scene.anvil_prefab_libraries.add()
        props_library.filepath = props_path
        barrel = props_library.objects.add()
        barrel.name = "Barrel"
        barrel.asset_type = 'OBJECT'

        items = prefab_ops._prefab_browser_display_items(scene, "")

        self.assertEqual(
            items,
            [
                (0, castle_path, "castle.blend", 'OBJECT', "Door"),
                (1, props_path, "props.blend", 'OBJECT', "Barrel"),
            ],
        )

    def test_prefab_browser_display_items_search_matches_prefab_or_library_name(self):
        scene = bpy.context.scene
        scene.anvil_prefab_libraries.clear()
        test_output_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "test_outputs")
        )

        castle_library = scene.anvil_prefab_libraries.add()
        castle_library.filepath = os.path.join(test_output_root, "castle.blend")
        door = castle_library.objects.add()
        door.name = "Door"
        door.asset_type = 'OBJECT'

        props_library = scene.anvil_prefab_libraries.add()
        props_library.filepath = os.path.join(test_output_root, "props.blend")
        barrel = props_library.objects.add()
        barrel.name = "Barrel"
        barrel.asset_type = 'OBJECT'

        library_matches = prefab_ops._prefab_browser_display_items(scene, "castle")
        asset_matches = prefab_ops._prefab_browser_display_items(scene, "bar")

        self.assertEqual([item[4] for item in library_matches], ["Door"])
        self.assertEqual([item[4] for item in asset_matches], ["Barrel"])

    def test_prefab_browser_preview_scale_maps_percentage_to_icon_scale(self):
        self.assertEqual(prefab_ops._prefab_browser_preview_icon_scale(0.0), 5.0)
        self.assertEqual(prefab_ops._prefab_browser_preview_icon_scale(100.0), 12.5)

    def test_prefab_browser_grid_columns_use_preview_size_as_cell_width(self):
        small_preview_columns = prefab_ops._prefab_browser_grid_columns(1200, 0.0, 1.0, 1)
        large_preview_columns = prefab_ops._prefab_browser_grid_columns(1200, 100.0, 1.0, 1)
        widget_unit = prefab_ops._prefab_browser_widget_unit(1.0, 1)
        small_target_width = prefab_ops._prefab_browser_target_cell_width(0.0, 1.0, 1)
        small_icon_width = (
            prefab_ops._prefab_browser_preview_icon_scale(0.0)
            * widget_unit
        )
        small_margin = max(12, int(round(widget_unit * 0.6)))
        small_gap = 0
        two_column_width = small_target_width * 2 + small_gap + small_margin * 2

        self.assertGreater(small_preview_columns, large_preview_columns)
        self.assertGreater(small_target_width, small_icon_width)
        self.assertEqual(prefab_ops._prefab_browser_grid_columns(100, 100.0, 1.0, 1), 1)
        self.assertEqual(
            prefab_ops._prefab_browser_grid_columns(two_column_width - 1, 0.0, 1.0, 1),
            1,
        )
        self.assertEqual(
            prefab_ops._prefab_browser_grid_columns(two_column_width, 0.0, 1.0, 1),
            2,
        )

    def test_prefab_browser_short_label_preserves_cell_width_for_long_names(self):
        label = prefab_ops._prefab_browser_short_label(
            "Very Long Prefab Name That Would Otherwise Stretch The Grid Cell",
            0.0,
        )

        self.assertTrue(label.endswith("..."))
        self.assertLessEqual(len(label), int(prefab_ops._prefab_browser_preview_icon_scale(0.0) * 2.8))
