import os

import bpy

from .base_test import AnvilTestCase
from ..operators.prefab_ops import (
    _make_all_free_objects_assets,
    _scan_library_prefab_assets,
)


def _new_object(name):
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


class PrefabAssetsTest(AnvilTestCase):

    def test_prefab_make_all_free_objects_assets_skips_nested_object_prefab_contents(self):
        scene = bpy.context.scene
        root_asset = _new_object("root_asset")
        child = _new_object("child")
        child.parent = root_asset
        parent_with_child = _new_object("parent_with_child")
        plain_child = _new_object("plain_child")
        plain_child.parent = parent_with_child
        parent_with_asset_child = _new_object("parent_with_asset_child")
        asset_child = _new_object("asset_child")
        asset_child.parent = parent_with_asset_child
        parent_with_asset_grandchild = _new_object("parent_with_asset_grandchild")
        middle_child = _new_object("middle_child")
        asset_grandchild = _new_object("asset_grandchild")
        middle_child.parent = parent_with_asset_grandchild
        asset_grandchild.parent = middle_child
        free_obj = _new_object("free_obj")
        collection_obj = _new_object("collection_obj")
        asset_collection = bpy.data.collections.new("asset_collection")
        scene.collection.children.link(asset_collection)
        scene.collection.objects.unlink(collection_obj)
        asset_collection.objects.link(collection_obj)

        root_asset.asset_mark()
        asset_child.asset_mark()
        asset_grandchild.asset_mark()
        asset_collection.asset_mark()

        marked_count = _make_all_free_objects_assets(scene)

        self.assertEqual(marked_count, 3)
        self.assertIsNotNone(free_obj.asset_data)
        self.assertIsNone(child.asset_data)
        self.assertIsNotNone(parent_with_child.asset_data)
        self.assertIsNone(plain_child.asset_data)
        self.assertIsNone(parent_with_asset_child.asset_data)
        self.assertIsNotNone(asset_child.asset_data)
        self.assertIsNone(parent_with_asset_grandchild.asset_data)
        self.assertIsNone(middle_child.asset_data)
        self.assertIsNotNone(asset_grandchild.asset_data)
        self.assertIsNotNone(collection_obj.asset_data)

    def test_prefab_library_scan_reads_object_assets_only(self):
        scene = bpy.context.scene
        asset_obj = _new_object("asset_obj")
        plain_obj = _new_object("plain_obj")
        asset_collection = bpy.data.collections.new("asset_collection")
        plain_collection = bpy.data.collections.new("plain_collection")
        scene.collection.children.link(asset_collection)
        scene.collection.children.link(plain_collection)

        asset_obj.asset_mark()
        asset_collection.asset_mark()

        output_root = os.path.join(os.path.dirname(__file__), "..", "test_outputs")
        os.makedirs(output_root, exist_ok=True)
        filepath = os.path.abspath(os.path.join(output_root, "prefab_asset_scan.blend"))

        try:
            bpy.data.libraries.write(
                filepath,
                {asset_obj, plain_obj, asset_collection, plain_collection},
            )
            assets = _scan_library_prefab_assets(filepath)
        finally:
            if os.path.isfile(filepath):
                os.remove(filepath)

        self.assertIn(('OBJECT', "asset_obj"), assets)
        self.assertNotIn(('OBJECT', "plain_obj"), assets)
        self.assertNotIn(('COLLECTION', "asset_collection"), assets)
        self.assertNotIn(('COLLECTION', "plain_collection"), assets)

    def test_prefab_instantiate_after_delete_and_refresh_recreates_override(self):
        scene = bpy.context.scene
        asset_obj = _new_object("Box")
        asset_obj.asset_mark()

        output_root = os.path.join(os.path.dirname(__file__), "..", "test_outputs")
        os.makedirs(output_root, exist_ok=True)
        filepath = os.path.abspath(os.path.join(output_root, "prefab_refresh_reinstantiate.blend"))

        try:
            bpy.data.libraries.write(filepath, {asset_obj})
            bpy.data.objects.remove(asset_obj, do_unlink=True)

            lib_entry = scene.anvil_prefab_libraries.add()
            lib_entry.filepath = filepath
            item = lib_entry.objects.add()
            item.name = "Box"
            item.asset_type = 'OBJECT'

            first_result = bpy.ops.leveldesign.prefab_instantiate(
                library_index=0,
                object_name="Box",
                asset_type='OBJECT',
            )
            self.assertEqual(first_result, {'FINISHED'})
            first_override = bpy.context.view_layer.objects.active
            self.assertIsNotNone(first_override)
            self.assertIsNotNone(first_override.override_library)

            bpy.data.objects.remove(first_override, do_unlink=True)
            bpy.context.view_layer.update()

            refresh_result = bpy.ops.leveldesign.prefab_refresh_libraries()
            self.assertEqual(refresh_result, {'FINISHED'})

            second_result = bpy.ops.leveldesign.prefab_instantiate(
                library_index=0,
                object_name="Box",
                asset_type='OBJECT',
            )
            self.assertEqual(second_result, {'FINISHED'})
            second_override = bpy.context.view_layer.objects.active
            self.assertIsNotNone(second_override)
            self.assertIsNotNone(second_override.override_library)
        finally:
            if os.path.isfile(filepath):
                os.remove(filepath)
