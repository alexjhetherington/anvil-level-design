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


def _add_prefab_library(scene, filepath, object_name):
    lib_entry = scene.anvil_prefab_libraries.add()
    lib_entry.filepath = filepath
    item = lib_entry.objects.add()
    item.name = object_name
    item.asset_type = 'OBJECT'
    return len(scene.anvil_prefab_libraries) - 1


def _write_instanced_hierarchy_prefab(filepath, object_name):
    root = bpy.data.objects.new(object_name, None)
    contents = bpy.data.collections.new(f"{object_name}Contents")
    children = []
    meshes = []
    for child_name in (f"{object_name}ChildA", f"{object_name}ChildB"):
        mesh = bpy.data.meshes.new(f"{child_name}Mesh")
        mesh.from_pydata(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [],
            [(0, 1, 2)],
        )
        mesh.update()
        child = bpy.data.objects.new(child_name, mesh)
        contents.objects.link(child)
        children.append(child)
        meshes.append(mesh)

    root.instance_type = 'COLLECTION'
    root.instance_collection = contents
    root.asset_mark()
    try:
        bpy.data.libraries.write(filepath, {root})
    finally:
        bpy.data.objects.remove(root, do_unlink=True)
        for child in children:
            bpy.data.objects.remove(child, do_unlink=True)
        bpy.data.collections.remove(contents)
        for mesh in meshes:
            bpy.data.meshes.remove(mesh)


class PrefabAssetsTest(AnvilTestCase):

    def test_prefab_instantiate_action_settings_hide_source_fields_and_use_readable_names(self):
        properties = bpy.ops.leveldesign.prefab_instantiate.get_rna_type().properties

        self.assertTrue(properties["library_index"].is_hidden)
        self.assertTrue(properties["source_object_name"].is_hidden)
        self.assertTrue(properties["asset_type"].is_hidden)
        self.assertFalse(properties["object_name"].is_hidden)
        self.assertEqual(properties["object_name"].name, "Object Name")
        self.assertEqual(properties["name_suffix"].name, "Suffix")
        self.assertEqual(properties["name_suffix"].default, "")
        self.assertEqual(properties["make_fully_local"].name, "Make Local")
        self.assertFalse(properties["make_fully_local"].default)

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

    def test_prefab_instantiate_after_delete_and_refresh_recreates_local_object_with_linked_data(self):
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
                source_object_name="Box",
                object_name="Box",
                asset_type='OBJECT',
            )
            self.assertEqual(first_result, {'FINISHED'})
            first_object = bpy.context.view_layer.objects.active
            self.assertIsNotNone(first_object)
            self.assertIsNone(first_object.library)
            self.assertIsNone(first_object.override_library)
            self.assertIsNotNone(first_object.data.library)

            bpy.data.objects.remove(first_object, do_unlink=True)
            bpy.context.view_layer.update()

            refresh_result = bpy.ops.leveldesign.prefab_refresh_libraries()
            self.assertEqual(refresh_result, {'FINISHED'})

            second_result = bpy.ops.leveldesign.prefab_instantiate(
                library_index=0,
                source_object_name="Box",
                object_name="Box",
                asset_type='OBJECT',
            )
            self.assertEqual(second_result, {'FINISHED'})
            second_object = bpy.context.view_layer.objects.active
            self.assertIsNotNone(second_object)
            self.assertIsNone(second_object.library)
            self.assertIsNone(second_object.override_library)
            self.assertIsNotNone(second_object.data.library)
        finally:
            if os.path.isfile(filepath):
                os.remove(filepath)

    def test_prefab_default_instantiation_makes_root_local_and_preserves_linked_hierarchy(self):
        scene = bpy.context.scene
        output_root = os.path.join(os.path.dirname(__file__), "..", "test_outputs")
        os.makedirs(output_root, exist_ok=True)
        filepath = os.path.abspath(os.path.join(output_root, "prefab_linked_hierarchy.blend"))
        object_name = "LinkedHierarchyPrefab"

        try:
            _write_instanced_hierarchy_prefab(filepath, object_name)
            library_index = _add_prefab_library(scene, filepath, object_name)

            result = bpy.ops.leveldesign.prefab_instantiate(
                library_index=library_index,
                source_object_name=object_name,
                object_name=object_name,
                asset_type='OBJECT',
            )

            self.assertEqual(result, {'FINISHED'})
            placed_object = bpy.context.view_layer.objects.active
            self.assertIsNone(placed_object.library)
            self.assertIsNone(placed_object.override_library)
            self.assertIsNotNone(placed_object.instance_collection)
            self.assertIsNotNone(placed_object.instance_collection.library)

            children = sorted(
                placed_object.instance_collection.all_objects,
                key=lambda child: child.name,
            )
            self.assertEqual(
                [child.name for child in children],
                [f"{object_name}ChildA", f"{object_name}ChildB"],
            )
            for child in children:
                self.assertIsNotNone(child.library)
                self.assertIsNotNone(child.data.library)
        finally:
            if os.path.isfile(filepath):
                os.remove(filepath)

    def test_prefab_make_local_instantiation_localizes_new_hierarchy_and_preserves_previous_linked_hierarchy(self):
        scene = bpy.context.scene
        output_root = os.path.join(os.path.dirname(__file__), "..", "test_outputs")
        os.makedirs(output_root, exist_ok=True)
        filepath = os.path.abspath(os.path.join(output_root, "prefab_isolated_local_hierarchy.blend"))
        object_name = "IsolatedLocalHierarchyPrefab"

        try:
            _write_instanced_hierarchy_prefab(filepath, object_name)
            library_index = _add_prefab_library(scene, filepath, object_name)

            linked_result = bpy.ops.leveldesign.prefab_instantiate(
                library_index=library_index,
                source_object_name=object_name,
                object_name=object_name,
                asset_type='OBJECT',
            )
            self.assertEqual(linked_result, {'FINISHED'})
            previous_object = bpy.context.view_layer.objects.active
            previous_collection = previous_object.instance_collection
            previous_children = list(previous_collection.all_objects)

            local_result = bpy.ops.leveldesign.prefab_instantiate(
                library_index=library_index,
                source_object_name=object_name,
                object_name=object_name,
                asset_type='OBJECT',
                make_fully_local=True,
            )

            self.assertEqual(local_result, {'FINISHED'})
            local_object = bpy.context.view_layer.objects.active
            self.assertIsNone(local_object.library)
            self.assertIsNone(local_object.override_library)
            self.assertIsNotNone(local_object.instance_collection)
            self.assertIsNone(local_object.instance_collection.library)

            local_children = list(local_object.instance_collection.all_objects)
            self.assertEqual(len(local_children), len(previous_children))
            for child in local_children:
                self.assertIsNone(child.library)
                self.assertIsNone(child.override_library)
                self.assertIsNone(child.data.library)

            self.assertIs(previous_object.instance_collection, previous_collection)
            self.assertIsNotNone(previous_collection.library)
            for child in previous_children:
                self.assertIsNotNone(child.library)
                self.assertIsNotNone(child.data.library)
        finally:
            if os.path.isfile(filepath):
                os.remove(filepath)

    def test_prefab_name_suffix_follows_blender_numbering_before_suffix(self):
        scene = bpy.context.scene
        asset_obj = _new_object("Tree")
        asset_obj.asset_mark()
        output_root = os.path.join(os.path.dirname(__file__), "..", "test_outputs")
        os.makedirs(output_root, exist_ok=True)
        filepath = os.path.abspath(os.path.join(output_root, "prefab_name_suffix.blend"))

        try:
            bpy.data.libraries.write(filepath, {asset_obj})
            bpy.data.objects.remove(asset_obj, do_unlink=True)
            library_index = _add_prefab_library(scene, filepath, "Tree")

            first_result = bpy.ops.leveldesign.prefab_instantiate(
                library_index=library_index,
                source_object_name="Tree",
                object_name="RenamedTree",
                asset_type='OBJECT',
                name_suffix="-col",
            )
            self.assertEqual(first_result, {'FINISHED'})
            first_object = bpy.context.view_layer.objects.active

            second_result = bpy.ops.leveldesign.prefab_instantiate(
                library_index=library_index,
                source_object_name="Tree",
                object_name="RenamedTree",
                asset_type='OBJECT',
                name_suffix="-col",
            )
            self.assertEqual(second_result, {'FINISHED'})
            second_object = bpy.context.view_layer.objects.active

            self.assertEqual(first_object.name, "RenamedTree-col")
            self.assertEqual(second_object.name, "RenamedTree.001-col")
        finally:
            if os.path.isfile(filepath):
                os.remove(filepath)
