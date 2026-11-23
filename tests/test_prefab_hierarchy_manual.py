"""Manual visual output for the supported multi-object prefab hierarchy.

The method is intentionally named with ``manual_`` so normal unittest
discovery skips it. Run it by exact name with ``--save`` to keep both the
review scene and its companion prefab library in ``test_outputs``.
"""

import os

import bpy

from . import base_test
from .base_test import AnvilTestCase


ROOT_OBJECT_NAME = "SupportedPrefabRoot"
CONTENTS_COLLECTION_NAME = "SupportedPrefabContents"
MESH_OBJECT_NAME = "SupportedMeshObject"
MESH_DATA_NAME = "SupportedMesh"
LIGHT_OBJECT_NAME = "SupportedLightObject"
LIGHT_DATA_NAME = "SupportedLight"
LIBRARY_FILENAME = "prefab_supported_hierarchy_library.blend"


def _test_output_root():
    if base_test.save_outputs:
        output_root = os.path.abspath(base_test.output_dir)
    else:
        output_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "test_outputs")
        )
    os.makedirs(output_root, exist_ok=True)
    return output_root


def _create_supported_prefab_mesh():
    mesh = bpy.data.meshes.new(MESH_DATA_NAME)
    mesh.from_pydata(
        [
            (-1.0, -1.0, -0.5),
            (1.0, -1.0, -0.5),
            (1.0, 1.0, -0.5),
            (-1.0, 1.0, -0.5),
            (-1.0, -1.0, 0.5),
            (1.0, -1.0, 0.5),
            (1.0, 1.0, 0.5),
            (-1.0, 1.0, 0.5),
        ],
        [],
        [
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ],
    )
    mesh.update()
    return mesh


def _write_supported_prefab_hierarchy(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)

    root = bpy.data.objects.new(ROOT_OBJECT_NAME, None)
    contents = bpy.data.collections.new(CONTENTS_COLLECTION_NAME)
    mesh = _create_supported_prefab_mesh()
    mesh_object = bpy.data.objects.new(MESH_OBJECT_NAME, mesh)
    light = bpy.data.lights.new(LIGHT_DATA_NAME, 'POINT')
    light.energy = 1000.0
    light_object = bpy.data.objects.new(LIGHT_OBJECT_NAME, light)
    light_object.location = (0.0, 0.0, 3.0)
    contents.objects.link(mesh_object)
    contents.objects.link(light_object)

    root.instance_type = 'COLLECTION'
    root.instance_collection = contents
    root.asset_mark()
    try:
        bpy.data.libraries.write(filepath, {root})
    finally:
        bpy.data.objects.remove(root, do_unlink=True)
        bpy.data.objects.remove(mesh_object, do_unlink=True)
        bpy.data.objects.remove(light_object, do_unlink=True)
        bpy.data.collections.remove(contents)
        bpy.data.meshes.remove(mesh)
        bpy.data.lights.remove(light)


def _add_prefab_library(scene, filepath):
    library = scene.anvil_prefab_libraries.add()
    library.filepath = filepath
    item = library.objects.add()
    item.name = ROOT_OBJECT_NAME
    item.asset_type = 'OBJECT'
    return len(scene.anvil_prefab_libraries) - 1


def _print_supported_hierarchy(root, contents, mesh_object, light_object):
    print("Anvil Level Design: Supported prefab hierarchy", flush=True)
    print(f"{root.name} [local root object]", flush=True)
    print(
        f"`-- instance_collection -> {contents.name} [linked collection]",
        flush=True,
    )
    print(
        f"    |-- objects -> {mesh_object.name} [linked MESH object]",
        flush=True,
    )
    print(
        f"    |   `-- data -> {mesh_object.data.name} [linked Mesh]",
        flush=True,
    )
    print(
        f"    `-- objects -> {light_object.name} [linked LIGHT object]",
        flush=True,
    )
    print(
        f"        `-- data -> {light_object.data.name} [linked Light]",
        flush=True,
    )


class PrefabHierarchyManualVisualTest(AnvilTestCase):

    def manual_prefab_supported_root_collection_instance_hierarchy_visible_check(self):
        output_root = _test_output_root()
        library_filepath = os.path.join(output_root, LIBRARY_FILENAME)
        _write_supported_prefab_hierarchy(library_filepath)
        library_index = _add_prefab_library(
            bpy.context.scene,
            library_filepath,
        )

        result = bpy.ops.leveldesign.prefab_instantiate(
            library_index=library_index,
            source_object_name=ROOT_OBJECT_NAME,
            object_name=ROOT_OBJECT_NAME,
            asset_type='OBJECT',
        )

        self.assertEqual(result, {'FINISHED'})
        placed_root = bpy.context.view_layer.objects.active
        self.assertIsNotNone(placed_root)
        self.assertEqual(placed_root.name, ROOT_OBJECT_NAME)
        self.assertIsNone(placed_root.library)
        self.assertIsNone(placed_root.override_library)
        self.assertEqual(placed_root.instance_type, 'COLLECTION')
        self.assertEqual(len(placed_root.children), 0)
        self.assertTrue(placed_root.select_get())

        contents = placed_root.instance_collection
        self.assertIsNotNone(contents)
        self.assertEqual(contents.name, CONTENTS_COLLECTION_NAME)
        self.assertIsNotNone(contents.library)

        objects_by_type = {
            obj.type: obj
            for obj in contents.all_objects
        }
        self.assertEqual(set(objects_by_type), {'MESH', 'LIGHT'})
        mesh_object = objects_by_type['MESH']
        light_object = objects_by_type['LIGHT']
        self.assertEqual(mesh_object.name, MESH_OBJECT_NAME)
        self.assertEqual(mesh_object.data.name, MESH_DATA_NAME)
        self.assertEqual(light_object.name, LIGHT_OBJECT_NAME)
        self.assertEqual(light_object.data.name, LIGHT_DATA_NAME)
        for child in (mesh_object, light_object):
            self.assertIsNotNone(child.library)
            self.assertIsNotNone(child.data.library)

        _print_supported_hierarchy(
            placed_root,
            contents,
            mesh_object,
            light_object,
        )
        print(
            f"Anvil Level Design: Companion prefab library: {library_filepath}",
            flush=True,
        )
        if base_test.save_outputs:
            review_filepath = os.path.join(
                base_test.output_dir,
                f"{self.id()}.blend",
            )
            print(
                f"Anvil Level Design: Review scene: {review_filepath}",
                flush=True,
            )
