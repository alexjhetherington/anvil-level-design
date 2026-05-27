import json
import os

import bpy

from .base_test import AnvilTestCase


FULL_EXPORT_ROUTE = "full"
COLLECTION_EXPORT_ROUTE = "collection"


def _export_output_path(filename):
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def _read_gltf(filepath):
    with open(filepath, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _create_export_collection(scene):
    collection = bpy.data.collections.new("ExportMe")
    scene.collection.children.link(collection)
    return collection


def _create_triangle_object(collection, object_name, mesh_name):
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [],
        [(0, 1, 2)],
    )
    mesh.update()

    obj = bpy.data.objects.new(object_name, mesh)
    collection.objects.link(obj)
    return obj


def _create_mirror_modifier_object(collection, object_name, mesh_name):
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(
        [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        [],
        [(0, 1, 2)],
    )
    mesh.update()

    obj = bpy.data.objects.new(object_name, mesh)
    collection.objects.link(obj)
    modifier = obj.modifiers.new("MirrorForExport", 'MIRROR')
    modifier.use_axis = (True, False, False)
    return obj


def _create_loose_mesh_object(collection, object_name, mesh_name):
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        ],
        [],
        [(0, 1, 2), (3, 4, 5)],
    )
    mesh.update()

    obj = bpy.data.objects.new(object_name, mesh)
    collection.objects.link(obj)
    return obj


def _create_linked_duplicate(collection, source_object, object_name):
    duplicate = source_object.copy()
    duplicate.name = object_name
    duplicate.data = source_object.data
    collection.objects.link(duplicate)
    return duplicate


def _set_anvil_export_settings(scene, scale, apply_modifiers, separate_loose):
    props = scene.level_design_props
    props.gltf_anvil_enabled = True
    props.gltf_anvil_scale = scale
    props.gltf_anvil_apply_modifiers = apply_modifiers
    props.gltf_anvil_separate_loose = separate_loose
    props.gltf_anvil_debug = False


def _scene_names(gltf_data):
    return [scene.get("name") for scene in gltf_data.get("scenes", [])]


def _node_names(gltf_data):
    return [node.get("name") for node in gltf_data.get("nodes", [])]


def _mesh_names(gltf_data):
    return [mesh.get("name") for mesh in gltf_data.get("meshes", [])]


def _assert_export_names(test_case, gltf_data, scene_names, node_names, mesh_names):
    test_case.assertCountEqual(_scene_names(gltf_data), scene_names)
    test_case.assertEqual(len(_scene_names(gltf_data)), len(scene_names))
    test_case.assertCountEqual(_node_names(gltf_data), node_names)
    test_case.assertEqual(len(_node_names(gltf_data)), len(node_names))
    test_case.assertCountEqual(_mesh_names(gltf_data), mesh_names)
    test_case.assertEqual(len(_mesh_names(gltf_data)), len(mesh_names))


def _position_accessor_for_mesh(gltf_data, mesh_name):
    mesh_index = _mesh_names(gltf_data).index(mesh_name)
    mesh = gltf_data["meshes"][mesh_index]
    primitive = mesh["primitives"][0]
    accessor_index = primitive["attributes"]["POSITION"]
    return gltf_data["accessors"][accessor_index]


def _assert_mesh_position_bounds(test_case, gltf_data, mesh_name, minimum, maximum):
    accessor = _position_accessor_for_mesh(gltf_data, mesh_name)
    actual_minimum = accessor["min"]
    actual_maximum = accessor["max"]
    for index in range(3):
        test_case.assertAlmostEqual(actual_minimum[index], minimum[index], places=5)
        test_case.assertAlmostEqual(actual_maximum[index], maximum[index], places=5)


def _active_layer_collection_for_collection(layer_collection, collection):
    if layer_collection.collection == collection:
        return layer_collection

    for child in layer_collection.children:
        found = _active_layer_collection_for_collection(child, collection)
        if found is not None:
            return found

    return None


def _activate_collection(collection):
    bpy.context.view_layer.update()
    layer_collection = _active_layer_collection_for_collection(
        bpy.context.view_layer.layer_collection,
        collection,
    )
    if layer_collection is None:
        raise RuntimeError(f"Could not find layer collection for '{collection.name}'")
    bpy.context.view_layer.active_layer_collection = layer_collection


def _export_full(filepath):
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLTF_SEPARATE',
        export_animations=False,
        export_apply=False,
        collection="",
        use_active_scene=False,
    )


def _export_collection(collection, filepath):
    _activate_collection(collection)
    bpy.ops.collection.exporter_add(name="IO_FH_gltf2")
    exporter = collection.exporters[0]
    exporter.filepath = filepath
    exporter.export_properties.export_format = 'GLTF_SEPARATE'
    exporter.export_properties.export_animations = False
    exporter.export_properties.export_apply = False
    bpy.ops.collection.exporter_export(index=0)


def _export_using_route(route, collection, filename):
    filepath = _export_output_path(filename)
    if route == FULL_EXPORT_ROUTE:
        _export_full(filepath)
    elif route == COLLECTION_EXPORT_ROUTE:
        _export_collection(collection, filepath)
    else:
        raise RuntimeError(f"Unknown export route '{route}'")

    return filepath, _read_gltf(filepath)


def _expected_scene_name_for_route(route):
    if route == FULL_EXPORT_ROUTE:
        return "Scene"
    if route == COLLECTION_EXPORT_ROUTE:
        return "ExportMe"
    raise RuntimeError(f"Unknown export route '{route}'")


def _assert_no_temp_export_datablocks(test_case):
    for scene in bpy.data.scenes:
        test_case.assertFalse(scene.name.startswith("Anvil_Export_Temp"))
    for collection in bpy.data.collections:
        test_case.assertFalse(collection.name.endswith(".001"))
    for obj in bpy.data.objects:
        test_case.assertFalse(obj.name.endswith(".001"))
    for mesh in bpy.data.meshes:
        test_case.assertFalse(mesh.name.endswith(".001"))


class GltfExportFeatureMatrixTest(AnvilTestCase):
    def _run_scale_export_route_test(self, route):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        _create_triangle_object(collection, "ScaleBlock", "ScaleBlockMesh")
        _set_anvil_export_settings(scene, 2.0, False, False)

        filepath, gltf_data = _export_using_route(
            route,
            collection,
            f"matrix_scale_{route}.gltf",
        )

        _assert_export_names(
            self,
            gltf_data,
            [_expected_scene_name_for_route(route)],
            ["ScaleBlock"],
            ["ScaleBlockMesh"],
        )
        _assert_mesh_position_bounds(
            self,
            gltf_data,
            "ScaleBlockMesh",
            [0.0, 0.0, -2.0],
            [2.0, 0.0, 0.0],
        )
        self.assertEqual(scene.level_design_props.last_export_filepath, filepath)

    def _run_apply_modifiers_export_route_test(self, route):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        _create_mirror_modifier_object(collection, "MirroredBlock", "MirroredBlockMesh")
        _set_anvil_export_settings(scene, 1.0, True, False)

        filepath, gltf_data = _export_using_route(
            route,
            collection,
            f"matrix_apply_modifiers_{route}.gltf",
        )

        _assert_export_names(
            self,
            gltf_data,
            [_expected_scene_name_for_route(route)],
            ["MirroredBlock"],
            ["MirroredBlockMesh"],
        )
        _assert_mesh_position_bounds(
            self,
            gltf_data,
            "MirroredBlockMesh",
            [-2.0, 0.0, -1.0],
            [2.0, 0.0, 0.0],
        )
        self.assertEqual(scene.level_design_props.last_export_filepath, filepath)

    def _run_separate_loose_export_route_test(self, route):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        _create_loose_mesh_object(collection, "LooseBlock", "LooseBlockMesh")
        _set_anvil_export_settings(scene, 1.0, False, True)

        filepath, gltf_data = _export_using_route(
            route,
            collection,
            f"matrix_separate_loose_{route}.gltf",
        )

        _assert_export_names(
            self,
            gltf_data,
            [_expected_scene_name_for_route(route)],
            ["LooseBlock", "LooseBlock"],
            ["LooseBlockMesh", "LooseBlockMesh"],
        )
        self.assertEqual(len(gltf_data.get("meshes", [])), 2)
        self.assertEqual(len(gltf_data.get("nodes", [])), 2)
        self.assertEqual(scene.level_design_props.last_export_filepath, filepath)

    def _run_separate_loose_linked_duplicate_export_route_test(self, route):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        obj = _create_loose_mesh_object(collection, "LooseBlock", "LooseBlockMesh")
        _create_linked_duplicate(collection, obj, "LooseBlockLinked")
        _set_anvil_export_settings(scene, 1.0, False, True)

        filepath, gltf_data = _export_using_route(
            route,
            collection,
            f"matrix_separate_loose_linked_duplicate_{route}.gltf",
        )

        _assert_export_names(
            self,
            gltf_data,
            [_expected_scene_name_for_route(route)],
            ["LooseBlock", "LooseBlockLinked"],
            ["LooseBlockMesh"],
        )
        self.assertEqual(len(gltf_data.get("meshes", [])), 1)
        self.assertEqual(len(gltf_data.get("nodes", [])), 2)
        self.assertEqual(scene.level_design_props.last_export_filepath, filepath)

    def test_gltf_anvil_scale_full_export_route_writes_original_names_and_scaled_geometry(self):
        self._run_scale_export_route_test(FULL_EXPORT_ROUTE)

    def test_gltf_anvil_scale_collection_export_route_writes_original_names_and_scaled_geometry(self):
        self._run_scale_export_route_test(COLLECTION_EXPORT_ROUTE)

    def test_gltf_anvil_apply_modifiers_full_export_route_writes_original_names_and_applied_geometry(self):
        self._run_apply_modifiers_export_route_test(FULL_EXPORT_ROUTE)

    def test_gltf_anvil_apply_modifiers_collection_export_route_writes_original_names_and_applied_geometry(self):
        self._run_apply_modifiers_export_route_test(COLLECTION_EXPORT_ROUTE)

    def test_gltf_anvil_separate_loose_full_export_route_writes_original_names_and_two_meshes(self):
        self._run_separate_loose_export_route_test(FULL_EXPORT_ROUTE)

    def test_gltf_anvil_separate_loose_collection_export_route_writes_original_names_and_two_meshes(self):
        self._run_separate_loose_export_route_test(COLLECTION_EXPORT_ROUTE)

    def test_gltf_anvil_separate_loose_full_export_route_with_linked_duplicate_skips_splitting_and_writes_one_shared_mesh(self):
        self._run_separate_loose_linked_duplicate_export_route_test(FULL_EXPORT_ROUTE)

    def test_gltf_anvil_separate_loose_collection_export_route_with_linked_duplicate_skips_splitting_and_writes_one_shared_mesh(self):
        self._run_separate_loose_linked_duplicate_export_route_test(COLLECTION_EXPORT_ROUTE)


class GltfExportRouteFlowTest(AnvilTestCase):
    def test_gltf_full_export_route_preprocesses_active_scene_and_cleans_up_temp_datablocks(self):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        obj = _create_triangle_object(collection, "FlowBlock", "FlowBlockMesh")
        original_scene_pointer = scene.as_pointer()
        original_object_pointer = obj.as_pointer()
        original_mesh_pointer = obj.data.as_pointer()
        other_scene = bpy.data.scenes.new("OtherScene")
        _create_triangle_object(other_scene.collection, "OtherSceneBlock", "OtherSceneBlockMesh")
        _set_anvil_export_settings(scene, 2.0, False, False)

        filepath, gltf_data = _export_using_route(
            FULL_EXPORT_ROUTE,
            collection,
            "route_flow_full.gltf",
        )

        _assert_export_names(self, gltf_data, ["Scene"], ["FlowBlock"], ["FlowBlockMesh"])
        _assert_mesh_position_bounds(
            self,
            gltf_data,
            "FlowBlockMesh",
            [0.0, 0.0, -2.0],
            [2.0, 0.0, 0.0],
        )
        self.assertEqual(bpy.context.scene.as_pointer(), original_scene_pointer)
        self.assertIn(other_scene.name, bpy.data.scenes)
        self.assertEqual(obj.as_pointer(), original_object_pointer)
        self.assertEqual(obj.data.as_pointer(), original_mesh_pointer)
        self.assertEqual(scene.level_design_props.last_export_filepath, filepath)
        _assert_no_temp_export_datablocks(self)

    def test_gltf_collection_export_route_preprocesses_collection_and_cleans_up_temp_datablocks(self):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        obj = _create_triangle_object(collection, "FlowBlock", "FlowBlockMesh")
        original_scene_pointer = scene.as_pointer()
        original_collection_pointer = collection.as_pointer()
        original_object_pointer = obj.as_pointer()
        original_mesh_pointer = obj.data.as_pointer()
        _set_anvil_export_settings(scene, 2.0, False, False)

        filepath, gltf_data = _export_using_route(
            COLLECTION_EXPORT_ROUTE,
            collection,
            "route_flow_collection.gltf",
        )

        _assert_export_names(self, gltf_data, ["ExportMe"], ["FlowBlock"], ["FlowBlockMesh"])
        _assert_mesh_position_bounds(
            self,
            gltf_data,
            "FlowBlockMesh",
            [0.0, 0.0, -2.0],
            [2.0, 0.0, 0.0],
        )
        self.assertEqual(bpy.context.scene.as_pointer(), original_scene_pointer)
        self.assertEqual(collection.as_pointer(), original_collection_pointer)
        self.assertEqual(obj.as_pointer(), original_object_pointer)
        self.assertEqual(obj.data.as_pointer(), original_mesh_pointer)
        self.assertEqual(len(collection.exporters), 1)
        self.assertEqual(collection.exporters[0].filepath, filepath)
        self.assertEqual(scene.level_design_props.last_export_filepath, filepath)
        _assert_no_temp_export_datablocks(self)
