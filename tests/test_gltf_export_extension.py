import json
import os

import bpy

from .base_test import AnvilTestCase


FULL_EXPORT_ROUTE = "full"
COLLECTION_EXPORT_ROUTE = "collection"
_ANVIL_WELD_MODE_PROP = "_aw_mode"


def _export_output_path(filename):
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def _test_texture_path():
    return os.path.join(os.path.dirname(__file__), "dev_orange_wall.png")


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


def _remove_datablock_when_unused(datablocks, datablock):
    if datablock.users == 0:
        datablocks.remove(datablock)


def _write_object_prefab_library(filepath, obj, cleanup_datablocks):
    if os.path.isfile(filepath):
        os.remove(filepath)

    obj.asset_mark()
    try:
        bpy.data.libraries.write(filepath, {obj})
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        for datablocks, datablock in cleanup_datablocks:
            _remove_datablock_when_unused(datablocks, datablock)


def _write_mirror_modifier_prefab_library(filepath, object_name, mesh_name):
    obj = _create_mirror_modifier_object(
        bpy.context.scene.collection,
        object_name,
        mesh_name,
    )
    _write_object_prefab_library(
        filepath,
        obj,
        [(bpy.data.meshes, obj.data)],
    )


def _write_plain_prefab_library(filepath, object_name, mesh_name):
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(
        [(4.0, 0.0, 0.0), (5.0, 0.0, 0.0), (4.0, 1.0, 0.0)],
        [],
        [(0, 1, 2)],
    )
    mesh.update()

    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    _write_object_prefab_library(
        filepath,
        obj,
        [(bpy.data.meshes, mesh)],
    )


def _create_textured_material(material_name, image, roughness, metallic):
    material = _create_basic_material(material_name, roughness, metallic)
    nodes = material.node_tree.nodes
    bsdf = next(
        node for node in nodes
        if node.type == 'BSDF_PRINCIPLED'
    )
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = image
    material.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def _write_textured_prefab_library(
        filepath,
        object_name,
        mesh_name,
        material_name,
        image_filepath,
        roughness,
        metallic):
    obj = _create_triangle_object(
        bpy.context.scene.collection,
        object_name,
        mesh_name,
    )
    mesh = obj.data
    image = bpy.data.images.load(image_filepath)
    material = _create_textured_material(
        material_name,
        image,
        roughness,
        metallic,
    )
    mesh.materials.append(material)
    _write_object_prefab_library(
        filepath,
        obj,
        [
            (bpy.data.meshes, mesh),
            (bpy.data.materials, material),
            (bpy.data.images, image),
        ],
    )


def _add_prefab_library_entry(scene, filepath, object_name):
    lib_entry = scene.anvil_prefab_libraries.add()
    lib_entry.filepath = filepath
    item = lib_entry.objects.add()
    item.name = object_name
    item.asset_type = 'OBJECT'
    return len(scene.anvil_prefab_libraries) - 1


def _instantiate_prefab_for_export(collection, library_index, object_name):
    result = bpy.ops.leveldesign.prefab_instantiate(
        library_index=library_index,
        source_object_name=object_name,
        object_name=object_name,
        asset_type='OBJECT',
        use_placement=True,
        action_pivot=(0.0, 0.0, 0.0),
        action_normal=(0.0, 0.0, 1.0),
    )
    if result != {'FINISHED'}:
        raise RuntimeError(f"Could not instantiate prefab '{object_name}': {result}")

    obj = bpy.context.view_layer.objects.active
    if obj.name not in collection.objects:
        collection.objects.link(obj)
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


def _create_basic_material(material_name, roughness, metallic):
    material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    bsdf = next(
        node for node in material.node_tree.nodes
        if node.type == 'BSDF_PRINCIPLED'
    )
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return material


def _write_material_library(filepath, material_name, roughness, metallic):
    material = _create_basic_material(material_name, roughness, metallic)
    try:
        bpy.data.libraries.write(filepath, {material})
    finally:
        bpy.data.materials.remove(material)


def _load_linked_material(filepath, material_name):
    with bpy.data.libraries.load(filepath, link=True) as (data_from, data_to):
        data_to.materials = [material_name]
    return data_to.materials[0]


def _create_material_object(collection, object_name, mesh_name, material):
    obj = _create_triangle_object(collection, object_name, mesh_name)
    obj.data.materials.append(material)
    return obj


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


def _node_for_name(gltf_data, node_name):
    node_index = _node_names(gltf_data).index(node_name)
    return gltf_data["nodes"][node_index]


def _mesh_name_for_node(gltf_data, node_name):
    node = _node_for_name(gltf_data, node_name)
    mesh_index = node.get("mesh")
    if mesh_index is None:
        return None
    return _mesh_names(gltf_data)[mesh_index]


def _mesh_index_for_node(gltf_data, node_name):
    return _node_for_name(gltf_data, node_name).get("mesh")


def _node_scale(gltf_data, node_name):
    return _node_for_name(gltf_data, node_name).get("scale")


def _material_names(gltf_data):
    return [material.get("name") for material in gltf_data.get("materials", [])]


def _material_roughness_factor(gltf_data, material_name):
    material_index = _material_names(gltf_data).index(material_name)
    material = gltf_data["materials"][material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    return pbr.get("roughnessFactor")


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
    for material in bpy.data.materials:
        test_case.assertFalse(material.name.startswith("ANVIL_EXPORT_Material_"))


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

    def _run_scale_linked_duplicate_export_route_test(self, route):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        obj = _create_triangle_object(collection, "ScaleBlock", "ScaleBlockMesh")
        linked_obj = _create_linked_duplicate(collection, obj, "ScaleBlockLinked")
        linked_obj.scale = (3.0, 3.0, 3.0)
        _set_anvil_export_settings(scene, 2.0, False, False)

        filepath, gltf_data = _export_using_route(
            route,
            collection,
            f"matrix_scale_linked_duplicate_{route}.gltf",
        )

        _assert_export_names(
            self,
            gltf_data,
            [_expected_scene_name_for_route(route)],
            [obj.name, linked_obj.name],
            ["ScaleBlockMesh"],
        )
        self.assertEqual(len(gltf_data.get("meshes", [])), 1)
        self.assertEqual(
            _mesh_index_for_node(gltf_data, obj.name),
            _mesh_index_for_node(gltf_data, linked_obj.name),
        )
        self.assertIsNone(_node_scale(gltf_data, obj.name))
        self.assertEqual(_node_scale(gltf_data, linked_obj.name), [3.0, 3.0, 3.0])
        _assert_mesh_position_bounds(
            self,
            gltf_data,
            "ScaleBlockMesh",
            [0.0, 0.0, -2.0],
            [2.0, 0.0, 0.0],
        )
        self.assertEqual(scene.level_design_props.last_export_filepath, filepath)

    def _run_scale_linked_prefab_instances_export_route_test(self, route):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        library_path = _export_output_path(
            f"matrix_scale_linked_prefab_instances_{route}.blend"
        )
        _write_plain_prefab_library(
            library_path,
            "PrefabScaleBlock",
            "PrefabScaleBlockMesh",
        )
        library_index = _add_prefab_library_entry(
            scene,
            library_path,
            "PrefabScaleBlock",
        )
        obj = _instantiate_prefab_for_export(
            collection,
            library_index,
            "PrefabScaleBlock",
        )
        linked_obj = _instantiate_prefab_for_export(
            collection,
            library_index,
            "PrefabScaleBlock",
        )
        linked_obj.scale = (3.0, 3.0, 3.0)
        original_mesh_pointer = obj.data.as_pointer()
        self.assertEqual(linked_obj.data.as_pointer(), original_mesh_pointer)
        self.assertIsNotNone(obj.data.library)
        self.assertIsNotNone(linked_obj.data.library)
        _set_anvil_export_settings(scene, 2.0, False, False)

        filepath, gltf_data = _export_using_route(
            route,
            collection,
            f"matrix_scale_linked_prefab_instances_{route}.gltf",
        )

        _assert_export_names(
            self,
            gltf_data,
            [_expected_scene_name_for_route(route)],
            [obj.name, linked_obj.name],
            ["PrefabScaleBlockMesh"],
        )
        self.assertEqual(len(gltf_data.get("meshes", [])), 1)
        self.assertEqual(
            _mesh_index_for_node(gltf_data, obj.name),
            _mesh_index_for_node(gltf_data, linked_obj.name),
        )
        self.assertIsNone(_node_scale(gltf_data, obj.name))
        self.assertEqual(_node_scale(gltf_data, linked_obj.name), [3.0, 3.0, 3.0])
        _assert_mesh_position_bounds(
            self,
            gltf_data,
            "PrefabScaleBlockMesh",
            [8.0, 0.0, -2.0],
            [10.0, 0.0, 0.0],
        )
        self.assertEqual(obj.data.as_pointer(), original_mesh_pointer)
        self.assertEqual(linked_obj.data.as_pointer(), original_mesh_pointer)
        self.assertIsNotNone(obj.data.library)
        self.assertIsNotNone(linked_obj.data.library)
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

    def _run_apply_modifiers_placed_prefab_export_route_test(self, route):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        modifier_library_path = _export_output_path(
            f"matrix_apply_modifiers_placed_prefab_{route}.blend"
        )
        plain_library_path = _export_output_path(
            f"matrix_apply_modifiers_prefab_plain_{route}.blend"
        )
        _write_mirror_modifier_prefab_library(
            modifier_library_path,
            "PrefabMirroredBlock",
            "PrefabMirroredBlockMesh",
        )
        _write_plain_prefab_library(
            plain_library_path,
            "PrefabPlainBlock",
            "PrefabPlainBlockMesh",
        )
        modifier_library_index = _add_prefab_library_entry(
            scene,
            modifier_library_path,
            "PrefabMirroredBlock",
        )
        plain_library_index = _add_prefab_library_entry(
            scene,
            plain_library_path,
            "PrefabPlainBlock",
        )
        modifier_obj = _instantiate_prefab_for_export(
            collection,
            modifier_library_index,
            "PrefabMirroredBlock",
        )
        plain_obj = _instantiate_prefab_for_export(
            collection,
            plain_library_index,
            "PrefabPlainBlock",
        )
        plain_obj_2 = _instantiate_prefab_for_export(
            collection,
            plain_library_index,
            "PrefabPlainBlock",
        )
        del plain_obj[_ANVIL_WELD_MODE_PROP]
        del plain_obj_2[_ANVIL_WELD_MODE_PROP]
        original_modifier_mesh_pointer = modifier_obj.data.as_pointer()
        original_plain_mesh_pointer = plain_obj.data.as_pointer()
        original_plain_mesh_pointer_2 = plain_obj_2.data.as_pointer()
        _set_anvil_export_settings(scene, 2.0, True, True)

        self.assertEqual(modifier_obj.get(_ANVIL_WELD_MODE_PROP), 'PREFAB')
        self.assertIsNone(modifier_obj.library)
        self.assertIsNone(modifier_obj.override_library)
        self.assertIsNotNone(modifier_obj.data.library)
        self.assertEqual(len(modifier_obj.modifiers), 1)
        self.assertNotEqual(plain_obj.get(_ANVIL_WELD_MODE_PROP), 'PREFAB')
        self.assertIsNone(plain_obj.library)
        self.assertIsNone(plain_obj.override_library)
        self.assertIsNotNone(plain_obj.data.library)
        self.assertEqual(len(plain_obj.modifiers), 0)
        self.assertNotEqual(plain_obj_2.get(_ANVIL_WELD_MODE_PROP), 'PREFAB')
        self.assertIsNone(plain_obj_2.library)
        self.assertIsNone(plain_obj_2.override_library)
        self.assertIsNotNone(plain_obj_2.data.library)
        self.assertEqual(len(plain_obj_2.modifiers), 0)

        filepath, gltf_data = _export_using_route(
            route,
            collection,
            f"matrix_apply_modifiers_placed_prefab_{route}.gltf",
        )

        _assert_export_names(
            self,
            gltf_data,
            [_expected_scene_name_for_route(route)],
            [modifier_obj.name, plain_obj.name, plain_obj_2.name],
            ["PrefabMirroredBlockMesh", "PrefabPlainBlockMesh"],
        )
        self.assertEqual(len(gltf_data.get("meshes", [])), 2)
        self.assertEqual(
            _mesh_name_for_node(gltf_data, plain_obj.name),
            "PrefabPlainBlockMesh",
        )
        self.assertEqual(
            _mesh_name_for_node(gltf_data, plain_obj_2.name),
            "PrefabPlainBlockMesh",
        )
        self.assertEqual(
            _mesh_index_for_node(gltf_data, plain_obj.name),
            _mesh_index_for_node(gltf_data, plain_obj_2.name),
        )
        self.assertIsNone(_node_scale(gltf_data, modifier_obj.name))
        self.assertIsNone(_node_scale(gltf_data, plain_obj.name))
        self.assertIsNone(_node_scale(gltf_data, plain_obj_2.name))
        _assert_mesh_position_bounds(
            self,
            gltf_data,
            "PrefabMirroredBlockMesh",
            [-4.0, 0.0, -2.0],
            [4.0, 0.0, 0.0],
        )
        _assert_mesh_position_bounds(
            self,
            gltf_data,
            "PrefabPlainBlockMesh",
            [8.0, 0.0, -2.0],
            [10.0, 0.0, 0.0],
        )
        self.assertEqual(modifier_obj.data.as_pointer(), original_modifier_mesh_pointer)
        self.assertEqual(len(modifier_obj.modifiers), 1)
        self.assertIsNotNone(modifier_obj.data.library)
        self.assertEqual(plain_obj.data.as_pointer(), original_plain_mesh_pointer)
        self.assertEqual(len(plain_obj.modifiers), 0)
        self.assertIsNotNone(plain_obj.data.library)
        self.assertEqual(plain_obj_2.data.as_pointer(), original_plain_mesh_pointer_2)
        self.assertEqual(len(plain_obj_2.modifiers), 0)
        self.assertIsNotNone(plain_obj_2.data.library)
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

    def test_gltf_anvil_scale_full_export_route_with_linked_duplicate_applies_scale_once_and_preserves_shared_mesh(self):
        self._run_scale_linked_duplicate_export_route_test(FULL_EXPORT_ROUTE)

    def test_gltf_anvil_scale_collection_export_route_with_linked_duplicate_applies_scale_once_and_preserves_shared_mesh(self):
        self._run_scale_linked_duplicate_export_route_test(COLLECTION_EXPORT_ROUTE)

    def test_gltf_anvil_scale_full_export_route_with_linked_prefab_instances_bakes_scale_once_and_preserves_shared_mesh_and_source_links(self):
        self._run_scale_linked_prefab_instances_export_route_test(FULL_EXPORT_ROUTE)

    def test_gltf_anvil_scale_collection_export_route_with_linked_prefab_instances_bakes_scale_once_and_preserves_shared_mesh_and_source_links(self):
        self._run_scale_linked_prefab_instances_export_route_test(COLLECTION_EXPORT_ROUTE)

    def test_gltf_anvil_apply_modifiers_full_export_route_writes_original_names_and_applied_geometry(self):
        self._run_apply_modifiers_export_route_test(FULL_EXPORT_ROUTE)

    def test_gltf_anvil_apply_modifiers_collection_export_route_writes_original_names_and_applied_geometry(self):
        self._run_apply_modifiers_export_route_test(COLLECTION_EXPORT_ROUTE)

    def test_gltf_anvil_apply_modifiers_full_export_route_with_placed_prefabs_materialises_modifier_prefabs_only_and_preserves_unmodified_prefab_mesh_instancing(self):
        self._run_apply_modifiers_placed_prefab_export_route_test(FULL_EXPORT_ROUTE)

    def test_gltf_anvil_apply_modifiers_collection_export_route_with_placed_prefabs_materialises_modifier_prefabs_only_and_preserves_unmodified_prefab_mesh_instancing(self):
        self._run_apply_modifiers_placed_prefab_export_route_test(COLLECTION_EXPORT_ROUTE)

    def test_gltf_anvil_separate_loose_full_export_route_writes_original_names_and_two_meshes(self):
        self._run_separate_loose_export_route_test(FULL_EXPORT_ROUTE)

    def test_gltf_anvil_separate_loose_collection_export_route_writes_original_names_and_two_meshes(self):
        self._run_separate_loose_export_route_test(COLLECTION_EXPORT_ROUTE)

    def test_gltf_anvil_separate_loose_full_export_route_with_linked_duplicate_skips_splitting_and_writes_one_shared_mesh(self):
        self._run_separate_loose_linked_duplicate_export_route_test(FULL_EXPORT_ROUTE)

    def test_gltf_anvil_separate_loose_collection_export_route_with_linked_duplicate_skips_splitting_and_writes_one_shared_mesh(self):
        self._run_separate_loose_linked_duplicate_export_route_test(COLLECTION_EXPORT_ROUTE)


class GltfExportMaterialCanonicalizationTest(AnvilTestCase):
    def _run_linked_prefab_same_texture_strict_material_export_route_test(self, route):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        library_path = _export_output_path(
            f"materials_linked_prefab_same_texture_{route}.blend"
        )
        image_filepath = _test_texture_path()
        material_name = "IMG_export_wall.png"
        _write_textured_prefab_library(
            library_path,
            "PrefabTexturedBlock",
            "PrefabTexturedBlockMesh",
            material_name,
            image_filepath,
            0.35,
            0.0,
        )
        library_index = _add_prefab_library_entry(
            scene,
            library_path,
            "PrefabTexturedBlock",
        )
        prefab_obj = _instantiate_prefab_for_export(
            collection,
            library_index,
            "PrefabTexturedBlock",
        )
        prefab_obj_2 = _instantiate_prefab_for_export(
            collection,
            library_index,
            "PrefabTexturedBlock",
        )
        prefab_obj_2.name = "PrefabTexturedBlockSecond"
        original_prefab_mesh_pointer = prefab_obj.data.as_pointer()
        original_prefab_slot_link = prefab_obj.material_slots[0].link
        original_prefab_material = prefab_obj.material_slots[0].material
        original_prefab_mesh_pointer_2 = prefab_obj_2.data.as_pointer()
        original_prefab_slot_link_2 = prefab_obj_2.material_slots[0].link
        original_prefab_material_2 = prefab_obj_2.material_slots[0].material

        local_image = bpy.data.images.load(image_filepath)
        local_material = _create_textured_material(
            material_name,
            local_image,
            0.35,
            0.0,
        )
        _create_material_object(
            collection,
            "LocalWallBlock",
            "LocalWallBlockMesh",
            local_material,
        )
        _set_anvil_export_settings(scene, 1.0, False, False)
        scene.level_design_props.gltf_anvil_always_combine_materials = False

        filepath, gltf_data = _export_using_route(
            route,
            collection,
            f"materials_linked_prefab_same_texture_{route}.gltf",
        )

        _assert_export_names(
            self,
            gltf_data,
            [_expected_scene_name_for_route(route)],
            [prefab_obj.name, prefab_obj_2.name, "LocalWallBlock"],
            ["PrefabTexturedBlockMesh", "LocalWallBlockMesh"],
        )
        self.assertEqual(len(gltf_data.get("meshes", [])), 2)
        material_names = _material_names(gltf_data)
        self.assertCountEqual(
            material_names,
            [material_name],
        )
        self.assertEqual(len(material_names), 1)
        self.assertEqual(
            _mesh_name_for_node(gltf_data, prefab_obj.name),
            "PrefabTexturedBlockMesh",
        )
        self.assertEqual(
            _mesh_name_for_node(gltf_data, prefab_obj_2.name),
            "PrefabTexturedBlockMesh",
        )
        self.assertEqual(prefab_obj.data.as_pointer(), original_prefab_mesh_pointer)
        self.assertIsNotNone(prefab_obj.data.library)
        self.assertEqual(prefab_obj.material_slots[0].link, original_prefab_slot_link)
        self.assertEqual(prefab_obj.material_slots[0].material, original_prefab_material)
        self.assertEqual(prefab_obj_2.data.as_pointer(), original_prefab_mesh_pointer_2)
        self.assertIsNotNone(prefab_obj_2.data.library)
        self.assertEqual(prefab_obj_2.material_slots[0].link, original_prefab_slot_link_2)
        self.assertEqual(prefab_obj_2.material_slots[0].material, original_prefab_material_2)
        self.assertEqual(scene.level_design_props.last_export_filepath, filepath)
        _assert_no_temp_export_datablocks(self)

    def test_gltf_anvil_full_export_route_with_linked_prefab_and_local_materials_sharing_texture_strict_mode_combines_matching_materials(self):
        self._run_linked_prefab_same_texture_strict_material_export_route_test(FULL_EXPORT_ROUTE)

    def test_gltf_anvil_collection_export_route_with_linked_prefab_and_local_materials_sharing_texture_strict_mode_combines_matching_materials(self):
        self._run_linked_prefab_same_texture_strict_material_export_route_test(COLLECTION_EXPORT_ROUTE)

    def test_gltf_anvil_always_combine_materials_same_named_local_and_linked_materials_prefers_local_settings(self):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        material_name = "IMG_export_wall.png"
        library_path = _export_output_path("same_named_material_library.blend")
        if os.path.isfile(library_path):
            os.remove(library_path)

        _write_material_library(library_path, material_name, 0.8, 0.0)
        local_material = _create_basic_material(material_name, 0.2, 0.0)
        linked_material = _load_linked_material(library_path, material_name)
        self.assertEqual(local_material.name, linked_material.name)
        self.assertIsNone(local_material.library)
        self.assertIsNotNone(linked_material.library)

        _create_material_object(collection, "LocalMaterialBlock", "LocalMaterialBlockMesh", local_material)
        _create_material_object(collection, "LinkedMaterialBlock", "LinkedMaterialBlockMesh", linked_material)
        _set_anvil_export_settings(scene, 1.0, False, False)
        scene.level_design_props.gltf_anvil_always_combine_materials = True

        filepath, gltf_data = _export_using_route(
            FULL_EXPORT_ROUTE,
            collection,
            "materials_always_combine.gltf",
        )

        self.assertEqual(_material_names(gltf_data), [material_name])
        self.assertAlmostEqual(
            _material_roughness_factor(gltf_data, material_name),
            0.2,
            places=5,
        )
        self.assertEqual(scene.level_design_props.last_export_filepath, filepath)
        _assert_no_temp_export_datablocks(self)

    def test_gltf_anvil_strict_materials_same_named_different_settings_writes_readable_suffixes(self):
        scene = bpy.context.scene
        scene.name = "Scene"
        collection = _create_export_collection(scene)
        material_name = "IMG_export_wall.png"
        library_path = _export_output_path("strict_material_library.blend")
        if os.path.isfile(library_path):
            os.remove(library_path)

        _write_material_library(library_path, material_name, 0.8, 0.0)
        local_material = _create_basic_material(material_name, 0.2, 0.0)
        linked_material = _load_linked_material(library_path, material_name)
        self.assertEqual(local_material.name, linked_material.name)

        _create_material_object(collection, "StrictLocalBlock", "StrictLocalBlockMesh", local_material)
        _create_material_object(collection, "StrictLinkedBlock", "StrictLinkedBlockMesh", linked_material)
        _set_anvil_export_settings(scene, 1.0, False, False)
        scene.level_design_props.gltf_anvil_always_combine_materials = False

        filepath, gltf_data = _export_using_route(
            FULL_EXPORT_ROUTE,
            collection,
            "materials_strict_suffixes.gltf",
        )

        names = _material_names(gltf_data)
        self.assertEqual(len(names), 2)
        self.assertIn(material_name, names)
        suffixed_names = [name for name in names if name != material_name]
        self.assertEqual(len(suffixed_names), 1)
        self.assertTrue(suffixed_names[0].startswith(material_name + "__"))
        self.assertIn("strict_material_library", suffixed_names[0])
        self.assertIn("rough_080", suffixed_names[0])
        self.assertEqual(scene.level_design_props.last_export_filepath, filepath)
        _assert_no_temp_export_datablocks(self)


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
