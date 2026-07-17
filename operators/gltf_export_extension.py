import os

import bmesh
import bpy

from ..core.materials import (
    get_image_from_material,
    infer_primary_image_from_shader,
    get_principled_bsdf_from_material,
    is_texture_alpha_connected,
    is_vertex_colors_enabled,
)
from ..core.logging import debug_log


_ANVIL_GLTF_PANEL_KEY = "anvil_level_design"
_ANVIL_WELD_MODE_PROP = "_aw_mode"
_ANVIL_PREFAB_MODE = "PREFAB"


def draw_anvil_gltf_export_panel(context, layout):
    props = context.scene.level_design_props

    # Enable "Remember Export Settings" so quick re-export works correctly
    # when drawing inside the regular File > Export glTF dialog.
    space_data = context.space_data
    operator = getattr(space_data, "active_operator", None)
    if operator and hasattr(operator, 'will_save_settings'):
        operator.will_save_settings = True

    _draw_anvil_gltf_settings(layout, props)


def _draw_anvil_gltf_settings(layout, props):
    header, body = layout.panel("GLTF_export_anvil", default_closed=True)
    header.use_property_split = False
    header.prop(props, "gltf_anvil_enabled", text="")
    header.label(text="Anvil")
    if body:
        body.active = props.gltf_anvil_enabled
        body.use_property_split = True
        body.use_property_decorate = False

        col = body.column()
        col.prop(props, "gltf_anvil_scale")
        col.prop(props, "gltf_anvil_apply_modifiers")
        col.prop(props, "gltf_anvil_separate_loose")
        col.prop(props, "gltf_anvil_always_combine_materials")
        col.prop(props, "gltf_anvil_debug")


class glTF2ExportUserExtension:
    """Hook into Blender's glTF exporter to apply Anvil preprocessing.

    Uses pre_export_hook to clone the scene and apply destructive operations
    before the exporter collects any data, then post_export_hook to clean up.
    """

    def __init__(self):
        # Required by the glTF exporter extension system
        from io_scene_gltf2.io.com.gltf2_io_extensions import Extension
        self.Extension = Extension
        self._export_scene = None
        self._source_scene = None
        self._original_collection_name = None
        self._debug_keep_export_scene = False
        self._original_names_by_pointer = {}
        self._export_material_names_by_pointer = {}

    def pre_export_hook(self, export_settings):
        original_collection_name = export_settings.get('gltf_collection', '')
        source_scene = _choose_source_scene(bpy.context.scene, original_collection_name)
        props = source_scene.level_design_props
        if not props.gltf_anvil_enabled:
            return

        debug_log("[glTF Anvil] Starting Anvil export preprocessing")

        self._source_scene = source_scene
        self._original_collection_name = original_collection_name
        self._debug_keep_export_scene = props.gltf_anvil_debug

        collection_path = None
        if original_collection_name:
            original_collection = bpy.data.collections.get(original_collection_name)
            if original_collection is None:
                raise RuntimeError(f"Collection '{original_collection_name}' was not found")

            collection_path = _find_collection_path(source_scene, original_collection)
            if collection_path is None:
                raise RuntimeError(
                    f"Collection '{original_collection_name}' was not found in scene '{source_scene.name}'"
                )

            debug_log(
                f"[glTF Anvil] Collection export detected: {original_collection_name}"
            )

        # Clone the scene (FULL_COPY duplicates objects + data)
        export_scene = _create_export_scene_copy(source_scene)
        export_scene.name = "Anvil_Export_Temp"
        self._export_scene = export_scene
        self._original_names_by_pointer = _map_original_export_names(source_scene, export_scene)

        # Keep Blender and glTF exporter operators in the scene being destructively prepared.
        bpy.context.window.scene = export_scene

        debug_log(f"[glTF Anvil] Created temp scene: {export_scene.name}")

        try:
            material_scope_collection = None
            if collection_path is not None:
                export_collection = _collection_at_path(export_scene, collection_path)
                if export_collection is None:
                    raise RuntimeError(
                        f"Could not map collection '{original_collection_name}' to the temp export scene"
                    )
                export_settings['gltf_collection'] = export_collection.name
                debug_log(
                    f"[glTF Anvil] Remapped collection export to temp collection: {export_collection.name}"
                )
                material_scope_collection = export_collection

            self._export_material_names_by_pointer = _prepare_export_scene(
                export_scene,
                material_scope_collection,
                props,
                self._original_names_by_pointer,
            )
            _map_generated_export_names(export_scene, self._original_names_by_pointer)
        except Exception as e:
            print(f"Level Design Tools: Error during Anvil export preprocessing: {e}")
            _cleanup(self._export_scene, self._source_scene, self._debug_keep_export_scene)
            _restore_collection_export_setting(export_settings, self._original_collection_name)
            self._clear_export_state()
            raise

        # Force the exporter to only export the active (cloned) scene.
        # Without this, if "Active Scene" is unchecked the exporter would
        # iterate bpy.data.scenes and export the original unmodified scene too.
        export_settings['gltf_active_scene'] = True

        debug_log("[glTF Anvil] Preprocessing complete, handing off to glTF exporter")

    def post_export_hook(self, export_settings):
        # Store the filepath for quick re-export (regardless of Anvil enabled state)
        filepath = export_settings.get('gltf_filepath', '')
        if filepath:
            # Use original scene props if we cloned, otherwise current scene
            target_scene = _valid_scene_or_fallback(self._source_scene, self._export_scene)
            if target_scene is None:
                target_scene = bpy.context.scene
            target_scene.level_design_props.last_export_filepath = filepath

        if self._export_scene is None:
            return

        try:
            _cleanup(self._export_scene, self._source_scene, self._debug_keep_export_scene)
        finally:
            _restore_collection_export_setting(export_settings, self._original_collection_name)
            self._clear_export_state()

    def gather_scene_hook(self, gltf_scene, blender_scene, export_settings):
        if self._export_scene is None:
            return

        if blender_scene != self._export_scene:
            return

        if export_settings.get('gltf_collection', ''):
            collection = bpy.data.collections.get(export_settings['gltf_collection'])
            original_name = _get_original_export_name(collection, self._original_names_by_pointer)
            if original_name:
                gltf_scene.name = original_name
            return

        if _is_valid_scene(self._source_scene):
            gltf_scene.name = self._source_scene.name

    def gather_node_name_hook(self, gltf_name, blender_datablock, export_settings):
        original_name = _get_original_export_name(blender_datablock, self._original_names_by_pointer)
        if original_name:
            gltf_name.name = original_name

    def gather_mesh_hook(
        self,
        gltf_mesh,
        blender_data,
        blender_object,
        vertex_groups,
        modifiers,
        materials,
        export_settings,
    ):
        original_name = _get_original_export_name(blender_data, self._original_names_by_pointer)
        if original_name:
            gltf_mesh.name = original_name

    def gather_material_hook(self, gltf_material, blender_material, export_settings):
        material_name = _get_export_material_name(
            blender_material,
            self._export_material_names_by_pointer,
        )
        if material_name:
            gltf_material.name = material_name

    def _clear_export_state(self):
        self._export_scene = None
        self._source_scene = None
        self._original_collection_name = None
        self._debug_keep_export_scene = False
        self._original_names_by_pointer = {}
        self._export_material_names_by_pointer = {}


def _prepare_export_scene(scene, material_scope_collection, props, original_names_by_pointer):
    prefab_object_pointers, linked_prefab_object_pointers = _materialize_linked_prefab_meshes_for_export(
        scene,
        props.gltf_anvil_apply_modifiers,
        original_names_by_pointer,
    )
    _apply_modifiers(scene, props.gltf_anvil_apply_modifiers)
    _apply_scale(scene, props.gltf_anvil_scale, linked_prefab_object_pointers)
    _separate_loose(scene, props.gltf_anvil_separate_loose, prefab_object_pointers)
    return _canonicalize_materials(
        scene,
        material_scope_collection,
        props.gltf_anvil_always_combine_materials,
        linked_prefab_object_pointers,
    )


def _canonicalize_materials(
        scene,
        material_scope_collection,
        always_combine_materials,
        readonly_mesh_object_pointers):
    scope_objects = _material_scope_objects(scene, material_scope_collection)
    used_materials = _collect_used_materials(scope_objects)
    if not used_materials:
        return {}

    groups = _group_materials_for_export(used_materials, always_combine_materials)
    export_names_by_group = _build_export_material_names(groups)

    material_remap = {}
    export_material_names_by_pointer = {}
    for group_index, group in enumerate(groups):
        source_material = _choose_export_material_source(group)
        export_material = source_material.copy()
        export_material.name = f"ANVIL_EXPORT_Material_{group_index + 1}"
        export_name = export_names_by_group[group_index]

        for material in group:
            material_remap[material] = export_material

        export_material_names_by_pointer[export_material.as_pointer()] = export_name
        debug_log(
            f"[glTF Anvil]   Export material {export_name!r} "
            f"from {source_material.name!r} users={len(group)}"
        )

    _remap_scene_materials(scope_objects, material_remap, readonly_mesh_object_pointers)
    return export_material_names_by_pointer


def _material_scope_objects(scene, material_scope_collection):
    if material_scope_collection is None:
        return list(scene.objects)

    objects = []
    _append_collection_objects_recursive(material_scope_collection, objects)
    return objects


def _append_collection_objects_recursive(collection, objects):
    for obj in collection.objects:
        if obj not in objects:
            objects.append(obj)

    for child_collection in collection.children:
        _append_collection_objects_recursive(child_collection, objects)


def _collect_used_materials(objects):
    used_materials = []
    for obj in objects:
        if obj.type != 'MESH' or obj.data is None:
            continue

        for material in _object_materials_for_export(obj):
            if material is None:
                continue
            if material not in used_materials:
                used_materials.append(material)

    return used_materials


def _cleanup_materials_for_object(obj):
    materials = []
    if obj.data is not None:
        for material in obj.data.materials:
            if material is not None and material not in materials:
                materials.append(material)

    for material in _object_materials_for_export(obj):
        if material is not None and material not in materials:
            materials.append(material)

    return materials


def _object_materials_for_export(obj):
    materials = []
    for slot in obj.material_slots:
        materials.append(slot.material)
    return materials


def _group_materials_for_export(materials, always_combine_materials):
    groups_by_key = {}
    groups = []
    for material in materials:
        key = _material_export_key(material, always_combine_materials)
        if key not in groups_by_key:
            groups_by_key[key] = []
            groups.append(groups_by_key[key])
        groups_by_key[key].append(material)
    return groups


def _material_export_key(material, always_combine_materials):
    if always_combine_materials:
        return ('name', material.name)

    bsdf = get_principled_bsdf_from_material(material)
    return (
        'settings',
        material.name,
        _material_image_export_key(material),
        material.blend_method,
        is_texture_alpha_connected(material),
        is_vertex_colors_enabled(material),
        _rounded_export_value(_bsdf_socket_default_value(bsdf, "Roughness")),
        _rounded_export_value(_bsdf_socket_default_value(bsdf, "Metallic")),
        _rounded_export_value(_bsdf_socket_default_value(bsdf, "Emission Strength")),
        _rounded_export_value(_bsdf_socket_default_value(bsdf, "Emission Color")),
        _rounded_export_value(_bsdf_socket_default_value(bsdf, "Specular IOR Level")),
    )


def _material_image_export_key(material):
    image = (
        get_image_from_material(material)
        or infer_primary_image_from_shader(material)
    )
    if image is None:
        return ""

    filepath = ""
    try:
        filepath = image.filepath_from_user()
    except RuntimeError:
        filepath = ""
    if not filepath:
        filepath = image.filepath
    if filepath:
        return _normalized_blender_filepath(filepath)

    return image.name


def _bsdf_socket_default_value(bsdf, socket_name):
    if bsdf is None:
        return None

    socket = bsdf.inputs.get(socket_name)
    if socket is None:
        return None

    return socket.default_value


def _rounded_export_value(value):
    if value is None:
        return None

    if isinstance(value, float) or isinstance(value, int):
        return round(float(value), 5)

    try:
        return tuple(round(float(component), 5) for component in value)
    except TypeError:
        return value


def _choose_export_material_source(group):
    local_materials = [material for material in group if material.library is None]
    if local_materials:
        return local_materials[0]
    return group[0]


def _build_export_material_names(groups):
    grouped_indexes_by_base_name = {}
    for index, group in enumerate(groups):
        base_name = _choose_export_material_source(group).name
        grouped_indexes_by_base_name.setdefault(base_name, []).append(index)

    export_names_by_group = {}
    used_names = set()
    for base_name, group_indexes in grouped_indexes_by_base_name.items():
        sorted_group_indexes = sorted(
            group_indexes,
            key=lambda index: _group_export_name_priority(groups[index], index),
        )
        for sorted_index, group_index in enumerate(sorted_group_indexes):
            group = groups[group_index]
            if sorted_index == 0:
                export_name = base_name
            else:
                export_name = base_name + "__" + _readable_material_suffix(group)
            export_names_by_group[group_index] = _unique_export_name(export_name, used_names)

    return export_names_by_group


def _group_export_name_priority(group, group_index):
    source_material = _choose_export_material_source(group)
    library_path = _material_library_path(source_material)
    return (source_material.library is not None, library_path, group_index)


def _readable_material_suffix(group):
    material = _choose_export_material_source(group)
    bsdf = get_principled_bsdf_from_material(material)
    tokens = []

    library_name = _material_library_name(material)
    if library_name:
        tokens.append(_safe_export_name_token(library_name))

    roughness = _bsdf_socket_default_value(bsdf, "Roughness")
    if roughness is not None:
        tokens.append(f"rough_{_percent_token(roughness)}")

    metallic = _bsdf_socket_default_value(bsdf, "Metallic")
    if metallic is not None:
        tokens.append(f"metal_{_percent_token(metallic)}")

    emission_strength = _bsdf_socket_default_value(bsdf, "Emission Strength")
    if emission_strength is not None and float(emission_strength) > 0.0:
        tokens.append(f"emit_{_percent_token(emission_strength)}")

    if material.blend_method != 'OPAQUE' or is_texture_alpha_connected(material):
        tokens.append("alpha")

    if is_vertex_colors_enabled(material):
        tokens.append("vcol")

    if not tokens:
        tokens.append("variant")

    return "_".join(tokens)


def _material_library_name(material):
    if material.library is None:
        return ""

    filepath = material.library.filepath
    if not filepath:
        return material.library.name

    basename = os.path.basename(bpy.path.abspath(filepath))
    return os.path.splitext(basename)[0]


def _material_library_path(material):
    if material.library is None:
        return ""
    if not material.library.filepath:
        return material.library.name
    return _normalized_blender_filepath(material.library.filepath)


def _percent_token(value):
    return f"{int(round(float(value) * 100.0)):03d}"


def _safe_export_name_token(value):
    result = []
    for character in value:
        if character.isalnum():
            result.append(character)
        else:
            result.append("_")

    token = "".join(result).strip("_")
    if token:
        return token
    return "material"


def _unique_export_name(name, used_names):
    if name not in used_names:
        used_names.add(name)
        return name

    suffix = 2
    while True:
        candidate = f"{name}_{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1


def _remap_scene_materials(objects, material_remap, readonly_mesh_object_pointers):
    for obj in objects:
        if obj.type != 'MESH' or obj.data is None:
            continue
        if obj.as_pointer() in readonly_mesh_object_pointers:
            _remap_readonly_object_material_slots(obj, material_remap)
        else:
            _remap_writable_object_material_slots(obj, material_remap)


def _remap_readonly_object_material_slots(obj, material_remap):
    for slot in obj.material_slots:
        material = slot.material
        if material not in material_remap:
            continue

        slot.link = 'OBJECT'
        slot.material = material_remap[material]
        debug_log(
            f"[glTF Anvil]   Object material override on linked mesh {obj.name}"
        )


def _remap_writable_object_material_slots(obj, material_remap):
    materials = obj.data.materials
    for index, slot in enumerate(obj.material_slots):
        material = slot.material
        if material not in material_remap:
            continue

        if slot.link == 'OBJECT':
            slot.material = material_remap[material]
        elif index < len(materials):
            materials[index] = material_remap[material]


def _get_export_material_name(material, export_material_names_by_pointer):
    if material is None:
        return None

    try:
        return export_material_names_by_pointer.get(material.as_pointer())
    except ReferenceError:
        return None


def _map_original_export_names(source_scene, export_scene):
    mapped_objects = set()
    mapped_data = set()
    original_names_by_pointer = {}
    _map_collection_tree_names(
        source_scene.collection,
        export_scene.collection,
        mapped_objects,
        mapped_data,
        original_names_by_pointer,
    )
    return original_names_by_pointer


def _map_collection_tree_names(
    source_collection,
    export_collection,
    mapped_objects,
    mapped_data,
    original_names_by_pointer,
):
    _store_original_export_name(export_collection, source_collection.name, original_names_by_pointer)
    _map_collection_object_names(
        source_collection,
        export_collection,
        mapped_objects,
        mapped_data,
        original_names_by_pointer,
    )

    source_children = list(source_collection.children)
    export_children = list(export_collection.children)
    child_count = min(len(source_children), len(export_children))
    for index in range(child_count):
        _map_collection_tree_names(
            source_children[index],
            export_children[index],
            mapped_objects,
            mapped_data,
            original_names_by_pointer,
        )


def _map_collection_object_names(
    source_collection,
    export_collection,
    mapped_objects,
    mapped_data,
    original_names_by_pointer,
):
    source_objects = list(source_collection.objects)
    export_objects = list(export_collection.objects)
    object_count = min(len(source_objects), len(export_objects))
    for index in range(object_count):
        source_object = source_objects[index]
        export_object = export_objects[index]
        export_pointer = export_object.as_pointer()
        if export_pointer in mapped_objects:
            continue

        mapped_objects.add(export_pointer)
        _store_original_export_name(export_object, source_object.name, original_names_by_pointer)
        _map_object_data_name(source_object, export_object, mapped_data, original_names_by_pointer)


def _map_object_data_name(source_object, export_object, mapped_data, original_names_by_pointer):
    if source_object.data is None or export_object.data is None:
        return

    export_data_pointer = export_object.data.as_pointer()
    if export_data_pointer in mapped_data:
        return

    mapped_data.add(export_data_pointer)
    _store_original_export_name(export_object.data, source_object.data.name, original_names_by_pointer)


def _map_generated_export_names(scene, original_names_by_pointer):
    object_names_by_base = {}
    data_names_by_base = {}

    for obj in scene.objects:
        original_object_name = _get_original_export_name(obj, original_names_by_pointer)
        if original_object_name:
            object_names_by_base[_strip_blender_numeric_suffix(obj.name)] = original_object_name

        if obj.data is None:
            continue

        original_data_name = _get_original_export_name(obj.data, original_names_by_pointer)
        if original_data_name:
            data_names_by_base[_strip_blender_numeric_suffix(obj.data.name)] = original_data_name

    for obj in scene.objects:
        if _get_original_export_name(obj, original_names_by_pointer) is None:
            original_object_name = object_names_by_base.get(_strip_blender_numeric_suffix(obj.name))
            if original_object_name:
                _store_original_export_name(obj, original_object_name, original_names_by_pointer)

        if obj.data is None:
            continue

        if _get_original_export_name(obj.data, original_names_by_pointer) is None:
            original_data_name = data_names_by_base.get(_strip_blender_numeric_suffix(obj.data.name))
            if original_data_name:
                _store_original_export_name(obj.data, original_data_name, original_names_by_pointer)


def _strip_blender_numeric_suffix(name):
    if len(name) < 4:
        return name
    if name[-4] != ".":
        return name
    if not name[-3:].isdigit():
        return name
    return name[:-4]


def _store_original_export_name(datablock, name, original_names_by_pointer):
    if datablock is None:
        return

    try:
        original_names_by_pointer[datablock.as_pointer()] = name
    except ReferenceError:
        return
    except AttributeError:
        return


def _get_original_export_name(datablock, original_names_by_pointer):
    if datablock is None:
        return None

    try:
        return original_names_by_pointer.get(datablock.as_pointer())
    except ReferenceError:
        return None
    except AttributeError:
        return None


def _choose_source_scene(active_scene, collection_name):
    if not collection_name:
        return active_scene

    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        return active_scene

    if not _scene_contains_collection(active_scene, collection):
        containing_scenes = _scenes_containing_collection(collection, active_scene)
        if len(containing_scenes) == 1:
            debug_log(
                f"[glTF Anvil] Using scene '{containing_scenes[0].name}' because it contains collection '{collection_name}'"
            )
            return containing_scenes[0]

    return active_scene


def _scenes_containing_collection(collection, scene_to_exclude):
    scenes = []
    for scene in bpy.data.scenes:
        if scene == scene_to_exclude:
            continue
        if _scene_contains_collection(scene, collection):
            scenes.append(scene)
    return scenes


def _scene_contains_collection(scene, target_collection):
    return _find_collection_path(scene, target_collection) is not None


def _find_collection_path(scene, target_collection):
    return _find_child_collection_path(scene.collection, target_collection, [])


def _find_child_collection_path(parent_collection, target_collection, parent_path):
    for index, child_collection in enumerate(parent_collection.children):
        child_path = parent_path + [index]
        if child_collection == target_collection:
            return child_path

        nested_path = _find_child_collection_path(child_collection, target_collection, child_path)
        if nested_path is not None:
            return nested_path

    return None


def _collection_at_path(scene, collection_path):
    collection = scene.collection
    for index in collection_path:
        if index >= len(collection.children):
            return None
        collection = collection.children[index]
    return collection


def _create_export_scene_copy(source_scene):
    existing_scene_pointers = _scene_pointer_set()
    bpy.context.window.scene = source_scene
    bpy.ops.scene.new(type='FULL_COPY')

    export_scene = _find_new_scene(existing_scene_pointers)
    if export_scene is None:
        raise RuntimeError("Could not identify the duplicated export scene")
    if export_scene == source_scene:
        raise RuntimeError("Duplicated export scene resolved to the original scene")

    return export_scene


def _scene_pointer_set():
    scene_pointers = set()
    for scene in bpy.data.scenes:
        scene_pointers.add(scene.as_pointer())
    return scene_pointers


def _find_new_scene(existing_scene_pointers):
    for scene in bpy.data.scenes:
        if scene.as_pointer() not in existing_scene_pointers:
            return scene
    return None


def _restore_collection_export_setting(export_settings, collection_name):
    if collection_name is None:
        return
    export_settings['gltf_collection'] = collection_name


def _apply_modifiers(scene, enabled):
    if not enabled:
        return

    debug_log("[glTF Anvil] Applying modifiers")

    for obj in scene.objects:
        if obj.type != 'MESH' or not obj.modifiers:
            continue
        if _has_armature_modifier(obj):
            debug_log(
                f"[glTF Anvil]   Skipping modifiers on {obj.name} because it has an armature modifier"
            )
            continue

        # Make this object active so modifier_apply works
        for o in scene.objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        while obj.modifiers:
            mod_name = obj.modifiers[0].name
            try:
                bpy.ops.object.modifier_apply(modifier=mod_name)
                debug_log(f"[glTF Anvil]   Applied modifier '{mod_name}' on {obj.name}")
            except RuntimeError:
                debug_log(f"[glTF Anvil]   Could not apply modifier '{mod_name}' on {obj.name}, removing")
                obj.modifiers.remove(obj.modifiers[0])


def _materialize_linked_prefab_meshes_for_export(
        scene,
        apply_modifiers,
        original_names_by_pointer):
    prefab_object_pointers = set()
    linked_prefab_object_pointers = set()
    prefab_library_paths = _scene_prefab_library_paths(scene)

    for obj in list(scene.objects):
        if not _is_linked_prefab_mesh_export_target(obj, prefab_library_paths):
            continue

        obj = _ensure_prefab_export_object_is_writable(
            scene,
            obj,
            original_names_by_pointer,
        )
        prefab_object_pointers.add(obj.as_pointer())
        if apply_modifiers and obj.modifiers and not _has_armature_modifier(obj):
            _materialize_prefab_evaluated_mesh_for_export(
                obj,
                original_names_by_pointer,
            )
        else:
            linked_prefab_object_pointers.add(obj.as_pointer())

    return prefab_object_pointers, linked_prefab_object_pointers


def _ensure_prefab_export_object_is_writable(scene, obj, original_names_by_pointer):
    if _object_material_slots_are_writable(obj):
        return obj

    local_obj = obj.copy()
    local_obj.data = obj.data
    _store_original_export_name(
        local_obj,
        _mesh_object_export_name(obj, original_names_by_pointer),
        original_names_by_pointer,
    )

    collections = _scene_object_collections(scene, obj)
    if not collections:
        scene.collection.objects.link(local_obj)
    else:
        for collection in collections:
            collection.objects.link(local_obj)
            collection.objects.unlink(obj)

    debug_log(
        f"[glTF Anvil]   Created local export object for linked prefab {obj.name}"
    )
    return local_obj


def _object_material_slots_are_writable(obj):
    if obj.library is None:
        return True
    return obj.override_library is not None


def _mesh_object_export_name(obj, original_names_by_pointer):
    original_name = _get_original_export_name(obj, original_names_by_pointer)
    if original_name:
        return original_name
    return obj.name


def _scene_object_collections(scene, obj):
    collections = []
    _append_object_collections(scene.collection, obj, collections)
    return collections


def _append_object_collections(collection, obj, collections):
    for collection_obj in collection.objects:
        if collection_obj == obj:
            collections.append(collection)
            break

    for child_collection in collection.children:
        _append_object_collections(child_collection, obj, collections)


def _is_linked_prefab_mesh_export_target(obj, prefab_library_paths):
    if obj.type != 'MESH':
        return False
    if obj.data is None:
        return False
    if obj.data.library is None:
        return False
    if _object_or_ancestor_is_prefab(obj):
        return True
    return _datablock_library_path(obj.data) in prefab_library_paths


def _has_armature_modifier(obj):
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE':
            return True
    return False


def _scene_prefab_library_paths(scene):
    paths = set()
    libraries = getattr(scene, "anvil_prefab_libraries", [])
    for lib_entry in libraries:
        path = _normalized_blender_filepath(lib_entry.filepath)
        if path:
            paths.add(path)
    return paths


def _datablock_library_path(datablock):
    library = getattr(datablock, "library", None)
    if library is None:
        return ""
    return _normalized_blender_filepath(library.filepath)


def _normalized_blender_filepath(filepath):
    if not filepath:
        return ""
    return os.path.normcase(os.path.normpath(bpy.path.abspath(filepath)))


def _object_or_ancestor_is_prefab(obj):
    current = obj
    while current is not None:
        if current.get(_ANVIL_WELD_MODE_PROP, "") == _ANVIL_PREFAB_MODE:
            return True
        current = current.parent
    return False


def _materialize_prefab_evaluated_mesh_for_export(obj, original_names_by_pointer):
    if obj.data is None:
        raise RuntimeError(
            f"Could not materialize prefab modifiers for '{obj.name}': no mesh data"
        )

    mesh_export_name = _mesh_export_name_for_modifier_object(
        obj,
        original_names_by_pointer,
    )
    old_mesh = obj.data

    try:
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        materialized_mesh = bpy.data.meshes.new_from_object(
            eval_obj,
            depsgraph=depsgraph,
            preserve_all_data_layers=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not materialize prefab modifiers for '{obj.name}': {exc}"
        )

    materialized_mesh.name = mesh_export_name
    _copy_missing_material_slots(materialized_mesh, old_mesh)
    obj.data = materialized_mesh
    _clear_object_modifiers(obj)
    _store_original_export_name(
        materialized_mesh,
        mesh_export_name,
        original_names_by_pointer,
    )
    _remove_replaced_local_mesh(old_mesh)
    debug_log(f"[glTF Anvil]   Materialized modifiers on {obj.name} for export")


def _mesh_export_name_for_modifier_object(obj, original_names_by_pointer):
    if obj.data is None:
        return obj.name + "Mesh"

    return _mesh_export_name_for_data(obj.data, original_names_by_pointer)


def _mesh_export_name_for_data(mesh, original_names_by_pointer):
    original_name = _get_original_export_name(mesh, original_names_by_pointer)
    if original_name:
        return original_name
    return mesh.name


def _copy_missing_material_slots(target_mesh, source_mesh):
    if source_mesh is None:
        return
    if len(target_mesh.materials) > 0:
        return

    for material in source_mesh.materials:
        target_mesh.materials.append(material)


def _clear_object_modifiers(obj):
    while obj.modifiers:
        obj.modifiers.remove(obj.modifiers[0])


def _remove_replaced_local_mesh(mesh):
    if mesh is None:
        return
    if mesh.library is not None:
        return
    if not _is_registered_mesh(mesh):
        return
    if mesh.users != 0:
        return

    bpy.data.meshes.remove(mesh)


def _apply_scale(scene, scale, linked_prefab_object_pointers):
    import math

    if math.isclose(scale, 1.0, rel_tol=1e-6):
        return

    debug_log(f"[glTF Anvil] Applying scale: {scale}")

    for obj in scene.objects:
        # Scale location relative to world origin
        obj.location = obj.location * scale

        # Combined scale factor
        scale_vec = obj.scale * scale

        if obj.type == 'MESH':
            if obj.as_pointer() in linked_prefab_object_pointers:
                obj.scale = scale_vec
                continue

            mesh = obj.data
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.scale(bm, vec=scale_vec, verts=bm.verts)
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()

        elif obj.type == 'CURVE':
            curve = obj.data
            for spline in curve.splines:
                if spline.type == 'BEZIER':
                    for point in spline.bezier_points:
                        point.co.x *= scale_vec.x
                        point.co.y *= scale_vec.y
                        point.co.z *= scale_vec.z
                        point.handle_left.x *= scale_vec.x
                        point.handle_left.y *= scale_vec.y
                        point.handle_left.z *= scale_vec.z
                        point.handle_right.x *= scale_vec.x
                        point.handle_right.y *= scale_vec.y
                        point.handle_right.z *= scale_vec.z
                else:  # NURBS or POLY
                    for point in spline.points:
                        point.co.x *= scale_vec.x
                        point.co.y *= scale_vec.y
                        point.co.z *= scale_vec.z
        else:
            continue

        # Reset scale to 1 since we baked it into the data
        obj.scale = (1.0, 1.0, 1.0)


def _separate_loose(scene, enabled, prefab_object_pointers):
    if not enabled:
        return

    debug_log("[glTF Anvil] Separating loose meshes")

    # Collect mesh objects (list() since we'll be adding new objects)
    mesh_objects = [obj for obj in scene.objects if obj.type == 'MESH']
    shared_mesh_pointers = _shared_mesh_pointers(mesh_objects)

    for obj in mesh_objects:
        if obj.as_pointer() in prefab_object_pointers:
            debug_log(f"[glTF Anvil]   Skipping prefab mesh {obj.name}")
            continue

        if obj.data is not None and obj.data.as_pointer() in shared_mesh_pointers:
            debug_log(f"[glTF Anvil]   Skipping linked mesh {obj.name}")
            continue

        for o in scene.objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Need to enter edit mode for mesh.separate
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            bpy.ops.mesh.separate(type='LOOSE')
        except RuntimeError:
            debug_log(f"[glTF Anvil]   Could not separate {obj.name}")
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')


def _shared_mesh_pointers(mesh_objects):
    mesh_user_counts = {}
    for obj in mesh_objects:
        if obj.data is None:
            continue

        pointer = obj.data.as_pointer()
        mesh_user_counts[pointer] = mesh_user_counts.get(pointer, 0) + 1

    # This snapshot is defensive rather than strictly necessary: linked meshes
    # would skip before separation anyway, but this keeps the policy based on
    # the pre-separation scene even as mesh.separate mutates object/data users.
    shared_mesh_pointers = set()
    for pointer, user_count in mesh_user_counts.items():
        if user_count > 1:
            shared_mesh_pointers.add(pointer)
    return shared_mesh_pointers


def _cleanup(export_scene, source_scene, debug):
    if debug:
        debug_log("[glTF Anvil] Debug mode: keeping export scene")
        # Switch back to original but leave the export scene intact
        _restore_window_scene(source_scene, export_scene)
        return

    debug_log("[glTF Anvil] Cleaning up temp scene")

    export_datablocks = _collect_export_scene_datablocks(export_scene)

    # Switch back to original scene first
    _restore_window_scene(source_scene, export_scene)

    # Delete the export scene and its objects
    if _is_valid_scene(export_scene):
        bpy.data.scenes.remove(export_scene, do_unlink=True)

    _remove_orphan_export_datablocks(export_datablocks)


def _collect_export_scene_datablocks(scene):
    datablocks = {
        'collections': [],
        'objects': [],
        'meshes': [],
        'curves': [],
        'materials': [],
    }
    if not _is_valid_scene(scene):
        return datablocks

    _collect_collection_objects(scene.collection, datablocks)
    for collection in scene.collection.children:
        _collect_collection_tree(collection, datablocks)

    return datablocks


def _collect_collection_tree(collection, datablocks):
    _append_unique_datablock(datablocks['collections'], collection)
    _collect_collection_objects(collection, datablocks)
    for child_collection in collection.children:
        _collect_collection_tree(child_collection, datablocks)


def _collect_collection_objects(collection, datablocks):
    for obj in collection.objects:
        _append_unique_datablock(datablocks['objects'], obj)
        if obj.type == 'MESH':
            _append_unique_datablock(datablocks['meshes'], obj.data)
            for material in _cleanup_materials_for_object(obj):
                if material is not None:
                    _append_unique_datablock(datablocks['materials'], material)
        elif obj.type == 'CURVE':
            _append_unique_datablock(datablocks['curves'], obj.data)


def _append_unique_datablock(datablocks, datablock):
    if datablock not in datablocks:
        datablocks.append(datablock)


def _remove_orphan_export_datablocks(datablocks):
    for collection in datablocks['collections']:
        if _is_registered_collection(collection) and collection.users == 0:
            bpy.data.collections.remove(collection)

    for obj in datablocks['objects']:
        if _is_registered_object(obj) and obj.users == 0:
            bpy.data.objects.remove(obj)

    for mesh in datablocks['meshes']:
        if _is_registered_mesh(mesh) and mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    for curve in datablocks['curves']:
        if _is_registered_curve(curve) and curve.users == 0:
            bpy.data.curves.remove(curve)

    for material in datablocks['materials']:
        if _is_registered_material(material) and material.users == 0:
            bpy.data.materials.remove(material)


def _restore_window_scene(scene, scene_to_skip):
    target_scene = _valid_scene_or_fallback(scene, scene_to_skip)
    if target_scene is not None:
        bpy.context.window.scene = target_scene


def _valid_scene_or_fallback(scene, scene_to_skip):
    if _is_valid_scene(scene):
        return scene
    return _fallback_scene(scene_to_skip)


def _is_valid_scene(scene):
    if scene is None:
        return False
    try:
        scene.name
    except ReferenceError:
        return False

    for existing_scene in bpy.data.scenes:
        if existing_scene == scene:
            return True
    return False


def _fallback_scene(scene_to_skip):
    for scene in bpy.data.scenes:
        if scene != scene_to_skip:
            return scene
    return None


def _is_registered_collection(collection):
    return _is_registered_datablock(collection, bpy.data.collections)


def _is_registered_object(obj):
    return _is_registered_datablock(obj, bpy.data.objects)


def _is_registered_mesh(mesh):
    return _is_registered_datablock(mesh, bpy.data.meshes)


def _is_registered_curve(curve):
    return _is_registered_datablock(curve, bpy.data.curves)


def _is_registered_material(material):
    return _is_registered_datablock(material, bpy.data.materials)


def _is_registered_datablock(datablock, registered_datablocks):
    if datablock is None:
        return False
    try:
        datablock.name
    except ReferenceError:
        return False

    for registered_datablock in registered_datablocks:
        if registered_datablock == datablock:
            return True
    return False


def _register_gltf_export_panel():
    try:
        import io_scene_gltf2
    except Exception:
        return

    layout_draw = getattr(io_scene_gltf2, "exporter_extension_layout_draw", None)
    if layout_draw is None:
        return

    layout_draw[_ANVIL_GLTF_PANEL_KEY] = draw_anvil_gltf_export_panel


def _unregister_gltf_export_panel():
    try:
        import io_scene_gltf2
    except Exception:
        return

    layout_draw = getattr(io_scene_gltf2, "exporter_extension_layout_draw", None)
    if layout_draw is None:
        return

    layout_draw.pop(_ANVIL_GLTF_PANEL_KEY, None)


def register():
    _register_gltf_export_panel()


def unregister():
    _unregister_gltf_export_panel()
