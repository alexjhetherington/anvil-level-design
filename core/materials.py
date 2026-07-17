import os
import re

import bpy

from .logging import debug_log
from .material_shader import (
    build_canonical_material_shader,
    image_has_transparency,
    infer_primary_shader_image,
    validate_material_shader,
)


_BLENDER_SUFFIX = re.compile(r'\.\d{3,}$')
_UNASSIGNED_MATERIAL_NAME = "ANVIL_Unassigned"
MATERIAL_SCHEMA_VERSION = 1
DEFAULT_MATERIAL_NAME_PATTERN = "{relativePath}{filename}{extension}"
_last_material_count = 0


class MaterialMappingConflictError(RuntimeError):
    pass


def _addon_preferences():
    package_name = __package__.split('.', 1)[0]
    addon = bpy.context.preferences.addons.get(package_name)
    if addon is None:
        return None
    return addon.preferences


def remember_pixels_per_meter(props):
    """Store the current file's pixels per meter in addon preferences."""
    prefs = _addon_preferences()
    if prefs is None:
        return

    prefs.pref_pixels_per_meter = props.pixels_per_meter


def apply_remembered_pixels_per_meter(props):
    """Apply the last remembered pixels per meter to file properties."""
    prefs = _addon_preferences()
    if prefs is None:
        return

    props.pixels_per_meter = prefs.pref_pixels_per_meter


def remember_default_material_settings(props):
    """Store the current file's default material settings in addon preferences."""
    prefs = _addon_preferences()
    if prefs is None:
        return

    prefs.pref_default_interpolation = props.default_interpolation
    prefs.pref_default_texture_as_alpha = props.default_texture_as_alpha
    prefs.pref_default_vertex_colors = props.default_vertex_colors
    prefs.pref_default_roughness = props.default_roughness
    prefs.pref_default_metallic = props.default_metallic
    prefs.pref_default_emission_strength = props.default_emission_strength
    prefs.pref_default_emission_color = props.default_emission_color[:]
    prefs.pref_default_specular = props.default_specular
    prefs.pref_default_material_name_pattern = props.default_material_name_pattern


def apply_remembered_default_material_settings(props):
    """Apply the last remembered default material settings to file properties."""
    prefs = _addon_preferences()
    if prefs is None:
        return

    props.default_interpolation = prefs.pref_default_interpolation
    props.default_texture_as_alpha = prefs.pref_default_texture_as_alpha
    props.default_vertex_colors = prefs.pref_default_vertex_colors
    props.default_roughness = prefs.pref_default_roughness
    props.default_metallic = prefs.pref_default_metallic
    props.default_emission_strength = prefs.pref_default_emission_strength
    props.default_emission_color = prefs.pref_default_emission_color[:]
    props.default_specular = prefs.pref_default_specular
    props.default_material_name_pattern = prefs.pref_default_material_name_pattern


def _material_has_current_mapping(material):
    try:
        return (
            material is not None
            and material.library is None
            and getattr(material, "anvil_material_schema_version", 0)
            == MATERIAL_SCHEMA_VERSION
            and getattr(material, "anvil_primary_image", None) is not None
        )
    except ReferenceError:
        return False


def materials_mapped_to_image(image):
    if image is None:
        return []
    return [
        material for material in bpy.data.materials
        if _material_has_current_mapping(material)
        and material.anvil_primary_image == image
    ]


def get_primary_image_from_material(material):
    if not _material_has_current_mapping(material):
        return None
    try:
        return material.anvil_primary_image
    except ReferenceError:
        return None


def clear_material_mapping(material):
    if material is None or material.library is not None:
        return
    material.anvil_primary_image = None
    material.anvil_material_schema_version = 0


def set_material_primary_image(material, image):
    if material is None or material.library is not None:
        raise ValueError("Only local materials can be mapped")
    if image is None:
        clear_material_mapping(material)
        return

    conflicts = [
        mapped for mapped in materials_mapped_to_image(image)
        if mapped != material
    ]
    if conflicts:
        names = ", ".join(mapped.name for mapped in conflicts)
        raise MaterialMappingConflictError(
            f"Image {image.name!r} is already mapped to {names}"
        )

    material.anvil_primary_image = image
    material.anvil_material_schema_version = MATERIAL_SCHEMA_VERSION


def reset_duplicate_material_consolidation():
    """Reset material dedupe state after file lifecycle changes."""
    global _last_material_count
    _last_material_count = 0


def consolidate_duplicate_materials():
    """Find and merge duplicate IMG_ materials created by copy/paste.

    When objects are duplicated, Blender creates copies of materials with
    suffixes like .001, .002, etc. This function finds these duplicates
    and consolidates them to the base material name.
    """
    global _last_material_count

    current_count = len(bpy.data.materials)

    if current_count <= _last_material_count:
        _last_material_count = current_count
        return

    _last_material_count = current_count

    duplicate_pattern = re.compile(r'^(IMG_.+)\.(\d{3,})$')

    material_groups = {}

    for mat in bpy.data.materials:
        match = duplicate_pattern.match(mat.name)
        if match:
            base_name = match.group(1)
            suffix_num = int(match.group(2))
            if base_name not in material_groups:
                material_groups[base_name] = []
            material_groups[base_name].append((suffix_num, mat))

    if not material_groups:
        return

    for base_name, duplicates in material_groups.items():
        if base_name not in bpy.data.materials:
            duplicates.sort(key=lambda x: x[0])
            duplicates[0][1].name = base_name

    replacements = {}
    for base_name, duplicates in material_groups.items():
        canonical = bpy.data.materials[base_name]
        for suffix_num, mat in duplicates:
            if mat != canonical:
                # Explicit mappings are never a license to rewrite scene slots.
                # Fix Material Mappings resolves mapping conflicts without
                # merging distinct material datablocks.
                if (
                    _material_has_current_mapping(canonical)
                    or _material_has_current_mapping(mat)
                ):
                    continue
                replacements[mat] = canonical

    if not replacements:
        return

    for obj in bpy.data.objects:
        if obj.type != 'MESH' or not obj.data:
            continue

        materials = obj.data.materials
        for i, mat in enumerate(materials):
            if mat in replacements:
                materials[i] = replacements[mat]

    for dup_mat in replacements.keys():
        if dup_mat.users == 0:
            bpy.data.materials.remove(dup_mat)

    _last_material_count = len(bpy.data.materials)


def get_image_from_material(mat):
    """Return a material's explicit Anvil primary image, or None."""
    return get_primary_image_from_material(mat)


def infer_primary_image_from_shader(mat):
    """Return an unambiguous image suggestion for Fix Material Mappings."""
    return infer_primary_shader_image(mat)


def get_texture_dimensions_from_material(mat, ppm, default_size=128):
    """Get texture dimensions in meters from a specific material.

    Args:
        mat: Blender material (can be None)
        ppm: Pixels per meter setting
        default_size: Default texture size if no texture found (default 128)

    Returns:
        Tuple of (tex_meters_u, tex_meters_v)
    """
    image = get_image_from_material(mat)
    if image:
        tex_width = image.size[0] if image.size[0] > 0 else default_size
        tex_height = image.size[1] if image.size[1] > 0 else default_size
    else:
        tex_width = default_size
        tex_height = default_size
    return (tex_width / ppm, tex_height / ppm)


def _normalized_material_path(path):
    return path.replace('\\', '/')


def _material_relative_path(image_folder, blend_filepath):
    if not blend_filepath:
        return ""

    blend_folder = os.path.dirname(os.path.abspath(blend_filepath))
    try:
        relative_path = os.path.relpath(image_folder, blend_folder)
    except ValueError:
        return ""

    if relative_path == '.':
        return ""
    if relative_path:
        return _normalized_material_path(relative_path).rstrip('/') + '/'
    return ""


def _image_name_parts(image):
    filepath = ""
    try:
        filepath = image.filepath_from_user()
    except RuntimeError:
        filepath = image.filepath

    absolute_path = ""
    if filepath:
        absolute_path = os.path.abspath(bpy.path.abspath(filepath))

    if absolute_path:
        basename = os.path.basename(absolute_path)
        filename, extension = os.path.splitext(basename)
        image_folder = os.path.dirname(absolute_path)
        relative_path = _material_relative_path(
            image_folder,
            bpy.data.filepath,
        )
        return relative_path, filename, extension

    image_name = _BLENDER_SUFFIX.sub('', image.name)
    filename, extension = os.path.splitext(image_name)
    return "", filename, extension


def material_name_for_image(image, pattern):
    relative_path, filename, extension = _image_name_parts(image)
    return (
        pattern
        .replace("{relativePath}", relative_path)
        .replace("{filename}", filename)
        .replace("{extension}", extension)
    )


def _material_name_for_image(image):
    props = getattr(getattr(bpy.context, "scene", None), "level_design_props", None)
    pattern = (
        props.default_material_name_pattern
        if props is not None
        else DEFAULT_MATERIAL_NAME_PATTERN
    )
    return material_name_for_image(image, pattern)


def ensure_material_slot(mesh, mat):
    """Ensure this exact material datablock is in the mesh slots."""
    for index, slot_mat in enumerate(mesh.materials):
        if slot_mat == mat:
            return index

    mesh.materials.append(mat)
    return len(mesh.materials) - 1


def find_material_with_image(image):
    """Return the one local material explicitly mapped to this image, or None."""
    if image is None:
        return None

    materials = materials_mapped_to_image(image)
    if len(materials) == 1:
        mat = materials[0]
        debug_log(
            f"[FindMaterial] image={image.name!r} -> local material {mat.name!r}"
        )
        return mat

    if len(materials) > 1:
        names = ", ".join(material.name for material in materials)
        debug_log(
            f"[FindMaterial] image={image.name!r} has conflicting mappings: {names}"
        )
        raise MaterialMappingConflictError(
            f"Image {image.name!r} is mapped to multiple materials: {names}"
        )

    debug_log(
        f"[FindMaterial] image={image.name!r} -> no mapped local material"
    )
    return None


def resolve_material_for_image(image):
    """Return the unique mapped material, creating one only when none exists."""
    material = find_material_with_image(image)
    if material is not None:
        return material
    return create_material_with_image(image)


def get_unassigned_material():
    """Return a neutral material used to preserve untextured face slots."""
    mat = bpy.data.materials.get(_UNASSIGNED_MATERIAL_NAME)
    if mat:
        return mat

    mat = bpy.data.materials.new(name=_UNASSIGNED_MATERIAL_NAME)
    mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    return mat


def is_unassigned_material(material):
    return material is not None and material.name == _UNASSIGNED_MATERIAL_NAME


def get_principled_bsdf_from_material(mat):
    """Return the Principled BSDF node from a material, or None"""
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None
    validation = validate_material_shader(mat, get_primary_image_from_material(mat))
    if validation.bsdf_node is not None:
        return validation.bsdf_node
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node
    return None


def get_texture_node_from_material(mat):
    """Return the mapped/canonical texture node from a material, or None."""
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None
    validation = validate_material_shader(mat, get_primary_image_from_material(mat))
    if validation.texture_node is not None:
        return validation.texture_node
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE':
            return node
    return None


def is_texture_alpha_connected(mat):
    """Check if the texture's alpha output is connected to the BSDF alpha input"""
    if not mat or not mat.use_nodes or not mat.node_tree:
        return False
    tex = get_texture_node_from_material(mat)
    bsdf = get_principled_bsdf_from_material(mat)
    if not tex or not bsdf:
        return False
    for link in mat.node_tree.links:
        if link.from_node == tex and link.from_socket.name == "Alpha":
            if link.to_node == bsdf and link.to_socket.name == "Alpha":
                return True
    return False


def is_vertex_colors_enabled(mat):
    """Check if vertex colors multiply node setup is present in the material"""
    if not mat or not mat.use_nodes or not mat.node_tree:
        return False
    for node in mat.node_tree.nodes:
        if node.type == 'MIX' and node.data_type == 'RGBA' and node.blend_type == 'MULTIPLY':
            for link in mat.node_tree.links:
                if link.to_node == node and link.from_node.type == 'VERTEX_COLOR':
                    return True
    return False


def remove_unused_nodes(mat):
    """Recursively remove nodes that have no connected outputs.

    Preserves Output Material and Principled BSDF nodes.
    """
    if not mat or not mat.use_nodes or not mat.node_tree:
        return
    nt = mat.node_tree
    protected_types = {'OUTPUT_MATERIAL', 'BSDF_PRINCIPLED', 'TEX_IMAGE'}
    changed = True
    while changed:
        changed = False
        for node in list(nt.nodes):
            if node.type in protected_types:
                continue
            has_connected_output = False
            for output in node.outputs:
                if output.links:
                    has_connected_output = True
                    break
            if not has_connected_output:
                nt.nodes.remove(node)
                changed = True


def get_default_material_settings():
    """Get the default material settings from the current scene."""
    props = bpy.context.scene.level_design_props
    return {
        'interpolation': props.default_interpolation,
        'texture_as_alpha': props.default_texture_as_alpha,
        'vertex_colors': props.default_vertex_colors,
        'roughness': props.default_roughness,
        'metallic': props.default_metallic,
        'emission_strength': props.default_emission_strength,
        'emission_color': tuple(props.default_emission_color),
        'specular': props.default_specular,
        'name_pattern': props.default_material_name_pattern,
    }


def create_material_with_image(image):
    """Create a new material using the given image texture with scene default settings"""
    mapped_materials = materials_mapped_to_image(image)
    if mapped_materials:
        names = ", ".join(material.name for material in mapped_materials)
        raise MaterialMappingConflictError(
            f"Cannot create a material: {image.name!r} is already mapped to {names}"
        )

    material_name = _material_name_for_image(image)
    debug_log(f"[CreateMaterial] creating {material_name} for image={image.name!r}")
    defaults = get_default_material_settings()
    settings = dict(defaults)
    settings['texture_as_alpha'] = (
        settings['texture_as_alpha'] or image_has_transparency(image)
    )

    mat = bpy.data.materials.new(name=material_name)
    try:
        build_canonical_material_shader(mat, image, settings)
        set_material_primary_image(mat, image)
    except Exception:
        if mat.users == 0:
            bpy.data.materials.remove(mat)
        raise
    return mat


def _socket_value(bsdf, socket_name, fallback):
    if bsdf is None:
        return fallback
    socket = bsdf.inputs.get(socket_name)
    if socket is None:
        return fallback
    value = socket.default_value
    try:
        return tuple(value)
    except TypeError:
        return value


def repair_material_shader(material):
    """Rebuild a mapped material's shader while retaining supported values."""
    image = get_primary_image_from_material(material)
    if image is None:
        raise ValueError("Material has no primary image")

    defaults = get_default_material_settings()
    validation = validate_material_shader(material, image)
    texture = validation.texture_node or get_texture_node_from_material(material)
    bsdf = validation.bsdf_node or get_principled_bsdf_from_material(material)
    settings = {
        'interpolation': (
            texture.interpolation if texture is not None else defaults['interpolation']
        ),
        'texture_as_alpha': (
            is_texture_alpha_connected(material) or image_has_transparency(image)
        ),
        'vertex_colors': is_vertex_colors_enabled(material),
        'roughness': _socket_value(bsdf, "Roughness", defaults['roughness']),
        'metallic': _socket_value(bsdf, "Metallic", defaults['metallic']),
        'emission_strength': _socket_value(
            bsdf, "Emission Strength", defaults['emission_strength']
        ),
        'emission_color': _socket_value(
            bsdf, "Emission Color", defaults['emission_color']
        ),
        'specular': _socket_value(bsdf, "Specular IOR Level", defaults['specular']),
    }
    return build_canonical_material_shader(material, image, settings)
