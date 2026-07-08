import re

import bpy

from .logging import debug_log


_BLENDER_SUFFIX = re.compile(r'\.\d{3,}$')
_UNASSIGNED_MATERIAL_NAME = "ANVIL_Unassigned"
_last_material_count = 0


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
    """Return the image plugged into the Base Color of the Principled BSDF, or None.

    Falls back to the first TEX_IMAGE node if no Principled BSDF is found.
    """
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            base_color = node.inputs.get('Base Color')
            if base_color and base_color.links:
                linked_node = base_color.links[0].from_node
                if linked_node.type == 'TEX_IMAGE' and linked_node.image:
                    return linked_node.image
            break
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image:
            return node.image
    return None


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


def find_local_material_named(material_name):
    """Return the local material with this exact name, or None."""
    for mat in bpy.data.materials:
        if mat.name == material_name and mat.library is None:
            return mat
    return None


def _material_name_for_image(image):
    # Blender may suffix the image datablock name after conflicts or undo/redo.
    # The material name should stay canonical, e.g. foo.png.001 -> IMG_foo.png.
    image_name = _BLENDER_SUFFIX.sub('', image.name)
    return f"IMG_{image_name}"


def ensure_material_slot(mesh, mat):
    """Ensure this exact material datablock is in the mesh slots."""
    for index, slot_mat in enumerate(mesh.materials):
        if slot_mat == mat:
            return index

    mesh.materials.append(mat)
    return len(mesh.materials) - 1


def find_material_with_image(image):
    """Return the existing local material named for this image, or None."""
    if image is None:
        return None

    material_name = _material_name_for_image(image)
    mat = find_local_material_named(material_name)
    if mat:
        debug_log(
            f"[FindMaterial] image={image.name!r} -> local material {mat.name!r}"
        )
        return mat

    debug_log(
        f"[FindMaterial] image={image.name!r} -> no local material named {material_name!r}"
    )
    return None


def get_unassigned_material():
    """Return a neutral material used to preserve untextured face slots."""
    mat = bpy.data.materials.get(_UNASSIGNED_MATERIAL_NAME)
    if mat:
        return mat

    mat = bpy.data.materials.new(name=_UNASSIGNED_MATERIAL_NAME)
    mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    return mat


def get_principled_bsdf_from_material(mat):
    """Return the Principled BSDF node from a material, or None"""
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node
    return None


def get_texture_node_from_material(mat):
    """Return the first TEX_IMAGE node from a material, or None"""
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None
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
    }


def create_material_with_image(image):
    """Create a new material using the given image texture with scene default settings"""
    material_name = _material_name_for_image(image)
    debug_log(f"[CreateMaterial] creating {material_name} for image={image.name!r}")
    defaults = get_default_material_settings()

    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    mat.use_backface_culling = True

    nt = mat.node_tree
    nt.nodes.clear()

    output = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexImage")

    tex.image = image
    tex.interpolation = defaults['interpolation']

    bsdf.inputs["Roughness"].default_value = defaults['roughness']
    bsdf.inputs["Metallic"].default_value = defaults['metallic']
    bsdf.inputs["Emission Strength"].default_value = defaults['emission_strength']
    bsdf.inputs["Emission Color"].default_value = defaults['emission_color']
    bsdf.inputs["Specular IOR Level"].default_value = defaults['specular']

    tex.location = (-400, 0)
    bsdf.location = (-200, 0)
    output.location = (0, 0)

    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    if defaults['texture_as_alpha']:
        nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
        mat.blend_method = 'CLIP'

    if defaults['vertex_colors']:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = 'RGBA'
        mix.blend_type = 'MULTIPLY'
        mix.clamp_result = True
        mix.inputs["Factor"].default_value = 1.0
        mix.location = (-200, 200)

        vc = nt.nodes.new("ShaderNodeVertexColor")
        vc.location = (-400, -200)

        # Remove existing tex Color -> BSDF Base Color link
        for link in list(nt.links):
            if (
                link.from_node == tex
                and link.from_socket.name == "Color"
                and link.to_node == bsdf
                and link.to_socket.name == "Base Color"
            ):
                nt.links.remove(link)

        nt.links.new(tex.outputs["Color"], mix.inputs[6])
        nt.links.new(vc.outputs["Color"], mix.inputs[7])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])

    return mat
