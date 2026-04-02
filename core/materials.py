import re

import bpy

from .logging import debug_log


_BLENDER_SUFFIX = re.compile(r'\.\d{3,}$')


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


def find_material_with_image(image):
    """Return existing material that uses this image, or None"""
    expected_name = f"IMG_{image.name}"
    mat = bpy.data.materials.get(expected_name)
    if mat:
        debug_log(f"[FindMaterial] image={image.name!r} -> lookup={expected_name!r} -> FOUND")
        return mat
    # HACK: Blender appends .001/.002/etc. to image datablock names when there
    # are naming conflicts. This happens when: (1) the same filename exists in
    # different directories, (2) file append/link brings in a same-named image,
    # or (3) undo/redo recreates datablocks. The material was created under the
    # original unsuffixed name (IMG_foo.png), but the image it references may
    # now be named foo.png.001. Strip the suffix to find the existing material.
    base_name = _BLENDER_SUFFIX.sub('', image.name)
    if base_name != image.name:
        fallback_name = f"IMG_{base_name}"
        mat = bpy.data.materials.get(fallback_name)
        if mat:
            debug_log(f"[FindMaterial] image={image.name!r} -> fallback={fallback_name!r} -> FOUND")
            return mat
    debug_log(f"[FindMaterial] image={image.name!r} -> NOT FOUND")
    return None


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
    debug_log(f"[CreateMaterial] creating IMG_{image.name} for image={image.name!r}")
    defaults = get_default_material_settings()

    mat = bpy.data.materials.new(name=f"IMG_{image.name}")
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


def get_selected_image_path(context):
    """Return absolute filepath of selected image in File Browser, or None"""
    if not context.window or not context.window.screen:
        return None

    for area in context.window.screen.areas:
        if area.type == 'FILE_BROWSER':
            space = area.spaces.active
            params = space.params
            if not params or not params.filename:
                continue

            # directory is bytes, filename is str
            directory = params.directory.decode('utf-8')
            filepath = bpy.path.abspath(directory + params.filename)
            return filepath
    return None
