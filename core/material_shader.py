"""Canonical Anvil material shader construction and semantic validation."""

from collections import namedtuple


ShaderValidationResult = namedtuple(
    "ShaderValidationResult",
    (
        "is_canonical",
        "variant",
        "issues",
        "has_shader_images",
        "primary_image_in_shader",
        "texture_node",
        "bsdf_node",
        "texture_as_alpha",
        "vertex_colors",
    ),
)


_NODE_DEFINITIONS = {
    "output": ("ShaderNodeOutputMaterial", "OUTPUT_MATERIAL", (0, 0)),
    "bsdf": ("ShaderNodeBsdfPrincipled", "BSDF_PRINCIPLED", (-200, 0)),
    "texture": ("ShaderNodeTexImage", "TEX_IMAGE", (-400, 0)),
    "mix": ("ShaderNodeMix", "MIX", (-200, 200)),
    "vertex_color": ("ShaderNodeVertexColor", "VERTEX_COLOR", (-400, -200)),
}


def _canonical_spec(texture_as_alpha, vertex_colors):
    roles = ["output", "bsdf", "texture"]
    links = [
        ("bsdf", "BSDF", "output", "Surface"),
    ]

    if vertex_colors:
        roles.extend(("mix", "vertex_color"))
        links.extend((
            ("texture", "Color", "mix", 6),
            ("vertex_color", "Color", "mix", 7),
            ("mix", 2, "bsdf", "Base Color"),
        ))
    else:
        links.append(("texture", "Color", "bsdf", "Base Color"))

    if texture_as_alpha:
        links.append(("texture", "Alpha", "bsdf", "Alpha"))

    return {
        "roles": tuple(roles),
        "links": tuple(links),
        "texture_as_alpha": texture_as_alpha,
        "vertex_colors": vertex_colors,
        "variant": (
            "ALPHA_VERTEX_COLORS" if texture_as_alpha and vertex_colors
            else "ALPHA" if texture_as_alpha
            else "VERTEX_COLORS" if vertex_colors
            else "BASIC"
        ),
    }


def _socket(node, output, selector):
    sockets = node.outputs if output else node.inputs
    if isinstance(selector, int):
        if selector < 0 or selector >= len(sockets):
            return None
        return sockets[selector]
    return sockets.get(selector)


def _nodes_by_expected_role(node_tree, spec):
    nodes_by_role = {}
    issues = []
    expected_types = {}
    for role in spec["roles"]:
        _bl_idname, node_type, _location = _NODE_DEFINITIONS[role]
        expected_types.setdefault(node_type, []).append(role)

    if len(node_tree.nodes) != len(spec["roles"]):
        issues.append(
            f"Expected {len(spec['roles'])} nodes, found {len(node_tree.nodes)}"
        )

    for node_type, roles in expected_types.items():
        matching_nodes = [node for node in node_tree.nodes if node.type == node_type]
        if len(matching_nodes) != len(roles):
            issues.append(
                f"Expected {len(roles)} {node_type} node(s), found {len(matching_nodes)}"
            )
            continue
        for role, node in zip(roles, matching_nodes):
            nodes_by_role[role] = node

    expected_node_types = set(expected_types)
    unexpected_types = sorted(
        node.type for node in node_tree.nodes if node.type not in expected_node_types
    )
    if unexpected_types:
        issues.append(f"Unexpected nodes: {', '.join(unexpected_types)}")

    return nodes_by_role, issues


def _link_matches(link, from_node, from_socket, to_node, to_socket):
    return (
        link.from_node == from_node
        and link.from_socket == from_socket
        and link.to_node == to_node
        and link.to_socket == to_socket
    )


def _match_spec(node_tree, spec):
    nodes_by_role, issues = _nodes_by_expected_role(node_tree, spec)
    if issues:
        return None, tuple(issues)

    output = nodes_by_role["output"]
    if not output.is_active_output:
        issues.append("The Material Output node is not active")

    if spec["vertex_colors"]:
        mix = nodes_by_role["mix"]
        if mix.data_type != 'RGBA' or mix.blend_type != 'MULTIPLY':
            issues.append("The vertex-colour Mix node is not an RGBA Multiply")
        if not mix.clamp_result:
            issues.append("The vertex-colour Mix node does not clamp its result")

    expected_links = []
    for from_role, from_selector, to_role, to_selector in spec["links"]:
        from_node = nodes_by_role[from_role]
        to_node = nodes_by_role[to_role]
        from_socket = _socket(from_node, True, from_selector)
        to_socket = _socket(to_node, False, to_selector)
        if from_socket is None or to_socket is None:
            issues.append(
                f"Required socket is missing for {from_role} to {to_role}"
            )
            continue
        expected_links.append((from_node, from_socket, to_node, to_socket))

    if len(node_tree.links) != len(expected_links):
        issues.append(
            f"Expected {len(expected_links)} links, found {len(node_tree.links)}"
        )

    for from_node, from_socket, to_node, to_socket in expected_links:
        if not any(
                _link_matches(link, from_node, from_socket, to_node, to_socket)
                for link in node_tree.links):
            issues.append(
                f"Missing link: {from_node.type}.{from_socket.name} to "
                f"{to_node.type}.{to_socket.name}"
            )

    if issues:
        return None, tuple(issues)
    return nodes_by_role, ()


def shader_images(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return []
    images = []
    for node in material.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image is not None and node.image not in images:
            images.append(node.image)
    return images


def _connected_principled_bsdf(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return None
    outputs = [
        node for node in material.node_tree.nodes
        if node.type == 'OUTPUT_MATERIAL' and node.is_active_output
    ]
    if len(outputs) != 1:
        return None
    surface = outputs[0].inputs.get("Surface")
    if surface is None or len(surface.links) != 1:
        return None
    node = surface.links[0].from_node
    return node if node.type == 'BSDF_PRINCIPLED' else None


def _upstream_images(input_socket):
    images = []
    pending_nodes = [link.from_node for link in input_socket.links]
    visited = set()
    while pending_nodes:
        node = pending_nodes.pop(0)
        pointer = node.as_pointer()
        if pointer in visited:
            continue
        visited.add(pointer)
        if node.type == 'TEX_IMAGE' and node.image is not None:
            if node.image not in images:
                images.append(node.image)
            continue
        for node_input in node.inputs:
            pending_nodes.extend(link.from_node for link in node_input.links)
    return images


def infer_primary_shader_image(material):
    """Return an unambiguous shader image suggestion for mapping repair."""
    bsdf = _connected_principled_bsdf(material)
    if bsdf is not None:
        base_color = bsdf.inputs.get("Base Color")
        if base_color is not None:
            upstream = _upstream_images(base_color)
            if len(upstream) == 1:
                return upstream[0]

    images = shader_images(material)
    return images[0] if len(images) == 1 else None


def validate_material_shader(material, primary_image):
    images = shader_images(material)
    has_images = bool(images)
    primary_in_shader = primary_image is not None and primary_image in images

    if material is None or not material.use_nodes or material.node_tree is None:
        return ShaderValidationResult(
            False,
            "NONE",
            ("Material does not use nodes",),
            has_images,
            primary_in_shader,
            None,
            None,
            False,
            False,
        )

    closest_issues = None
    closest_spec = None
    for texture_as_alpha, vertex_colors in (
            (False, False), (True, False), (False, True), (True, True)):
        spec = _canonical_spec(texture_as_alpha, vertex_colors)
        nodes_by_role, issues = _match_spec(material.node_tree, spec)
        if not issues:
            return ShaderValidationResult(
                True,
                spec["variant"],
                (),
                has_images,
                primary_in_shader,
                nodes_by_role["texture"],
                nodes_by_role["bsdf"],
                texture_as_alpha,
                vertex_colors,
            )
        if closest_issues is None or len(issues) < len(closest_issues):
            closest_issues = issues
            closest_spec = spec

    texture_node = None
    if primary_image is not None:
        texture_node = next(
            (
                node for node in material.node_tree.nodes
                if node.type == 'TEX_IMAGE' and node.image == primary_image
            ),
            None,
        )
    if texture_node is None:
        texture_node = next(
            (node for node in material.node_tree.nodes if node.type == 'TEX_IMAGE'),
            None,
        )

    bsdf_node = _connected_principled_bsdf(material)
    if bsdf_node is None:
        bsdf_node = next(
            (node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED'),
            None,
        )

    return ShaderValidationResult(
        False,
        closest_spec["variant"] if closest_spec is not None else "NONE",
        closest_issues or ("Shader is not canonical",),
        has_images,
        primary_in_shader,
        texture_node,
        bsdf_node,
        False,
        False,
    )


def build_canonical_material_shader(material, image, settings):
    """Replace a material node tree with the canonical Anvil shader."""
    material.use_nodes = True
    material.use_backface_culling = True
    node_tree = material.node_tree
    node_tree.nodes.clear()

    spec = _canonical_spec(
        settings["texture_as_alpha"],
        settings["vertex_colors"],
    )
    nodes_by_role = {}
    for role in spec["roles"]:
        bl_idname, _node_type, location = _NODE_DEFINITIONS[role]
        node = node_tree.nodes.new(bl_idname)
        node.location = location
        nodes_by_role[role] = node

    texture = nodes_by_role["texture"]
    texture.image = image
    texture.interpolation = settings["interpolation"]

    bsdf = nodes_by_role["bsdf"]
    bsdf.inputs["Roughness"].default_value = settings["roughness"]
    bsdf.inputs["Metallic"].default_value = settings["metallic"]
    bsdf.inputs["Emission Strength"].default_value = settings["emission_strength"]
    bsdf.inputs["Emission Color"].default_value = settings["emission_color"]
    bsdf.inputs["Specular IOR Level"].default_value = settings["specular"]

    if spec["vertex_colors"]:
        mix = nodes_by_role["mix"]
        mix.data_type = 'RGBA'
        mix.blend_type = 'MULTIPLY'
        mix.clamp_result = True
        mix.inputs["Factor"].default_value = 1.0

    for from_role, from_selector, to_role, to_selector in spec["links"]:
        from_node = nodes_by_role[from_role]
        to_node = nodes_by_role[to_role]
        node_tree.links.new(
            _socket(from_node, True, from_selector),
            _socket(to_node, False, to_selector),
        )

    material.blend_method = 'CLIP' if spec["texture_as_alpha"] else 'OPAQUE'
    return validate_material_shader(material, image)


def image_has_transparency(image):
    """Return whether an image contains any pixel below maximum alpha."""
    if image is None:
        return False
    channels = int(image.channels)
    if channels == 2:
        alpha_offset = 1
    elif channels == 4:
        alpha_offset = 3
    else:
        return False

    pixels = image.pixels
    chunk_length = 262144 * channels
    try:
        for start in range(0, len(pixels), chunk_length):
            chunk = pixels[start:min(start + chunk_length, len(pixels))]
            for index in range(alpha_offset, len(chunk), channels):
                if chunk[index] < 1.0:
                    return True
    except RuntimeError:
        return False
    return False
