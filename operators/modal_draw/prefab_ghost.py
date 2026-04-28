"""GPU albedo ghost drawing for prefab placement."""

import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from ...core.materials import is_texture_alpha_connected


_GHOST_ALPHA = 0.42
_GHOST_GREY_MIX = 0.35
_DEFAULT_COLOR = (0.72, 0.72, 0.72, 1.0)

_texture_shader = None
_color_shader = None
_texture_shader_error_reported = False
_color_shader_error_reported = False


_TEXTURE_VERTEX_SOURCE = (
    "void main()"
    "{"
    "  uv_interp = uv;"
    "  gl_Position = viewProjectionMatrix * ghostMatrix * vec4(pos, 1.0);"
    "  gl_Position.z -= 0.000001 * gl_Position.w;"
    "}"
)

_TEXTURE_FRAGMENT_SOURCE = (
    "void main()"
    "{"
    "  vec4 tex_color = texture(image, uv_interp);"
    "  float grey = dot(tex_color.rgb, vec3(0.299, 0.587, 0.114));"
    "  vec3 ghost_color = mix(tex_color.rgb, vec3(grey), greyMix);"
    "  float tex_alpha = mix(1.0, tex_color.a, textureAlpha);"
    "  FragColor = vec4(ghost_color, tex_alpha * baseAlpha * alpha);"
    "}"
)

_COLOR_VERTEX_SOURCE = (
    "void main()"
    "{"
    "  gl_Position = viewProjectionMatrix * ghostMatrix * vec4(pos, 1.0);"
    "  gl_Position.z -= 0.000001 * gl_Position.w;"
    "}"
)

_COLOR_FRAGMENT_SOURCE = (
    "void main()"
    "{"
    "  float grey = dot(color.rgb, vec3(0.299, 0.587, 0.114));"
    "  vec3 ghost_color = mix(color.rgb, vec3(grey), greyMix);"
    "  FragColor = vec4(ghost_color, color.a * alpha);"
    "}"
)


class PrefabGhostDrawGroup:
    """Local-space triangles for one material slot in the prefab ghost."""

    def __init__(
            self,
            positions,
            uvs,
            image,
            use_linear_filter,
            use_texture_alpha,
            color):
        self.positions = positions
        self.uvs = uvs
        self.image = image
        self.use_linear_filter = use_linear_filter
        self.use_texture_alpha = use_texture_alpha
        self.color = color
        self._texture_batch = None
        self._color_batch = None


class PrefabAlbedoGhost:
    """GPU-ready albedo preview data for a prefab object."""

    def __init__(self, textured_groups, color_groups):
        self.textured_groups = textured_groups
        self.color_groups = color_groups

    def has_geometry(self):
        return bool(self.textured_groups or self.color_groups)


def _ensure_texture_shader():
    global _texture_shader, _texture_shader_error_reported
    if _texture_shader is not None:
        return _texture_shader

    try:
        vert_info = gpu.types.GPUStageInterfaceInfo("prefab_ghost_texture_iface")
        vert_info.smooth('VEC2', "uv_interp")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', "viewProjectionMatrix")
        shader_info.push_constant('MAT4', "ghostMatrix")
        shader_info.push_constant('FLOAT', "alpha")
        shader_info.push_constant('FLOAT', "greyMix")
        shader_info.push_constant('FLOAT', "baseAlpha")
        shader_info.push_constant('FLOAT', "textureAlpha")
        shader_info.sampler(0, 'FLOAT_2D', "image")
        shader_info.vertex_in(0, 'VEC3', "pos")
        shader_info.vertex_in(1, 'VEC2', "uv")
        shader_info.vertex_out(vert_info)
        shader_info.fragment_out(0, 'VEC4', "FragColor")
        shader_info.vertex_source(_TEXTURE_VERTEX_SOURCE)
        shader_info.fragment_source(_TEXTURE_FRAGMENT_SOURCE)

        _texture_shader = gpu.shader.create_from_info(shader_info)
        del vert_info
        del shader_info
    except Exception as exc:
        _texture_shader = None
        if not _texture_shader_error_reported:
            print(f"Level Design Tools: Prefab albedo ghost texture shader failed: {exc}")
            _texture_shader_error_reported = True

    return _texture_shader


def _ensure_color_shader():
    global _color_shader, _color_shader_error_reported
    if _color_shader is not None:
        return _color_shader

    try:
        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', "viewProjectionMatrix")
        shader_info.push_constant('MAT4', "ghostMatrix")
        shader_info.push_constant('FLOAT', "alpha")
        shader_info.push_constant('FLOAT', "greyMix")
        shader_info.push_constant('VEC4', "color")
        shader_info.vertex_in(0, 'VEC3', "pos")
        shader_info.fragment_out(0, 'VEC4', "FragColor")
        shader_info.vertex_source(_COLOR_VERTEX_SOURCE)
        shader_info.fragment_source(_COLOR_FRAGMENT_SOURCE)

        _color_shader = gpu.shader.create_from_info(shader_info)
        del shader_info
    except Exception as exc:
        _color_shader = None
        if not _color_shader_error_reported:
            print(f"Level Design Tools: Prefab albedo ghost color shader failed: {exc}")
            _color_shader_error_reported = True

    return _color_shader


def _get_view_projection_matrix():
    return gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()


def _base_color_texture_node(material):
    if material is None or not material.use_nodes or not material.node_tree:
        return None

    for node in material.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            base_color = node.inputs.get('Base Color')
            if base_color is not None and base_color.links:
                linked_node = base_color.links[0].from_node
                if linked_node.type == 'TEX_IMAGE' and linked_node.image is not None:
                    return linked_node
            break

    for node in material.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image is not None:
            return node

    return None


def _material_color(material):
    if material is None:
        return _DEFAULT_COLOR

    if material.use_nodes and material.node_tree:
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                base_color = node.inputs.get('Base Color')
                if base_color is not None and hasattr(base_color, "default_value"):
                    return tuple(base_color.default_value)
                break

    return tuple(material.diffuse_color)


def _material_uses_texture_alpha(material):
    if material is None:
        return False
    return material.blend_method != 'OPAQUE' or is_texture_alpha_connected(material)


def _material_for_index(mesh, material_index):
    if material_index < 0 or material_index >= len(mesh.materials):
        return None
    return mesh.materials[material_index]


def _render_uv_layer(mesh):
    if not mesh.uv_layers:
        return None

    for uv_map in mesh.uv_layers:
        if uv_map.active_render:
            return uv_map

    return mesh.uv_layers[0]


def _append_bbox_triangles(positions, corners, indices):
    for index in indices:
        positions.append(corners[index][:])


def _bbox_positions(bound_box):
    corners = [Vector(corner) for corner in bound_box]
    if len(corners) != 8:
        return []

    positions = []
    _append_bbox_triangles(positions, corners, (0, 1, 2, 0, 2, 3))
    _append_bbox_triangles(positions, corners, (4, 7, 6, 4, 6, 5))
    _append_bbox_triangles(positions, corners, (0, 4, 5, 0, 5, 1))
    _append_bbox_triangles(positions, corners, (1, 5, 6, 1, 6, 2))
    _append_bbox_triangles(positions, corners, (2, 6, 7, 2, 7, 3))
    _append_bbox_triangles(positions, corners, (3, 7, 4, 3, 4, 0))
    return positions


def _group_for_material(groups, material_index, material, uv_layer):
    group = groups.get(material_index)
    if group is not None:
        return group

    texture_node = _base_color_texture_node(material)
    image = texture_node.image if texture_node is not None else None
    use_texture = image is not None and uv_layer is not None
    use_linear_filter = True
    if texture_node is not None:
        use_linear_filter = texture_node.interpolation != 'Closest'

    group = PrefabGhostDrawGroup(
        [],
        [],
        image if use_texture else None,
        use_linear_filter,
        _material_uses_texture_alpha(material),
        _material_color(material),
    )
    groups[material_index] = group
    return group


def _build_mesh_ghost(mesh):
    mesh.calc_loop_triangles()
    if len(mesh.loop_triangles) == 0:
        return PrefabAlbedoGhost([], [])

    uv_layer = _render_uv_layer(mesh)
    groups_by_material = {}

    for loop_triangle in mesh.loop_triangles:
        polygon = mesh.polygons[loop_triangle.polygon_index]
        material_index = polygon.material_index
        material = _material_for_index(mesh, material_index)
        group = _group_for_material(groups_by_material, material_index, material, uv_layer)

        for vertex_index, loop_index in zip(loop_triangle.vertices, loop_triangle.loops):
            group.positions.append(mesh.vertices[vertex_index].co[:])
            if group.image is not None:
                group.uvs.append(uv_layer.data[loop_index].uv[:])

    textured_groups = []
    color_groups = []
    for group in groups_by_material.values():
        if group.image is not None and len(group.uvs) == len(group.positions):
            textured_groups.append(group)
        elif group.positions:
            color_groups.append(group)

    return PrefabAlbedoGhost(textured_groups, color_groups)


def _build_bbox_ghost(linked_asset):
    positions = _bbox_positions(linked_asset.bound_box)
    if not positions:
        return PrefabAlbedoGhost([], [])
    return PrefabAlbedoGhost(
        [],
        [PrefabGhostDrawGroup(positions, [], None, True, False, _DEFAULT_COLOR)],
    )


def build_prefab_albedo_ghost(linked_asset):
    """Build local-space albedo ghost draw data from a linked prefab object."""
    if linked_asset.type == 'MESH' and linked_asset.data is not None:
        ghost = _build_mesh_ghost(linked_asset.data)
        if ghost.has_geometry():
            return ghost

    return _build_bbox_ghost(linked_asset)


def _draw_color_group(shader, group, view_projection_matrix, ghost_matrix):
    if group._color_batch is None:
        group._color_batch = batch_for_shader(
            shader,
            'TRIS',
            {"pos": group.positions},
        )

    shader.bind()
    shader.uniform_float("viewProjectionMatrix", view_projection_matrix)
    shader.uniform_float("ghostMatrix", ghost_matrix)
    shader.uniform_float("alpha", _GHOST_ALPHA)
    shader.uniform_float("greyMix", _GHOST_GREY_MIX)
    shader.uniform_float("color", group.color)
    group._color_batch.draw(shader)


def _draw_textured_group(texture_shader, group, view_projection_matrix, ghost_matrix):
    if group._texture_batch is None:
        group._texture_batch = batch_for_shader(
            texture_shader,
            'TRIS',
            {"pos": group.positions, "uv": group.uvs},
        )

    try:
        gpu_texture = gpu.texture.from_image(group.image)
        gpu_texture.filter_mode(group.use_linear_filter)
    except Exception:
        color_shader = _ensure_color_shader()
        if color_shader is not None:
            _draw_color_group(
                color_shader,
                group,
                view_projection_matrix,
                ghost_matrix,
            )
        return

    texture_shader.bind()
    texture_shader.uniform_float("viewProjectionMatrix", view_projection_matrix)
    texture_shader.uniform_float("ghostMatrix", ghost_matrix)
    texture_shader.uniform_sampler("image", gpu_texture)
    texture_shader.uniform_float("alpha", _GHOST_ALPHA)
    texture_shader.uniform_float("greyMix", _GHOST_GREY_MIX)
    texture_shader.uniform_float("baseAlpha", group.color[3])
    texture_shader.uniform_float(
        "textureAlpha",
        1.0 if group.use_texture_alpha else 0.0,
    )
    group._texture_batch.draw(texture_shader)


def draw_prefab_albedo_ghost(ghost, ghost_matrix):
    """Draw the prefab albedo ghost with no wire outline."""
    if ghost is None or ghost_matrix is None:
        return
    if not ghost.has_geometry():
        return

    texture_shader = _ensure_texture_shader() if ghost.textured_groups else None
    color_shader = _ensure_color_shader() if ghost.color_groups else None
    if texture_shader is None and color_shader is None:
        return

    view_projection_matrix = _get_view_projection_matrix()

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    try:
        if texture_shader is not None:
            for group in ghost.textured_groups:
                _draw_textured_group(
                    texture_shader,
                    group,
                    view_projection_matrix,
                    ghost_matrix,
                )

        if color_shader is not None:
            for group in ghost.color_groups:
                _draw_color_group(
                    color_shader,
                    group,
                    view_projection_matrix,
                    ghost_matrix,
                )
    finally:
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
