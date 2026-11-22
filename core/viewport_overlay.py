import gpu
from gpu_extras.batch import batch_for_shader


_screen_stripe_shader = None
_screen_stripe_instance_shader = None
_screen_stripe_last_draw_error = None
_screen_stripe_last_draw_count = 0
_screen_stripe_last_draw_instance_count = 0
SCREEN_STRIPE_INSTANCE_CHUNK_SIZE = 1024


_STRIPE_VERTEX_SOURCE = (
    "void main()"
    "{"
    "  gl_Position = viewProjectionMatrix * vec4(pos, 1.0);"
    "}"
)


_STRIPE_INSTANCE_VERTEX_SOURCE = (
    "void main()"
    "{"
    "  mat4 model = mat4("
    "    texelFetch(model_matrices, ivec2(0, gl_InstanceID), 0),"
    "    texelFetch(model_matrices, ivec2(1, gl_InstanceID), 0),"
    "    texelFetch(model_matrices, ivec2(2, gl_InstanceID), 0),"
    "    texelFetch(model_matrices, ivec2(3, gl_InstanceID), 0)"
    "  );"
    "  gl_Position = viewProjectionMatrix * model * vec4(pos, 1.0);"
    "}"
)


_STRIPE_FRAGMENT_SOURCE = (
    "void main()"
    "{"
    "  float stripe_pos = gl_FragCoord.x + gl_FragCoord.y;"
    "  float band = mod(stripe_pos, stripe_width * 2.0);"
    "  FragColor = band < stripe_width ? light_color : dark_color;"
    "}"
)


class TriangleOverlayCache:
    """Small lazy cache for overlays that draw world-space triangles."""

    def __init__(self, rebuild_fn):
        self._rebuild_fn = rebuild_fn
        self._tris = []
        self._needs_rebuild = True

    def invalidate(self):
        self._needs_rebuild = True

    def clear(self):
        self._tris = []
        self._needs_rebuild = True

    def get_tris(self):
        if self._needs_rebuild:
            self._tris = self._rebuild_fn()
            self._needs_rebuild = False
        return self._tris


def triangulate_face_world(face, matrix_world):
    """Fan-triangulate a BMesh face into world-space vertex tuples."""
    verts = [matrix_world @ vert.co for vert in face.verts]
    tris = []
    for index in range(1, len(verts) - 1):
        tris.append(verts[0][:])
        tris.append(verts[index][:])
        tris.append(verts[index + 1][:])
    return tris


def triangulate_mesh_world(mesh, matrix_world):
    """Fan-triangulate mesh polygons into world-space vertex tuples."""
    tris = []
    vertices = mesh.vertices
    for polygon in mesh.polygons:
        indices = list(polygon.vertices)
        if len(indices) < 3:
            continue
        first = matrix_world @ vertices[indices[0]].co
        for index in range(1, len(indices) - 1):
            second = matrix_world @ vertices[indices[index]].co
            third = matrix_world @ vertices[indices[index + 1]].co
            tris.append(first[:])
            tris.append(second[:])
            tris.append(third[:])
    return tris


def indexed_mesh_triangles_local(mesh):
    """Return local-space vertices and fan-triangulated polygon indices."""
    positions = [vertex.co[:] for vertex in mesh.vertices]
    triangle_indices = []
    for polygon in mesh.polygons:
        indices = list(polygon.vertices)
        if len(indices) < 3:
            continue
        for index in range(1, len(indices) - 1):
            triangle_indices.append((
                indices[0],
                indices[index],
                indices[index + 1],
            ))
    return positions, triangle_indices


def draw_tris(tris, color):
    """Draw world-space triangles as a depth-tested translucent overlay."""
    if not tris:
        return

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.uniform_float("color", color)
        batch = batch_for_shader(shader, 'TRIS', {"pos": tris})
        batch.draw(shader)
    except Exception:
        pass
    finally:
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)


def _ensure_screen_stripe_shader():
    global _screen_stripe_shader

    if _screen_stripe_shader is not None:
        return _screen_stripe_shader

    try:
        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', "viewProjectionMatrix")
        shader_info.push_constant('VEC4', "light_color")
        shader_info.push_constant('VEC4', "dark_color")
        shader_info.push_constant('FLOAT', "stripe_width")
        shader_info.vertex_in(0, 'VEC3', "pos")
        shader_info.fragment_out(0, 'VEC4', "FragColor")
        shader_info.vertex_source(_STRIPE_VERTEX_SOURCE)
        shader_info.fragment_source(_STRIPE_FRAGMENT_SOURCE)

        _screen_stripe_shader = gpu.shader.create_from_info(shader_info)
        del shader_info
    except Exception:
        _screen_stripe_shader = None

    return _screen_stripe_shader


def _ensure_screen_stripe_instance_shader():
    global _screen_stripe_instance_shader

    if _screen_stripe_instance_shader is not None:
        return _screen_stripe_instance_shader

    try:
        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', "viewProjectionMatrix")
        shader_info.push_constant('VEC4', "light_color")
        shader_info.push_constant('VEC4', "dark_color")
        shader_info.push_constant('FLOAT', "stripe_width")
        shader_info.sampler(0, 'FLOAT_2D', "model_matrices")
        shader_info.vertex_in(0, 'VEC3', "pos")
        shader_info.fragment_out(0, 'VEC4', "FragColor")
        shader_info.vertex_source(_STRIPE_INSTANCE_VERTEX_SOURCE)
        shader_info.fragment_source(_STRIPE_FRAGMENT_SOURCE)

        _screen_stripe_instance_shader = gpu.shader.create_from_info(shader_info)
        del shader_info
    except Exception:
        _screen_stripe_instance_shader = None

    return _screen_stripe_instance_shader


def build_screen_striped_batch(tris):
    """Build a reusable GPU batch for screen-striped world-space triangles."""
    if not tris:
        return None

    shader = _ensure_screen_stripe_shader()
    if shader is None:
        return None

    try:
        return batch_for_shader(shader, 'TRIS', {"pos": tris})
    except Exception:
        return None


def build_screen_striped_instance_batch(vertices, triangle_indices):
    """Build one reusable local-space batch for instanced striped drawing."""
    if not vertices or not triangle_indices:
        return None

    shader = _ensure_screen_stripe_instance_shader()
    if shader is None:
        return None

    try:
        return batch_for_shader(
            shader,
            'TRIS',
            {"pos": vertices},
            indices=triangle_indices,
        )
    except Exception:
        return None


def screen_striped_matrix_values(matrix):
    """Flatten one transform into the column vectors consumed by GLSL."""
    values = []
    for row in matrix.transposed():
        values.extend(float(value) for value in row)
    return values


def build_screen_striped_matrix_texture(values, instance_count):
    """Build one RGBA matrix texture for an instance chunk."""
    buffer = gpu.types.Buffer('FLOAT', len(values), values)
    texture = gpu.types.GPUTexture(
        (4, instance_count),
        format='RGBA32F',
        data=buffer,
    )
    texture.filter_mode(False)
    return texture


def draw_screen_striped_batches(batches, light_color, dark_color, stripe_width):
    """Draw reusable GPU batches with screen-space diagonal stripes."""
    if not batches:
        return

    shader = _ensure_screen_stripe_shader()
    if shader is None:
        return

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    try:
        shader.bind()
        shader.uniform_float(
            "viewProjectionMatrix",
            gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
        )
        shader.uniform_float("light_color", light_color)
        shader.uniform_float("dark_color", dark_color)
        shader.uniform_float("stripe_width", stripe_width)
        for batch in batches:
            if batch is not None:
                batch.draw(shader)
    except Exception:
        pass
    finally:
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)


def draw_screen_striped_instances(draw_groups, light_color, dark_color, stripe_width):
    """Draw local-space mesh batches using cached instance matrix buffers."""
    global _screen_stripe_last_draw_error
    global _screen_stripe_last_draw_count
    global _screen_stripe_last_draw_instance_count
    _screen_stripe_last_draw_error = None
    _screen_stripe_last_draw_count = 0
    _screen_stripe_last_draw_instance_count = 0

    if not draw_groups:
        return

    shader = _ensure_screen_stripe_instance_shader()
    if shader is None:
        _screen_stripe_last_draw_error = "Instance shader unavailable"
        return

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    try:
        shader.bind()
        shader.uniform_float(
            "viewProjectionMatrix",
            gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
        )
        shader.uniform_float("light_color", light_color)
        shader.uniform_float("dark_color", dark_color)
        shader.uniform_float("stripe_width", stripe_width)
        for batch, instance_buffers in draw_groups:
            if batch is None:
                continue
            for matrix_texture, instance_count in instance_buffers:
                shader.uniform_sampler("model_matrices", matrix_texture)
                batch.draw_instanced(shader, instance_count=instance_count)
                _screen_stripe_last_draw_count += 1
                _screen_stripe_last_draw_instance_count += instance_count
    except Exception as error:
        _screen_stripe_last_draw_error = repr(error)
    finally:
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)


def screen_stripe_instance_draw_status():
    """Return latest draw calls, instances, and captured GPU error."""
    return (
        _screen_stripe_last_draw_count,
        _screen_stripe_last_draw_instance_count,
        _screen_stripe_last_draw_error,
    )


def draw_screen_striped_tris(tris, light_color, dark_color, stripe_width):
    """Draw world-space triangles with screen-space diagonal stripes."""
    batch = build_screen_striped_batch(tris)
    if batch is None:
        return
    draw_screen_striped_batches(
        (batch,),
        light_color,
        dark_color,
        stripe_width,
    )
