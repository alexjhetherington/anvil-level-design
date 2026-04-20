"""UV Transform Modal - GPU drawing for ghost texture preview and handles.

Draws:
1. A semi-transparent textured quad showing the full texture tile in 3D space
2. Interactive handles at corners (scale), center (move), and top (rotation)
3. A face outline highlight
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from .interaction import compute_handle_positions


# Visual constants
GHOST_ALPHA = 0.4
FACE_OUTLINE_COLOR = (1.0, 1.0, 1.0, 0.6)
HANDLE_COLOR_CORNER = (1.0, 0.8, 0.0, 0.9)
HANDLE_COLOR_MOVE = (0.3, 0.7, 1.0, 0.9)
HANDLE_COLOR_ROTATION = (0.3, 1.0, 0.5, 0.9)
# U axis (horizontal/left-right) — red. V axis (vertical/up-down) — purple.
HANDLE_COLOR_AXIS_U = (1.0, 0.35, 0.3, 0.9)
HANDLE_COLOR_AXIS_V = (0.75, 0.4, 1.0, 0.9)
HANDLE_COLOR_HOVER = (1.0, 1.0, 1.0, 1.0)
QUAD_OUTLINE_COLOR = (1.0, 1.0, 1.0, 0.35)

# Shader source for textured quad (image sampling in 3D)
_VERT_SRC = (
    "void main()"
    "{"
    "  uv_interp = uv;"
    "  gl_Position = viewProjectionMatrix * vec4(pos, 1.0);"
    "}"
)

_FRAG_SRC = (
    "void main()"
    "{"
    "  vec4 tex_color = texture(image, uv_interp);"
    "  FragColor = vec4(tex_color.rgb, tex_color.a * alpha);"
    "}"
)

_image_shader = None


def _ensure_image_shader():
    """Create the textured quad shader on first use."""
    global _image_shader
    if _image_shader is not None:
        return _image_shader

    vert_info = gpu.types.GPUStageInterfaceInfo("uv_transform_iface")
    vert_info.smooth('VEC2', "uv_interp")

    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant('MAT4', "viewProjectionMatrix")
    shader_info.push_constant('FLOAT', "alpha")
    shader_info.sampler(0, 'FLOAT_2D', "image")
    shader_info.vertex_in(0, 'VEC3', "pos")
    shader_info.vertex_in(1, 'VEC2', "uv")
    shader_info.vertex_out(vert_info)
    shader_info.fragment_out(0, 'VEC4', "FragColor")
    shader_info.vertex_source(_VERT_SRC)
    shader_info.fragment_source(_FRAG_SRC)

    _image_shader = gpu.shader.create_from_info(shader_info)
    return _image_shader


def _get_view_projection_matrix():
    """Get the current view-projection matrix from the GPU stack."""
    return gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()


def _get_viewport_size():
    """Get the current viewport size from bpy.context.region."""
    region = bpy.context.region
    return (region.width, region.height)


def draw_ghost_texture(quad_corners, blender_image):
    """Draw a semi-transparent textured quad in 3D space.

    Args:
        quad_corners: list of 4 Vector3 [BL, BR, TR, TL]
        blender_image: bpy.types.Image to sample from
    """
    if blender_image is None:
        return

    shader = _ensure_image_shader()

    # Get or create GPU texture from the Blender image
    gpu_texture = gpu.texture.from_image(blender_image)

    # Build triangulated quad (two triangles: BL-BR-TR, BL-TR-TL)
    positions = [
        quad_corners[0][:], quad_corners[1][:], quad_corners[2][:],
        quad_corners[0][:], quad_corners[2][:], quad_corners[3][:],
    ]
    uvs = [
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
        (0.0, 0.0), (1.0, 1.0), (0.0, 1.0),
    ]

    batch = batch_for_shader(shader, 'TRIS', {"pos": positions, "uv": uvs})

    shader.bind()
    shader.uniform_float("viewProjectionMatrix", _get_view_projection_matrix())
    shader.uniform_sampler("image", gpu_texture)
    shader.uniform_float("alpha", GHOST_ALPHA)
    batch.draw(shader)


def draw_quad_outline(quad_corners):
    """Draw the outline of the texture quad."""
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')

    # Quad edges: BL->BR->TR->TL->BL
    positions = [c[:] for c in quad_corners] + [quad_corners[0][:]]

    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": positions})

    shader.bind()
    shader.uniform_float("color", QUAD_OUTLINE_COLOR)
    shader.uniform_float("lineWidth", 1.5)
    shader.uniform_float("viewportSize", _get_viewport_size())
    batch.draw(shader)


def draw_face_outline(face_corners_3d):
    """Draw a highlight outline around the selected face."""
    if not face_corners_3d:
        return

    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')

    positions = [c[:] for c in face_corners_3d] + [face_corners_3d[0][:]]
    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": positions})

    shader.bind()
    shader.uniform_float("color", FACE_OUTLINE_COLOR)
    shader.uniform_float("lineWidth", 2.0)
    shader.uniform_float("viewportSize", _get_viewport_size())
    batch.draw(shader)


def draw_handles_3d(quad_corners, hover_type, hover_index):
    """Draw handle indicators in 3D space (POST_VIEW).

    Draws small diamond shapes at each handle position using 3D coordinates.
    """
    handle_info = compute_handle_positions(quad_corners)

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    # Get face-plane directions from the quad edges
    bl, br, tr, tl = quad_corners
    right_dir = (br - bl)
    up_dir = (tl - bl)
    r_len = right_dir.length
    u_len = up_dir.length
    if r_len > 0.0001:
        right_dir = right_dir / r_len
    if u_len > 0.0001:
        up_dir = up_dir / u_len
    avg_size = (r_len + u_len) * 0.5

    def _draw_diamond_3d(center, size_factor, color):
        """Draw a diamond shape in the face plane using 3D coords."""
        s = avg_size * size_factor

        top = center + up_dir * s
        bottom = center - up_dir * s
        left = center - right_dir * s
        right = center + right_dir * s

        positions = [
            center[:], top[:], right[:],
            center[:], right[:], bottom[:],
            center[:], bottom[:], left[:],
            center[:], left[:], top[:],
        ]
        batch = batch_for_shader(shader, 'TRIS', {"pos": positions})
        shader.uniform_float("color", color)
        batch.draw(shader)

    def _draw_bar_3d(center, along_dir, across_dir, length_factor,
                     width_factor, color):
        """Draw a thin filled rectangle aligned to the face plane."""
        half_len = avg_size * length_factor
        half_wid = avg_size * width_factor
        a = along_dir * half_len
        b = across_dir * half_wid
        p0 = center - a - b
        p1 = center + a - b
        p2 = center + a + b
        p3 = center - a + b
        positions = [p0[:], p1[:], p2[:], p0[:], p2[:], p3[:]]
        batch = batch_for_shader(shader, 'TRIS', {"pos": positions})
        shader.uniform_float("color", color)
        batch.draw(shader)

    shader.bind()

    # Corner handles (scale both axes)
    for i, pos in enumerate(handle_info['corners']):
        color = HANDLE_COLOR_HOVER if (hover_type == 'corner' and hover_index == i) else HANDLE_COLOR_CORNER
        _draw_diamond_3d(pos, 0.03, color)

    # Edge handles (axis-locked resize). Even indices are horizontal edges
    # (bottom/top) which scale V; odd indices are vertical edges which
    # scale U. Draw as bars aligned with the edge they sit on.
    for i, pos in enumerate(handle_info['edge_midpoints']):
        if i % 2 == 0:  # horizontal edge → V axis resize
            base_color = HANDLE_COLOR_AXIS_V
            along = right_dir
            across = up_dir
        else:           # vertical edge → U axis resize
            base_color = HANDLE_COLOR_AXIS_U
            along = up_dir
            across = right_dir
        color = HANDLE_COLOR_HOVER if (hover_type == 'edge' and hover_index == i) else base_color
        _draw_bar_3d(pos, along, across, 0.04, 0.012, color)

    # Axis-constrained move handles (bars aligned with their active axis)
    color = HANDLE_COLOR_HOVER if hover_type == 'move_v' else HANDLE_COLOR_AXIS_V
    _draw_bar_3d(handle_info['move_axis_v'], up_dir, right_dir, 0.045, 0.012, color)

    color = HANDLE_COLOR_HOVER if hover_type == 'move_h' else HANDLE_COLOR_AXIS_U
    _draw_bar_3d(handle_info['move_axis_h'], right_dir, up_dir, 0.045, 0.012, color)

    # Free-move center handle (unconstrained)
    color = HANDLE_COLOR_HOVER if hover_type == 'move_free' else HANDLE_COLOR_MOVE
    _draw_diamond_3d(handle_info['center'], 0.035, color)

    # Rotation handle
    color = HANDLE_COLOR_HOVER if hover_type == 'rotation' else HANDLE_COLOR_ROTATION
    _draw_diamond_3d(handle_info['rotation'], 0.025, color)

    # Line from top midpoint to rotation handle
    top_mid = handle_info['edge_midpoints'][2]
    rot_pos = handle_info['rotation']
    line_shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    line_batch = batch_for_shader(line_shader, 'LINE_STRIP', {
        "pos": [top_mid[:], rot_pos[:]]
    })
    line_shader.bind()
    line_shader.uniform_float("color", HANDLE_COLOR_ROTATION[:3] + (0.5,))
    line_shader.uniform_float("lineWidth", 1.0)
    line_shader.uniform_float("viewportSize", _get_viewport_size())
    line_batch.draw(line_shader)
