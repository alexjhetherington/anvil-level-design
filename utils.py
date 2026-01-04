import bpy
import bmesh
import os
import math
from mathutils import Vector, Matrix


def normalize_offset(offset):
    """Normalize a UV offset value to the 0-1 range.

    UV offsets can be any value, but visually only the fractional part matters.
    This function wraps the offset to stay within 0-1 range.

    Args:
        offset: Float offset value (can be any positive or negative value)

    Returns:
        Float offset normalized to 0-1 range
    """
    return offset % 1.0


def get_selected_faces_or_report(operator, bm):
    """Get selected faces from bmesh, or report warning and return None.

    Args:
        operator: The operator instance (for self.report)
        bm: BMesh instance

    Returns:
        List of selected faces, or None if no faces selected (warning reported)
    """
    selected_faces = [f for f in bm.faces if f.select]
    if not selected_faces:
        operator.report({'WARNING'}, "No faces selected")
        return None
    return selected_faces


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


def get_face_local_axes(face):
    """Compute local X and Y axes for a face.

    X-axis: Along the first edge (vertex 0 -> vertex 1), normalized
    Y-axis: Perpendicular to X, in the plane of the first 3 vertices

    Returns (local_x, local_y) as Vector3, or None if face has < 2 vertices.
    """
    loops = list(face.loops)
    if len(loops) < 2:
        return None

    # Get first edge direction as local X
    p0 = loops[0].vert.co
    p1 = loops[1].vert.co
    local_x = (p1 - p0).normalized()

    if len(loops) < 3:
        # For 2 vertices, create arbitrary perpendicular
        if abs(local_x.z) < 0.9:
            local_y = local_x.cross(Vector((0, 0, 1))).normalized()
        else:
            local_y = local_x.cross(Vector((1, 0, 0))).normalized()
    else:
        # Get face normal from first 3 vertices
        p2 = loops[2].vert.co
        edge1 = p1 - p0
        edge2 = p2 - p0
        face_normal = edge1.cross(edge2).normalized()

        # Local Y is perpendicular to X, in the face plane
        local_y = face_normal.cross(local_x).normalized()

    return (local_x, local_y)


def get_uv_local_axes(uvs):
    """Compute local X and Y axes for a UV shape.

    X-axis: Along the first UV edge (uv[0] -> uv[1]), normalized
    Y-axis: Perpendicular to X in 2D space

    Args:
        uvs: List of Vector2 or tuples representing UV coordinates

    Returns (local_x, local_y) as 2D vectors (tuples), or None if < 2 UVs.
    """
    if len(uvs) < 2:
        return None

    # Get first UV edge as local X
    uv0 = uvs[0] if isinstance(uvs[0], (tuple, list)) else (uvs[0].x, uvs[0].y)
    uv1 = uvs[1] if isinstance(uvs[1], (tuple, list)) else (uvs[1].x, uvs[1].y)

    dx = uv1[0] - uv0[0]
    dy = uv1[1] - uv0[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < 0.0001:
        return None

    local_x = (dx / length, dy / length)
    # Perpendicular in 2D: rotate 90 degrees counter-clockwise
    local_y = (-local_x[1], local_x[0])

    return (local_x, local_y)


def get_bbox_in_local_axes(points, local_x, local_y):
    """Compute bounding box of points projected onto local axes.

    Args:
        points: List of Vector3 (for 3D) or Vector2/tuples (for 2D)
        local_x: Local X axis (Vector3 or 2-tuple)
        local_y: Local Y axis (Vector3 or 2-tuple)

    Returns dict with 'min_x', 'max_x', 'min_y', 'max_y', 'width', 'height'.
    """
    if not points:
        return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0, 'width': 0, 'height': 0}

    # Determine if 2D or 3D
    is_3d = hasattr(points[0], 'z') or (hasattr(points[0], '__len__') and len(points[0]) >= 3)

    projected_x = []
    projected_y = []

    for p in points:
        if is_3d:
            # 3D projection using dot product
            if isinstance(local_x, tuple):
                px = p[0] * local_x[0] + p[1] * local_x[1] + (p[2] * local_x[2] if len(local_x) > 2 else 0)
                py = p[0] * local_y[0] + p[1] * local_y[1] + (p[2] * local_y[2] if len(local_y) > 2 else 0)
            else:
                pv = Vector(p) if not isinstance(p, Vector) else p
                px = pv.dot(local_x)
                py = pv.dot(local_y)
        else:
            # 2D projection
            if isinstance(p, (tuple, list)):
                px_val, py_val = p[0], p[1]
            else:
                px_val, py_val = p.x, p.y
            px = px_val * local_x[0] + py_val * local_x[1]
            py = px_val * local_y[0] + py_val * local_y[1]

        projected_x.append(px)
        projected_y.append(py)

    min_x = min(projected_x)
    max_x = max(projected_x)
    min_y = min(projected_y)
    max_y = max(projected_y)

    return {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'width': max_x - min_x,
        'height': max_y - min_y
    }


def derive_transform_from_uvs(face, uv_layer, ppm, me):
    """Derive scale, rotation, and offset from a face's current UVs.

    Scale is extracted in texture space, meaning scale_u represents scaling
    along the texture's horizontal axis and scale_v along the vertical axis,
    regardless of rotation.

    The forward transform is: rotate projection axes → project → scale → offset
    This function reverses that to extract the parameters.

    Args:
        face: BMesh face to derive transform from
        uv_layer: BMesh UV layer
        ppm: Pixels per meter setting
        me: Mesh data (for accessing materials)

    Returns dict with 'scale_u', 'scale_v', 'rotation', 'offset_x', 'offset_y'
    or None if derivation fails.
    """
    loops = list(face.loops)
    uvs = [loop[uv_layer].uv.copy() for loop in loops]
    if len(uvs) < 3:
        return None

    # Get texture dimensions from face's material
    mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
    tu, tv = get_texture_dimensions_from_material(mat, ppm)

    # Get face local axes
    face_axes = get_face_local_axes(face)
    if not face_axes:
        return None
    face_local_x, face_local_y = face_axes

    # Project face vertices to face-local 2D space
    first_vert = loops[0].vert.co
    face_local = []
    for loop in loops:
        delta = loop.vert.co - first_vert
        x = delta.dot(face_local_x)
        y = delta.dot(face_local_y)
        face_local.append((x, y))

    # Offset is UV[0] (first vertex is at face-local origin)
    offset_x = uvs[0].x
    offset_y = uvs[0].y

    # Forward transform: u = (x*cos - y*sin) / (su*tu) + offset_x
    #                    v = (x*sin + y*cos) / (sv*tv) + offset_y
    #
    # For vertex 1 at face_local (L, 0):
    #   du = L * cos / (su * tu)
    #   dv = L * sin / (sv * tv)

    L = face_local[1][0]
    if abs(L) < 0.0001:
        return None

    du = uvs[1].x - offset_x
    dv = uvs[1].y - offset_y

    # Find a vertex with non-zero y for additional equations
    y_i = None
    x_i = None
    u_i = None
    v_i = None
    for i in range(2, len(face_local)):
        if abs(face_local[i][1]) > 0.0001:
            x_i, y_i = face_local[i]
            u_i = uvs[i].x - offset_x
            v_i = uvs[i].y - offset_y
            break

    if y_i is None:
        # Degenerate face (all vertices collinear)
        uv_edge_len = math.sqrt(du * du + dv * dv)
        if uv_edge_len > 0.0001:
            rotation = math.degrees(math.atan2(dv, du))
            scale_u = L / (uv_edge_len * tu)
            scale_v = scale_u
        else:
            rotation = 0.0
            scale_u = 1.0
            scale_v = 1.0
        return {
            'scale_u': scale_u,
            'scale_v': scale_v,
            'rotation': rotation,
            'offset_x': normalize_offset(offset_x),
            'offset_y': normalize_offset(offset_y)
        }

    epsilon = 0.0001

    if abs(du) < epsilon and abs(dv) < epsilon:
        # Degenerate: first edge maps to zero UV length
        return None

    elif abs(dv) < epsilon:
        # sin ≈ 0, rotation ≈ 0° or 180°
        # cos = sign(du), since du = L * cos / (su * tu) and L, su, tu > 0
        if du > 0:
            cos_rot = 1.0
            rotation = 0.0
        else:
            cos_rot = -1.0
            rotation = 180.0

        # su * tu = L * |cos| / |du| = L / |du|
        su_tu = L / abs(du)
        scale_u = su_tu / tu

        # From vertex i: v_i = (x_i * sin + y_i * cos) / (sv * tv) = y_i * cos / (sv * tv)
        # sv * tv = y_i * cos / v_i
        if abs(v_i) > epsilon:
            sv_tv = y_i * cos_rot / v_i
            scale_v = abs(sv_tv) / tv
        else:
            scale_v = scale_u  # Fallback to uniform

    elif abs(du) < epsilon:
        # cos ≈ 0, rotation ≈ 90° or -90°
        # sin = sign(dv)
        if dv > 0:
            sin_rot = 1.0
            rotation = 90.0
        else:
            sin_rot = -1.0
            rotation = -90.0

        # sv * tv = L * |sin| / |dv| = L / |dv|
        sv_tv = L / abs(dv)
        scale_v = sv_tv / tv

        # From vertex i: u_i = (x_i * cos - y_i * sin) / (su * tu) = -y_i * sin / (su * tu)
        # su * tu = -y_i * sin / u_i
        if abs(u_i) > epsilon:
            su_tu = -y_i * sin_rot / u_i
            scale_u = abs(su_tu) / tu
        else:
            scale_u = scale_v  # Fallback to uniform

    else:
        # General case: solve using both vertices
        # r = -y_i * dv / (u_i * L - x_i * du)
        denominator = u_i * L - x_i * du
        if abs(denominator) < epsilon:
            # Fallback: use simpler calculation
            uv_edge_len = math.sqrt(du * du + dv * dv)
            rotation = math.degrees(math.atan2(dv, du))
            scale_u = L / (uv_edge_len * tu)
            scale_v = scale_u
        else:
            r = -y_i * dv / denominator

            # B = L / sqrt(du² * r² + dv²)
            # A = r * B
            denom_sq = du * du * r * r + dv * dv
            if denom_sq < epsilon:
                return None

            B = L / math.sqrt(denom_sq)
            A = r * B

            # su = A / tu, sv = B / tv
            scale_u = abs(A) / tu
            scale_v = abs(B) / tv

            # rotation = atan2(dv, du * r)
            rotation = math.degrees(math.atan2(dv, du * r))

    return {
        'scale_u': scale_u,
        'scale_v': scale_v,
        'rotation': rotation,
        'offset_x': normalize_offset(offset_x),
        'offset_y': normalize_offset(offset_y)
    }


def align_2d_shape_to_square(shape, shape_edge_index, square_edge_index):
    """
    Align a 2D shape so that the specified shape edge aligns with a unit square edge.

    The shape is rotated so the edge becomes parallel to the square edge, then
    translated so the edge midpoint lies on that square edge.

    Args:
        shape: List of (x, y) tuples forming a polygon
        shape_edge_index: Index of the shape edge to align (edge from point[i] to point[i+1])
        square_edge_index: Which edge of unit square to align to:
            0 = bottom (y=0, direction +X)
            1 = right  (x=1, direction +Y)
            2 = top    (y=1, direction -X)
            3 = left   (x=0, direction -Y)

    Returns:
        List of transformed (x, y) tuples
    """
    n = len(shape)
    if n < 2:
        return list(shape)

    # Get shape edge vertices
    p1 = shape[shape_edge_index]
    p2 = shape[(shape_edge_index + 1) % n]

    # Calculate shape edge angle
    edge_dx = p2[0] - p1[0]
    edge_dy = p2[1] - p1[1]
    shape_angle = math.atan2(edge_dy, edge_dx)

    # Target angle for each square edge (direction the edge points)
    # bottom: +X (0), right: +Y (90), top: -X (180), left: -Y (-90)
    target_angles = [0, math.pi / 2, math.pi, -math.pi / 2]
    target_angle = target_angles[square_edge_index]

    # Calculate rotation needed
    rotation = target_angle - shape_angle

    # Calculate shape center
    center_x = sum(p[0] for p in shape) / n
    center_y = sum(p[1] for p in shape) / n

    # Rotate all points around center
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    rotated = []
    for p in shape:
        dx = p[0] - center_x
        dy = p[1] - center_y
        new_x = dx * cos_r - dy * sin_r + center_x
        new_y = dx * sin_r + dy * cos_r + center_y
        rotated.append((new_x, new_y))

    # Get the rotated edge midpoint
    r_p1 = rotated[shape_edge_index]
    r_p2 = rotated[(shape_edge_index + 1) % n]
    edge_mid_x = (r_p1[0] + r_p2[0]) / 2
    edge_mid_y = (r_p1[1] + r_p2[1]) / 2

    # Calculate translation to snap edge to square edge
    if square_edge_index == 0:    # bottom: y = 0
        offset_x = 0.5 - edge_mid_x
        offset_y = -edge_mid_y
    elif square_edge_index == 1:  # right: x = 1
        offset_x = 1.0 - edge_mid_x
        offset_y = 0.5 - edge_mid_y
    elif square_edge_index == 2:  # top: y = 1
        offset_x = 0.5 - edge_mid_x
        offset_y = 1.0 - edge_mid_y
    else:                         # left: x = 0
        offset_x = -edge_mid_x
        offset_y = 0.5 - edge_mid_y

    # Apply translation
    result = [(p[0] + offset_x, p[1] + offset_y) for p in rotated]

    return result


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


def get_file_browser_image(context):
    """Get the image from the File Browser selection (load if needed), or None"""
    filepath = get_selected_image_path(context)
    if not filepath or not os.path.exists(filepath):
        return None

    # Check if it's an image file
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tiff', '.exr', '.hdr'}
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in valid_extensions:
        return None

    # Load or get existing image
    try:
        image = bpy.data.images.load(filepath, check_existing=True)
        return image
    except RuntimeError:
        return None


def find_material_with_image(image):
    """Return existing material that uses this image, or None"""
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image == image:
                return mat
    return None


def get_image_from_material(mat):
    """Return the first image used by a material, or None"""
    if not mat or not mat.use_nodes:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image:
            return node.image
    return None


def get_principled_bsdf_from_material(mat):
    """Return the Principled BSDF node from a material, or None"""
    if not mat or not mat.use_nodes:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node
    return None


def get_texture_node_from_material(mat):
    """Return the first TEX_IMAGE node from a material, or None"""
    if not mat or not mat.use_nodes:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE':
            return node
    return None


def is_texture_alpha_connected(mat):
    """Check if the texture's alpha output is connected to the BSDF alpha input"""
    if not mat or not mat.use_nodes:
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


def create_material_with_image(image):
    """Create a new material using the given image texture"""
    mat = bpy.data.materials.new(name=f"IMG_{image.name}")
    mat.use_nodes = True
    mat.use_backface_culling = True

    nt = mat.node_tree
    nt.nodes.clear()

    output = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexImage")

    tex.image = image

    tex.location = (-400, 0)
    bsdf.location = (-200, 0)
    output.location = (0, 0)

    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat


def get_viewport_grid_settings(context):
    """Get the viewport overlay settings for grid"""
    space = context.space_data
    if space and space.type == 'VIEW_3D':
        return space.overlay
    return None


def get_selected_face_count(context):
    """Get number of selected faces"""
    if context.mode != 'EDIT_MESH' or not context.object or context.object.type != 'MESH':
        return 0

    me = context.object.data
    bm = bmesh.from_edit_mesh(me)
    return sum(1 for f in bm.faces if f.select)


def find_shared_edge(face1, face2):
    """Find the shared edge between two faces, or None if not adjacent.

    Returns tuple of (edge, vert1, vert2) where vert1 and vert2 are the
    shared vertices in order as they appear in face1's loop.
    """
    face1_verts = set(face1.verts)
    face2_verts = set(face2.verts)
    shared_verts = face1_verts & face2_verts

    if len(shared_verts) < 2:
        return None

    # Find the edge that contains both shared vertices
    for edge in face1.edges:
        if edge.verts[0] in shared_verts and edge.verts[1] in shared_verts:
            # Return verts in the order they appear in face1's loops
            for i, loop in enumerate(face1.loops):
                if loop.vert in shared_verts:
                    next_loop = face1.loops[(i + 1) % len(face1.loops)]
                    if next_loop.vert in shared_verts:
                        return (edge, loop.vert, next_loop.vert)

    return None


def compute_uv_projection_from_face(face, uv_layer):
    """Compute the UV projection axes from a face's existing UVs.

    This extracts the transformation that maps 3D vertex positions to UV coordinates.
    The axes are computed in world space, lying in the source face's plane.

    Returns (u_axis, v_axis, origin_uv, origin_pos, face_normal) or None if computation fails.

    The transformation is: UV = origin_uv + (pos - origin_pos) dot [u_axis, v_axis]
    """
    if len(face.loops) < 3:
        return None

    # Get 3 vertices and their UVs
    loops = list(face.loops)
    p0 = loops[0].vert.co.copy()
    p1 = loops[1].vert.co.copy()
    p2 = loops[2].vert.co.copy()

    uv0 = loops[0][uv_layer].uv.copy()
    uv1 = loops[1][uv_layer].uv.copy()
    uv2 = loops[2][uv_layer].uv.copy()

    # Compute edge vectors in 3D and UV space
    e1 = p1 - p0
    e2 = p2 - p0

    duv1 = uv1 - uv0
    duv2 = uv2 - uv0

    # We want to find u_axis and v_axis (3D vectors in face plane) such that:
    # e1 · u_axis = duv1.x
    # e2 · u_axis = duv2.x
    # e1 · v_axis = duv1.y
    # e2 · v_axis = duv2.y
    #
    # The axes lie in the plane spanned by e1 and e2.
    # u_axis = a * e1 + b * e2
    # v_axis = c * e1 + d * e2
    #
    # Substituting:
    # a * (e1·e1) + b * (e1·e2) = duv1.x
    # a * (e1·e2) + b * (e2·e2) = duv2.x

    e1e1 = e1.dot(e1)
    e2e2 = e2.dot(e2)
    e1e2 = e1.dot(e2)

    det = e1e1 * e2e2 - e1e2 * e1e2
    if abs(det) < 1e-10:
        return None

    inv_det = 1.0 / det

    # Solve for u_axis coefficients
    a = inv_det * (e2e2 * duv1.x - e1e2 * duv2.x)
    b = inv_det * (e1e1 * duv2.x - e1e2 * duv1.x)
    u_axis = e1 * a + e2 * b

    # Solve for v_axis coefficients
    c = inv_det * (e2e2 * duv1.y - e1e2 * duv2.y)
    d = inv_det * (e1e1 * duv2.y - e1e2 * duv1.y)
    v_axis = e1 * c + e2 * d

    return (u_axis, v_axis, uv0, p0, face.normal.copy())


def apply_uv_projection_to_face(target_face, uv_layer, u_axis, v_axis, origin_uv, origin_pos, source_normal):
    """Apply a UV projection to a face, projecting vertices along target face's normal.

    Each vertex is projected onto the source texture plane (defined by origin_pos and
    source_normal) along the target face's normal. This makes the texture appear as if
    it's being projected onto the surface like a decal.

    For faces with multiple normals (smooth shading), each loop uses the face normal,
    ensuring consistent UVs within each face.
    """
    target_normal = target_face.normal

    for loop in target_face.loops:
        pos = loop.vert.co.copy()

        # Project vertex onto source texture plane along target face's normal
        # Plane equation: (P - origin_pos) · source_normal = 0
        # Ray: pos + t * target_normal
        # Solve: (pos + t * target_normal - origin_pos) · source_normal = 0
        # t = (origin_pos - pos) · source_normal / (target_normal · source_normal)

        denom = target_normal.dot(source_normal)

        if abs(denom) > 1e-6:
            # Project along target face's normal
            t = (origin_pos - pos).dot(source_normal) / denom
            projected = pos + target_normal * t
        else:
            # Target face is nearly perpendicular to source plane
            # Project orthogonally onto source plane instead
            t = (origin_pos - pos).dot(source_normal) / source_normal.length_squared
            projected = pos + source_normal * t

        # Calculate UV from projected position
        delta = projected - origin_pos
        u = origin_uv.x + delta.dot(u_axis)
        v = origin_uv.y + delta.dot(v_axis)

        loop[uv_layer].uv = (u, v)


def transfer_uvs_from_projection(source_face, target_face, uv_layer):
    """Transfer UVs from source face to target face using the same world-space projection.

    This implements TrenchBroom-style UV alignment where the texture is projected
    onto the target face along the target face's normal. This makes the texture
    appear continuous in world space, as if projected from a fixed plane.

    Works for both connected and non-connected faces.

    Returns True if UVs were transferred, False if transfer failed.
    """
    # Get the UV projection from the source face
    projection = compute_uv_projection_from_face(source_face, uv_layer)
    if projection is None:
        return False

    u_axis, v_axis, origin_uv, origin_pos, source_normal = projection

    # Apply the projection to the target face
    apply_uv_projection_to_face(target_face, uv_layer, u_axis, v_axis, origin_uv, origin_pos, source_normal)

    return True
