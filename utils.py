import bpy
import bmesh
import math
from mathutils import Vector


# Global debug logging toggle for the addon
DEBUG_LOGGING = True


def debug_log(msg):
    """Print debug message if logging is enabled."""
    if DEBUG_LOGGING:
        print(msg)


LEVEL_DESIGN_WORKSPACE_NAME = "Level Design"
HOTSPOT_MAPPING_WORKSPACE_NAME = "Hotspot Mapping"


def is_level_design_workspace():
    """Check if the current workspace is the Level Design workspace."""
    try:
        return bpy.context.workspace.name == LEVEL_DESIGN_WORKSPACE_NAME
    except AttributeError:
        return False


def is_hotspot_mapping_workspace():
    """Check if the current workspace is the Hotspot Mapping workspace."""
    try:
        return bpy.context.workspace.name == HOTSPOT_MAPPING_WORKSPACE_NAME
    except AttributeError:
        return False



def get_local_x_from_verts_3d(verts):
    """Get the first non-zero edge direction from a list of 3D vertices.

    Iterates through consecutive vertex pairs until finding an edge with
    non-zero length. This handles cases where vertices may be coincident
    (e.g., after moving a vertex on top of another).

    Args:
        verts: List of Vector3 vertex positions (at least 2 elements)

    Returns:
        Normalized Vector3 direction of the first valid edge, or None if
        no valid edge exists (all vertices coincident or < 2 vertices).
    """
    if len(verts) < 2:
        return None

    for i in range(len(verts)):
        p0 = verts[i]
        p1 = verts[(i + 1) % len(verts)]
        edge = p1 - p0
        if edge.length > 0.0001:
            return edge.normalized()

    return None


def compute_normal_from_verts(verts):
    """Compute a normal vector from a list of vertices by finding 3 non-collinear points.

    Searches through vertex triplets until finding three that are non-collinear,
    then computes the normal from their cross product.

    Args:
        verts: List of BMVerts or objects with .co attribute (at least 3 elements)

    Returns:
        Normalized Vector3 normal, or None if all vertices are collinear or < 3 vertices.
    """
    if len(verts) < 3:
        return None

    # Get position from vert (handles both BMVert and Vector)
    def get_pos(v):
        return v.co if hasattr(v, 'co') else v

    n = len(verts)
    p0 = get_pos(verts[0])

    for i in range(1, n):
        p1 = get_pos(verts[i])
        e1 = p1 - p0
        e1e1 = e1.dot(e1)

        # Skip if e1 is too short
        if e1e1 < 1e-10:
            continue

        for j in range(i + 1, n):
            p2 = get_pos(verts[j])
            e2 = p2 - p0
            e2e2 = e2.dot(e2)

            # Skip if e2 is too short
            if e2e2 < 1e-10:
                continue

            e1e2 = e1.dot(e2)
            det = e1e1 * e2e2 - e1e2 * e1e2

            # Check if vertices are non-collinear (det is non-zero)
            if abs(det) < 1e-10:
                continue

            # Found valid triplet - compute normal
            normal = e1.cross(e2)
            if normal.length > 1e-10:
                return normal.normalized()

    return None


def get_local_x_from_verts_2d(verts):
    """Get the first non-zero edge direction from a list of 2D vertices.

    Iterates through consecutive vertex pairs until finding an edge with
    non-zero length. This handles cases where UV vertices may be coincident.

    Args:
        verts: List of 2D coordinates as Vector2, tuples, or objects with x/y attributes
               (at least 2 elements)

    Returns:
        Tuple (dx, dy) normalized direction of the first valid edge, or None if
        no valid edge exists (all vertices coincident or < 2 vertices).
    """
    if len(verts) < 2:
        return None

    for i in range(len(verts)):
        v0 = verts[i]
        v1 = verts[(i + 1) % len(verts)]

        # Handle different input types
        if isinstance(v0, (tuple, list)):
            x0, y0 = v0[0], v0[1]
        else:
            x0, y0 = v0.x, v0.y

        if isinstance(v1, (tuple, list)):
            x1, y1 = v1[0], v1[1]
        else:
            x1, y1 = v1.x, v1.y

        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)

        if length > 0.0001:
            return (dx / length, dy / length)

    return None


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

    X-axis: Along the first non-zero edge, normalized
    Y-axis: Perpendicular to X, in the face plane

    Returns (local_x, local_y) as Vector3, or None if face has < 2 vertices,
    face is invalid, or no valid edge exists.
    """
    if face is None or not face.is_valid:
        return None

    loops = list(face.loops)
    if len(loops) < 2:
        return None

    # Get first non-zero edge direction as local X
    verts = [loop.vert.co for loop in loops]
    local_x = get_local_x_from_verts_3d(verts)
    if local_x is None:
        return None

    if len(loops) < 3:
        # For 2 vertices, create arbitrary perpendicular
        if abs(local_x.z) < 0.9:
            local_y = local_x.cross(Vector((0, 0, 1))).normalized()
        else:
            local_y = local_x.cross(Vector((1, 0, 0))).normalized()
    else:
        # Use Blender's computed face normal (handles degenerate cases)
        face_normal = face.normal

        # Local Y is perpendicular to X, in the face plane
        local_y = face_normal.cross(local_x).normalized()

    return (local_x, local_y)


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
    if face is None or not face.is_valid:
        return None

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


def compute_uv_projection_from_face(face, uv_layer):
    """Compute the UV projection axes from a face's existing UVs.

    This extracts the transformation that maps 3D vertex positions to UV coordinates.
    The axes are computed in world space, lying in the source face's plane.

    Returns (u_axis, v_axis, origin_uv, origin_pos, face_normal) or None if computation fails.

    The transformation is: UV = origin_uv + (pos - origin_pos) dot [u_axis, v_axis]
    """
    if face is None or not face.is_valid:
        return None

    if len(face.loops) < 3:
        return None

    loops = list(face.loops)
    n = len(loops)

    # Search for 3 non-collinear vertices that form the best basis
    # (highest determinant = most orthogonal = most stable UV computation)
    best_result = None
    best_det = 0

    for origin_idx in range(n):
        p0 = loops[origin_idx].vert.co.copy()
        uv0 = loops[origin_idx][uv_layer].uv.copy()

        for i in range(n):
            if i == origin_idx:
                continue
            p1 = loops[i].vert.co.copy()
            e1 = p1 - p0
            e1e1 = e1.dot(e1)

            # Skip if e1 is too short
            if e1e1 < 1e-10:
                continue

            for j in range(i + 1, n):
                if j == origin_idx:
                    continue
                p2 = loops[j].vert.co.copy()
                e2 = p2 - p0
                e2e2 = e2.dot(e2)

                # Skip if e2 is too short
                if e2e2 < 1e-10:
                    continue

                e1e2 = e1.dot(e2)
                det = e1e1 * e2e2 - e1e2 * e1e2

                # Check if vertices are non-collinear (det is non-zero)
                if abs(det) < 1e-10:
                    continue

                # Keep track of the best (most orthogonal) triplet
                if det > best_det:
                    best_det = det
                    uv1 = loops[i][uv_layer].uv.copy()
                    uv2 = loops[j][uv_layer].uv.copy()

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

                    inv_det = 1.0 / det

                    # Solve for u_axis coefficients
                    a = inv_det * (e2e2 * duv1.x - e1e2 * duv2.x)
                    b = inv_det * (e1e1 * duv2.x - e1e2 * duv1.x)
                    u_axis = e1 * a + e2 * b

                    # Solve for v_axis coefficients
                    c = inv_det * (e2e2 * duv1.y - e1e2 * duv2.y)
                    d = inv_det * (e1e1 * duv2.y - e1e2 * duv1.y)
                    v_axis = e1 * c + e2 * d

                    best_result = (u_axis, v_axis, uv0, p0, face.normal.copy())

    return best_result


def apply_uv_projection_to_face(target_face, uv_layer, u_axis, v_axis, origin_uv, origin_pos, source_normal):
    """Apply a UV projection to a face, projecting vertices along target face's normal.

    Each vertex is projected onto the source texture plane (defined by origin_pos and
    source_normal) along the target face's normal. This makes the texture appear as if
    it's being projected onto the surface like a decal.

    For faces with multiple normals (smooth shading), each loop uses the face normal,
    ensuring consistent UVs within each face.
    """
    if target_face is None or not target_face.is_valid:
        return

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


def face_has_hotspot_material(face, me):
    """Check if a face has a material with a hotspottable texture.

    Args:
        face: BMesh face to check
        me: Mesh data (for accessing materials)

    Returns:
        True if the face's material has a hotspottable texture, False otherwise.
    """
    from .hotspot_mapping.json_storage import is_texture_hotspottable

    mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
    image = get_image_from_material(mat)
    if image and is_texture_hotspottable(image.name):
        return True
    return False


def get_connected_faces(face):
    """Get all faces connected to a face via shared edges.

    Args:
        face: BMesh face to get connected faces for

    Returns:
        Set of BMesh faces connected to the input face (excluding the input face itself).
    """
    connected = set()
    for edge in face.edges:
        for linked_face in edge.link_faces:
            if linked_face != face and linked_face.is_valid:
                connected.add(linked_face)
    return connected


def any_connected_face_has_hotspot(face, me):
    """Check if any face connected to the given face has a hotspot material.

    Args:
        face: BMesh face to check neighbors of
        me: Mesh data (for accessing materials)

    Returns:
        True if any connected face has a hotspottable texture, False otherwise.
    """
    for connected_face in get_connected_faces(face):
        if face_has_hotspot_material(connected_face, me):
            return True
    return False


def get_all_hotspot_faces(bm, me):
    """Get all faces in the bmesh that have hotspot materials.

    Args:
        bm: BMesh to search
        me: Mesh data (for accessing materials)

    Returns:
        List of BMesh faces that have hotspottable textures.
    """
    return [f for f in bm.faces if f.is_valid and face_has_hotspot_material(f, me)]


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
