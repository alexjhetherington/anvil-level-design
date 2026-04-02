import bmesh
import math
from mathutils import Vector, Matrix

from .geometry import get_local_x_from_verts_3d, normalize_offset
from .materials import get_image_from_material, get_texture_dimensions_from_material


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


def face_aligned_project(face, uv_layer, mat, ppm, scale=1.0):
    """Project UVs aligned to the face using world-axis planar projection.

    Chooses the best projection plane (XY, XZ, or YZ) based on the face
    normal's dominant component, then maps vertex positions to UVs using
    the texture's pixels-per-meter scale.

    Args:
        face: BMFace to project
        uv_layer: BMesh UV layer
        mat: Blender material assigned to the face
        ppm: Pixels per meter setting
        scale: Uniform scale multiplier for the projection
    """
    tex_meters_u, tex_meters_v = get_texture_dimensions_from_material(mat, ppm)
    uv_per_meter_u = 1.0 / (tex_meters_u * scale)
    uv_per_meter_v = 1.0 / (tex_meters_v * scale)

    normal = face.normal
    abs_x = abs(normal.x)
    abs_y = abs(normal.y)
    abs_z = abs(normal.z)

    if abs_z > abs_x and abs_z > abs_y:
        for loop in face.loops:
            loop[uv_layer].uv = (
                loop.vert.co.x * uv_per_meter_u,
                loop.vert.co.y * uv_per_meter_v,
            )
    elif abs_y > abs_x:
        for loop in face.loops:
            loop[uv_layer].uv = (
                loop.vert.co.x * uv_per_meter_u,
                loop.vert.co.z * uv_per_meter_v,
            )
    else:
        for loop in face.loops:
            loop[uv_layer].uv = (
                loop.vert.co.y * uv_per_meter_u,
                loop.vert.co.z * uv_per_meter_v,
            )


def derive_transform_from_uvs(face, uv_layer, ppm, me):
    """Derive scale, rotation, and offset from a face's current UVs.

    Scale is extracted in texture space, meaning scale_u represents scaling
    along the texture's horizontal axis and scale_v along the vertical axis,
    regardless of rotation.

    The forward transform is: rotate projection axes -> project -> scale -> offset
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

    # Check for zero-area UV (all UVs collapsed to same point)
    uv_area = 0.0
    for i in range(1, len(uvs) - 1):
        edge_a = uvs[i] - uvs[0]
        edge_b = uvs[i + 1] - uvs[0]
        uv_area += abs(edge_a.x * edge_b.y - edge_a.y * edge_b.x)
    if uv_area < 1e-8:
        return {
            'scale_u': 0.0,
            'scale_v': 0.0,
            'rotation': 0.0,
            'offset_x': normalize_offset(uvs[0].x),
            'offset_y': normalize_offset(uvs[0].y)
        }

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
        # sin ~ 0, rotation ~ 0 or 180
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
        # cos ~ 0, rotation ~ 90 or -90
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

            # B = L / sqrt(du^2 * r^2 + dv^2)
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


def _is_quad_rectangular(points_2d, tolerance=0.01):
    """Check if 4 points in 2D form a rectangle (all angles ~90 degrees).

    Args:
        points_2d: List of 4 (x, y) tuples.
        tolerance: Maximum allowed absolute cosine of each corner angle.

    Returns:
        True if all four corner angles are within tolerance of 90 degrees.
    """
    if len(points_2d) != 4:
        return False
    for i in range(4):
        px, py = points_2d[(i - 1) % 4]
        cx, cy = points_2d[i]
        nx, ny = points_2d[(i + 1) % 4]
        e1x, e1y = px - cx, py - cy
        e2x, e2y = nx - cx, ny - cy
        len1 = math.sqrt(e1x * e1x + e1y * e1y)
        len2 = math.sqrt(e2x * e2x + e2y * e2y)
        if len1 < 1e-8 or len2 < 1e-8:
            return False
        cos_angle = (e1x * e2x + e1y * e2y) / (len1 * len2)
        if abs(cos_angle) > tolerance:
            return False
    return True


def needs_affine_transfer(face, uv_layer):
    """Check whether affine UV transfer should be used for this face.

    Returns True when the source face's UVs form a rectangle but the face
    geometry does not.  In that case the scalar (scale/rotation/offset)
    extraction cannot faithfully represent the mapping and the affine path
    gives better results.

    Args:
        face: BMesh face (source).
        uv_layer: BMesh UV layer.

    Returns:
        True if affine transfer is recommended, False otherwise.
    """
    loops = list(face.loops)
    if len(loops) != 4:
        return False

    # Check if UVs form a rectangle
    uvs = [(loop[uv_layer].uv.x, loop[uv_layer].uv.y) for loop in loops]
    if not _is_quad_rectangular(uvs):
        return False

    # Check if face geometry is also rectangular (in face-local 2D)
    face_axes = get_face_local_axes(face)
    if not face_axes:
        return False
    local_x, local_y = face_axes
    first_vert = loops[0].vert.co
    face_2d = []
    for loop in loops:
        delta = loop.vert.co - first_vert
        face_2d.append((delta.dot(local_x), delta.dot(local_y)))

    if _is_quad_rectangular(face_2d):
        return False  # Both rectangular -- scalar path works fine

    return True


def _solve_affine_3pt(local_2d, uvs, i0, i1, i2):
    """Solve for exact affine transform from 3 vertex correspondences.

    Args:
        local_2d: List of (x, y) tuples for all face verts in local 2D.
        uvs: List of (u, v) tuples for all face verts.
        i0, i1, i2: Indices into local_2d/uvs for the 3 triangle vertices.

    Returns:
        (M, t) where M is a 2x2 Matrix and t is a 2D Vector, or None if degenerate.
    """
    x0, y0 = local_2d[i0]
    x1, y1 = local_2d[i1]
    x2, y2 = local_2d[i2]
    u0, v0 = uvs[i0]
    u1, v1 = uvs[i1]
    u2, v2 = uvs[i2]

    dx1 = x1 - x0
    dy1 = y1 - y0
    dx2 = x2 - x0
    dy2 = y2 - y0
    du1 = u1 - u0
    dv1 = v1 - v0
    du2 = u2 - u0
    dv2 = v2 - v0

    det = dx1 * dy2 - dx2 * dy1
    if abs(det) < 1e-10:
        return None

    inv_det = 1.0 / det

    a = (du1 * dy2 - du2 * dy1) * inv_det
    b = (du2 * dx1 - du1 * dx2) * inv_det
    c = (dv1 * dy2 - dv2 * dy1) * inv_det
    d = (dv2 * dx1 - dv1 * dx2) * inv_det

    M = Matrix(((a, b), (c, d)))
    tx = u0 - a * x0 - b * y0
    ty = v0 - c * x0 - d * y0
    t = Vector((tx, ty))

    return M, t


def _solve_affine_least_squares(local_2d, uvs):
    """Solve for best-fit affine transform from n vertex correspondences.

    Minimises sum ||M @ xy_i + t - uv_i||^2 over all vertices.

    Args:
        local_2d: List of (x, y) tuples.
        uvs: List of (u, v) tuples.

    Returns:
        (M, t) where M is a 2x2 Matrix and t is a 2D Vector, or None if degenerate.
    """
    n = len(local_2d)

    # Build normal equations A^T A and A^T b for the system:
    #   u_i = a * x_i + b * y_i + tx
    #   v_i = c * x_i + d * y_i + ty
    sxx = sxy = sx = syy = sy = 0.0
    sxu = syu = su = sxv = syv = sv = 0.0

    for i in range(n):
        x, y = local_2d[i]
        u, v = uvs[i]
        sxx += x * x
        sxy += x * y
        sx += x
        syy += y * y
        sy += y
        sxu += x * u
        syu += y * u
        su += u
        sxv += x * v
        syv += y * v
        sv += v

    # A^T A (3x3 symmetric)
    ATA = Matrix((
        (sxx, sxy, sx),
        (sxy, syy, sy),
        (sx, sy, n),
    ))

    try:
        ATA_inv = ATA.inverted()
    except ValueError:
        return None

    # Solve for u-row: [a, b, tx]
    rhs_u = Vector((sxu, syu, su))
    params_u = ATA_inv @ rhs_u

    # Solve for v-row: [c, d, ty]
    rhs_v = Vector((sxv, syv, sv))
    params_v = ATA_inv @ rhs_v

    M = Matrix(((params_u.x, params_u.y), (params_v.x, params_v.y)))
    t = Vector((params_u.z, params_v.z))

    return M, t


def extract_affine_from_face(face, uv_layer, shared_edge_verts, loop_triangles):
    """Extract affine transform (2x2 matrix + offset) mapping face-local 2D to UV.

    When shared_edge_verts is provided and tessellation data is available, uses
    the tessellation triangle touching that edge for an exact solve. Otherwise
    uses least-squares across all vertices.

    Args:
        face: BMesh face to extract from.
        uv_layer: BMesh UV layer.
        shared_edge_verts: Tuple of 2 BMVert on the shared edge, or None.
        loop_triangles: Result of bm.calc_loop_triangles(), or None.

    Returns:
        (M, t, local_axes) where M is a 2x2 Matrix, t is a 2D Vector,
        and local_axes is (local_x, local_y), or None on failure.
    """
    face_axes = get_face_local_axes(face)
    if not face_axes:
        return None
    local_x, local_y = face_axes

    loops = list(face.loops)
    if len(loops) < 3:
        return None

    # Build local 2D coords and UVs
    first_vert = loops[0].vert.co
    local_2d = []
    uvs = []
    vert_to_loop_idx = {}

    for i, loop in enumerate(loops):
        delta = loop.vert.co - first_vert
        x = delta.dot(local_x)
        y = delta.dot(local_y)
        local_2d.append((x, y))
        uvs.append((loop[uv_layer].uv.x, loop[uv_layer].uv.y))
        vert_to_loop_idx[loop.vert.index] = i

    # Try exact solve from tessellation triangle touching the shared edge
    if shared_edge_verts is not None:
        sv0_idx = shared_edge_verts[0].index
        sv1_idx = shared_edge_verts[1].index

        if sv0_idx in vert_to_loop_idx and sv1_idx in vert_to_loop_idx:
            tri_indices = None

            if loop_triangles is not None:
                # Find Blender's tessellation triangle for this face that
                # contains both shared edge vertices
                for tri in loop_triangles:
                    if tri[0].face != face:
                        continue
                    tri_vert_indices = {l.vert.index for l in tri}
                    if sv0_idx in tri_vert_indices and sv1_idx in tri_vert_indices:
                        tri_indices = tuple(
                            vert_to_loop_idx[l.vert.index] for l in tri
                        )
                        break

            if tri_indices is None:
                # Fallback: fan triangulation from vertex 0
                li0 = vert_to_loop_idx[sv0_idx]
                li1 = vert_to_loop_idx[sv1_idx]
                # Pick a third vertex that isn't on the shared edge
                for i in range(len(loops)):
                    if i != li0 and i != li1:
                        tri_indices = (li0, li1, i)
                        break

            if tri_indices is not None:
                result = _solve_affine_3pt(local_2d, uvs, *tri_indices)
                if result is not None:
                    return (result[0], result[1], (local_x, local_y))

    # Least-squares across all vertices
    if len(loops) == 3:
        result = _solve_affine_3pt(local_2d, uvs, 0, 1, 2)
    else:
        result = _solve_affine_least_squares(local_2d, uvs)
    if result is None:
        return None
    return (result[0], result[1], (local_x, local_y))


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
        # Plane equation: (P - origin_pos) . source_normal = 0
        # Ray: pos + t * target_normal
        # Solve: (pos + t * target_normal - origin_pos) . source_normal = 0
        # t = (origin_pos - pos) . source_normal / (target_normal . source_normal)

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


def apply_uv_to_face(face, uv_layer, scale_u, scale_v, rotation_deg, offset_x, offset_y,
                     mat, ppm, me):
    """Apply UV coordinates to a single face from scratch with given parameters.

    This is the low-level function that does the actual UV projection.
    Order: rotate projection axes -> project -> scale -> offset

    Rotation is applied by rotating the projection axes in face space, which
    avoids distortion with non-square textures. Scale is applied in texture space,
    meaning scale_u always affects the texture's horizontal axis and scale_v
    always affects the texture's vertical axis, regardless of rotation.

    Args:
        face: BMesh face to apply UVs to
        uv_layer: BMesh UV layer
        scale_u, scale_v: Texture scale factors
        rotation_deg: Rotation in degrees
        offset_x, offset_y: UV offset
        mat: Material to get texture dimensions from (can be None)
        ppm: Pixels per meter setting
        me: Mesh data (for bmesh.update_edit_mesh)
    """
    # Guard against invalid face data during modal operators (e.g., loop cut)
    try:
        loops = list(face.loops)
    except (ReferenceError, RuntimeError, OSError):
        return

    if len(loops) < 2:
        return

    face_axes = get_face_local_axes(face)
    if not face_axes:
        return
    face_local_x, face_local_y = face_axes

    # Get texture dimensions from material
    tex_meters_u, tex_meters_v = get_texture_dimensions_from_material(mat, ppm)

    # Rotate the projection axes in face space
    # This determines which face direction maps to texture U vs V
    rotation_rad = math.radians(rotation_deg)
    cos_rot = math.cos(rotation_rad)
    sin_rot = math.sin(rotation_rad)

    # Rotated projection axes (3D vectors in face plane)
    proj_x = face_local_x * cos_rot - face_local_y * sin_rot
    proj_y = face_local_x * sin_rot + face_local_y * cos_rot

    first_vert = loops[0].vert.co

    for loop in loops:
        delta = loop.vert.co - first_vert

        # Project onto rotated axes
        x = delta.dot(proj_x)
        y = delta.dot(proj_y)

        # Convert to UV with scale (texture-based since x is along texture U direction)
        u = x / (scale_u * tex_meters_u)
        v = y / (scale_v * tex_meters_v)

        # Offset
        loop[uv_layer].uv.x = u + offset_x
        loop[uv_layer].uv.y = v + offset_y

    if me.is_editmode:
        bmesh.update_edit_mesh(me)


def apply_affine_to_face(face, uv_layer, M, t, me):
    """Apply an affine transform to a face's UVs.

    Computes UV = M @ local_2d + t for each vertex, where local_2d is the
    vertex position in face-local 2D space (relative to first loop vertex).

    Args:
        face: BMesh face to apply UVs to.
        uv_layer: BMesh UV layer.
        M: 2x2 Matrix mapping face-local 2D to UV.
        t: 2D Vector offset.
        me: Mesh data (for bmesh.update_edit_mesh).
    """
    try:
        loops = list(face.loops)
    except (ReferenceError, RuntimeError, OSError):
        return

    if len(loops) < 2:
        return

    face_axes = get_face_local_axes(face)
    if not face_axes:
        return
    local_x, local_y = face_axes

    first_vert = loops[0].vert.co

    for loop in loops:
        delta = loop.vert.co - first_vert
        x = delta.dot(local_x)
        y = delta.dot(local_y)

        xy = Vector((x, y))
        uv = M @ xy + t

        loop[uv_layer].uv.x = uv.x
        loop[uv_layer].uv.y = uv.y

    if me.is_editmode:
        bmesh.update_edit_mesh(me)
