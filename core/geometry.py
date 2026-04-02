import math
from mathutils import Vector


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
    """Compute a normal vector from a list of vertices using Newell's method.

    Sums cross-product contributions from all edges of the polygon, which
    correctly handles concave polygons (unlike a simple 3-point cross product).

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
    normal = Vector((0.0, 0.0, 0.0))

    for i in range(n):
        curr = get_pos(verts[i])
        next_ = get_pos(verts[(i + 1) % n])
        normal.x += (curr.y - next_.y) * (curr.z + next_.z)
        normal.y += (curr.z - next_.z) * (curr.x + next_.x)
        normal.z += (curr.x - next_.x) * (curr.y + next_.y)

    if normal.length < 1e-10:
        return None

    return normal.normalized()


def are_verts_coplanar(verts, tolerance=0.001):
    """Check if all vertices lie on a single plane.

    Finds a plane normal from the first three non-collinear points
    and verifies all remaining points lie on that plane.

    Args:
        verts: Iterable of BMVerts or objects with .co attribute
        tolerance: Maximum distance from the plane to be considered coplanar

    Returns:
        True if all vertices are coplanar (or fewer than 3 vertices).
    """
    positions = [v.co if hasattr(v, 'co') else v for v in verts]
    if len(positions) <= 2:
        return True

    p0 = positions[0]
    normal = None
    for i in range(1, len(positions)):
        for j in range(i + 1, len(positions)):
            n = (positions[i] - p0).cross(positions[j] - p0)
            if n.length > tolerance:
                normal = n.normalized()
                break
        if normal:
            break

    if not normal:
        return True  # All points are collinear

    for p in positions:
        if abs((p - p0).dot(normal)) > tolerance:
            return False

    return True


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
