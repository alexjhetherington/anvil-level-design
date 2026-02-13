"""
Modal Draw - Utility Functions

Coordinate math, plane intersections, and helper functions.
"""

import bpy
import math
from mathutils import Vector, Matrix
from mathutils.geometry import intersect_line_plane
from bpy_extras.view3d_utils import (
    region_2d_to_origin_3d,
    region_2d_to_vector_3d,
)


def tag_redraw_all_3d_views():
    """
    Tag all 3D viewports for redraw.

    This ensures previews appear in all open views simultaneously.
    """
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def get_view_type(context):
    """
    Determine if we're in a 2D orthographic view or 3D perspective view.

    Returns:
        str: 'TOP', 'FRONT', 'RIGHT', 'BACK', 'LEFT', 'BOTTOM', or 'PERSPECTIVE'
    """
    rv3d = context.region_data
    if rv3d is None:
        return 'PERSPECTIVE'

    if not rv3d.is_perspective:
        # Check view orientation
        view_matrix = rv3d.view_matrix
        view_dir = Vector((view_matrix[2][0], view_matrix[2][1], view_matrix[2][2]))

        # Determine which axis we're looking along
        abs_x = abs(view_dir.x)
        abs_y = abs(view_dir.y)
        abs_z = abs(view_dir.z)

        threshold = 0.99

        if abs_z > threshold:
            return 'TOP' if view_dir.z < 0 else 'BOTTOM'
        elif abs_y > threshold:
            return 'FRONT' if view_dir.y > 0 else 'BACK'
        elif abs_x > threshold:
            return 'RIGHT' if view_dir.x < 0 else 'LEFT'

    return 'PERSPECTIVE'


def is_2d_view(context):
    """Check if current view is a 2D orthographic view."""
    view_type = get_view_type(context)
    return view_type in ('TOP', 'FRONT', 'RIGHT', 'BACK', 'LEFT', 'BOTTOM')


def get_2d_view_plane(context):
    """
    Get the plane for 2D view drawing.

    Returns:
        tuple: (plane_point, plane_normal, axis1, axis2) where axis1 and axis2
               are the drawing axes, or None if not a 2D view

    TODO: The plane positions (Z=0, Y=0, X=0) may need future refinement
          to use 3D cursor position or other reference.
    """
    view_type = get_view_type(context)

    if view_type in ('TOP', 'BOTTOM'):
        # XY plane at Z=0
        return (Vector((0, 0, 0)), Vector((0, 0, 1)), Vector((1, 0, 0)), Vector((0, 1, 0)))
    elif view_type in ('FRONT', 'BACK'):
        # XZ plane at Y=0
        return (Vector((0, 0, 0)), Vector((0, 1, 0)), Vector((1, 0, 0)), Vector((0, 0, 1)))
    elif view_type in ('RIGHT', 'LEFT'):
        # YZ plane at X=0
        return (Vector((0, 0, 0)), Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)))

    return None


def get_2d_view_tangents(context):
    """
    Get the tangent axes for the cross indicator in 2D view.

    Returns:
        tuple: (tangent1, tangent2) unit vectors on the view plane
    """
    view_type = get_view_type(context)

    if view_type in ('TOP', 'BOTTOM'):
        return (Vector((1, 0, 0)), Vector((0, 1, 0)))
    elif view_type in ('FRONT', 'BACK'):
        return (Vector((1, 0, 0)), Vector((0, 0, 1)))
    elif view_type in ('RIGHT', 'LEFT'):
        return (Vector((0, 1, 0)), Vector((0, 0, 1)))

    # Default fallback
    return (Vector((1, 0, 0)), Vector((0, 1, 0)))


def get_face_tangents(face_normal):
    """
    Get tangent axes for a face (for cross indicator orientation).

    The cross lies on the face plane, perpendicular to the normal.

    Args:
        face_normal: The face normal vector

    Returns:
        tuple: (tangent1, tangent2) unit vectors on the face plane
    """
    normal = face_normal.normalized()

    # Choose an "up" vector that's not parallel to the normal
    up = Vector((0, 0, 1))
    if abs(normal.dot(up)) > 0.99:
        up = Vector((0, 1, 0))

    # First tangent: perpendicular to normal and up
    tangent1 = normal.cross(up)
    tangent1.normalize()

    # Second tangent: perpendicular to normal and tangent1
    tangent2 = normal.cross(tangent1)
    tangent2.normalize()

    return (tangent1, tangent2)


def get_snap_aligned_tangents(face_normal):
    """
    Get world-aligned tangent axes for snapping grid display.

    These tangents align with the grid axes used by snap_to_grid_on_face,
    ensuring the visual grid matches actual snap points.

    Args:
        face_normal: The face normal vector

    Returns:
        tuple: (tangent1, tangent2) world-aligned unit vectors on the face plane
    """
    normal_abs = Vector([abs(n) for n in face_normal])

    # Determine dominant axis and return the perpendicular world axes
    if normal_abs.x >= normal_abs.y and normal_abs.x >= normal_abs.z:
        # Normal is closest to X axis - grid is on YZ plane
        return (Vector((0, 1, 0)), Vector((0, 0, 1)))
    elif normal_abs.y >= normal_abs.x and normal_abs.y >= normal_abs.z:
        # Normal is closest to Y axis - grid is on XZ plane
        return (Vector((1, 0, 0)), Vector((0, 0, 1)))
    else:
        # Normal is closest to Z axis - grid is on XY plane
        return (Vector((1, 0, 0)), Vector((0, 1, 0)))


def mouse_to_3d_on_plane(context, event, plane_point, plane_normal):
    """
    Project mouse position onto a plane in 3D space.

    Args:
        context: Blender context
        event: Mouse event
        plane_point: A point on the plane
        plane_normal: The plane's normal vector

    Returns:
        Vector or None: The 3D point on the plane, or None if no intersection
    """
    region = context.region
    rv3d = context.region_data

    if rv3d is None:
        return None

    coord = (event.mouse_region_x, event.mouse_region_y)

    origin = region_2d_to_origin_3d(region, rv3d, coord)
    direction = region_2d_to_vector_3d(region, rv3d, coord)

    if origin is None or direction is None:
        return None

    return intersect_line_plane(origin, origin + direction * 10000, plane_point, plane_normal)


def raycast_scene(context, event):
    """
    Raycast from mouse position into the scene.

    Returns:
        tuple: (hit, location, normal, face_index, object, matrix) or
               (False, None, None, None, None, None) if no hit
    """
    region = context.region
    rv3d = context.region_data

    if rv3d is None:
        return (False, None, None, None, None, None)

    coord = (event.mouse_region_x, event.mouse_region_y)

    origin = region_2d_to_origin_3d(region, rv3d, coord)
    direction = region_2d_to_vector_3d(region, rv3d, coord)

    if origin is None or direction is None:
        return (False, None, None, None, None, None)

    depsgraph = context.evaluated_depsgraph_get()
    result = context.scene.ray_cast(depsgraph, origin, direction)

    return result


def get_face_rotation_angle(normal):
    """
    Calculate the rotation angle around the vertical (Z) axis from a face normal.

    The rectangle should be perpendicular to this direction in the horizontal plane.

    Args:
        normal: The face normal vector

    Returns:
        float: Rotation angle in radians around Z axis
    """
    # Project normal onto XY plane
    horizontal = Vector((normal.x, normal.y, 0))

    if horizontal.length < 0.001:
        # Face is horizontal (floor/ceiling), return 0 (will be handled separately)
        return None

    horizontal.normalize()

    # Calculate angle from X axis
    angle = math.atan2(horizontal.y, horizontal.x)

    return angle


def get_rectangle_axes(face_normal, context):
    """
    Get the local X and Y axes for the rectangle based on face normal.

    The rectangle's front face aligns with the face normal.
    local_x stays horizontal (no roll along the cube's length).
    local_y tilts with the face normal to match the surface angle.

    For horizontal faces (floor/ceiling), uses world-aligned axes
    with the view direction to pick orientation.

    Args:
        face_normal: The face normal of the hit face
        context: Blender context (used for view direction if face is horizontal)

    Returns:
        tuple: (local_x, local_y, local_z) unit vectors
               local_z is the face normal (depth direction)
               local_x is horizontal, perpendicular to face
               local_y is perpendicular to both (tilts with face)
    """
    local_z = face_normal.normalized()

    # Check if face is horizontal (normal nearly parallel to world Z)
    horizontal_component = Vector((face_normal.x, face_normal.y, 0))

    if horizontal_component.length < 0.001:
        # Face is horizontal (floor/ceiling)
        # Use view direction to determine which world axis to use for local_x
        rv3d = context.region_data
        if rv3d is not None:
            view_forward = Vector((
                rv3d.view_matrix[2][0],
                rv3d.view_matrix[2][1],
                0
            ))
            if view_forward.length > 0.001:
                view_forward.normalize()
                dot_x = abs(view_forward.dot(Vector((1, 0, 0))))
                dot_y = abs(view_forward.dot(Vector((0, 1, 0))))

                if dot_x > dot_y:
                    local_x = Vector((0, 1, 0))
                else:
                    local_x = Vector((1, 0, 0))
            else:
                local_x = Vector((1, 0, 0))
        else:
            local_x = Vector((1, 0, 0))

        local_y = local_z.cross(local_x)
        local_y.normalize()
        return (local_x, local_y, local_z)

    # Non-horizontal face: local_x is horizontal, perpendicular to face normal
    world_up = Vector((0, 0, 1))
    local_x = world_up.cross(local_z)
    local_x.normalize()

    # local_y tilts with the face, perpendicular to both
    local_y = local_z.cross(local_x)
    local_y.normalize()

    return (local_x, local_y, local_z)


def get_face_vertices_world(obj, face_index):
    """
    Get the world-space vertices of a face.

    Args:
        obj: The mesh object
        face_index: Index of the face in the mesh

    Returns:
        list: List of Vector positions in world space, or None if invalid
    """
    if obj is None or obj.type != 'MESH':
        return None

    # Get the evaluated mesh (includes modifiers)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()

    if face_index < 0 or face_index >= len(mesh.polygons):
        obj_eval.to_mesh_clear()
        return None

    face = mesh.polygons[face_index]
    world_matrix = obj.matrix_world

    # Get vertices in world space
    vertices = []
    for vert_idx in face.vertices:
        local_co = mesh.vertices[vert_idx].co
        world_co = world_matrix @ local_co
        vertices.append(Vector(world_co))

    obj_eval.to_mesh_clear()
    return vertices


def closest_point_on_line_segment(point, seg_start, seg_end):
    """
    Find the closest point on a line segment to a given point.

    Args:
        point: The point to find closest to
        seg_start: Start of line segment
        seg_end: End of line segment

    Returns:
        Vector: Closest point on the segment
    """
    seg_vec = seg_end - seg_start
    seg_len_sq = seg_vec.length_squared

    if seg_len_sq < 1e-10:
        return seg_start.copy()

    # Project point onto line, clamped to segment
    t = max(0, min(1, (point - seg_start).dot(seg_vec) / seg_len_sq))
    return seg_start + seg_vec * t


def point_in_polygon_2d(point, polygon, axis1, axis2):
    """
    Check if a point is inside a polygon using 2D projection.

    Uses ray casting algorithm projected onto the plane defined by axis1/axis2.

    Args:
        point: 3D point to test
        polygon: List of 3D polygon vertices
        axis1: First axis of the projection plane
        axis2: Second axis of the projection plane

    Returns:
        bool: True if point is inside polygon
    """
    # Project all points to 2D
    def to_2d(p):
        return (p.dot(axis1), p.dot(axis2))

    px, py = to_2d(point)
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = to_2d(polygon[i])
        xj, yj = to_2d(polygon[j])

        # Ray casting algorithm
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def clamp_point_to_face(point, face_vertices, face_normal):
    """
    Clamp a point to be inside a face polygon.

    If the point is inside the face, returns it unchanged.
    If outside, returns the closest point on any face edge.

    Args:
        point: The 3D point to clamp
        face_vertices: List of face vertices in world space
        face_normal: The face normal vector

    Returns:
        Vector: The clamped point (on face or on edge)
    """
    if len(face_vertices) < 3:
        return point.copy()

    # Get projection axes (perpendicular to normal)
    normal_abs = Vector([abs(n) for n in face_normal])
    if normal_abs.x >= normal_abs.y and normal_abs.x >= normal_abs.z:
        axis1 = Vector((0, 1, 0))
        axis2 = Vector((0, 0, 1))
    elif normal_abs.y >= normal_abs.x and normal_abs.y >= normal_abs.z:
        axis1 = Vector((1, 0, 0))
        axis2 = Vector((0, 0, 1))
    else:
        axis1 = Vector((1, 0, 0))
        axis2 = Vector((0, 1, 0))

    # Check if point is inside the polygon
    if point_in_polygon_2d(point, face_vertices, axis1, axis2):
        return point.copy()

    # Point is outside - find closest point on any edge
    closest = None
    closest_dist_sq = float('inf')

    n = len(face_vertices)
    for i in range(n):
        edge_start = face_vertices[i]
        edge_end = face_vertices[(i + 1) % n]

        edge_closest = closest_point_on_line_segment(point, edge_start, edge_end)
        dist_sq = (point - edge_closest).length_squared

        if dist_sq < closest_dist_sq:
            closest_dist_sq = dist_sq
            closest = edge_closest

    return closest if closest is not None else point.copy()


def project_point_onto_plane(point, plane_point, plane_normal):
    """
    Project a point onto a plane.

    Args:
        point: The point to project
        plane_point: A point on the plane
        plane_normal: The plane's normal vector (doesn't need to be normalized)

    Returns:
        Vector: The projected point on the plane
    """
    normal = plane_normal.normalized()
    # Distance from point to plane along normal
    dist = (point - plane_point).dot(normal)
    # Project point onto plane
    return point - normal * dist


def get_grid_size(context):
    """Get the current grid size from the viewport overlay."""
    space = context.space_data
    if space and hasattr(space, 'overlay'):
        return space.overlay.grid_scale
    return 1.0


def is_snapping_enabled(context):
    """Check if snapping is enabled in tool settings."""
    return context.tool_settings.use_snap


def build_cuboid_vertices(corner1, corner2, depth, local_x, local_y, local_z):
    """
    Build the 8 vertices of a cuboid from two corners and depth.

    Args:
        corner1: First corner of the rectangle (Vector)
        corner2: Opposite corner of the rectangle (Vector)
        depth: Depth of the cuboid (can be negative)
        local_x: Rectangle's local X axis
        local_y: Rectangle's local Y axis (vertical)
        local_z: Rectangle's local Z axis (depth direction)

    Returns:
        list: 8 Vector positions for cuboid vertices
              [front_bl, front_br, front_tr, front_tl, back_bl, back_br, back_tr, back_tl]
    """
    # Calculate rectangle corners in local space
    # corner1 and corner2 define opposite corners

    # Get the 4 corners of the front face
    # We need to handle that corner1 and corner2 are opposite corners
    # but we don't know which configuration they're in

    # Project corners onto local axes relative to corner1
    diff = corner2 - corner1
    local_dx = diff.dot(local_x)
    local_dy = diff.dot(local_y)

    # Build front face corners (clockwise from bottom-left when looking at front)
    front_bl = corner1.copy()
    front_br = corner1 + local_x * local_dx
    front_tr = corner1 + local_x * local_dx + local_y * local_dy
    front_tl = corner1 + local_y * local_dy

    # Build back face corners (offset by depth)
    depth_offset = local_z * depth
    back_bl = front_bl + depth_offset
    back_br = front_br + depth_offset
    back_tr = front_tr + depth_offset
    back_tl = front_tl + depth_offset

    return [front_bl, front_br, front_tr, front_tl, back_bl, back_br, back_tr, back_tl]


def get_cuboid_edges():
    """
    Get the edge indices for a cuboid.

    Returns:
        list: List of (start_idx, end_idx) tuples for the 12 edges
    """
    return [
        # Front face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Back face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Connecting edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
