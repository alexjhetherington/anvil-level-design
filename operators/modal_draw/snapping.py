"""
Modal Draw - Snapping Logic

All snapping calculations are isolated here for easy modification.
"""

import bpy
import math
from mathutils import Vector

from . import utils


def snap_to_grid(point, grid_size):
    """
    Snap a point to the nearest grid intersection.

    Args:
        point: Vector to snap
        grid_size: Size of grid cells

    Returns:
        Vector: Snapped position
    """
    return Vector([
        round(c / grid_size) * grid_size
        for c in point
    ])


def snap_to_grid_on_face(hit_location, face_normal, grid_size):
    """
    Snap to a grid-aligned position ON the mesh surface.

    This is used for the first vertex in 3D view.

    NOTE: This method is intentionally isolated for future modifications.

    The snapping behavior:
    - Determine which world axis the face normal is closest to (X, Y, or Z)
    - Snap to grid on the perpendicular axes
    - Solve for the third coordinate using line-plane intersection

    Args:
        hit_location: The point where the ray hit the face
        face_normal: The face's normal vector
        grid_size: Current grid size

    Returns:
        Vector: Snapped position on the mesh surface
    """
    normal_abs = Vector([abs(n) for n in face_normal])
    n = face_normal
    p0 = hit_location

    # Determine dominant axis and snap the perpendicular coordinates.
    # Then solve for the dominant coordinate using the plane equation:
    # n · (p - p0) = 0
    if normal_abs.x >= normal_abs.y and normal_abs.x >= normal_abs.z:
        # X-dominant: snap Y and Z, solve for X
        y_snap = round(hit_location.y / grid_size) * grid_size
        z_snap = round(hit_location.z / grid_size) * grid_size
        x = p0.x - (n.y / n.x) * (y_snap - p0.y) - (n.z / n.x) * (z_snap - p0.z)
        snapped = Vector((x, y_snap, z_snap))
    elif normal_abs.y >= normal_abs.x and normal_abs.y >= normal_abs.z:
        # Y-dominant: snap X and Z, solve for Y
        x_snap = round(hit_location.x / grid_size) * grid_size
        z_snap = round(hit_location.z / grid_size) * grid_size
        y = p0.y - (n.x / n.y) * (x_snap - p0.x) - (n.z / n.y) * (z_snap - p0.z)
        snapped = Vector((x_snap, y, z_snap))
    else:
        # Z-dominant: snap X and Y, solve for Z
        x_snap = round(hit_location.x / grid_size) * grid_size
        y_snap = round(hit_location.y / grid_size) * grid_size
        z = p0.z - (n.x / n.z) * (x_snap - p0.x) - (n.y / n.z) * (y_snap - p0.z)
        snapped = Vector((x_snap, y_snap, z))

    return snapped


def snap_relative_to_origin(point, origin, grid_size, local_x, local_y):
    """
    Snap a point relative to an origin point using local axes.

    This is used for second vertex and depth snapping.
    Instead of snapping to absolute grid positions, we snap the
    distance from the origin to grid increments.

    NOTE: This method is intentionally isolated for future modifications.

    Args:
        point: The point to snap
        origin: The reference origin (first vertex)
        grid_size: Current grid size
        local_x: Local X axis for the rectangle
        local_y: Local Y axis for the rectangle

    Returns:
        Vector: Snapped position
    """
    # Calculate offset from origin in local space
    offset = point - origin

    # Project onto local axes
    dx = offset.dot(local_x)
    dy = offset.dot(local_y)

    # Snap the distances
    snapped_dx = round(dx / grid_size) * grid_size
    snapped_dy = round(dy / grid_size) * grid_size

    # Reconstruct the snapped point
    return origin + local_x * snapped_dx + local_y * snapped_dy


def snap_depth_relative(depth, grid_size):
    """
    Snap depth value to grid increments.

    Args:
        depth: The raw depth value
        grid_size: Current grid size

    Returns:
        float: Snapped depth value
    """
    return round(depth / grid_size) * grid_size


def calculate_first_vertex_snap_3d(context, event):
    """
    Calculate the snapped position for the first vertex in 3D view.

    Args:
        context: Blender context
        event: Mouse event

    Returns:
        tuple: (snapped_position, face_normal, hit_object, was_clamped) or (None, None, None, False) if no valid hit
               was_clamped is True if the position had to be clamped to the face edge
    """
    hit, location, normal, face_index, obj, matrix = utils.raycast_scene(context, event)

    if not hit or location is None or normal is None:
        return (None, None, None, False)

    grid_size = utils.get_grid_size(context)
    was_clamped = False

    if utils.is_snapping_enabled(context):
        snapped = snap_to_grid_on_face(location, normal, grid_size)

        # Clamp snapped position to face boundary
        face_verts = utils.get_face_vertices_world(obj, face_index)
        if face_verts is not None and len(face_verts) >= 3:
            clamped = utils.clamp_point_to_face(snapped, face_verts, normal)
            # Check if clamping changed the position
            if (clamped - snapped).length_squared > 1e-10:
                was_clamped = True
                snapped = clamped
    else:
        # Without snapping, use the exact hit location (on the mesh surface)
        snapped = location.copy()

    return (snapped, normal, obj, was_clamped)


def calculate_first_vertex_snap_2d(context, event):
    """
    Calculate the snapped position for the first vertex in 2D view.

    Args:
        context: Blender context
        event: Mouse event

    Returns:
        tuple: (snapped_position, plane_normal) or (None, None) if invalid
    """
    plane_data = utils.get_2d_view_plane(context)
    if plane_data is None:
        return (None, None)

    plane_point, plane_normal, axis1, axis2 = plane_data

    point = utils.mouse_to_3d_on_plane(context, event, plane_point, plane_normal)
    if point is None:
        return (None, None)

    grid_size = utils.get_grid_size(context)

    if utils.is_snapping_enabled(context):
        snapped = snap_to_grid(point, grid_size)
    else:
        snapped = point

    return (snapped, plane_normal)


def calculate_second_vertex_snap(context, event, first_vertex, local_x, local_y, plane_point, plane_normal):
    """
    Calculate the snapped position for the second vertex.

    Uses relative snapping from the first vertex.

    Args:
        context: Blender context
        event: Mouse event
        first_vertex: The first vertex position
        local_x: Rectangle's local X axis
        local_y: Rectangle's local Y axis
        plane_point: A point on the rectangle's plane
        plane_normal: The rectangle's plane normal

    Returns:
        Vector or None: Snapped position, or None if invalid
    """
    point = utils.mouse_to_3d_on_plane(context, event, plane_point, plane_normal)
    if point is None:
        return None

    grid_size = utils.get_grid_size(context)

    if utils.is_snapping_enabled(context):
        return snap_relative_to_origin(point, first_vertex, grid_size, local_x, local_y)
    else:
        return point


def calculate_depth_from_mouse_3d(context, event, first_vertex, second_vertex, local_z, initial_mouse_pos):
    """
    Calculate depth based on geometric mouse movement in 3D view.

    Uses mouse movement projected onto the depth axis direction as seen from
    the camera. Moving mouse "into" the face (toward face center when looking
    at the face) gives negative depth, moving "away" gives positive depth.

    Args:
        context: Blender context
        event: Mouse event
        first_vertex: First corner of rectangle
        second_vertex: Second corner of rectangle
        local_z: Depth axis direction (points "outward" from rectangle)
        initial_mouse_pos: (x, y) tuple of mouse position when depth phase started

    Returns:
        float: Depth value (can be negative)
    """
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    region = context.region
    rv3d = context.region_data

    if rv3d is None:
        return 0.0

    # Get rectangle center
    rect_center = (first_vertex + second_vertex) / 2

    # Project rectangle center and a point along depth axis to screen space
    # depth_dir_length tells us how many pixels = 1 world unit of depth
    center_2d = location_3d_to_region_2d(region, rv3d, rect_center)
    depth_point_3d = rect_center + local_z  # Point 1 unit in depth direction
    depth_point_2d = location_3d_to_region_2d(region, rv3d, depth_point_3d)

    if center_2d is None or depth_point_2d is None:
        return 0.0

    # Get the screen-space vector for 1 unit of depth
    depth_dir_2d = Vector((depth_point_2d[0] - center_2d[0], depth_point_2d[1] - center_2d[1]))
    depth_dir_length = depth_dir_2d.length

    # Minimum threshold to avoid division issues when looking edge-on
    MIN_DEPTH_PROJECTION = 5.0  # pixels

    if depth_dir_length < MIN_DEPTH_PROJECTION:
        # Depth axis barely visible on screen - use fallback horizontal movement
        # Scale based on view distance for consistent feel
        if rv3d.is_perspective:
            view_pos = rv3d.view_matrix.inverted().translation
            dist = (view_pos - rect_center).length
            if dist < 0.001:
                dist = 10.0
            pixels_per_unit = region.width / (dist * 2.0)
        else:
            pixels_per_unit = region.width / (rv3d.view_distance * 2)

        if pixels_per_unit < 1.0:
            pixels_per_unit = 50.0

        # Use horizontal mouse movement as fallback
        initial_x, initial_y = initial_mouse_pos
        delta_x = event.mouse_region_x - initial_x
        depth = delta_x / pixels_per_unit
    else:
        # Normal case: project mouse movement onto depth axis direction
        depth_dir_2d /= depth_dir_length

        # Calculate mouse delta from initial position
        initial_x, initial_y = initial_mouse_pos
        delta = Vector((event.mouse_region_x - initial_x, event.mouse_region_y - initial_y))

        # Project mouse delta onto the depth direction in screen space
        projected_delta = delta.dot(depth_dir_2d)

        # depth_dir_length is pixels per world unit, so divide to get world units
        depth = projected_delta / depth_dir_length

    # Apply snapping
    grid_size = utils.get_grid_size(context)
    if utils.is_snapping_enabled(context):
        depth = snap_depth_relative(depth, grid_size)

    return depth


def calculate_depth_from_mouse_2d(context, event, initial_mouse_x):
    """
    Calculate depth based on mouse movement in 2D view.

    Right = outward (positive depth), Left = inward (negative depth)

    Args:
        context: Blender context
        event: Mouse event
        initial_mouse_x: Mouse X position when depth phase started

    Returns:
        float: Depth value (can be negative)
    """
    region = context.region
    delta_x = event.mouse_region_x - initial_mouse_x

    # Scale: full screen width = 10 grid units (hardcoded, can iterate)
    grid_size = utils.get_grid_size(context)
    screen_to_depth_scale = (10 * grid_size) / region.width

    depth = delta_x * screen_to_depth_scale

    # Apply snapping
    if utils.is_snapping_enabled(context):
        depth = snap_depth_relative(depth, grid_size)

    return depth
