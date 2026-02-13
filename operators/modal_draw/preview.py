"""
Modal Draw - GPU Drawing / Preview

Handles all visual feedback: snap indicators, rectangle preview, cuboid preview.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import utils


# Colors for visual feedback
COLOR_SNAP_POINT = (1.0, 1.0, 1.0, 1.0)      # White - snap indicator
COLOR_RECTANGLE = (1.0, 0.5, 0.0, 0.9)        # Orange - rectangle preview
COLOR_CUBOID = (1.0, 0.5, 0.0, 0.9)           # Orange - cuboid preview
COLOR_DEPTH_INDICATOR = (0.0, 1.0, 0.5, 0.9)  # Green - depth direction
COLOR_GRID_POINT = (0.7, 0.7, 0.7)            # Light grey (no alpha - set per-point)

POINT_SIZE = 10.0
LINE_WIDTH = 2.0
GRID_LINE_WIDTH = 1.5

# Grid overlay settings
GRID_FADE_RADIUS = 5  # Number of grid cells before fully faded
GRID_EXTENT = 6       # Number of grid cells to draw in each direction from cursor


class ModalDrawPreview:
    """
    Manages all preview drawing for modal draw tools.

    Registers draw handlers for all 3D viewports and draws based on current state.
    """

    def __init__(self):
        self._handlers = []
        self._state = 'NONE'  # 'NONE', 'FIRST_VERTEX', 'SECOND_VERTEX', 'DEPTH'

        # State data
        self._snap_point = None        # Current snap preview point
        self._snap_tangent1 = None     # First tangent axis for cross orientation
        self._snap_tangent2 = None     # Second tangent axis for cross orientation
        self._first_vertex = None      # Confirmed first vertex
        self._second_vertex = None     # Current/confirmed second vertex
        self._depth = 0.0              # Current depth
        self._local_x = None           # Rectangle local X axis
        self._local_y = None           # Rectangle local Y axis
        self._local_z = None           # Rectangle local Z axis (depth direction)

        # Face grid overlay data (for 3D view first vertex phase)
        self._grid_size = 1.0          # Current grid size for overlay
        self._face_plane_point = None  # Point on the face plane
        self._face_plane_normal = None # Face normal for grid orientation
        self._snap_was_clamped = False # Whether snap position was clamped to edge

    def register_handlers(self):
        """Register draw handlers for all 3D view spaces."""
        self.unregister_handlers()

        # Register for POST_VIEW (3D drawing)
        self._handlers.append(
            bpy.types.SpaceView3D.draw_handler_add(
                self._draw_3d, (), 'WINDOW', 'POST_VIEW'
            )
        )

    def unregister_handlers(self):
        """Remove all draw handlers."""
        for handler in self._handlers:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')
            except:
                pass
        self._handlers.clear()

    def set_state(self, state):
        """Set the current drawing state."""
        self._state = state

    def update_snap_point(self, point, tangent1, tangent2):
        """Update the snap preview point and its orientation axes."""
        self._snap_point = point
        self._snap_tangent1 = tangent1
        self._snap_tangent2 = tangent2

    def update_face_grid(self, plane_point, plane_normal, grid_size, snap_was_clamped):
        """
        Update the face grid overlay data for 3D view.

        Args:
            plane_point: A point on the face plane (for grid positioning)
            plane_normal: The face normal (for grid orientation)
            grid_size: Current grid size
            snap_was_clamped: True if snap position was clamped to face edge
        """
        self._face_plane_point = plane_point
        self._face_plane_normal = plane_normal
        self._grid_size = grid_size
        self._snap_was_clamped = snap_was_clamped

    def clear_face_grid(self):
        """Clear the face grid overlay data."""
        self._face_plane_point = None
        self._face_plane_normal = None
        self._snap_was_clamped = False

    def set_first_vertex(self, vertex, local_x, local_y, local_z):
        """Set the confirmed first vertex and local axes."""
        self._first_vertex = vertex
        self._local_x = local_x
        self._local_y = local_y
        self._local_z = local_z

    def update_second_vertex(self, vertex):
        """Update the current second vertex position."""
        self._second_vertex = vertex

    def update_depth(self, depth):
        """Update the current depth value."""
        self._depth = depth

    def clear(self):
        """Clear all preview data."""
        self._state = 'NONE'
        self._snap_point = None
        self._snap_tangent1 = None
        self._snap_tangent2 = None
        self._first_vertex = None
        self._second_vertex = None
        self._depth = 0.0
        self._local_x = None
        self._local_y = None
        self._local_z = None
        self._face_plane_point = None
        self._face_plane_normal = None
        self._grid_size = 1.0
        self._snap_was_clamped = False

    def _draw_3d(self):
        """Main 3D drawing callback."""
        if self._state == 'NONE':
            return

        # Set up GPU state - draw on top of scene for visibility
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')  # Always visible
        gpu.state.depth_mask_set(False)
        gpu.state.line_width_set(LINE_WIDTH)

        try:
            if self._state == 'FIRST_VERTEX':
                self._draw_face_grid()
                self._draw_snap_point()
            elif self._state == 'SECOND_VERTEX':
                self._draw_rectangle_preview()
            elif self._state == 'DEPTH':
                self._draw_cuboid_preview()
        finally:
            # Restore GPU state
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('NONE')
            gpu.state.depth_mask_set(True)
            gpu.state.line_width_set(1.0)

    def _draw_snap_point(self):
        """Draw the snap preview cross."""
        if self._snap_point is None:
            return

        # Use face tangents for uniform cross size (not stretched on angled faces)
        if self._face_plane_normal is not None:
            tangent1, tangent2 = utils.get_face_tangents(self._face_plane_normal)
        else:
            tangent1 = self._snap_tangent1
            tangent2 = self._snap_tangent2

        self._draw_cross(self._snap_point, tangent1, tangent2, COLOR_SNAP_POINT)

    def _draw_face_grid(self):
        """Draw a fading grid overlay on the face under the cursor.

        The grid shows world-aligned grid positions on the face. For each grid
        coordinate pair, we find the exact point on the face that has those
        coordinates (line-plane intersection along the dominant axis).
        """
        if self._snap_point is None:
            return
        if self._face_plane_point is None or self._face_plane_normal is None:
            return
        if self._snap_tangent1 is None or self._snap_tangent2 is None:
            return

        grid_size = self._grid_size
        if grid_size <= 0:
            return

        fade_center = self._snap_point
        axis1 = self._snap_tangent1
        axis2 = self._snap_tangent2
        normal = self._face_plane_normal.normalized()
        p0 = self._face_plane_point

        # Determine which axis is dominant (closest to face normal)
        # This determines which coordinate we solve for via line-plane intersection
        normal_abs = Vector([abs(n) for n in normal])
        if normal_abs.x >= normal_abs.y and normal_abs.x >= normal_abs.z:
            dominant = 'X'
        elif normal_abs.y >= normal_abs.x and normal_abs.y >= normal_abs.z:
            dominant = 'Y'
        else:
            dominant = 'Z'

        # Find base grid coordinates (nearest grid intersection to fade center)
        center_coord1 = fade_center.dot(axis1)
        center_coord2 = fade_center.dot(axis2)
        base_coord1 = round(center_coord1 / grid_size) * grid_size
        base_coord2 = round(center_coord2 / grid_size) * grid_size

        # Compute orthonormal tangent vectors in the face plane for drawing crosses.
        # These ensure crosses appear uniform (not stretched) on angled faces.
        face_tan1, face_tan2 = utils.get_face_tangents(normal)

        grid_positions = []
        grid_colors = []
        cross_size = grid_size * 0.15

        for i in range(-GRID_EXTENT, GRID_EXTENT + 1):
            for j in range(-GRID_EXTENT, GRID_EXTENT + 1):
                coord1 = base_coord1 + i * grid_size
                coord2 = base_coord2 + j * grid_size

                # Find the point on the face with these grid coordinates.
                grid_point = self._compute_grid_point_on_face(
                    coord1, coord2, dominant, normal, p0
                )
                if grid_point is None:
                    continue

                # Calculate fade based on distance from cursor
                dist = (grid_point - fade_center).length
                dist_in_grid_units = dist / grid_size

                if dist_in_grid_units >= GRID_FADE_RADIUS:
                    continue
                if dist_in_grid_units < 0.3 and not self._snap_was_clamped:
                    continue

                alpha = 0.6 * (1.0 - (dist_in_grid_units / GRID_FADE_RADIUS))
                color = (COLOR_GRID_POINT[0], COLOR_GRID_POINT[1], COLOR_GRID_POINT[2], alpha)

                # Build cross endpoints using face tangents (uniform size on face)
                p1 = grid_point + face_tan1 * cross_size
                p2 = grid_point - face_tan1 * cross_size
                p3 = grid_point + face_tan2 * cross_size
                p4 = grid_point - face_tan2 * cross_size

                grid_positions.extend([p1[:], p2[:], p3[:], p4[:]])
                grid_colors.extend([color, color, color, color])

        if len(grid_positions) < 2:
            return

        try:
            shader = gpu.shader.from_builtin('POLYLINE_SMOOTH_COLOR')
            region = bpy.context.region
            if region is None:
                return
            shader.uniform_float("viewportSize", (region.width, region.height))
            shader.uniform_float("lineWidth", GRID_LINE_WIDTH)
            batch = batch_for_shader(shader, 'LINES', {
                "pos": grid_positions,
                "color": grid_colors
            })
            batch.draw(shader)
        except Exception:
            try:
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                avg_alpha = sum(c[3] for c in grid_colors) / len(grid_colors) if grid_colors else 0.3
                shader.uniform_float("color", (COLOR_GRID_POINT[0], COLOR_GRID_POINT[1], COLOR_GRID_POINT[2], avg_alpha))
                batch = batch_for_shader(shader, 'LINES', {"pos": grid_positions})
                batch.draw(shader)
            except Exception:
                pass

    def _compute_grid_point_on_face(self, coord1, coord2, dominant, normal, p0):
        """
        Find the point on the face plane with the given grid coordinates.

        For a face with X-dominant normal, coord1/coord2 are Y/Z values.
        We solve the plane equation for X.

        Args:
            coord1: First grid coordinate (Y for X-dominant, X for Y/Z-dominant)
            coord2: Second grid coordinate (Z for X/Y-dominant, Y for Z-dominant)
            dominant: 'X', 'Y', or 'Z' - which axis is closest to face normal
            normal: Face normal vector
            p0: A point on the face plane

        Returns:
            Vector: Point on face with exact grid coordinates, or None if degenerate
        """
        if dominant == 'X':
            # coord1=Y, coord2=Z, solve for X
            if abs(normal.x) < 1e-10:
                return None
            x = p0.x - (normal.y / normal.x) * (coord1 - p0.y) - (normal.z / normal.x) * (coord2 - p0.z)
            return Vector((x, coord1, coord2))
        elif dominant == 'Y':
            # coord1=X, coord2=Z, solve for Y
            if abs(normal.y) < 1e-10:
                return None
            y = p0.y - (normal.x / normal.y) * (coord1 - p0.x) - (normal.z / normal.y) * (coord2 - p0.z)
            return Vector((coord1, y, coord2))
        else:
            # coord1=X, coord2=Y, solve for Z
            if abs(normal.z) < 1e-10:
                return None
            z = p0.z - (normal.x / normal.z) * (coord1 - p0.x) - (normal.y / normal.z) * (coord2 - p0.y)
            return Vector((coord1, coord2, z))

    def _draw_cross(self, position, tangent1, tangent2, color):
        """Draw a cross at the given position oriented along tangent axes."""
        if tangent1 is None or tangent2 is None:
            # Fallback: draw a small world-aligned cross
            tangent1 = Vector((1, 0, 0))
            tangent2 = Vector((0, 1, 0))

        # Cross arm half-length (in world units)
        cross_size = 0.1

        # Build the four endpoints of the cross
        p1 = position + tangent1 * cross_size
        p2 = position - tangent1 * cross_size
        p3 = position + tangent2 * cross_size
        p4 = position - tangent2 * cross_size

        line_points = [p1[:], p2[:], p3[:], p4[:]]

        try:
            shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')

            region = bpy.context.region
            if region is None:
                return
            shader.uniform_float("viewportSize", (region.width, region.height))
            shader.uniform_float("lineWidth", LINE_WIDTH)
            shader.uniform_float("color", color)

            batch = batch_for_shader(shader, 'LINES', {"pos": line_points})
            batch.draw(shader)
        except Exception:
            # Fallback to simple lines
            try:
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                shader.uniform_float("color", color)

                batch = batch_for_shader(shader, 'LINES', {"pos": line_points})
                batch.draw(shader)
            except Exception:
                pass  # Silent fail

    def _draw_rectangle_preview(self):
        """Draw the rectangle preview from first to second vertex."""
        if self._first_vertex is None or self._second_vertex is None:
            return
        if self._local_x is None or self._local_y is None:
            return

        corners = self._get_rectangle_corners(self._first_vertex, self._second_vertex)
        self._draw_line_loop(corners, COLOR_RECTANGLE)

    def _draw_cuboid_preview(self):
        """Draw the full cuboid preview."""
        if self._first_vertex is None or self._second_vertex is None:
            return
        if self._local_x is None or self._local_y is None or self._local_z is None:
            return

        vertices = utils.build_cuboid_vertices(
            self._first_vertex,
            self._second_vertex,
            self._depth,
            self._local_x,
            self._local_y,
            self._local_z
        )

        edges = utils.get_cuboid_edges()
        self._draw_edges(vertices, edges, COLOR_CUBOID)

    def _get_rectangle_corners(self, corner1, corner2):
        """Get the 4 corners of the rectangle."""
        diff = corner2 - corner1
        local_dx = diff.dot(self._local_x)
        local_dy = diff.dot(self._local_y)

        bl = corner1.copy()
        br = corner1 + self._local_x * local_dx
        tr = corner1 + self._local_x * local_dx + self._local_y * local_dy
        tl = corner1 + self._local_y * local_dy

        return [bl, br, tr, tl]

    def _draw_line_loop(self, points, color):
        """Draw a closed line loop through the given points."""
        if len(points) < 2:
            return

        # Close the loop
        line_points = [p[:] for p in points] + [points[0][:]]

        try:
            shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')

            # Get viewport size for line shader
            region = bpy.context.region
            if region is None:
                return
            shader.uniform_float("viewportSize", (region.width, region.height))
            shader.uniform_float("lineWidth", LINE_WIDTH)
            shader.uniform_float("color", color)

            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": line_points})
            batch.draw(shader)
        except Exception:
            # Fallback to simple lines if POLYLINE shader fails
            try:
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                shader.uniform_float("color", color)

                edges = []
                for i in range(len(points)):
                    edges.append(points[i][:])
                    edges.append(points[(i + 1) % len(points)][:])

                batch = batch_for_shader(shader, 'LINES', {"pos": edges})
                batch.draw(shader)
            except Exception:
                pass  # Silent fail if drawing is not possible

    def _draw_edges(self, vertices, edges, color):
        """Draw edges of a shape."""
        if len(vertices) < 2 or len(edges) < 1:
            return

        # Build line segments
        line_points = []
        for start_idx, end_idx in edges:
            line_points.append(vertices[start_idx][:])
            line_points.append(vertices[end_idx][:])

        try:
            shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')

            region = bpy.context.region
            if region is None:
                return
            shader.uniform_float("viewportSize", (region.width, region.height))
            shader.uniform_float("lineWidth", LINE_WIDTH)
            shader.uniform_float("color", color)

            batch = batch_for_shader(shader, 'LINES', {"pos": line_points})
            batch.draw(shader)
        except Exception:
            # Fallback
            try:
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                shader.uniform_float("color", color)

                batch = batch_for_shader(shader, 'LINES', {"pos": line_points})
                batch.draw(shader)
            except Exception:
                pass  # Silent fail


# Global preview instance (created by operator)
_preview_instance = None


def get_preview():
    """Get the global preview instance."""
    global _preview_instance
    if _preview_instance is None:
        _preview_instance = ModalDrawPreview()
    return _preview_instance


def cleanup_preview():
    """Clean up the global preview instance."""
    global _preview_instance
    if _preview_instance is not None:
        _preview_instance.unregister_handlers()
        _preview_instance.clear()
        _preview_instance = None
