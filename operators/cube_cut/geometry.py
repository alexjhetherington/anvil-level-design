"""
Cube Cut Tool - Custom Intersection Geometry

Custom intersection algorithm that:
1. Finds intersection vertices (edge-plane and cuboid-face)
2. Splits edges at intersection points
3. Removes faces inside the cuboid
4. Reconstructs partial faces
"""

import bmesh
from mathutils import Vector
from mathutils.geometry import intersect_line_plane

from ...utils import compute_uv_projection_from_face, apply_uv_projection_to_face, debug_log


# Epsilon for floating point comparisons
EPSILON = 1e-5


class CuboidPlanes:
    """Represents the 6 planes of a cuboid for intersection testing."""

    def __init__(self, first_vertex, second_vertex, depth, local_x, local_y, local_z):
        # Calculate rectangle dimensions
        diff = second_vertex - first_vertex
        dx = diff.dot(local_x)
        dy = diff.dot(local_y)

        # Normalize direction so min/max work correctly
        if dx < 0:
            dx = -dx
            local_x = -local_x
        if dy < 0:
            dy = -dy
            local_y = -local_y

        # Handle depth direction
        if depth < 0:
            self.depth_min = depth
            self.depth_max = 0
        else:
            self.depth_min = 0
            self.depth_max = depth

        self.local_x = local_x
        self.local_y = local_y
        self.local_z = local_z
        self.dx = dx
        self.dy = dy
        self.origin = first_vertex.copy()

        # Track which plane index is the "rectangle plane" (at depth=0)
        if depth >= 0:
            self.rectangle_plane_idx = 0
        else:
            self.rectangle_plane_idx = 1

        # Build the 6 planes: (point_on_plane, outward_normal)
        self.planes = self._build_planes()

    def _build_planes(self):
        """Build the 6 bounding planes with outward-pointing normals."""
        planes = []

        # Plane 0: "Front" at depth_min
        planes.append((
            self.origin + self.local_z * self.depth_min,
            -self.local_z.copy()
        ))

        # Plane 1: "Back" at depth_max
        planes.append((
            self.origin + self.local_z * self.depth_max,
            self.local_z.copy()
        ))

        # Plane 2: "Left" at x=0
        planes.append((
            self.origin.copy(),
            -self.local_x.copy()
        ))

        # Plane 3: "Right" at x=dx
        planes.append((
            self.origin + self.local_x * self.dx,
            self.local_x.copy()
        ))

        # Plane 4: "Bottom" at y=0
        planes.append((
            self.origin.copy(),
            -self.local_y.copy()
        ))

        # Plane 5: "Top" at y=dy
        planes.append((
            self.origin + self.local_y * self.dy,
            self.local_y.copy()
        ))

        return planes

    def point_inside(self, point):
        """Test if a point is inside the cuboid (or on boundary)."""
        local = self.to_local(point)
        x, y, z = local.x, local.y, local.z

        return (
            -EPSILON <= x <= self.dx + EPSILON and
            -EPSILON <= y <= self.dy + EPSILON and
            self.depth_min - EPSILON <= z <= self.depth_max + EPSILON
        )

    def point_strictly_inside(self, point):
        """Test if a point is strictly inside the cuboid (not on boundary)."""
        local = self.to_local(point)
        x, y, z = local.x, local.y, local.z

        return (
            EPSILON < x < self.dx - EPSILON and
            EPSILON < y < self.dy - EPSILON and
            self.depth_min + EPSILON < z < self.depth_max - EPSILON
        )

    def to_local(self, point):
        """Convert point to local cuboid coordinates (x, y, z)."""
        offset = point - self.origin
        return Vector((
            offset.dot(self.local_x),
            offset.dot(self.local_y),
            offset.dot(self.local_z)
        ))


def execute_cube_cut(context, first_vertex, second_vertex, depth, local_x, local_y, local_z):
    """
    Execute the cube cut operation.

    Algorithm:
    1. Find all intersection points (edge-plane and cuboid-face)
    2. Split mesh edges at intersection points
    3. Delete faces entirely inside the cuboid
    4. Reconstruct faces that are partially inside
    """
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        return (False, "No active mesh object")

    # Handle zero depth
    effective_depth = depth
    if abs(depth) < EPSILON:
        effective_depth = EPSILON * 2 if depth >= 0 else -EPSILON * 2

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # Get pixels per meter for UV calculations
    props = context.scene.level_design_props
    ppm = props.pixels_per_meter

    # Transform cuboid to object local space
    world_to_local = obj.matrix_world.inverted()

    local_first = world_to_local @ first_vertex
    local_second = world_to_local @ second_vertex
    local_x_trans = (world_to_local.to_3x3() @ local_x).normalized()
    local_y_trans = (world_to_local.to_3x3() @ local_y).normalized()
    local_z_trans = (world_to_local.to_3x3() @ local_z).normalized()

    scale_factor = (world_to_local.to_3x3() @ local_z).length
    local_depth = effective_depth * scale_factor

    cuboid = CuboidPlanes(
        local_first, local_second, local_depth,
        local_x_trans, local_y_trans, local_z_trans
    )

    # === STEP 1: Determine which faces will be cut (have interior intersections) ===
    # We must do this BEFORE splitting edges, so we only split edges belonging to faces that will be cut
    debug_log(f"\n[CubeCut] === STEP 1: Find cuboid-face intersections ===")
    bm.faces.ensure_lookup_table()
    face_interior_points = _find_cuboid_face_intersections(bm, cuboid)

    # Apply selection filter: only cut selected faces (or all if none selected)
    any_faces_selected = any(f.select for f in bm.faces if f.is_valid)
    if any_faces_selected:
        debug_log(f"[CubeCut] Selection mode: only cutting selected faces")
    else:
        debug_log(f"[CubeCut] No faces selected: cutting all intersecting faces")

    # Determine which faces will actually be cut
    faces_to_be_cut = set()
    skipped_unselected_count = 0
    for face_idx, points in face_interior_points.items():
        face = bm.faces[face_idx] if face_idx < len(bm.faces) else None
        if face is None or not face.is_valid:
            continue
        if not points:
            continue

        # Only cut selected faces (unless no faces are selected, then cut all)
        if any_faces_selected and not face.select:
            skipped_unselected_count += 1
            debug_log(f"[CubeCut] Face {face_idx} skipped (not selected)")
            continue

        faces_to_be_cut.add(face)
        debug_log(f"[CubeCut] Face {face_idx} will be cut ({len(points)} interior points)")

    debug_log(f"[CubeCut] Faces to be cut: {len(faces_to_be_cut)}")
    if skipped_unselected_count > 0:
        debug_log(f"[CubeCut] Skipped {skipped_unselected_count} unselected faces")

    # === STEP 2: Find edge-plane intersections and split edges ===
    # Only split edges that belong to faces that will be cut
    debug_log(f"\n[CubeCut] === STEP 2: Find edge-plane intersections ===")
    edge_splits = _find_edge_plane_intersections(bm, cuboid, faces_to_be_cut)
    debug_log(f"[CubeCut] Found {len(edge_splits)} edges to split")

    # Split edges (must do this before face operations)
    # Also track which faces had their edges split
    split_verts, faces_with_split_edges = _split_edges_at_intersections(bm, edge_splits)
    debug_log(f"[CubeCut] Created {len(split_verts)} split vertices")
    debug_log(f"[CubeCut] Faces with split edges: {len(faces_with_split_edges)}")

    # Debug: print all edges in mesh
    debug_log(f"[CubeCut] === ALL EDGES AFTER SPLITS ===")
    bm.edges.ensure_lookup_table()
    for e in bm.edges:
        if e.is_valid:
            debug_log(f"[CubeCut]   Edge id={id(e)}: {e.verts[0].co[:]} -> {e.verts[1].co[:]}")

    # Debug: print face loops
    debug_log(f"[CubeCut] === FACE LOOPS AFTER SPLITS ===")
    bm.faces.ensure_lookup_table()
    for f in bm.faces:
        if f.is_valid:
            debug_log(f"[CubeCut] Face {f.index}:")
            for loop in f.loops:
                debug_log(f"[CubeCut]   Loop: vert={loop.vert.co[:]} -> edge={loop.edge.verts[0].co[:]}->{loop.edge.verts[1].co[:]}")

    # === STEP 3: Create interior vertices for faces to be cut ===
    debug_log(f"\n[CubeCut] === STEP 3: Create interior vertices ===")
    face_interior_verts = []  # List of (face, interior_verts) tuples
    for face in faces_to_be_cut:
        if face is None or not face.is_valid:
            continue

        points = face_interior_points.get(face.index, [])
        if not points:
            continue

        interior_verts = []
        for point in points:
            new_vert = bm.verts.new(point)
            interior_verts.append(new_vert)
            debug_log(f"[CubeCut] VERTEX CREATED (interior): pos={point}, for face {face.index}")
            debug_log(f"[CubeCut]   No edges created yet (floating vertex)")

        if interior_verts:
            face_interior_verts.append((face, interior_verts))

    debug_log(f"[CubeCut] Total interior vertices created: {sum(len(v) for _, v in face_interior_verts)}")


    # === STEP 4: Connect interior vertices to face boundaries ===
    # For faces with interior vertices, we need to split the face
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    faces_actually_cut = set()
    split_verts_set = set(split_verts)
    for face, interior_verts in face_interior_verts:
        if face is None or not face.is_valid:
            continue
        # Find edge-split vertices that are on this face's boundary and on cuboid boundary
        edge_verts_on_face = []
        for v in face.verts:
            if v in split_verts_set:
                edge_verts_on_face.append(v)
        _connect_new_verts_to_face(bm, face, interior_verts, edge_verts_on_face, cuboid, me, ppm)
        faces_actually_cut.add(face)

    # === STEP 5: Quadrilate n-gons created by edge splits ===
    # Faces that had edges split but weren't cut are now n-gons and need to be quadrilated
    faces_to_quadrilate = []
    for face in faces_with_split_edges:
        if not face.is_valid:
            continue
        if face in faces_actually_cut:
            # This face was cut - it's been replaced with new faces, skip
            continue
        if len(face.verts) > 4:
            # This is an n-gon that needs quadrilating
            faces_to_quadrilate.append(face)
            debug_log(f"[CubeCut] Face needs quadrilating: {len(face.verts)} verts")

    if faces_to_quadrilate:
        debug_log(f"[CubeCut] === STEP 5: Quadrilating {len(faces_to_quadrilate)} n-gon faces ===")
        # Triangulate the n-gons first
        result = bmesh.ops.triangulate(bm, faces=faces_to_quadrilate)
        new_faces = result['faces']
        debug_log(f"[CubeCut] Triangulated into {len(new_faces)} triangles")

        # Join triangles into quads where possible
        if new_faces:
            bmesh.ops.join_triangles(
                bm,
                faces=new_faces,
                angle_face_threshold=3.14159,  # ~180 degrees - allow any face angle
                angle_shape_threshold=3.14159  # ~180 degrees - allow any shape
            )
            debug_log(f"[CubeCut] Joined triangles into quads where possible")

    # === STEP 6: Cleanup ===
    # Remove loose vertices
    loose_verts = [v for v in bm.verts if v.is_valid and not v.link_faces]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')

    # Merge very close vertices
    bmesh.ops.remove_doubles(bm, verts=[v for v in bm.verts if v.is_valid], dist=EPSILON)

    bmesh.update_edit_mesh(me)

    return (True, "Cut complete")


def _find_edge_plane_intersections(bm, cuboid, faces_to_cut):
    """
    Find all points where mesh edges cross cuboid planes.

    Only considers edges that belong to at least one face in faces_to_cut.
    This prevents adding edge splits to faces that won't actually be cut.

    Args:
        bm: BMesh
        cuboid: CuboidPlanes instance
        faces_to_cut: Set of BMFace that will be cut (have interior intersections)

    Returns:
        dict: edge -> list of (intersection_point, plane_idx)
    """
    edge_splits = {}

    debug_log(f"[CubeCut] Checking {len(bm.edges)} edges for plane intersections")

    for edge in bm.edges:
        if not edge.is_valid:
            continue

        # Only split edges that belong to faces that will be cut
        edge_belongs_to_cut_face = any(f in faces_to_cut for f in edge.link_faces)
        if not edge_belongs_to_cut_face:
            continue

        v1_co = edge.verts[0].co
        v2_co = edge.verts[1].co

        intersections = []

        for plane_idx, (plane_point, plane_normal) in enumerate(cuboid.planes):
            d1 = (v1_co - plane_point).dot(plane_normal)
            d2 = (v2_co - plane_point).dot(plane_normal)

            # Edge crosses plane only if endpoints are on strictly opposite sides
            crosses = (d1 > EPSILON and d2 < -EPSILON) or (d1 < -EPSILON and d2 > EPSILON)
            if not crosses:
                continue

            intersection = intersect_line_plane(v1_co, v2_co, plane_point, plane_normal)
            if intersection is None:
                continue

            # Check if intersection is within cuboid bounds on this plane
            within_bounds = _point_within_plane_bounds(intersection, plane_idx, cuboid)
            if not within_bounds:
                debug_log(f"[CubeCut] Edge {edge.index} crosses plane {plane_idx} at {intersection} but OUTSIDE bounds")
                continue

            # Calculate parameter t along edge (for ordering multiple splits)
            edge_vec = v2_co - v1_co
            t = (intersection - v1_co).dot(edge_vec) / edge_vec.length_squared

            intersections.append((intersection.copy(), plane_idx, t))
            debug_log(f"[CubeCut] Edge {edge.index} ({v1_co} -> {v2_co}) crosses plane {plane_idx} at {intersection}, t={t:.3f}")

        if intersections:
            # Sort by t parameter so we split in order from v1 to v2
            intersections.sort(key=lambda x: x[2])
            edge_splits[edge] = intersections

    return edge_splits


def _split_edges_at_intersections(bm, edge_splits):
    """
    Split edges at intersection points.

    Returns:
        tuple: (newly created vertices, set of faces that had edges split)
    """
    new_verts = []
    faces_with_split_edges = set()

    for edge, intersections in edge_splits.items():
        if not edge.is_valid:
            continue

        # Track all faces linked to this edge BEFORE splitting
        # (after split, edge.link_faces will only show faces on one segment)
        for face in edge.link_faces:
            if face.is_valid:
                faces_with_split_edges.add(face)

        # Split from end to start (reverse order) so indices stay valid
        # We need to track the original v1 endpoint to keep splitting toward it
        original_v1 = edge.verts[0]
        original_v2 = edge.verts[1]
        current_edge = edge
        edges_to_keep = []  # Track all edges created by splits

        for intersection_point, plane_idx, t in reversed(intersections):
            if not current_edge.is_valid:
                break

            # Find the correct position along current edge segment
            # Ensure v1 is the original starting vertex we're splitting toward
            if current_edge.verts[0] == original_v1 or (
                current_edge.verts[0].co - original_v1.co
            ).length < EPSILON:
                v1 = current_edge.verts[0]
                v2 = current_edge.verts[1]
            else:
                v1 = current_edge.verts[1]
                v2 = current_edge.verts[0]

            # Recalculate t for current edge segment
            edge_vec = v2.co - v1.co
            if edge_vec.length_squared < EPSILON * EPSILON:
                continue

            new_t = (intersection_point - v1.co).dot(edge_vec) / edge_vec.length_squared
            new_t = max(0.01, min(0.99, new_t))  # Clamp to avoid degenerate splits

            # Split the edge
            old_edge_verts = (current_edge.verts[0].index, current_edge.verts[1].index)
            old_edge_coords = (current_edge.verts[0].co.copy(), current_edge.verts[1].co.copy())
            linked_faces = [f.index for f in current_edge.link_faces if f.is_valid]

            debug_log(f"[CubeCut] BEFORE edge_split:")
            debug_log(f"[CubeCut]   current_edge id={id(current_edge)} verts={old_edge_coords}")
            debug_log(f"[CubeCut]   edge (original) id={id(edge)} verts={[v.co[:] for v in edge.verts]}")
            debug_log(f"[CubeCut]   splitting at t={new_t:.3f}, intersection={intersection_point}")

            new_edge, new_vert = bmesh.utils.edge_split(current_edge, v1, new_t)
            new_vert.co = intersection_point.copy()  # Ensure exact position
            new_verts.append(new_vert)

            debug_log(f"[CubeCut] AFTER edge_split:")
            debug_log(f"[CubeCut]   new_vert pos={new_vert.co[:]}")
            debug_log(f"[CubeCut]   new_edge id={id(new_edge)} verts={[v.co[:] for v in new_edge.verts]}")
            debug_log(f"[CubeCut]   current_edge id={id(current_edge)} verts={[v.co[:] for v in current_edge.verts]} (same object as before split)")
            debug_log(f"[CubeCut]   edge (original) id={id(edge)} verts={[v.co[:] for v in edge.verts]}")
            debug_log(f"[CubeCut]   original_v1={original_v1.co[:]}, original_v2={original_v2.co[:]}")

            # After edge_split, find which edge contains original_v1 for next iteration
            # and which edge is the "far" segment to keep
            if new_edge.is_valid:
                if original_v1 in new_edge.verts:
                    debug_log(f"[CubeCut]   -> new_edge contains original_v1, setting current_edge = new_edge")
                    current_edge = new_edge
                else:
                    debug_log(f"[CubeCut]   -> new_edge does NOT contain original_v1, keeping current_edge")
                    edges_to_keep.append(new_edge)

    return new_verts, faces_with_split_edges


def _find_cuboid_face_intersections(bm, cuboid):
    """
    Find where cuboid edges pierce mesh face interiors.

    Returns:
        dict: face_index -> list of intersection points
    """
    face_interior_verts = {}

    # Build cuboid vertices and edges
    cuboid_verts = _build_cuboid_vertices_local(cuboid)
    cuboid_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting edges
    ]

    debug_log(f"[CubeCut] Checking {len(bm.faces)} faces for cuboid-face intersections")
    debug_log(f"[CubeCut] Cuboid vertices: {[str(v) for v in cuboid_verts]}")

    for face in bm.faces:
        if not face.is_valid:
            continue

        face_normal = face.normal
        if face_normal.length < EPSILON:
            continue
        face_point = face.verts[0].co
        face_verts = [v.co for v in face.verts]

        for edge_idx, (v1_idx, v2_idx) in enumerate(cuboid_edges):
            edge_start = cuboid_verts[v1_idx]
            edge_end = cuboid_verts[v2_idx]

            d1 = (edge_start - face_point).dot(face_normal)
            d2 = (edge_end - face_point).dot(face_normal)

            # Check if edge crosses the face plane (endpoints on opposite sides)
            crosses = (d1 > EPSILON and d2 < -EPSILON) or (d1 < -EPSILON and d2 > EPSILON)

            # Special case: endpoint ON the face plane, other endpoint on one side
            # Only allow exact alignment cutting for front/back faces (parallel to local_z)
            # Side faces (left/right/top/bottom) still require crossing through
            endpoint_on_face = None
            if not crosses:
                # Check if mesh face is parallel to front/back planes (normal parallel to local_z)
                face_parallel_to_depth = abs(abs(face_normal.dot(cuboid.local_z)) - 1.0) < EPSILON * 10

                if face_parallel_to_depth:
                    # Check if one endpoint is on the face plane
                    if abs(d1) <= EPSILON and abs(d2) > EPSILON:
                        # edge_start (v1) is on the face plane
                        endpoint_on_face = edge_start.copy()
                        debug_log(f"[CubeCut] Cuboid vertex {v1_idx} is ON face {face.index} (depth-aligned)")

                    elif abs(d2) <= EPSILON and abs(d1) > EPSILON:
                        # edge_end (v2) is on the face plane
                        endpoint_on_face = edge_end.copy()
                        debug_log(f"[CubeCut] Cuboid vertex {v2_idx} is ON face {face.index} (depth-aligned)")

            if not crosses and endpoint_on_face is None:
                if edge_idx >= 8:  # Connecting edges are indices 8-11
                    debug_log(f"[CubeCut] Cuboid edge {edge_idx} ({v1_idx}->{v2_idx}) did NOT cross face {face.index}: d1={d1:.4f}, d2={d2:.4f}")
                continue

            # Determine intersection point
            if endpoint_on_face is not None:
                intersection = endpoint_on_face
            else:
                intersection = intersect_line_plane(edge_start, edge_end, face_point, face_normal)
                if intersection is None:
                    continue

            debug_log(f"[CubeCut] Cuboid edge {edge_idx} intersects face {face.index} plane at {intersection}")

            # Check if inside face polygon (not on edge)
            in_polygon = _point_in_polygon(intersection, face_verts, face_normal)
            in_interior = _point_in_face_interior(intersection, face_verts, face_normal)
            debug_log(f"[CubeCut]   in_polygon={in_polygon}, in_interior={in_interior}")

            if not in_interior:
                continue

            if face.index not in face_interior_verts:
                face_interior_verts[face.index] = []
            face_interior_verts[face.index].append(intersection.copy())
            debug_log(f"[CubeCut]   Added interior intersection!")

    debug_log(f"[CubeCut] Found {len(face_interior_verts)} faces with interior intersections")
    return face_interior_verts


def _connect_new_verts_to_face(bm, face, interior_verts, edge_verts, cuboid, me, ppm):
    """
    Connect new vertices to form a hole in the face.

    Args:
        interior_verts: Vertices from cuboid corners piercing the face interior (need connecting)
        edge_verts: Vertices from edge splits on the face boundary (part of hole but already connected)

    When creating the hole boundary, edges are only created when at least one vertex is interior.
    """
    if not face.is_valid:
        return

    face_normal = face.normal.copy()
    face_material_index = face.material_index

    # Combine all new vertices for the hole boundary
    all_cut_verts = list(interior_verts) + list(edge_verts)
    interior_verts_set = set(interior_verts)

    debug_log(f"[CubeCut] Face {face.index}: {len(interior_verts)} interior verts + {len(edge_verts)} edge verts = {len(all_cut_verts)} total")

    if len(all_cut_verts) < 3:
        debug_log(f"[CubeCut] Not enough cut verts to form hole (need at least 3)")
        return

    # Get UV layer if it exists
    uv_layer = bm.loops.layers.uv.active

    # Compute UV projection from original face BEFORE deleting it
    # This captures the world-space projection axes that map 3D positions to UVs
    uv_projection = None
    if uv_layer:
        uv_projection = compute_uv_projection_from_face(face, uv_layer)

        if uv_projection:
            u_axis, v_axis, origin_uv, origin_pos, source_normal = uv_projection
            debug_log(f"[CubeCut] Computed UV projection from face {face.index}:")
            debug_log(f"[CubeCut]   u_axis={u_axis}, v_axis={v_axis}")
            debug_log(f"[CubeCut]   origin_uv={origin_uv}, origin_pos={origin_pos}")
        else:
            debug_log(f"[CubeCut] Failed to compute UV projection from face {face.index}")

    # Store the original face vertices in order (the outer boundary)
    outer_loop = list(face.verts)

    # Sort cut verts to form a proper polygon (the hole boundary)
    sorted_cut_verts = _sort_verts_by_angle(all_cut_verts, face_normal)

    debug_log(f"[CubeCut] Sorted cut verts: {[v.index for v in sorted_cut_verts]}")

    # Create edges between cut verts to form the hole boundary
    # Only create edges when at least one vertex is an interior vertex
    debug_log(f"[CubeCut] Creating hole boundary edges...")
    for i in range(len(sorted_cut_verts)):
        v1 = sorted_cut_verts[i]
        v2 = sorted_cut_verts[(i + 1) % len(sorted_cut_verts)]
        if v1.is_valid and v2.is_valid:
            # Skip edge if both vertices are non-interior (edge verts)
            v1_is_interior = v1 in interior_verts_set
            v2_is_interior = v2 in interior_verts_set
            if not v1_is_interior and not v2_is_interior:
                debug_log(f"[CubeCut] EDGE SKIPPED (both non-interior): ({v1.co}) -> ({v2.co})")
                continue

            edge_exists = any(v2 in e.verts for e in v1.link_edges)
            if not edge_exists:
                try:
                    bm.edges.new([v1, v2])
                    debug_log(f"[CubeCut] EDGE CREATED (hole boundary): ({v1.co}) -> ({v2.co})")
                except ValueError:
                    debug_log(f"[CubeCut] EDGE FAILED: ({v1.co}) -> ({v2.co})")
            else:
                debug_log(f"[CubeCut] EDGE EXISTS (skipped): ({v1.co}) -> ({v2.co})")

    # Now we need to create faces around the hole
    # Strategy: walk around the outer boundary, creating faces that connect to the inner boundary

    # Delete the original face first
    face_edges_before = [(e.verts[0].co.copy(), e.verts[1].co.copy()) for e in face.edges if e.is_valid]
    debug_log(f"[CubeCut] FACE DELETED: face had edges: {len(face_edges_before)}")
    bmesh.ops.delete(bm, geom=[face], context='FACES_ONLY')

    # Delete edges between non-interior vertices (created during edge splitting)
    # These edges shouldn't exist as part of the hole boundary
    # BUT only delete if no other faces depend on this edge
    edges_to_delete = []
    for i in range(len(sorted_cut_verts)):
        v1 = sorted_cut_verts[i]
        v2 = sorted_cut_verts[(i + 1) % len(sorted_cut_verts)]
        if v1.is_valid and v2.is_valid:
            v1_is_interior = v1 in interior_verts_set
            v2_is_interior = v2 in interior_verts_set
            if not v1_is_interior and not v2_is_interior:
                # Find edge between v1 and v2
                for e in v1.link_edges:
                    if e.is_valid and v2 in e.verts:
                        # Only delete if no other faces use this edge
                        # (the original face was already deleted, so link_faces shows remaining faces)
                        remaining_faces = [f for f in e.link_faces if f.is_valid]
                        if len(remaining_faces) == 0:
                            edges_to_delete.append(e)
                            debug_log(f"[CubeCut] EDGE MARKED FOR DELETION (both non-interior, no linked faces): ({v1.co}) -> ({v2.co})")
                        else:
                            debug_log(f"[CubeCut] EDGE KEPT (used by {len(remaining_faces)} other face(s)): ({v1.co}) -> ({v2.co})")
                        break

    if edges_to_delete:
        bmesh.ops.delete(bm, geom=edges_to_delete, context='EDGES')
        debug_log(f"[CubeCut] Deleted {len(edges_to_delete)} edges between non-interior vertices")

    # Identify which outer verts are "outside" the cut (to be kept) vs "inside" (to be removed)
    outer_kept = []  # Verts outside the cut region
    outer_cut = []   # Verts on the cuboid boundary (these are in sorted_cut_verts too)
    outer_inside = []  # Verts inside the cut region (to be deleted)

    # Set of actual cut vertices (edge splits and interior) - these are kept as transition points
    cut_verts_set = set(sorted_cut_verts)

    # Track vertices that are excluded from this face but kept for other faces
    excluded_verts = []

    for v in outer_loop:
        if v in cut_verts_set:
            # This is an actual cut boundary vertex (edge split or interior) - keep it
            outer_cut.append(v)
            outer_kept.append(v)
        elif cuboid.point_strictly_inside(v.co):
            # Strictly inside 3D cuboid - candidate for removal
            # But check if other faces still use this vertex
            other_faces = [f for f in v.link_faces if f.is_valid]
            if len(other_faces) > 0:
                # Other faces still use this vertex - DON'T delete it
                # But also DON'T include it in outer_kept or sorted_cut_verts
                # because it's inside the cut region for THIS face
                debug_log(f"[CubeCut] Vertex at {v.co} is inside cuboid but used by {len(other_faces)} other face(s) - keeping vertex but excluding from this face")
                excluded_verts.append(v)
            else:
                outer_inside.append(v)
                debug_log(f"[CubeCut] Vertex at {v.co} is strictly inside cuboid - removing")
        elif cuboid.point_inside(v.co):
            # On cuboid boundary but NOT a cut vertex - this is an original face vertex
            # that falls on/inside the cut region
            # But check if other faces still use this vertex
            other_faces = [f for f in v.link_faces if f.is_valid]
            if len(other_faces) > 0:
                # Other faces still use this vertex - DON'T delete it
                # But also DON'T include it in outer_kept or sorted_cut_verts
                # because it's inside the cut region for THIS face
                debug_log(f"[CubeCut] Vertex at {v.co} is on cuboid boundary, used by {len(other_faces)} other face(s) - keeping vertex but excluding from this face")
                excluded_verts.append(v)
            else:
                outer_inside.append(v)
                debug_log(f"[CubeCut] Vertex at {v.co} is on cuboid boundary (not a cut vert) - removing")
        else:
            # Outside - keep
            outer_kept.append(v)

    debug_log(f"[CubeCut] Outer loop: {len(outer_loop)} verts, {len(outer_kept)} kept, {len(outer_cut)} on cut boundary, {len(outer_inside)} inside cut, {len(excluded_verts)} excluded (kept for other faces)")

    # Delete vertices that are inside the cut region
    if outer_inside:
        bmesh.ops.delete(bm, geom=outer_inside, context='VERTS')
        debug_log(f"[CubeCut] Deleted {len(outer_inside)} vertices inside cut region")

    # Clean up edges from excluded vertices that are no longer used by any face
    # (these are edges that were part of the deleted face but connect to excluded verts)
    if excluded_verts:
        edges_to_cleanup = []
        for v in excluded_verts:
            if not v.is_valid:
                continue
            for e in v.link_edges:
                if e.is_valid:
                    # Check if this edge is used by any face
                    linked_faces = [f for f in e.link_faces if f.is_valid]
                    if len(linked_faces) == 0:
                        edges_to_cleanup.append(e)
                        debug_log(f"[CubeCut] EDGE CLEANUP (dangling from excluded vert): {e.verts[0].co[:]} -> {e.verts[1].co[:]}")
        if edges_to_cleanup:
            # Remove duplicates
            edges_to_cleanup = list(set(edges_to_cleanup))
            bmesh.ops.delete(bm, geom=edges_to_cleanup, context='EDGES')
            debug_log(f"[CubeCut] Cleaned up {len(edges_to_cleanup)} dangling edges from excluded vertices")

    # Create faces by walking around and connecting outer to inner (using kept verts only)
    new_faces = _create_faces_around_hole(bm, outer_kept, sorted_cut_verts, cuboid)

    # Apply UV projection to all new faces (same projection as original face)
    if uv_projection and uv_layer and new_faces:
        u_axis, v_axis, origin_uv, origin_pos, source_normal = uv_projection
        debug_log(f"[CubeCut] Applying UV projection to {len(new_faces)} new faces")
        for i, new_face in enumerate(new_faces):
            if new_face.is_valid:
                apply_uv_projection_to_face(
                    new_face, uv_layer,
                    u_axis, v_axis, origin_uv, origin_pos, source_normal
                )
                # Preserve material index
                new_face.material_index = face_material_index
                debug_log(f"[CubeCut]   Face {i}: {len(new_face.verts)} verts - UV projection applied")
    elif not uv_projection:
        debug_log(f"[CubeCut] No UV projection available - skipping UV application")
    elif not uv_layer:
        debug_log(f"[CubeCut] No UV layer - skipping UV application")
    elif not new_faces:
        debug_log(f"[CubeCut] No new faces created - skipping UV application")


def _face_spans_cut_edge(face_verts, outer_on_cut):
    """Check if a face would include an edge between two outer_on_cut vertices.

    This detects faces that would span across the open edge where the cut
    meets the mesh boundary.
    """
    n = len(face_verts)
    for i in range(n):
        v1 = face_verts[i]
        v2 = face_verts[(i + 1) % n]
        if v1 in outer_on_cut and v2 in outer_on_cut:
            return True
    return False


def _vert_pos_key(v):
    """Convert a vertex to a hashable position tuple (rounded for comparison)."""
    co = v.co if hasattr(v, 'co') else v
    return (round(co.x, 5), round(co.y, 5), round(co.z, 5))


def _verts_to_set(verts):
    """Convert a list of vertices to a frozenset of position keys."""
    return frozenset(_vert_pos_key(v) for v in verts)


def _is_subset_of_existing_face(new_verts, existing_face_vert_sets):
    """Check if new_verts are a subset of any existing face's vertices."""
    new_set = _verts_to_set(new_verts)
    for existing_set in existing_face_vert_sets:
        if new_set.issubset(existing_set):
            return True
    return False


def _create_faces_around_hole(bm, outer_loop, inner_loop, cuboid):
    """
    Create faces connecting the outer boundary to the inner hole boundary.

    Walks around both loops simultaneously, creating quads or triangles as needed.

    Returns:
        list: List of newly created BMFaces
    """
    n_outer = len(outer_loop)
    n_inner = len(inner_loop)

    debug_log(f"[CubeCut] Creating faces: outer={n_outer}, inner={n_inner}")

    created_faces = []
    # Track vertex sets of created faces to detect redundant triangles
    created_face_vert_sets = []

    if n_inner < 3:
        # Not enough inner verts, recreate outer face
        try:
            new_face = bm.faces.new(outer_loop)
            created_faces.append(new_face)
        except ValueError:
            pass
        return created_faces

    # Find which outer verts are on the cut boundary (shared with inner)
    outer_on_cut = set()
    for ov in outer_loop:
        if any((ov.co - iv.co).length < EPSILON for iv in inner_loop):
            outer_on_cut.add(ov)

    # Find the best starting alignment between outer and inner loops
    # Start from an outer vert that's on the cut boundary
    start_outer_idx = 0
    for i, ov in enumerate(outer_loop):
        if ov in outer_on_cut:
            start_outer_idx = i
            break

    # Find corresponding inner vert
    start_outer_vert = outer_loop[start_outer_idx]
    start_inner_idx = 0
    for i, iv in enumerate(inner_loop):
        if (iv.co - start_outer_vert.co).length < EPSILON:
            start_inner_idx = i
            break

    debug_log(f"[CubeCut] Start alignment: outer[{start_outer_idx}] <-> inner[{start_inner_idx}]")

    # Walk around outer loop, creating faces
    inner_idx = start_inner_idx

    for i in range(n_outer):
        outer_idx = (start_outer_idx + i) % n_outer
        next_outer_idx = (outer_idx + 1) % n_outer

        ov1 = outer_loop[outer_idx]
        ov2 = outer_loop[next_outer_idx]

        ov1_on_cut = ov1 in outer_on_cut
        ov2_on_cut = ov2 in outer_on_cut

        if ov1_on_cut and ov2_on_cut:
            # Both outer verts are on cut - this edge is part of cut, skip
            # But we need to advance inner_idx if ov2 matches next inner
            next_inner_idx = (inner_idx + 1) % n_inner
            if (inner_loop[next_inner_idx].co - ov2.co).length < EPSILON:
                inner_idx = next_inner_idx
            debug_log(f"[CubeCut]   Skipping edge (both on cut): ({ov1.co}) -> ({ov2.co})")
            continue

        elif ov1_on_cut and not ov2_on_cut:
            # Transitioning from cut to outside
            iv = inner_loop[inner_idx]
            if (iv.co - ov1.co).length < EPSILON:
                continue
            face_verts = [ov1, ov2, iv]
            if _face_spans_cut_edge(face_verts, outer_on_cut):
                debug_log(f"[CubeCut] FACE SKIPPED (spans cut edge): {ov1.co}, {ov2.co}, {iv.co}")
                continue
            # Check if this triangle's vertices are already covered by an existing face
            if _is_subset_of_existing_face(face_verts, created_face_vert_sets):
                debug_log(f"[CubeCut] FACE SKIPPED (redundant - subset of existing): {ov1.co}, {ov2.co}, {iv.co}")
                continue
            try:
                new_face = bm.faces.new(face_verts)
                created_faces.append(new_face)
                created_face_vert_sets.append(_verts_to_set(face_verts))
                debug_log(f"[CubeCut] FACE CREATED (cut->outside tri): verts at {ov1.co}, {ov2.co}, {iv.co}")
                debug_log(f"[CubeCut]   Edges implied: ({ov1.co})->({ov2.co}), ({ov2.co})->({iv.co}), ({iv.co})->({ov1.co})")
            except ValueError:
                debug_log(f"[CubeCut] FACE FAILED (cut->outside tri)")

        elif not ov1_on_cut and ov2_on_cut:
            # Transitioning from outside to cut
            for j in range(n_inner):
                test_idx = (inner_idx + j) % n_inner
                if (inner_loop[test_idx].co - ov2.co).length < EPSILON:
                    inner_idx = test_idx
                    break

            iv = inner_loop[inner_idx]
            if (iv.co - ov2.co).length < EPSILON:
                prev_inner_idx = (inner_idx - 1) % n_inner
                iv_prev = inner_loop[prev_inner_idx]
                face_verts = [ov1, ov2, iv_prev]
                if _face_spans_cut_edge(face_verts, outer_on_cut):
                    debug_log(f"[CubeCut] FACE SKIPPED (spans cut edge): {ov1.co}, {ov2.co}, {iv_prev.co}")
                    continue
                # Check if this triangle's vertices are already covered by an existing face
                if _is_subset_of_existing_face(face_verts, created_face_vert_sets):
                    debug_log(f"[CubeCut] FACE SKIPPED (redundant - subset of existing): {ov1.co}, {ov2.co}, {iv_prev.co}")
                    continue
                try:
                    new_face = bm.faces.new(face_verts)
                    created_faces.append(new_face)
                    created_face_vert_sets.append(_verts_to_set(face_verts))
                    debug_log(f"[CubeCut] FACE CREATED (outside->cut tri): verts at {ov1.co}, {ov2.co}, {iv_prev.co}")
                    debug_log(f"[CubeCut]   Edges implied: ({ov1.co})->({ov2.co}), ({ov2.co})->({iv_prev.co}), ({iv_prev.co})->({ov1.co})")
                except ValueError:
                    debug_log(f"[CubeCut] FACE FAILED (outside->cut tri)")

        else:
            # Both outside - create quad or triangle connecting to inner
            iv1 = min(inner_loop, key=lambda v: (v.co - ov1.co).length)
            iv2 = min(inner_loop, key=lambda v: (v.co - ov2.co).length)

            if iv1 == iv2:
                face_verts = [ov1, ov2, iv1]
                if _face_spans_cut_edge(face_verts, outer_on_cut):
                    debug_log(f"[CubeCut] FACE SKIPPED (spans cut edge): {ov1.co}, {ov2.co}, {iv1.co}")
                    continue
                try:
                    new_face = bm.faces.new(face_verts)
                    created_faces.append(new_face)
                    created_face_vert_sets.append(_verts_to_set(face_verts))
                    debug_log(f"[CubeCut] FACE CREATED (outside tri): verts at {ov1.co}, {ov2.co}, {iv1.co}")
                except ValueError:
                    debug_log(f"[CubeCut] FACE FAILED (outside tri)")
            elif _is_convex_quad(ov1.co, ov2.co, iv2.co, iv1.co):
                # Quad is convex, create it
                face_verts = [ov1, ov2, iv2, iv1]
                if _face_spans_cut_edge(face_verts, outer_on_cut):
                    debug_log(f"[CubeCut] FACE SKIPPED (spans cut edge): {ov1.co}, {ov2.co}, {iv2.co}, {iv1.co}")
                    continue
                try:
                    new_face = bm.faces.new(face_verts)
                    created_faces.append(new_face)
                    created_face_vert_sets.append(_verts_to_set(face_verts))
                    debug_log(f"[CubeCut] FACE CREATED (outside quad): verts at {ov1.co}, {ov2.co}, {iv2.co}, {iv1.co}")
                except ValueError:
                    debug_log(f"[CubeCut] FACE FAILED (outside quad)")
            else:
                # Quad would be non-convex, fall back to two triangles
                debug_log(f"[CubeCut] Non-convex quad detected, splitting into triangles")

                # Triangle 1: ov1, ov2, iv2
                face_verts_1 = [ov1, ov2, iv2]
                if not _face_spans_cut_edge(face_verts_1, outer_on_cut):
                    try:
                        new_face = bm.faces.new(face_verts_1)
                        created_faces.append(new_face)
                        created_face_vert_sets.append(_verts_to_set(face_verts_1))
                        debug_log(f"[CubeCut] FACE CREATED (non-convex split tri 1): verts at {ov1.co}, {ov2.co}, {iv2.co}")
                    except ValueError:
                        debug_log(f"[CubeCut] FACE FAILED (non-convex split tri 1)")

                # Triangle 2: ov1, iv2, iv1
                face_verts_2 = [ov1, iv2, iv1]
                if not _face_spans_cut_edge(face_verts_2, outer_on_cut):
                    try:
                        new_face = bm.faces.new(face_verts_2)
                        created_faces.append(new_face)
                        created_face_vert_sets.append(_verts_to_set(face_verts_2))
                        debug_log(f"[CubeCut] FACE CREATED (non-convex split tri 2): verts at {ov1.co}, {iv2.co}, {iv1.co}")
                    except ValueError:
                        debug_log(f"[CubeCut] FACE FAILED (non-convex split tri 2)")

    debug_log(f"[CubeCut] Created {len(created_faces)} faces around hole")
    return created_faces


def _is_convex_quad(p0, p1, p2, p3):
    """
    Check if a quad with vertices p0, p1, p2, p3 (in order) is convex.

    A quad is convex if all interior angles are less than 180 degrees,
    which means all cross products of consecutive edges point in the same direction.

    Args:
        p0, p1, p2, p3: Vector positions of the quad vertices in order

    Returns:
        bool: True if the quad is convex, False otherwise
    """
    # Edge vectors (going around the quad)
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p3 - p2
    e3 = p0 - p3

    # Cross products at each vertex (tells us which way we're turning)
    c0 = e0.cross(e1)
    c1 = e1.cross(e2)
    c2 = e2.cross(e3)
    c3 = e3.cross(e0)

    # For a convex quad, all cross products should point in the same direction
    # Check by comparing signs of dot products
    # Use a small epsilon to handle numerical precision
    eps = EPSILON * EPSILON

    # Check that all cross products agree in direction
    if c0.dot(c1) < -eps:
        return False
    if c0.dot(c2) < -eps:
        return False
    if c0.dot(c3) < -eps:
        return False

    return True


def _sort_verts_by_angle(verts, face_normal):
    """Sort vertices by angle around their centroid, consistent with face normal."""
    import math

    centroid = Vector((0, 0, 0))
    for v in verts:
        centroid += v.co
    centroid /= len(verts)

    # Create coordinate axes on the face plane
    up = Vector((0, 0, 1))
    if abs(face_normal.dot(up)) > 0.9:
        up = Vector((1, 0, 0))
    axis1 = face_normal.cross(up).normalized()
    axis2 = face_normal.cross(axis1).normalized()

    def angle_key(v):
        delta = v.co - centroid
        return math.atan2(delta.dot(axis2), delta.dot(axis1))

    return sorted(verts, key=angle_key)


def _reconstruct_partial_face(bm, face, inside_verts, outside_verts, boundary_verts, cuboid):
    """
    Reconstruct a face that is partially inside the cuboid.

    Keeps the portion outside the cuboid, removes the portion inside.
    """
    debug_log(f"[CubeCut] _reconstruct_partial_face: face {face.index}")
    if not face.is_valid:
        debug_log(f"[CubeCut]   Face invalid, skipping")
        return

    # Get the ordered list of face vertices
    face_vert_list = list(face.verts)
    n = len(face_vert_list)
    debug_log(f"[CubeCut]   Original verts ({n}): {[v.co[:] for v in face_vert_list]}")

    # Build new vertex sequence: keep outside and boundary verts, skip inside verts
    # This is simplified - just creates a face from non-inside vertices in order
    new_face_verts = []

    for v in face_vert_list:
        if v not in inside_verts:
            new_face_verts.append(v)

    debug_log(f"[CubeCut]   After removing inside verts ({len(new_face_verts)}): {[v.co[:] for v in new_face_verts]}")

    # Need at least 3 vertices for a valid face
    if len(new_face_verts) < 3:
        # Can't form a face, just delete the original
        debug_log(f"[CubeCut]   Not enough verts, deleting face")
        bmesh.ops.delete(bm, geom=[face], context='FACES')
        return

    # Check if we can just modify the existing face or need to recreate
    if len(new_face_verts) == len(face_vert_list):
        # No vertices removed, face doesn't need reconstruction
        debug_log(f"[CubeCut]   No verts removed, no action needed")
        return

    # Delete old face and create new one
    debug_log(f"[CubeCut]   Deleting old face, creating new with {len(new_face_verts)} verts")
    bmesh.ops.delete(bm, geom=[face], context='FACES_ONLY')

    try:
        new_face = bm.faces.new(new_face_verts)
        debug_log(f"[CubeCut]   FACE CREATED (reconstructed): {[v.co[:] for v in new_face_verts]}")
    except ValueError:
        # Face already exists or invalid vertex sequence
        debug_log(f"[CubeCut]   FACE FAILED (reconstructed)")


def _point_within_plane_bounds(point, plane_idx, cuboid):
    """Check if a point on a cuboid plane is within that plane's bounds."""
    local = cuboid.to_local(point)
    x, y, z = local.x, local.y, local.z

    if plane_idx in (0, 1):  # Front/back planes
        return (
            -EPSILON <= x <= cuboid.dx + EPSILON and
            -EPSILON <= y <= cuboid.dy + EPSILON
        )
    elif plane_idx in (2, 3):  # Left/right planes
        return (
            -EPSILON <= y <= cuboid.dy + EPSILON and
            cuboid.depth_min - EPSILON <= z <= cuboid.depth_max + EPSILON
        )
    else:  # Top/bottom planes
        return (
            -EPSILON <= x <= cuboid.dx + EPSILON and
            cuboid.depth_min - EPSILON <= z <= cuboid.depth_max + EPSILON
        )


def _build_cuboid_vertices_local(cuboid):
    """Build the 8 vertices of the cuboid in mesh local space."""
    o = cuboid.origin
    lx = cuboid.local_x
    ly = cuboid.local_y
    lz = cuboid.local_z
    dx = cuboid.dx
    dy = cuboid.dy
    d_min = cuboid.depth_min
    d_max = cuboid.depth_max

    return [
        o + lz * d_min,                          # 0: front_bl
        o + lx * dx + lz * d_min,                # 1: front_br
        o + lx * dx + ly * dy + lz * d_min,      # 2: front_tr
        o + ly * dy + lz * d_min,                # 3: front_tl
        o + lz * d_max,                          # 4: back_bl
        o + lx * dx + lz * d_max,                # 5: back_br
        o + lx * dx + ly * dy + lz * d_max,      # 6: back_tr
        o + ly * dy + lz * d_max,                # 7: back_tl
    ]


def _point_in_face_interior(point, face_verts, face_normal):
    """
    Test if a point is strictly inside a face (not on edges).
    """
    if len(face_verts) < 3:
        return False

    # First check if point is in the polygon at all
    if not _point_in_polygon(point, face_verts, face_normal):
        return False

    # Check distance to all edges - must not be too close
    n = len(face_verts)
    for i in range(n):
        v1 = face_verts[i]
        v2 = face_verts[(i + 1) % n]

        # Distance from point to edge
        edge_vec = v2 - v1
        edge_len_sq = edge_vec.length_squared
        if edge_len_sq < EPSILON * EPSILON:
            continue

        t = max(0, min(1, (point - v1).dot(edge_vec) / edge_len_sq))
        closest = v1 + edge_vec * t
        dist = (point - closest).length

        if dist < EPSILON * 10:  # Too close to edge
            return False

    return True


def _point_in_polygon(point, face_verts, face_normal):
    """Test if a point is inside a polygon using ray casting."""
    if len(face_verts) < 3:
        return False

    normal_abs = Vector([abs(n) for n in face_normal])

    if normal_abs.x >= normal_abs.y and normal_abs.x >= normal_abs.z:
        def to_2d(p):
            return (p.y, p.z)
    elif normal_abs.y >= normal_abs.x and normal_abs.y >= normal_abs.z:
        def to_2d(p):
            return (p.x, p.z)
    else:
        def to_2d(p):
            return (p.x, p.y)

    px, py = to_2d(point)
    n = len(face_verts)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = to_2d(face_verts[i])
        xj, yj = to_2d(face_verts[j])

        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


# Keep the diagnostic function available for testing
def find_cut_vertices(context, first_vertex, second_vertex, depth, local_x, local_y, local_z):
    """
    Diagnostic function: Find and create intersection vertices without cutting.
    """
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        return (False, "No active mesh object", [])

    effective_depth = depth
    if abs(depth) < EPSILON:
        effective_depth = EPSILON * 2 if depth >= 0 else -EPSILON * 2

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    world_to_local = obj.matrix_world.inverted()
    local_to_world = obj.matrix_world

    local_first = world_to_local @ first_vertex
    local_second = world_to_local @ second_vertex
    local_x_trans = (world_to_local.to_3x3() @ local_x).normalized()
    local_y_trans = (world_to_local.to_3x3() @ local_y).normalized()
    local_z_trans = (world_to_local.to_3x3() @ local_z).normalized()

    scale_factor = (world_to_local.to_3x3() @ local_z).length
    local_depth = effective_depth * scale_factor

    cuboid = CuboidPlanes(
        local_first, local_second, local_depth,
        local_x_trans, local_y_trans, local_z_trans
    )

    intersection_points = []

    # Edge-plane intersections
    for edge in bm.edges:
        if not edge.is_valid:
            continue
        v1_co = edge.verts[0].co
        v2_co = edge.verts[1].co

        for plane_idx, (plane_point, plane_normal) in enumerate(cuboid.planes):
            d1 = (v1_co - plane_point).dot(plane_normal)
            d2 = (v2_co - plane_point).dot(plane_normal)

            crosses = (d1 > EPSILON and d2 < -EPSILON) or (d1 < -EPSILON and d2 > EPSILON)
            if not crosses:
                continue

            intersection = intersect_line_plane(v1_co, v2_co, plane_point, plane_normal)
            if intersection is None:
                continue

            if not _point_within_plane_bounds(intersection, plane_idx, cuboid):
                continue

            is_duplicate = any((existing - intersection).length < EPSILON for existing in intersection_points)
            if not is_duplicate:
                intersection_points.append(intersection.copy())

    # Cuboid-face intersections
    cuboid_verts = _build_cuboid_vertices_local(cuboid)
    cuboid_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for face in bm.faces:
        if not face.is_valid:
            continue
        face_normal = face.normal
        if face_normal.length < EPSILON:
            continue
        face_point = face.verts[0].co
        face_verts = [v.co for v in face.verts]

        for v1_idx, v2_idx in cuboid_edges:
            edge_start = cuboid_verts[v1_idx]
            edge_end = cuboid_verts[v2_idx]

            d1 = (edge_start - face_point).dot(face_normal)
            d2 = (edge_end - face_point).dot(face_normal)

            crosses = (d1 > EPSILON and d2 < -EPSILON) or (d1 < -EPSILON and d2 > EPSILON)
            if not crosses:
                continue

            intersection = intersect_line_plane(edge_start, edge_end, face_point, face_normal)
            if intersection is None:
                continue

            if not _point_in_polygon(intersection, face_verts, face_normal):
                continue

            is_duplicate = any((existing - intersection).length < EPSILON for existing in intersection_points)
            if not is_duplicate:
                intersection_points.append(intersection.copy())

    # Create vertices
    created_verts = []
    for point in intersection_points:
        new_vert = bm.verts.new(point)
        created_verts.append(new_vert)

    # Select created vertices
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False
    for v in created_verts:
        v.select = True
    bm.select_flush_mode()

    world_positions = [local_to_world @ v.co for v in created_verts]
    bmesh.update_edit_mesh(me)

    return (True, f"Created {len(created_verts)} intersection vertices", world_positions)
