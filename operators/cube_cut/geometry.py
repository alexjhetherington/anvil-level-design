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

from ...utils import compute_uv_projection_from_face, apply_uv_projection_to_face, debug_log, compute_normal_from_verts


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

    # === PRE-STEP: Identify faces to delete (all vertices inside cuboid) ===
    debug_log(f"\n[CubeCut] === PRE-STEP: Find faces entirely inside cuboid ===")
    bm.faces.ensure_lookup_table()

    # Apply selection filter: only process selected faces (or all if none selected)
    any_faces_selected = any(f.select for f in bm.faces if f.is_valid)
    if any_faces_selected:
        debug_log(f"[CubeCut] Selection mode: only processing selected faces")
    else:
        debug_log(f"[CubeCut] No faces selected: processing all faces")

    faces_to_delete = set()
    for face in bm.faces:
        if not face.is_valid:
            continue
        if any_faces_selected and not face.select:
            continue

        # Check if ALL vertices are inside the cuboid
        all_inside = all(cuboid.point_inside(v.co) for v in face.verts)
        if all_inside:
            faces_to_delete.add(face)
            debug_log(f"[CubeCut] Face {face.index} has all vertices inside cuboid - will be deleted")

    # Delete faces that are entirely inside the cuboid
    if faces_to_delete:
        debug_log(f"[CubeCut] Deleting {len(faces_to_delete)} faces entirely inside cuboid")
        bmesh.ops.delete(bm, geom=list(faces_to_delete), context='FACES_ONLY')
        bm.faces.ensure_lookup_table()

    # === STEP 1: Determine which faces will be cut ===
    # A face needs cutting if:
    # 1. Any cuboid edge pierces the face interior, OR
    # 2. Any face edge crosses a cuboid plane (cube wider than face case)
    debug_log(f"\n[CubeCut] === STEP 1: Find faces to cut ===")
    face_interior_points = _find_cuboid_face_intersections(bm, cuboid)

    # Determine which faces will actually be cut
    faces_to_be_cut = set()
    skipped_unselected_count = 0

    # First: add faces with interior intersections (cuboid corners pierce face)
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

    # Second: check for faces where face edges cross cuboid planes (cube wider than face)
    # Only mark for cutting if crossings are on at least 2 DIFFERENT edges (cube passes through)
    # If all crossings are on the same edge, it's just an edge that got split multiple times
    for face in bm.faces:
        if not face.is_valid:
            continue
        if face in faces_to_be_cut:
            continue  # Already marked for cutting
        if any_faces_selected and not face.select:
            continue

        # Collect all edges that have crossings within cuboid bounds
        edges_with_crossings = set()
        for edge in face.edges:
            v1_co = edge.verts[0].co
            v2_co = edge.verts[1].co

            for plane_idx, (plane_point, plane_normal) in enumerate(cuboid.planes):
                d1 = (v1_co - plane_point).dot(plane_normal)
                d2 = (v2_co - plane_point).dot(plane_normal)

                # Edge crosses plane if endpoints are on strictly opposite sides
                crosses = (d1 > EPSILON and d2 < -EPSILON) or (d1 < -EPSILON and d2 > EPSILON)
                if not crosses:
                    continue

                intersection = intersect_line_plane(v1_co, v2_co, plane_point, plane_normal)
                if intersection is None:
                    continue

                # Check if intersection is within cuboid bounds on this plane
                if _point_within_plane_bounds(intersection, plane_idx, cuboid):
                    edges_with_crossings.add(edge)
                    debug_log(f"[CubeCut] Face {face.index} edge crosses cuboid plane {plane_idx}")
                    break  # This edge has a crossing, check next edge

        # Only mark for cutting if crossings are on at least 2 different edges
        # (meaning cube actually passes through the face, not just touches one edge)
        if len(edges_with_crossings) >= 2:
            faces_to_be_cut.add(face)
            # Initialize empty interior points list for this face
            if face.index not in face_interior_points:
                face_interior_points[face.index] = []
            debug_log(f"[CubeCut] Face {face.index} will be cut ({len(edges_with_crossings)} edges have crossings)")
        elif len(edges_with_crossings) == 1:
            debug_log(f"[CubeCut] Face {face.index} has only 1 edge with crossings - NOT cutting (just edge split)")

    # Third: check for faces where any vertex is inside the cuboid
    # (face partially overlaps cuboid but cuboid vertices land on existing edges)
    for face in bm.faces:
        if not face.is_valid:
            continue
        if face in faces_to_be_cut:
            continue
        if any_faces_selected and not face.select:
            continue

        for vert in face.verts:
            if cuboid.point_inside(vert.co):
                faces_to_be_cut.add(face)
                if face.index not in face_interior_points:
                    face_interior_points[face.index] = []
                debug_log(f"[CubeCut] Face {face.index} will be cut (vertex inside cuboid)")
                break

    debug_log(f"[CubeCut] Faces to be cut: {len(faces_to_be_cut)}")
    if skipped_unselected_count > 0:
        debug_log(f"[CubeCut] Skipped {skipped_unselected_count} unselected faces")

    # Check for degenerate geometry (duplicate vertices at same position) on faces to be cut.
    # This can happen from previous operations leaving zero-length edges. Proceeding would
    # cause face creation failures and orphaned geometry, so bail out early.
    for face in faces_to_be_cut:
        if not face.is_valid:
            continue
        seen_positions = {}
        for v in face.verts:
            key = (round(v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5))
            if key in seen_positions:
                print(f"Level Design Tools: Error - Face {face.index} has duplicate vertices at {v.co[:]}. "
                      f"Run Mesh > Clean Up > Merge by Distance first.")
                debug_log(f"[CubeCut] ABORTING: Face {face.index} has duplicate verts at {key}")
                bmesh.update_edit_mesh(me)
                return (False, "Face has duplicate vertices - run Merge by Distance first")
            seen_positions[key] = v

    # === STEP 2: Find edge-plane intersections and split edges ===
    # Only split edges that belong to faces that will be cut
    debug_log(f"\n[CubeCut] === STEP 2: Find edge-plane intersections ===")
    edge_splits = _find_edge_plane_intersections(bm, cuboid, faces_to_be_cut)
    debug_log(f"[CubeCut] Found {len(edge_splits)} edges to split")

    # Split edges (must do this before face operations)
    # Also track which faces had their edges split
    split_verts, faces_with_split_edges, vert_plane_map = _split_edges_at_intersections(bm, edge_splits)
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

        interior_verts = []
        for point in points:
            new_vert = bm.verts.new(point)
            interior_verts.append(new_vert)
            debug_log(f"[CubeCut] VERTEX CREATED (interior): pos={point}, for face {face.index}")
            debug_log(f"[CubeCut]   No edges created yet (floating vertex)")

        # Always add face to list (even with no interior verts) so it gets processed in STEP 4
        # This handles the "cube wider than face" case where only edge splits exist
        face_interior_verts.append((face, interior_verts))

    debug_log(f"[CubeCut] Total interior vertices created: {sum(len(v) for _, v in face_interior_verts)}")
    debug_log(f"[CubeCut] Faces to process: {len(face_interior_verts)}")

    # === STEP 4: Capture face data and delete faces being processed ===
    # Capture vertex data for each face BEFORE deleting faces
    debug_log(f"\n[CubeCut] === STEP 4: Capture face data and delete faces ===")
    split_verts_set = set(split_verts)
    face_data_list = []  # List of (new_verts, verts_on_original_exterior, verts_in_original_interior, face_normal, uv_projection) tuples
    faces_to_delete = []

    # Get UV layer for capturing projection data
    uv_layer = bm.loops.layers.uv.active

    for face, interior_verts in face_interior_verts:
        if face is None or not face.is_valid:
            continue

        # Capture face normal BEFORE deleting - used for consistent vertex sorting
        face_normal = face.normal.copy()

        # Find edge-split vertices that are on this face's boundary
        # Also include existing face vertices that lie on the cuboid boundary
        # (these are effectively "split" vertices where the cuboid edge meets an existing vertex)
        edge_verts_on_face = [v for v in face.verts if v in split_verts_set or
                             (cuboid.point_inside(v.co) and not cuboid.point_strictly_inside(v.co))]

        # Compute input variables for _verts_to_faces
        # Use face normal for consistent winding in angle sorting
        verts_to_delete = [v for v in face.verts if v not in split_verts_set and _should_delete_vertex_for_face(v, face, cuboid)]

        # Validate that this cut will create a valid shape
        # Count unique positions (zero-depth cuts create duplicate vertices at same positions)
        def unique_positions(verts):
            seen = set()
            for v in verts:
                key = (round(v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5))
                seen.add(key)
            return len(seen)

        num_interior_unique = unique_positions(interior_verts)
        num_edge_unique = unique_positions(edge_verts_on_face)
        num_deleted = len(verts_to_delete)

        # Skip if only interior verts and 2 or fewer unique (can't form a valid hole)
        if num_edge_unique == 0 and num_interior_unique <= 2:
            print(f"Level Design Tools: Skipping face {face.index} - only {num_interior_unique} unique interior verts, cannot form valid shape")
            # Remove orphaned interior vertices
            for v in interior_verts:
                if v.is_valid:
                    bm.verts.remove(v)
            continue

        # Skip if only edge verts, 2 or fewer unique, and no deleted verts (cut doesn't remove anything)
        if num_interior_unique == 0 and num_edge_unique <= 2 and num_deleted == 0:
            print(f"Level Design Tools: Skipping face {face.index} - only {num_edge_unique} unique edge verts with no deleted verts, cannot form valid shape")
            # Remove orphaned interior vertices
            for v in interior_verts:
                if v.is_valid:
                    bm.verts.remove(v)
            continue

        verts_to_delete_set = set(verts_to_delete)
        new_verts = _sort_verts_by_angle_with_normal(list(interior_verts) + [v for v in edge_verts_on_face if v not in verts_to_delete_set], face_normal)
        verts_in_original_interior = _sort_verts_by_angle_with_normal(list(interior_verts), face_normal)
        verts_on_original_exterior = _sort_verts_by_angle_with_normal([v for v in face.verts if v not in verts_to_delete], face_normal)

        # Capture UV projection data and material index before deleting the face
        uv_projection = None
        if uv_layer is not None:
            uv_projection = compute_uv_projection_from_face(face, uv_layer)
        material_index = face.material_index

        face_data_list.append((new_verts, verts_on_original_exterior, verts_in_original_interior, face_normal, uv_projection, material_index))
        faces_to_delete.append(face)
        debug_log(f"[CubeCut] Captured data for face {face.index}: {len(new_verts)} new_verts, {len(verts_on_original_exterior)} exterior, {len(verts_in_original_interior)} interior, uv={'yes' if uv_projection else 'no'}")

    # Delete only the faces we're processing
    debug_log(f"[CubeCut] Deleting {len(faces_to_delete)} faces")
    bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES_ONLY')

    # === STEP 5: Rebuild faces from captured vertex data ===
    debug_log(f"\n[CubeCut] === STEP 5: Rebuild faces ===")
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    newly_created_faces = []
    for new_verts, verts_on_original_exterior, verts_in_original_interior, face_normal, uv_projection, material_index in face_data_list:
        new_faces = _verts_to_faces(bm, new_verts, verts_on_original_exterior, verts_in_original_interior, face_normal, cuboid, me, ppm, vert_plane_map)
        if new_faces:
            for new_face in new_faces:
                # Apply material from original face
                new_face.material_index = material_index
                # Apply UV projection from the original face
                if uv_projection is not None and uv_layer is not None:
                    u_axis, v_axis, origin_uv, origin_pos, source_normal = uv_projection
                    apply_uv_projection_to_face(new_face, uv_layer, u_axis, v_axis, origin_uv, origin_pos, source_normal)
            newly_created_faces.extend(new_faces)

    # === STEP 6: Quadrilate n-gons created by edge splits ===
    # Faces that had edges split but weren't cut are now n-gons and need to be quadrilated
    # Also quadrilate any newly created faces from cutting that are n-gons
    faces_to_quadrilate = []

    # Check adjacent faces that had their edges split (but weren't deleted/cut)
    for face in faces_with_split_edges:
        if not face.is_valid:
            continue
        if len(face.verts) > 4:
            # This is an n-gon that needs quadrilating
            faces_to_quadrilate.append(face)
            debug_log(f"[CubeCut] Adjacent face needs quadrilating: {len(face.verts)} verts")

    # Check newly created faces from cutting
    for face in newly_created_faces:
        if not face.is_valid:
            continue
        if len(face.verts) > 4:
            faces_to_quadrilate.append(face)
            debug_log(f"[CubeCut] Newly created face needs quadrilating: {len(face.verts)} verts")

    if faces_to_quadrilate:
        debug_log(f"[CubeCut] === STEP 6: Quadrilating {len(faces_to_quadrilate)} n-gon faces ===")
        bm.normal_update()
        # Triangulate the n-gons first
        result = bmesh.ops.triangulate(bm, faces=faces_to_quadrilate)
        new_tris = result['faces']
        debug_log(f"[CubeCut] Triangulated into {len(new_tris)} triangles")

        # Join triangles into quads where possible
        if new_tris:
            bmesh.ops.join_triangles(
                bm,
                faces=new_tris,
                angle_face_threshold=3.14159,  # ~180 degrees - allow any face angle
                angle_shape_threshold=3.14159  # ~180 degrees - allow any shape
            )
            debug_log(f"[CubeCut] Joined triangles into quads where possible")

    # === STEP 7: Cleanup ===
    # Remove loose edges (edges not connected to any face)
    # This can happen when adjacent faces are both cut and their shared edges are inside the cube
    loose_edges = [e for e in bm.edges if e.is_valid and not e.link_faces]
    if loose_edges:
        debug_log(f"[CubeCut] Removing {len(loose_edges)} loose edges")
        bmesh.ops.delete(bm, geom=loose_edges, context='EDGES')

    # Remove loose vertices
    loose_verts = [v for v in bm.verts if v.is_valid and not v.link_faces]
    if loose_verts:
        debug_log(f"[CubeCut] Removing {len(loose_verts)} loose vertices")
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')

    # Merge very close vertices
    bmesh.ops.remove_doubles(bm, verts=[v for v in bm.verts if v.is_valid], dist=EPSILON)

    # Recalculate normals for newly created faces
    bm.normal_update()

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
        tuple: (newly created vertices, set of faces that had edges split,
                dict mapping each split vert to the set of cuboid plane indices it lies on)
    """
    new_verts = []
    vert_plane_map = {}  # BMVert -> set of plane indices
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

            # Check if intersection coincides with an existing vertex
            # (happens when a cuboid edge passes through a mesh edge)
            if (intersection_point - v1.co).length < EPSILON:
                if v1 not in vert_plane_map:
                    vert_plane_map[v1] = set()
                vert_plane_map[v1].add(plane_idx)
                debug_log(f"[CubeCut] Intersection at existing vert {v1.co[:]}, adding plane {plane_idx}")
                continue
            if (intersection_point - v2.co).length < EPSILON:
                if v2 not in vert_plane_map:
                    vert_plane_map[v2] = set()
                vert_plane_map[v2].add(plane_idx)
                debug_log(f"[CubeCut] Intersection at existing vert {v2.co[:]}, adding plane {plane_idx}")
                continue

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

            # Track which cuboid plane(s) this vertex lies on (as a set,
            # since a vertex at a cuboid edge/corner can be on multiple planes)
            if new_vert not in vert_plane_map:
                vert_plane_map[new_vert] = set()
            vert_plane_map[new_vert].add(plane_idx)

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

    return new_verts, faces_with_split_edges, vert_plane_map


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


def _sort_verts_by_angle(verts):
    """Sort vertices by angle around their centroid."""
    import math

    if len(verts) < 2:
        return list(verts)

    centroid = Vector((0, 0, 0))
    for v in verts:
        centroid += v.co
    centroid /= len(verts)

    # Compute a normal from the vertices (finds 3 non-collinear points)
    normal = compute_normal_from_verts(verts)
    if normal is None:
        normal = Vector((0, 0, 1))

    # Create coordinate axes on the plane
    up = Vector((0, 0, 1))
    if abs(normal.dot(up)) > 0.9:
        up = Vector((1, 0, 0))
    axis1 = normal.cross(up).normalized()
    axis2 = normal.cross(axis1).normalized()

    def angle_key(v):
        delta = v.co - centroid
        return math.atan2(delta.dot(axis2), delta.dot(axis1))

    return sorted(verts, key=angle_key)


def _sort_verts_by_angle_with_normal(verts, normal):
    """Sort vertices by angle around their centroid using the provided normal.

    This ensures consistent winding direction based on the original face normal.
    The normal is negated to produce winding that creates exterior faces (not the hole).
    """
    import math

    if len(verts) < 2:
        return list(verts)

    if normal is None or normal.length < EPSILON:
        return _sort_verts_by_angle(verts)

    # Negate normal to get correct winding for exterior faces
    normal = -normal

    centroid = Vector((0, 0, 0))
    for v in verts:
        centroid += v.co
    centroid /= len(verts)

    # Create coordinate axes on the plane using the provided normal
    up = Vector((0, 0, 1))
    if abs(normal.dot(up)) > 0.9:
        up = Vector((1, 0, 0))
    axis1 = normal.cross(up).normalized()
    axis2 = normal.cross(axis1).normalized()

    def angle_key(v):
        delta = v.co - centroid
        return math.atan2(delta.dot(axis2), delta.dot(axis1))

    return sorted(verts, key=angle_key)


def _should_delete_vertex_for_face(vertex, face, cuboid):
    """Check if a vertex should be deleted when processing a specific face.

    Returns True if:
    - Vertex is strictly inside the cuboid, OR
    - Vertex is on the cuboid boundary and has no edges (on this face) leading outside the cuboid

    Only considers edges that belong to the given face.
    """
    if cuboid.point_strictly_inside(vertex.co):
        return True

    if cuboid.point_inside(vertex.co):
        # On boundary - check if any edge on this face goes outside
        face_edges = set(face.edges)
        for edge in vertex.link_edges:
            if edge not in face_edges:
                continue  # Skip edges not on this face
            other_vert = edge.other_vert(vertex)
            if not cuboid.point_inside(other_vert.co):
                return False  # Has edge outside on this face, keep it
        return True  # No edges on this face go outside, delete it

    return False  # Outside cuboid, keep it


def _verts_to_faces(bm, new_verts, verts_on_original_exterior, verts_in_original_interior, face_normal, cuboid, me, ppm, vert_plane_map):
    import math

    debug_log(f"[CubeCut] _verts_to_faces: new_verts={[v.co[:] for v in new_verts]}")
    debug_log(f"[CubeCut] _verts_to_faces: verts_on_original_exterior={[v.co[:] for v in verts_on_original_exterior]}")
    debug_log(f"[CubeCut] _verts_to_faces: verts_in_original_interior={[v.co[:] for v in verts_in_original_interior]}")
    debug_log(f"[CubeCut] _verts_to_faces: face_normal={face_normal[:]}")

    if len(new_verts) < 2:
        return []

    # Helper to check if two verts are adjacent in the exterior loop
    exterior_set = set(verts_on_original_exterior)
    n_exterior = len(verts_on_original_exterior)

    def are_adjacent_on_exterior(v1, v2):
        if v1 not in exterior_set or v2 not in exterior_set:
            return False
        try:
            idx1 = verts_on_original_exterior.index(v1)
            idx2 = verts_on_original_exterior.index(v2)
            return (idx1 + 1) % n_exterior == idx2 or (idx2 + 1) % n_exterior == idx1
        except ValueError:
            return False

    # Connect new vertices in order, skipping where both are adjacent on exterior
    # Track created edges for face creation
    created_edges = []
    n_new = len(new_verts)
    for i in range(n_new):
        v1 = new_verts[i]
        v2 = new_verts[(i + 1) % n_new]

        # Skip if both vertices are on exterior and adjacent on exterior
        # UNLESS there are no interior verts (cutting off a piece, not making a hole)
        if v1 in exterior_set and v2 in exterior_set and are_adjacent_on_exterior(v1, v2) and len(verts_in_original_interior) > 0:
            debug_log(f"[CubeCut]   Skipping edge (both adjacent on exterior): {v1.co[:]} -> {v2.co[:]}")
            continue

        # Skip cross-hole edges between split vertices that share no cuboid plane.
        # When the cuboid cuts through a face, each cuboid plane creates split
        # vertices on the face edges. Valid closing edges connect splits from the
        # SAME plane (sealing one side of the cut). Edges between splits from
        # entirely DIFFERENT planes would bridge across the removed region.
        # On quad faces this is implicitly prevented because both splits on the
        # same original edge create an "already exists" barrier, but on triangles
        # (or other odd faces) the splits land on different original edges with
        # no such barrier. We use sets of plane indices (not single values) because
        # a vertex at a cuboid edge or corner can belong to multiple planes.
        v1_planes = vert_plane_map.get(v1)
        v2_planes = vert_plane_map.get(v2)
        if v1_planes is not None and v2_planes is not None and v1_planes.isdisjoint(v2_planes):
            debug_log(f"[CubeCut]   Skipping cross-hole edge (no shared plane {v1_planes} vs {v2_planes}): {v1.co[:]} -> {v2.co[:]}")
            continue

        # Check if edge already exists
        edge_exists = any(v2 in e.verts for e in v1.link_edges)
        if not edge_exists:
            try:
                new_edge = bm.edges.new([v1, v2])
                created_edges.append((v1, v2))
                debug_log(f"[CubeCut]   Created edge: {v1.co[:]} -> {v2.co[:]}")
            except ValueError:
                debug_log(f"[CubeCut]   Failed to create edge: {v1.co[:]} -> {v2.co[:]}")
        else:
            debug_log(f"[CubeCut]   Edge already exists: {v1.co[:]} -> {v2.co[:]}")

    # Connect interior vertices to closest exterior vertex (in the "away" direction)
    for interior_vert in verts_in_original_interior:
        # Find the two connected edges from the new_verts loop
        connected_verts = []
        for edge in interior_vert.link_edges:
            other = edge.other_vert(interior_vert)
            if other in new_verts:
                connected_verts.append(other)

        if len(connected_verts) != 2:
            debug_log(f"[CubeCut]   Interior vert {interior_vert.co[:]} has {len(connected_verts)} connections (expected 2)")
            continue

        # Get directions to the two connected vertices
        dir1 = (connected_verts[0].co - interior_vert.co).normalized()
        dir2 = (connected_verts[1].co - interior_vert.co).normalized()

        # Compute bisector of the angle between the two edges
        bisector = (dir1 + dir2).normalized()

        # The "away" direction is opposite to the bisector
        away_dir = -bisector

        # Compute the half-angle of the cut (angle between bisector and either edge)
        dot_val = max(-1.0, min(1.0, bisector.dot(dir1)))
        half_angle = math.acos(dot_val)

        # Find the exterior vertex closest to the "away" direction
        best_vert = None
        best_dot = -2.0  # Will look for highest dot product with away_dir

        for ext_vert in verts_on_original_exterior:
            if ext_vert == interior_vert:
                continue
            dir_to_ext = (ext_vert.co - interior_vert.co).normalized()
            dot = dir_to_ext.dot(away_dir)
            if dot > best_dot:
                best_dot = dot
                best_vert = ext_vert

        if best_vert is None:
            debug_log(f"[CubeCut]   No exterior vertex found for interior vert {interior_vert.co[:]}")
            continue

        # Check if the best vertex is inside the angle between the two connected edges
        # A vertex is inside if its direction from interior_vert has a higher dot with bisector
        # than the threshold (which is cos(half_angle))
        dir_to_best = (best_vert.co - interior_vert.co).normalized()
        dot_with_bisector = dir_to_best.dot(bisector)
        threshold = math.cos(half_angle)

        if dot_with_bisector >= threshold - EPSILON:
            # The best vertex is inside (or on) the angle between connected edges - ERROR
            print(f"Level Design Tools: Error - Interior vertex {interior_vert.co[:]} closest exterior vertex {best_vert.co[:]} is inside the cut angle")
            continue

        # Create edge to the best exterior vertex
        edge_exists = any(best_vert in e.verts for e in interior_vert.link_edges)
        if not edge_exists:
            try:
                bm.edges.new([interior_vert, best_vert])
                debug_log(f"[CubeCut]   Connected interior {interior_vert.co[:]} to exterior {best_vert.co[:]}")
            except ValueError:
                debug_log(f"[CubeCut]   Failed to connect interior {interior_vert.co[:]} to exterior {best_vert.co[:]}")
        else:
            debug_log(f"[CubeCut]   Edge already exists: interior {interior_vert.co[:]} to exterior {best_vert.co[:]}")

    # Create faces - one for each edge created between new verts
    # Use the provided face_normal for angular ordering (captured from original face)
    if face_normal is None or face_normal.length < EPSILON:
        debug_log(f"[CubeCut]   Invalid face normal for angular ordering")
        return []

    def signed_angle(v_from, v_to, normal):
        """Compute signed angle from v_from to v_to around normal axis."""
        cross = v_from.cross(v_to)
        dot = v_from.dot(v_to)
        angle = math.atan2(cross.dot(normal), dot)
        return angle

    valid_verts = set(verts_on_original_exterior) | set(verts_in_original_interior)

    def find_next_vert_angular(current, prev, target):
        """Find the next vertex by following edges in angular order (clockwise).

        Picks the edge that makes the largest counterclockwise angle from the incoming direction
        (i.e., the leftmost turn / clockwise traversal).
        Only considers edges leading to vertices belonging to this face.
        """
        incoming = (prev.co - current.co).normalized()

        best_vert = None
        best_angle = float('-inf')

        for edge in current.link_edges:
            other = edge.other_vert(current)
            if other == prev:
                continue

            if other not in valid_verts:
                continue

            outgoing = (other.co - current.co).normalized()
            # Compute angle from incoming to outgoing (counterclockwise positive)
            # We want the largest angle (leftmost turn / clockwise winding)
            angle = signed_angle(incoming, outgoing, face_normal)

            # Normalize to [0, 2*pi) for comparison
            if angle < 0:
                angle += 2 * math.pi

            if angle > best_angle:
                best_angle = angle
                best_vert = other

        return best_vert

    created_faces = []

    for v1, v2 in created_edges:
        # Check if starting from v2 produces correct winding by examining the first step
        first_next = find_next_vert_angular(v2, v1, v1)
        if first_next is not None:
            edge_dir = v2.co - v1.co
            next_dir = first_next.co - v2.co
            if edge_dir.cross(next_dir).dot(face_normal) < 0:
                v1, v2 = v2, v1

        # Walk around edges to form a closed loop starting with this edge
        # Start at v1, go to v2, then continue in angular order until we return to v1
        face_verts = [v1, v2]
        current = v2
        prev = v1
        max_steps = 100  # Safety limit

        for _ in range(max_steps):
            next_vert = find_next_vert_angular(current, prev, v1)

            if next_vert is None:
                debug_log(f"[CubeCut]   Could not find next vert from {current.co[:]}")
                break

            if next_vert == v1:
                # Completed the loop
                break

            face_verts.append(next_vert)
            prev = current
            current = next_vert

        if len(face_verts) >= 3:
            try:
                new_face = bm.faces.new(face_verts)
                created_faces.append(new_face)
                debug_log(f"[CubeCut]   Created face with {len(face_verts)} verts")
            except ValueError as e:
                debug_log(f"[CubeCut]   Failed to create face: {e}")

    return created_faces


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
