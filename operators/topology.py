"""
Topology utilities for UV island processing.

Handles face grouping by angle threshold and minimal cut graph computation
to ensure surfaces are genus 0 (topologically a disk) before unwrapping.
"""

import math
from ..core.logging import debug_log


def get_face_angle(face1, face2):
    """Get the angle between two face normals in radians."""
    dot = face1.normal.dot(face2.normal)
    # Clamp to avoid math domain errors
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def get_shared_edge(face1, face2):
    """Get the edge shared between two faces, or None if not adjacent."""
    edges1 = set(face1.edges)
    for edge in face2.edges:
        if edge in edges1:
            return edge
    return None


def group_quad_faces_by_angle_and_existing_seams(bm, faces, angle_threshold):
    """Group quad faces by connectivity, respecting existing seams and angle threshold.

    Groups are separated by:
    - Existing seams (user-defined seams are respected as group boundaries)
    - Angle threshold (edges where face normals differ by more than threshold)
    - Non-quad faces (triangles, ngons are treated as blocking boundaries)

    Args:
        bm: BMesh instance
        faces: List of faces to process
        angle_threshold: Maximum angle (radians) between normals before adding seam

    Returns:
        Tuple of:
            - List of groups, where each group is a dict with:
                'faces': set of faces in the group
                'adjacency': dict mapping face -> list of (neighbor_face, shared_edge)
            - Set of non-quad faces that were excluded
    """
    face_set = set(faces)
    visited = set()
    groups = []
    non_quad_faces = set()

    # First pass: identify non-quad faces
    for face in faces:
        if len(face.loops) != 4:
            non_quad_faces.add(face)
            visited.add(face)  # Mark as visited so they won't be added to groups

    quad_faces = [f for f in faces if f not in non_quad_faces]

    for start_face in quad_faces:
        if start_face in visited:
            continue

        # Start a new group
        group_faces = set()
        adjacency = {}  # face -> [(neighbor, edge), ...]

        # DFS stack: faces to process
        stack = [start_face]

        while stack:
            face = stack.pop()

            if face in visited:
                continue

            visited.add(face)
            group_faces.add(face)
            adjacency[face] = []

            # Collect all adjacent faces with their edges (for deterministic processing)
            neighbors_to_process = []
            for edge in face.edges:
                for neighbor in edge.link_faces:
                    if neighbor == face or neighbor not in face_set:
                        continue

                    # Check if edge already has a seam (respect existing seams)
                    if edge.seam:
                        continue

                    # Check if neighbor is a non-quad - treat as boundary
                    if neighbor in non_quad_faces:
                        edge.seam = True
                        continue

                    neighbors_to_process.append((neighbor, edge))

            # Sort by neighbor face index for deterministic order
            neighbors_to_process.sort(key=lambda x: x[0].index)

            for neighbor, edge in neighbors_to_process:
                # Check angle between faces
                angle = get_face_angle(face, neighbor)

                if angle > angle_threshold:
                    # Angle too large - mark seam
                    edge.seam = True
                else:
                    # Track adjacency within group
                    adjacency[face].append((neighbor, edge))

                    if neighbor not in visited:
                        stack.append(neighbor)

        if group_faces:
            groups.append({
                'faces': group_faces,
                'adjacency': adjacency,
            })
            debug_log(f"[topology] Created quad group with {len(group_faces)} faces")

    return groups, non_quad_faces


def build_dual_spanning_tree(group):
    """Build a spanning tree of faces (dual graph).

    Args:
        group: Group dict with 'faces' and 'adjacency'

    Returns:
        Tuple of:
            - Set of edges used in the dual spanning tree
            - Dict mapping face -> parent face (for tree structure)
    """
    faces = group['faces']
    adjacency = group['adjacency']

    if not faces:
        return set(), {}

    # BFS to build spanning tree (deterministic: start from lowest index face)
    start_face = min(faces, key=lambda f: f.index)
    visited = {start_face}
    parent = {start_face: None}
    tree_edges = set()  # Mesh edges used in dual tree

    queue = [start_face]
    queue_idx = 0

    while queue_idx < len(queue):
        face = queue[queue_idx]
        queue_idx += 1

        # Sort neighbors by index for deterministic traversal
        neighbors = sorted(adjacency.get(face, []), key=lambda x: x[0].index)
        for neighbor, edge in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = face
                tree_edges.add(edge)
                queue.append(neighbor)

    debug_log(f"[topology] Dual spanning tree: {len(tree_edges)} edges for {len(faces)} faces")
    return tree_edges, parent


def build_primal_spanning_tree(group, dual_tree_edges):
    """Build a spanning tree of vertices (primal graph), avoiding dual tree edges.

    Args:
        group: Group dict with 'faces'
        dual_tree_edges: Set of edges used in dual spanning tree

    Returns:
        Tuple of:
            - Set of edges used in the primal spanning tree
            - Dict mapping vertex -> (parent_vertex, edge_to_parent) for path tracing
    """
    faces = group['faces']

    # Collect all vertices and ALL edges in the group (including boundary edges)
    # The primal tree can use boundary edges freely since they're not crossed by the dual tree
    vertices = set()
    all_edges = set()
    for face in faces:
        for vert in face.verts:
            vertices.add(vert)
        for edge in face.edges:
            all_edges.add(edge)

    if not vertices:
        return set(), {}

    # Edges available for primal tree (all edges except those in dual tree)
    available_edges = all_edges - dual_tree_edges

    # BFS to build spanning tree using available edges (deterministic: start from lowest index vertex)
    start_vert = min(vertices, key=lambda v: v.index)
    visited = {start_vert}
    tree_edges = set()
    parent = {start_vert: None}  # vertex -> (parent_vertex, edge_to_parent)

    queue = [start_vert]
    queue_idx = 0

    while queue_idx < len(queue):
        vert = queue[queue_idx]
        queue_idx += 1

        # Sort edges by index for deterministic traversal
        sorted_edges = sorted(
            [e for e in vert.link_edges if e in available_edges],
            key=lambda e: e.index
        )
        for edge in sorted_edges:
            other_vert = edge.other_vert(vert)
            if other_vert not in visited and other_vert in vertices:
                visited.add(other_vert)
                tree_edges.add(edge)
                parent[other_vert] = (vert, edge)
                queue.append(other_vert)

    debug_log(f"[topology] Primal spanning tree: {len(tree_edges)} edges for {len(vertices)} vertices")
    return tree_edges, parent


def trace_path_in_tree(v1, v2, parent):
    """Trace the path between two vertices in a spanning tree.

    Args:
        v1: First vertex
        v2: Second vertex
        parent: Dict mapping vertex -> (parent_vertex, edge_to_parent)

    Returns:
        List of edges forming the path from v1 to v2
    """
    # Find path from v1 to root
    path1 = []
    ancestors1 = {v1}
    current = v1
    while parent.get(current) is not None:
        parent_vert, edge = parent[current]
        path1.append((current, parent_vert, edge))
        ancestors1.add(parent_vert)
        current = parent_vert

    # Find path from v2 to root, stopping when we hit an ancestor of v1
    path2 = []
    current = v2
    while current not in ancestors1:
        if parent.get(current) is None:
            # v2 is not connected to v1 in the tree
            debug_log(f"[topology] Warning: vertices not connected in tree")
            return []
        parent_vert, edge = parent[current]
        path2.append((current, parent_vert, edge))
        current = parent_vert

    # 'current' is now the lowest common ancestor (LCA)
    lca = current

    # Collect edges from v1 to LCA
    edges = []
    for v, pv, edge in path1:
        edges.append(edge)
        if pv == lca:
            break

    # Collect edges from v2 to LCA (these are already in order from v2 toward LCA)
    for v, pv, edge in path2:
        edges.append(edge)

    return edges


def compute_fundamental_cycles(group):
    """Compute the fundamental cycles for cutting a face group to genus 0.

    Each co-tree edge (edge not in dual or primal spanning tree) defines a
    fundamental cycle when combined with the path through the primal tree
    connecting its endpoints.

    Args:
        group: Group dict with 'faces' and 'adjacency'

    Returns:
        List of cycles, where each cycle is a list of edges
    """
    faces = group['faces']

    # Collect interior edges (shared by two faces in group)
    interior_edges = set()
    for face in faces:
        for edge in face.edges:
            faces_in_group = sum(1 for f in edge.link_faces if f in faces)
            if faces_in_group == 2:
                interior_edges.add(edge)

    # Build dual spanning tree
    dual_tree_edges, _ = build_dual_spanning_tree(group)

    # Build primal spanning tree (avoiding dual tree edges)
    primal_tree_edges, primal_parent = build_primal_spanning_tree(group, dual_tree_edges)

    # Co-tree edges = interior edges not in either tree
    cotree_edges = interior_edges - dual_tree_edges - primal_tree_edges

    debug_log(f"[topology] Co-tree edges: {len(cotree_edges)} (interior: {len(interior_edges)}, dual: {len(dual_tree_edges)}, primal: {len(primal_tree_edges)})")

    # For each co-tree edge, find the fundamental cycle
    cycles = []
    for edge in sorted(cotree_edges, key=lambda e: e.index):  # Deterministic order
        v1, v2 = edge.verts

        # Find path from v1 to v2 through the primal spanning tree
        tree_path = trace_path_in_tree(v1, v2, primal_parent)

        # The full cycle is the tree path plus the co-tree edge
        cycle = tree_path + [edge]
        cycles.append(cycle)

        debug_log(f"[topology] Fundamental cycle: {len(cycle)} edges (tree path: {len(tree_path)}, co-tree edge: 1)")

    return cycles


def mark_seams_for_genus_zero(bm, group):
    """Mark seams on edges to make a face group genus 0.

    For each co-tree edge, traces the fundamental cycle (the loop formed by
    the co-tree edge plus the path through the primal spanning tree) and
    marks all edges in that cycle as seams.

    Args:
        bm: BMesh instance
        group: Group dict with 'faces' and 'adjacency'

    Returns:
        Number of seams marked
    """
    cycles = compute_fundamental_cycles(group)

    marked_edges = set()
    for cycle in cycles:
        for edge in cycle:
            if edge not in marked_edges:
                edge.seam = True
                marked_edges.add(edge)

    if marked_edges:
        debug_log(f"[topology] Marked {len(marked_edges)} seams for genus 0 ({len(cycles)} cycles)")

    return len(marked_edges)


def get_boundary_edges_for_group(group):
    """Get all boundary edges of a face group.

    Boundary edges are edges that have only one face in the group,
    OR interior seam edges (both faces in group but marked as seam),
    since seams also act as UV boundaries.

    Args:
        group: Group dict with 'faces'

    Returns:
        Set of boundary edges
    """
    faces = group['faces']
    boundary_edges = set()

    for face in faces:
        for edge in face.edges:
            faces_in_group = sum(1 for f in edge.link_faces if f in faces)
            if faces_in_group == 1:
                boundary_edges.add(edge)
            elif faces_in_group == 2 and edge.seam:
                boundary_edges.add(edge)

    return boundary_edges


def find_boundary_loops(boundary_edges):
    """Group boundary edges into separate loops.

    Args:
        boundary_edges: Set of boundary edges

    Returns:
        List of boundary loops, where each loop is a set of vertices
    """
    if not boundary_edges:
        return []

    # Build adjacency for boundary vertices
    # A boundary vertex connects to other boundary vertices via boundary edges
    boundary_verts = set()
    vert_to_edges = {}
    for edge in boundary_edges:
        for vert in edge.verts:
            boundary_verts.add(vert)
            if vert not in vert_to_edges:
                vert_to_edges[vert] = []
            vert_to_edges[vert].append(edge)

    # Find connected components of boundary vertices
    visited = set()
    loops = []

    for start_vert in sorted(boundary_verts, key=lambda v: v.index):  # Deterministic
        if start_vert in visited:
            continue

        # BFS to find all vertices in this boundary loop
        loop_verts = set()
        queue = [start_vert]
        queue_idx = 0

        while queue_idx < len(queue):
            vert = queue[queue_idx]
            queue_idx += 1

            if vert in visited:
                continue

            visited.add(vert)
            loop_verts.add(vert)

            # Follow boundary edges to neighbors
            for edge in vert_to_edges.get(vert, []):
                other_vert = edge.other_vert(vert)
                if other_vert not in visited:
                    queue.append(other_vert)

        if loop_verts:
            loops.append(loop_verts)

    return loops


def find_path_between_boundaries(group, loop1_verts, loop2_verts):
    """Find the shortest path of edges connecting two boundary loops.

    Uses BFS through interior edges to find the shortest path.

    Args:
        group: Group dict with 'faces'
        loop1_verts: Set of vertices in first boundary loop
        loop2_verts: Set of vertices in second boundary loop

    Returns:
        List of edges forming the path, or empty list if no path found
    """
    faces = group['faces']

    # Collect all edges in the group (for traversal)
    all_edges = set()
    all_verts = set()
    for face in faces:
        for edge in face.edges:
            all_edges.add(edge)
        for vert in face.verts:
            all_verts.add(vert)

    # BFS from all vertices in loop1
    visited = set()
    parent = {}  # vert -> (parent_vert, edge)

    queue = []
    for vert in sorted(loop1_verts, key=lambda v: v.index):  # Deterministic
        queue.append(vert)
        visited.add(vert)
        parent[vert] = None

    queue_idx = 0
    target_vert = None

    while queue_idx < len(queue):
        vert = queue[queue_idx]
        queue_idx += 1

        # Check if we reached loop2
        if vert in loop2_verts and parent[vert] is not None:
            target_vert = vert
            break

        # Explore neighbors through group edges
        neighbor_edges = sorted(
            [e for e in vert.link_edges if e in all_edges],
            key=lambda e: e.index
        )
        for edge in neighbor_edges:
            other_vert = edge.other_vert(vert)
            if other_vert in visited or other_vert not in all_verts:
                continue

            visited.add(other_vert)
            parent[other_vert] = (vert, edge)
            queue.append(other_vert)

    if target_vert is None:
        debug_log("[topology] No path found between boundary loops")
        return []

    # Trace path back from target to source
    path = []
    current = target_vert
    while parent[current] is not None:
        prev_vert, edge = parent[current]
        path.append(edge)
        current = prev_vert

    debug_log(f"[topology] Found path of {len(path)} edges between boundaries")
    return path


def mark_seams_for_single_boundary(bm, group):
    """Mark seams to reduce multiple boundary loops to a single boundary.

    If a face group has multiple boundary loops (like a cylinder barrel),
    finds and marks a path connecting them.

    Args:
        bm: BMesh instance
        group: Group dict with 'faces'

    Returns:
        Number of seams marked
    """
    # Find boundary edges and group them into loops
    boundary_edges = get_boundary_edges_for_group(group)

    if not boundary_edges:
        debug_log("[topology] No boundary edges (closed surface)")
        return 0

    boundary_loops = find_boundary_loops(boundary_edges)
    debug_log(f"[topology] Found {len(boundary_loops)} boundary loops")

    if len(boundary_loops) <= 1:
        debug_log("[topology] Already has single boundary (or none)")
        return 0

    # Mark paths between consecutive boundary loops
    marked_count = 0
    for i in range(len(boundary_loops) - 1):
        loop1 = boundary_loops[i]
        loop2 = boundary_loops[i + 1]

        path = find_path_between_boundaries(group, loop1, loop2)
        for edge in path:
            if not edge.seam:
                edge.seam = True
                marked_count += 1

    if marked_count > 0:
        debug_log(f"[topology] Marked {marked_count} seams to connect {len(boundary_loops)} boundaries")

    return marked_count


def get_quad_islands(bm, faces, angle_threshold):
    """Process faces: group quads by angle and existing seams, mark additional seams, cut to disk.

    This function:
    1. Groups quad faces by angle threshold and existing seams (user seams are respected as boundaries)
    2. Treats non-quad faces (triangles, ngons) as blocking boundaries
    3. For each group, marks seams to reduce genus to 0 (fundamental cycles)
    4. For each group, marks seams to reduce boundary loops to 1 (path between boundaries)

    Args:
        bm: BMesh instance
        faces: List of faces to process
        angle_threshold: Maximum angle (radians) between normals before adding seam

    Returns:
        Tuple of:
            - List of quad face groups (each group is a set of faces)
            - Set of non-quad faces that were excluded from grouping
    """
    # Phase 1: Group quad faces by angle threshold and existing seams (non-quads excluded)
    groups, non_quad_faces = group_quad_faces_by_angle_and_existing_seams(bm, faces, angle_threshold)
    debug_log(f"[topology] Created {len(groups)} quad groups from {len(faces)} faces ({len(non_quad_faces)} non-quads excluded)")

    # Phase 2: For each group, mark seams for genus reduction (handles/holes)
    total_genus_cuts = 0
    for i, group in enumerate(groups):
        cuts = mark_seams_for_genus_zero(bm, group)
        total_genus_cuts += cuts
        if cuts > 0:
            debug_log(f"[topology] Group {i}: marked {cuts} seams for genus 0")

    debug_log(f"[topology] Total genus cuts: {total_genus_cuts}")

    # Phase 3: For each group, mark seams to connect multiple boundaries
    total_boundary_cuts = 0
    for i, group in enumerate(groups):
        cuts = mark_seams_for_single_boundary(bm, group)
        total_boundary_cuts += cuts
        if cuts > 0:
            debug_log(f"[topology] Group {i}: marked {cuts} seams for single boundary")

    debug_log(f"[topology] Total boundary cuts: {total_boundary_cuts}")

    # Return just the face sets and non-quad faces
    return [g['faces'] for g in groups], non_quad_faces
