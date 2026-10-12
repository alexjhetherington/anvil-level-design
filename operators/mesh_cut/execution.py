"""
Shared Profile Cut - Custom Intersection Geometry

Custom intersection algorithm that:
1. Finds intersection vertices (edge-plane and prism-face)
2. Splits edges at intersection points
3. Removes faces inside the prism
4. Reconstructs partial faces
"""

import bmesh
from mathutils import Vector
from mathutils.geometry import intersect_line_line_2d

from .analysis import (
    analyze_convex_prism_cut,
    face_is_available,
)
from .concave_reconstruction import build_canonical_cut_graph
from .convex_prism import EPSILON
from ...core.logging import debug_log
from ...core.geometry import (
    compute_normal_from_verts,
    polygon_has_negligible_area,
)
from ...core.uv_projection import compute_uv_projection_from_face, apply_uv_projection_to_face
from ...core.uv_layers import get_all_uv_layers


RECONSTRUCTION_MODE_QUADS = 'QUADS'
RECONSTRUCTION_MODE_NGONS = 'NGONS'
RECONSTRUCTION_MODES = {
    RECONSTRUCTION_MODE_QUADS,
    RECONSTRUCTION_MODE_NGONS,
}


def execute_convex_prism_cut(obj, tool_settings, ppm, prism):
    """Run a convex-prism cut with the original quad reconstruction."""
    return execute_convex_prism_cut_with_reconstruction(
        obj,
        tool_settings,
        ppm,
        prism,
        RECONSTRUCTION_MODE_QUADS,
    )


def execute_convex_prism_cut_with_reconstruction(
        obj, tool_settings, ppm, prism, reconstruction_mode):
    """Cut an arbitrary mesh-local convex prism from an edit-mode mesh.

    The caller owns construction of ``prism`` and any context-dependent cache
    updates. Convex profiles retain the legacy radial reconstruction behavior.
    """
    return execute_prism_cut_with_face_reconstruction(
        obj,
        tool_settings,
        ppm,
        prism,
        reconstruction_mode,
        analyze_convex_prism_cut,
        None,
    )


def execute_prism_cut_with_face_reconstruction(
        obj, tool_settings, ppm, prism, reconstruction_mode,
        analysis_function, face_reconstruction_function):
    """Run the shared cut pipeline with an explicit face reconstructor."""
    if obj is None or obj.type != 'MESH':
        return (False, "No active mesh object")
    if not obj.data.is_editmode:
        return (False, "Active mesh must be in edit mode")
    if reconstruction_mode not in RECONSTRUCTION_MODES:
        return (False, f"Unknown face reconstruction mode: {reconstruction_mode}")

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # Analyze the unchanged BMesh before any geometry is deleted or split.
    # The result keeps the face and edge references that execution consumes;
    # faces_fully_inside is disjoint from faces_to_cut, and FACES_ONLY deletion
    # preserves the boundary geometry referenced by the remaining analysis.
    analysis = analysis_function(bm, prism)
    faces_to_be_cut = analysis.faces_to_cut
    face_interior_points = analysis.face_interior_points

    # Delete faces that are entirely inside the prism
    if analysis.faces_fully_inside:
        debug_log(
            f"[CubeCut] Deleting {len(analysis.faces_fully_inside)} faces "
            "entirely inside prism"
        )
        bmesh.ops.delete(
            bm,
            geom=list(analysis.faces_fully_inside),
            context='FACES_ONLY',
        )
        bm.faces.ensure_lookup_table()

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
                      f"Run Mesh > Clean Up > Merge by Distance first.", flush=True)
                debug_log(f"[CubeCut] ABORTING: Face {face.index} has duplicate verts at {key}")
                bmesh.update_edit_mesh(me)
                return (False, "Face has duplicate vertices - run Merge by Distance first")
            seen_positions[key] = v

    # === STEP 2: Find edge-plane intersections and split edges ===
    # Only split edges that belong to faces that will be cut
    debug_log(f"\n[CubeCut] === STEP 2: Find edge-plane intersections ===")
    edge_splits = analysis.edge_splits
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

        points = face_interior_points.get(face, [])

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

    canonical_cut_graph = None
    if face_reconstruction_function is not None:
        canonical_cut_graph = build_canonical_cut_graph(
            bm,
            faces_to_be_cut,
            prism,
            split_verts,
            dict(face_interior_verts),
        )
        split_verts = [
            vertex for vertex in split_verts if vertex.is_valid
        ]
        face_interior_verts = [
            (
                face,
                [vertex for vertex in interior_verts if vertex.is_valid],
            )
            for face, interior_verts in face_interior_verts
            if face.is_valid
        ]

    # === STEP 4: Capture face data and delete faces being processed ===
    # Capture vertex data for each face BEFORE deleting faces
    debug_log(f"\n[CubeCut] === STEP 4: Capture face data and delete faces ===")
    split_verts_set = set(split_verts)
    face_data_list = []
    faces_to_delete = []

    # Get UV layer for capturing projection data
    uv_layer = bm.loops.layers.uv.active

    for face, interior_verts in face_interior_verts:
        if face is None or not face.is_valid:
            continue

        # Capture face normal BEFORE deleting - used for consistent vertex sorting
        face_normal = face.normal.copy()

        # Find edge-split vertices that are on this face's boundary
        # Also include existing face vertices that lie on the prism boundary.
        # These are effectively split vertices where a prism edge meets one.
        edge_verts_on_face = [
            vertex for vertex in face.verts
            if vertex in split_verts_set or (
                prism.point_inside(vertex.co) and
                not prism.point_strictly_inside(vertex.co)
            )
        ]

        # Compute input variables for _verts_to_faces
        # Use face normal for consistent winding in angle sorting
        verts_to_delete = [
            vertex for vertex in face.verts
            if vertex not in split_verts_set and
            _should_delete_vertex_for_face(vertex, face, prism)
        ]

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
        if (
                face_reconstruction_function is None
                and num_edge_unique == 0
                and num_interior_unique <= 2):
            print(f"Level Design Tools: Skipping face {face.index} - only {num_interior_unique} unique interior verts, cannot form valid shape", flush=True)
            # Remove orphaned interior vertices
            for v in interior_verts:
                if v.is_valid:
                    bm.verts.remove(v)
            continue

        # Skip if only edge verts, 2 or fewer unique, and no deleted verts (cut doesn't remove anything)
        if (
                face_reconstruction_function is None
                and num_interior_unique == 0
                and num_edge_unique <= 2
                and num_deleted == 0):
            print(f"Level Design Tools: Skipping face {face.index} - only {num_edge_unique} unique edge verts with no deleted verts, cannot form valid shape", flush=True)
            # Remove orphaned interior vertices
            for v in interior_verts:
                if v.is_valid:
                    bm.verts.remove(v)
            continue

        original_face_verts = list(face.verts)
        verts_to_delete_set = set(verts_to_delete)
        new_verts = _sort_verts_by_angle_with_normal(list(interior_verts) + [v for v in edge_verts_on_face if v not in verts_to_delete_set], face_normal)
        verts_in_original_interior = _sort_verts_by_angle_with_normal(list(interior_verts), face_normal)
        verts_on_original_exterior = _sort_verts_by_angle_with_normal([v for v in face.verts if v not in verts_to_delete], face_normal)

        # Capture UV projection data for ALL layers and material index before deleting the face
        uv_projections = {}
        all_layers = get_all_uv_layers(bm, me)
        for layer in all_layers:
            proj = compute_uv_projection_from_face(face, layer)
            if proj is not None:
                uv_projections[layer.name] = proj
        material_index = face.material_index

        # Snapshot the host polygon's edges so the bridge picker can
        # test segment-visibility against the original boundary. These
        # survive the 'FACES_ONLY' delete below.
        host_edges = list(face.edges)

        cut_segments = None
        suppressed_cap_indices = set()
        if canonical_cut_graph is not None:
            cut_segments = canonical_cut_graph.segments_for_face(face)
            suppressed_cap_indices = (
                canonical_cut_graph.suppressed_cap_indices_for_face(face)
            )
            new_verts.extend(
                vertex
                for vertex in canonical_cut_graph.vertices_for_face(face)
                if vertex not in new_verts
            )

        face_data_list.append((
            new_verts,
            verts_on_original_exterior,
            verts_in_original_interior,
            original_face_verts,
            face_normal,
            uv_projections,
            material_index,
            host_edges,
            cut_segments,
            suppressed_cap_indices,
        ))
        faces_to_delete.append(face)
        debug_log(f"[CubeCut] Captured data for face {face.index}: {len(new_verts)} new_verts, {len(verts_on_original_exterior)} exterior, {len(verts_in_original_interior)} interior, uv_layers={len(uv_projections)}")

    # Delete only the faces we're processing
    debug_log(f"[CubeCut] Deleting {len(faces_to_delete)} faces")
    bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES_ONLY')

    # === STEP 5: Rebuild faces from captured vertex data ===
    debug_log(f"\n[CubeCut] === STEP 5: Rebuild faces ===")
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    newly_created_faces = []
    for (
            new_verts,
            verts_on_original_exterior,
            verts_in_original_interior,
            original_face_verts,
            face_normal,
            uv_projections,
            material_index,
            host_edges,
            cut_segments,
            suppressed_cap_indices,
            ) in face_data_list:
        if face_reconstruction_function is None:
            new_faces, connector_edges = _verts_to_faces(
                bm, new_verts, verts_on_original_exterior,
                verts_in_original_interior, face_normal, prism, me, ppm,
                vert_plane_map, host_edges,
            )
        else:
            new_faces, connector_edges = face_reconstruction_function(
                bm,
                original_face_verts,
                new_verts,
                face_normal,
                prism,
                cut_segments,
                suppressed_cap_indices,
                canonical_cut_graph,
            )
        if reconstruction_mode == RECONSTRUCTION_MODE_NGONS:
            new_faces = _minimize_face_local_connector_edges(
                bm,
                new_faces,
                connector_edges,
                face_normal,
            )
        elif face_reconstruction_function is not None:
            new_faces = _join_face_local_triangles_to_quads(
                bm,
                new_faces,
                connector_edges,
                face_normal,
            )
        if new_faces:
            for new_face in new_faces:
                # Apply material from original face
                new_face.material_index = material_index
                # Apply UV projection from original face to ALL layers
                for layer_name, proj in uv_projections.items():
                    layer = bm.loops.layers.uv.get(layer_name)
                    if layer is not None:
                        u_axis, v_axis, origin_uv, origin_pos, source_normal = proj
                        apply_uv_projection_to_face(new_face, layer, u_axis, v_axis, origin_uv, origin_pos, source_normal)
            newly_created_faces.extend(new_faces)

    #
    # === STEP 6: Quadrilate n-gons created by edge splits ===
    # Faces that had edges split but weren't cut are now n-gons and need to be quadrilated
    # Also quadrilate any newly created faces from cutting that are n-gons
    faces_to_quadrilate = []

    if reconstruction_mode == RECONSTRUCTION_MODE_QUADS:
        if face_reconstruction_function is not None:
            # Rebuild each untouched neighbor independently. Its current
            # edges are the source-face boundary after edge splitting, so
            # they must remain protected while only face-local connectors
            # may be removed to form quads.
            for face in faces_with_split_edges:
                if not face_is_available(face) or len(face.verts) <= 4:
                    continue

                projections = {}
                for layer in get_all_uv_layers(bm, me):
                    projection = compute_uv_projection_from_face(face, layer)
                    if projection is not None:
                        projections[layer.name] = projection
                material_index = face.material_index
                face_normal = face.normal.copy()
                local_faces = _quadrangulate_face_locally(
                    bm,
                    face,
                    face_normal,
                )
                for local_face in local_faces:
                    local_face.material_index = material_index
                    for layer_name, projection in projections.items():
                        layer = bm.loops.layers.uv.get(layer_name)
                        if layer is None:
                            continue
                        (
                            u_axis,
                            v_axis,
                            origin_uv,
                            origin_pos,
                            source_normal,
                        ) = projection
                        apply_uv_projection_to_face(
                            local_face,
                            layer,
                            u_axis,
                            v_axis,
                            origin_uv,
                            origin_pos,
                            source_normal,
                        )
        else:
            # Preserve the legacy convex reconstruction behavior.
            for face in faces_with_split_edges:
                if not face_is_available(face):
                    continue
                if len(face.verts) > 4:
                    faces_to_quadrilate.append(face)
                    debug_log(
                        f"[CubeCut] Adjacent face needs quadrilating: "
                        f"{len(face.verts)} verts"
                    )

            for face in newly_created_faces:
                if not face.is_valid:
                    continue
                if len(face.verts) > 4:
                    faces_to_quadrilate.append(face)
                    debug_log(
                        f"[CubeCut] Newly created face needs quadrilating: "
                        f"{len(face.verts)} verts"
                    )

    if faces_to_quadrilate:
        debug_log(f"[CubeCut] === STEP 6: Quadrilating {len(faces_to_quadrilate)} n-gon faces ===")
        bm.normal_update()

        # Capture UV projections and vertex sets before triangulation destroys the n-gons.
        # After join_triangles, a resulting face may span two original n-gons
        # (e.g. merging triangles across a shared edge). We re-project its UVs
        # from the original n-gon whose vertices contain all of the result's vertices.
        all_layers = get_all_uv_layers(bm, me)
        ngon_uv_data = []  # list of (vert_set, {layer_name: proj}, material_index)
        for face in faces_to_quadrilate:
            projections = {}
            for layer in all_layers:
                proj = compute_uv_projection_from_face(face, layer)
                if proj is not None:
                    projections[layer.name] = proj
            ngon_uv_data.append((set(face.verts), projections, face.material_index))

        # Triangulate the n-gons first
        result = bmesh.ops.triangulate(bm, faces=faces_to_quadrilate)
        new_tris = result['faces']
        quadrangulation_result_faces = {
            face for face in new_tris
            if face_is_available(face)
        }
        debug_log(f"[CubeCut] Triangulated into {len(new_tris)} triangles")

        # Join triangles into quads where possible
        if new_tris:
            join_result = bmesh.ops.join_triangles(
                bm,
                faces=new_tris,
                angle_face_threshold=3.14159,  # ~180 degrees - allow any face angle
                angle_shape_threshold=3.14159  # ~180 degrees - allow any shape
            )
            quadrangulation_result_faces.update(
                face for face in join_result.get('faces', [])
                if face_is_available(face)
            )
            debug_log(f"[CubeCut] Joined triangles into quads where possible")

        # Re-project UVs on faces whose vertices span multiple original n-gons.
        # Restrict this to faces explicitly returned by the triangulate/join
        # operations. Other mesh faces can share vertices with these source
        # n-gons without having participated in quadrangulation.
        for face in quadrangulation_result_faces:
            if not face_is_available(face):
                continue
            face_verts = set(face.verts)
            # Find which original n-gon(s) contain this face's vertices
            containing = [data for data in ngon_uv_data if face_verts <= data[0]]
            if len(containing) >= 1:
                continue  # Entirely within one n-gon, UVs are fine
            # Face spans multiple n-gons - use the first n-gon with overlap
            for ngon_verts, projections, mat_idx in ngon_uv_data:
                if face_verts & ngon_verts:
                    for layer_name, proj in projections.items():
                        layer = bm.loops.layers.uv.get(layer_name)
                        if layer is not None:
                            u_axis, v_axis, origin_uv, origin_pos, source_normal = proj
                            apply_uv_projection_to_face(face, layer, u_axis, v_axis, origin_uv, origin_pos, source_normal)
                    debug_log(f"[CubeCut] Re-projected UVs on cross-boundary face with {len(face.verts)} verts")
                    break

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

    # Merge very close vertices without welding separate mesh islands that
    # happen to have colocated coordinates.
    _remove_doubles_within_connected_components(
        bm, [v for v in bm.verts if v.is_valid], EPSILON)

    # Recalculate normals for newly created faces
    bm.normal_update()

    # === STEP 8: Select cut boundary edges ===
    # Switch to edge select mode and select only edges on the cut boundary
    # (edges with exactly 1 linked face where both vertices lie on the prism surface)
    bm.select_mode = {'EDGE'}
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False

    boundary_count = 0
    for e in bm.edges:
        if not e.is_valid:
            continue
        if canonical_cut_graph is not None:
            is_cut_boundary = (
                len(e.link_faces) == 1
                and canonical_cut_graph.edge_is_boundary(e)
            )
        else:
            is_cut_boundary = (
                len(e.link_faces) == 1
                and prism.point_on_surface(e.verts[0].co)
                and prism.point_on_surface(e.verts[1].co)
            )
        if is_cut_boundary:
            e.select = True
            boundary_count += 1

    bm.select_flush_mode()
    debug_log(f"[CubeCut] Selected {boundary_count} cut boundary edges")

    bmesh.update_edit_mesh(me)

    # Set Blender's mesh select mode to edge
    tool_settings.mesh_select_mode = (False, True, False)

    return (True, "Cut complete")


def _split_edges_at_intersections(bm, edge_splits):
    """
    Split edges at intersection points.

    Returns:
        tuple: (newly created vertices, set of faces that had edges split,
                dict mapping each split vert to its prism plane indices)
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
            # (happens when a prism edge passes through a mesh edge)
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

            # Track which prism plane(s) this vertex lies on (as a set,
            # since a vertex at a prism edge can be on multiple planes)
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


def _connected_vertex_components(verts):
    """Return edge-connected components from the supplied vertices."""
    remaining = set(v for v in verts if v.is_valid)
    components = []

    while remaining:
        start = remaining.pop()
        component = [start]
        stack = [start]

        while stack:
            current = stack.pop()
            for edge in current.link_edges:
                if not edge.is_valid:
                    continue
                other = edge.other_vert(current)
                if other not in remaining:
                    continue
                remaining.remove(other)
                component.append(other)
                stack.append(other)

        components.append(component)

    return components


def _remove_doubles_within_connected_components(bm, verts, dist):
    """Merge close verts only within each existing edge-connected island."""
    components = _connected_vertex_components(verts)
    for component in components:
        valid_component = [v for v in component if v.is_valid]
        if len(valid_component) < 2:
            continue
        bmesh.ops.remove_doubles(bm, verts=valid_component, dist=dist)


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


def _quadrangulate_face_locally(bm, face, face_normal):
    """Triangulate one split source face without crossing its boundary."""
    protected_boundary_edges = set(face.edges)
    result = bmesh.ops.triangulate(bm, faces=[face])
    local_faces = _discard_numerically_degenerate_faces(
        bm,
        result.get('faces', []),
    )
    connector_edges = {
        edge
        for local_face in local_faces
        for edge in local_face.edges
        if edge not in protected_boundary_edges
    }
    return _join_face_local_triangles_to_quads(
        bm,
        local_faces,
        connector_edges,
        face_normal,
    )


def _discard_numerically_degenerate_faces(bm, faces):
    """Remove face-local triangulation cells with negligible altitude."""
    valid_faces = [
        face for face in faces
        if face is not None and face.is_valid
    ]
    degenerate_faces = [
        face for face in valid_faces
        if polygon_has_negligible_area(
            [vertex.co for vertex in face.verts],
            EPSILON,
        )
    ]

    if degenerate_faces:
        bmesh.ops.delete(
            bm,
            geom=degenerate_faces,
            context='FACES_ONLY',
        )
        debug_log(
            f"[PrismCut] Removed {len(degenerate_faces)} numerically "
            "degenerate face-local triangulation cells"
        )
    return [face for face in valid_faces if face.is_valid]


def _join_face_local_triangles_to_quads(
        bm, faces, connector_edges, face_normal):
    """Join triangle pairs only across one source face's interior edges."""
    region_faces = {
        face for face in faces
        if face is not None and face.is_valid
    }
    remaining_connectors = {
        edge for edge in connector_edges
        if edge is not None and edge.is_valid
    }

    for connector_edge in sorted(
            remaining_connectors,
            key=_face_local_connector_merge_key):
        if not connector_edge.is_valid:
            continue
        linked_faces = [
            face for face in connector_edge.link_faces
            if face in region_faces and face.is_valid
        ]
        if len(linked_faces) != 2:
            continue

        first_face, second_face = linked_faces
        if len(first_face.verts) != 3 or len(second_face.verts) != 3:
            continue
        shared_edges = set(first_face.edges) & set(second_face.edges)
        if shared_edges != {connector_edge}:
            continue

        boundary_edges = [
            edge
            for edge in set(first_face.edges) | set(second_face.edges)
            if edge is not connector_edge
        ]
        boundary_verts = _trace_simple_edge_cycle(boundary_edges)
        if boundary_verts is None or len(boundary_verts) != 4:
            continue

        merged_face = _merge_faces_across_connector(
            bm,
            first_face,
            second_face,
            connector_edge,
            face_normal,
        )
        if merged_face is None:
            continue
        region_faces.discard(first_face)
        region_faces.discard(second_face)
        region_faces.add(merged_face)

    return [face for face in region_faces if face.is_valid]


def _should_delete_vertex_for_face(vertex, face, prism):
    """Check if a vertex should be deleted when processing a specific face.

    Returns True if:
    - Vertex is strictly inside the prism, OR
    - Vertex is on the prism boundary and has no face edge leading outside

    Only considers edges that belong to the given face.
    """
    if prism.point_strictly_inside(vertex.co):
        return True

    if prism.point_inside(vertex.co):
        # On boundary - check if any edge on this face goes outside
        face_edges = set(face.edges)
        for edge in vertex.link_edges:
            if edge not in face_edges:
                continue  # Skip edges not on this face
            other_vert = edge.other_vert(vertex)
            if not prism.point_inside(other_vert.co):
                return False  # Has edge outside on this face, keep it
        return True  # No edges on this face go outside, delete it

    return False  # Outside prism, keep it


def _segment_visible_in_polygon(a, b, edges, face_normal):
    """True if segment a-b doesn't strictly cross any edge in `edges`,
    other than touching at a shared endpoint. Checks in the face plane
    via projection onto the 2D axes dominant relative to face_normal.
    """
    normal_abs = Vector((abs(face_normal.x), abs(face_normal.y), abs(face_normal.z)))
    if normal_abs.x >= normal_abs.y and normal_abs.x >= normal_abs.z:
        def to_2d(p):
            return Vector((p.y, p.z))
    elif normal_abs.y >= normal_abs.z:
        def to_2d(p):
            return Vector((p.x, p.z))
    else:
        def to_2d(p):
            return Vector((p.x, p.y))

    a2 = to_2d(a.co)
    b2 = to_2d(b.co)
    for e in edges:
        if not e.is_valid:
            continue
        h1 = e.verts[0]
        h2 = e.verts[1]
        # Edges adjacent to the candidate endpoint legitimately touch
        # the segment at `b`; that's not a crossing.
        if b is h1 or b is h2:
            continue
        if a is h1 or a is h2:
            continue
        if intersect_line_line_2d(a2, b2, to_2d(h1.co), to_2d(h2.co)) is not None:
            return False
    return True


def _minimize_face_local_connector_edges(
        bm, created_faces, connector_edges, face_normal):
    """Merge reconstruction cells without crossing an original-face boundary.

    ``connector_edges`` contains only the spokes added while rebuilding one
    deleted source face. Cutter-boundary edges, surviving source edges, and
    shared mesh edges are deliberately absent, so no merge can cross from one
    original face to another.

    A connector can be removed when its two faces share only that edge and its
    endpoints. If the faces share two connectors, removing either would create
    one face with a hole, which Blender cannot represent as a valid BMFace. The
    iteration therefore naturally leaves two connectors around each enclosed
    cut loop while removing every connector from a notch or split component.
    """
    region_faces = {
        face for face in created_faces
        if face is not None and face.is_valid
    }
    remaining_connectors = {
        edge for edge in connector_edges
        if edge is not None and edge.is_valid
    }
    rejected_connectors = set()
    initial_connector_count = len(remaining_connectors)

    while True:
        merged_any = False
        for connector_edge in sorted(
                remaining_connectors,
                key=_face_local_connector_merge_key):
            if not connector_edge.is_valid:
                remaining_connectors.discard(connector_edge)
                continue
            if connector_edge in rejected_connectors:
                continue
            if len(connector_edge.link_faces) != 2:
                continue

            first_face, second_face = connector_edge.link_faces
            if first_face not in region_faces or second_face not in region_faces:
                continue
            if not first_face.is_valid or not second_face.is_valid:
                continue

            shared_edges = set(first_face.edges) & set(second_face.edges)
            if shared_edges != {connector_edge}:
                continue
            shared_verts = set(first_face.verts) & set(second_face.verts)
            if shared_verts != set(connector_edge.verts):
                continue

            merged_face = _merge_faces_across_connector(
                bm,
                first_face,
                second_face,
                connector_edge,
                face_normal,
            )
            if merged_face is None:
                rejected_connectors.add(connector_edge)
                continue

            region_faces.discard(first_face)
            region_faces.discard(second_face)
            region_faces.add(merged_face)
            remaining_connectors.discard(connector_edge)
            rejected_connectors.clear()
            merged_any = True
            break

        if not merged_any:
            break

    final_faces = [face for face in region_faces if face.is_valid]
    final_connector_count = sum(
        1 for edge in remaining_connectors
        if edge.is_valid and len(edge.link_faces) == 2
    )
    debug_log(
        f"[CubeCut] N-gon reconstruction merged "
        f"{initial_connector_count - final_connector_count} face-local "
        f"connectors; kept {final_connector_count}"
    )
    return final_faces


def _face_local_connector_merge_key(edge):
    """Prefer removing longer connectors, with a coordinate-stable tie-break."""
    if not edge.is_valid:
        return (0.0, ())
    delta = edge.verts[1].co - edge.verts[0].co
    coordinates = tuple(sorted(
        (
            round(vertex.co.x, 6),
            round(vertex.co.y, 6),
            round(vertex.co.z, 6),
        )
        for vertex in edge.verts
    ))
    return (-delta.length_squared, coordinates)


def _merge_faces_across_connector(
        bm, first_face, second_face, connector_edge, face_normal):
    """Replace two cells sharing one connector with their simple union."""
    boundary_edges = [
        edge for edge in set(first_face.edges) | set(second_face.edges)
        if edge is not connector_edge
    ]
    boundary_verts = _trace_simple_edge_cycle(boundary_edges)
    if boundary_verts is None or len(boundary_verts) < 3:
        return None

    merged_normal = compute_normal_from_verts(
        [vertex.co for vertex in boundary_verts]
    )
    if merged_normal is None or merged_normal.length < EPSILON:
        return None
    if merged_normal.dot(face_normal) < 0:
        boundary_verts.reverse()

    # Create the replacement before deleting either source cell. BMesh allows
    # the short-lived non-manifold radial state, and this keeps a failed create
    # non-destructive.
    try:
        merged_face = bm.faces.new(boundary_verts)
    except ValueError:
        return None

    bmesh.ops.delete(
        bm,
        geom=[first_face, second_face],
        context='FACES_ONLY',
    )
    if connector_edge.is_valid and not connector_edge.link_faces:
        bmesh.ops.delete(bm, geom=[connector_edge], context='EDGES')
    return merged_face


def _trace_simple_edge_cycle(edges):
    """Return ordered vertices when ``edges`` form exactly one simple cycle."""
    if len(edges) < 3:
        return None

    adjacency = {}
    for edge in edges:
        if not edge.is_valid:
            return None
        for vertex in edge.verts:
            adjacency.setdefault(vertex, []).append(edge)

    if any(len(vertex_edges) != 2 for vertex_edges in adjacency.values()):
        return None

    start_vertex = edges[0].verts[0]
    ordered_verts = [start_vertex]
    used_edges = set()
    current_vertex = start_vertex
    previous_edge = None

    for _ in range(len(edges)):
        candidates = [
            edge for edge in adjacency[current_vertex]
            if edge is not previous_edge
        ]
        if not candidates:
            return None

        next_edge = candidates[0]
        if next_edge in used_edges:
            return None
        used_edges.add(next_edge)
        next_vertex = next_edge.other_vert(current_vertex)

        if next_vertex is start_vertex:
            if len(used_edges) != len(edges):
                return None
            return ordered_verts
        if next_vertex in ordered_verts:
            return None

        ordered_verts.append(next_vertex)
        current_vertex = next_vertex
        previous_edge = next_edge

    return None


def _verts_to_faces(bm, new_verts, verts_on_original_exterior, verts_in_original_interior, face_normal, prism, me, ppm, vert_plane_map, host_edges):
    import math

    debug_log(f"[CubeCut] _verts_to_faces: new_verts={[v.co[:] for v in new_verts]}")
    debug_log(f"[CubeCut] _verts_to_faces: verts_on_original_exterior={[v.co[:] for v in verts_on_original_exterior]}")
    debug_log(f"[CubeCut] _verts_to_faces: verts_in_original_interior={[v.co[:] for v in verts_in_original_interior]}")
    debug_log(f"[CubeCut] _verts_to_faces: face_normal={face_normal[:]}")

    if len(new_verts) < 2:
        return ([], [])

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

        # Skip cross-hole edges between split vertices that share no prism plane.
        # When the prism cuts through a face, each prism plane creates split
        # vertices on the face edges. Valid closing edges connect splits from the
        # SAME plane (sealing one side of the cut). Edges between splits from
        # entirely DIFFERENT planes would bridge across the removed region.
        # On quad faces this is implicitly prevented because both splits on the
        # same original edge create an "already exists" barrier, but on triangles
        # (or other odd faces) the splits land on different original edges with
        # no such barrier. We use sets of plane indices (not single values) because
        # a vertex at a prism edge or corner can belong to multiple planes.
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
    connector_edges = []
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

        # Find the exterior vertex closest to the "away" direction that
        # is also visible from the interior vert (the bridge segment must
        # stay inside the host polygon — no crossing the host boundary).
        best_vert = None
        best_dot = -2.0  # Will look for highest dot product with away_dir

        for ext_vert in verts_on_original_exterior:
            if ext_vert == interior_vert:
                continue
            if not _segment_visible_in_polygon(interior_vert, ext_vert, host_edges, face_normal):
                debug_log(f"[CubeCut]   Skipping bridge to {ext_vert.co[:]} - not visible from {interior_vert.co[:]}")
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
            print(f"Level Design Tools: Error - Interior vertex {interior_vert.co[:]} closest exterior vertex {best_vert.co[:]} is inside the cut angle", flush=True)
            continue

        # Create edge to the best exterior vertex
        edge_exists = any(best_vert in e.verts for e in interior_vert.link_edges)
        if not edge_exists:
            try:
                connector_edge = bm.edges.new([interior_vert, best_vert])
                connector_edges.append(connector_edge)
                debug_log(f"[CubeCut]   Connected interior {interior_vert.co[:]} to exterior {best_vert.co[:]}")
            except ValueError:
                debug_log(f"[CubeCut]   Failed to connect interior {interior_vert.co[:]} to exterior {best_vert.co[:]}")
        else:
            debug_log(f"[CubeCut]   Edge already exists: interior {interior_vert.co[:]} to exterior {best_vert.co[:]}")

    # Create faces - one for each edge created between new verts
    # Use the provided face_normal for angular ordering (captured from original face)
    if face_normal is None or face_normal.length < EPSILON:
        debug_log(f"[CubeCut]   Invalid face normal for angular ordering")
        return ([], connector_edges)

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
            # Check if the polygon's normal matches the expected face normal
            # If not, reverse the vertex order to correct the winding
            poly_normal = compute_normal_from_verts([v.co for v in face_verts])
            if poly_normal.dot(face_normal) < 0:
                face_verts.reverse()
                debug_log(f"[CubeCut]   Reversed winding for face with {len(face_verts)} verts")

            try:
                new_face = bm.faces.new(face_verts)
                created_faces.append(new_face)
                debug_log(f"[CubeCut]   Created face with {len(face_verts)} verts")
            except ValueError as e:
                debug_log(f"[CubeCut]   Failed to create face: {e}")

    return (created_faces, connector_edges)
