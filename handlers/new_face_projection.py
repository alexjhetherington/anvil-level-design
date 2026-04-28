"""UV projection for newly created faces after topology changes."""

import bmesh
import bpy
from mathutils import Matrix, Vector

from ..core.logging import debug_log
from ..core.face_id import get_face_id_layer
from ..core.geometry import get_local_x_from_verts_3d
from ..core.uv_layers import get_unlocked_uv_layers
from ..core.hotspot_queries import face_has_hotspot_material

from .face_cache import face_data_cache, get_cached_layer_data


def _is_translated_from_cache(face, cached):
    cached_normal = cached.get('normal')
    cached_verts = cached.get('verts')
    if not cached_normal or not cached_verts:
        return False

    if (face.normal - cached_normal).length >= 0.01:
        return False

    current_verts = [v.co for v in face.verts]
    if len(current_verts) != len(cached_verts):
        return False

    offset = current_verts[0] - cached_verts[0]
    if offset.length <= 0.0001:
        return False

    return all((current_vert - cached_vert - offset).length < 0.0001
               for current_vert, cached_vert in zip(current_verts[1:],
                                                    cached_verts[1:]))


def _empty_face_cache_buckets():
    return {
        'id_zero': set(),
        'id_duplicate': set(),
        'hotspot': set(),
        'spin_rotated': set(),
        'cached_missing': set(),
        'cached_normal_changed': set(),
        'cached_vertex_count_changed': set(),
        'cached_exact': set(),
        'cached_translated': set(),
        'cached_same_plane_exact': set(),
        'cached_same_plane_changed': set(),
        'cached_off_plane': set(),
        'cached_same_normal_shape_changed': set(),
    }


def _classify_faces_by_cache_state(bm, me, id_layer, spin_rotated):
    """Bucket faces by literal ID/cache/geometry facts.

    These buckets intentionally do not decide projection policy. Callers derive
    operation roles such as "projection seed" or "preserved" from the facts.
    """
    buckets = _empty_face_cache_buckets()
    buckets['spin_rotated'] = set(spin_rotated)

    seen_ids = set()
    duplicated_ids = set()
    for face in bm.faces:
        if not face.is_valid:
            continue
        face_id = face[id_layer]
        if face_id == 0:
            continue
        if face_id in seen_ids:
            duplicated_ids.add(face_id)
        seen_ids.add(face_id)

    for face in bm.faces:
        if not face.is_valid:
            continue
        face_id = face[id_layer]
        if face_id == 0:
            buckets['id_zero'].add(face)
            continue

        if face_id in duplicated_ids:
            buckets['id_duplicate'].add(face)
        if face_has_hotspot_material(face, me):
            buckets['hotspot'].add(face)

        cached = face_data_cache.get(face_id)
        if not cached:
            buckets['cached_missing'].add(face)
            continue

        cached_normal = cached.get('normal')
        cached_verts = cached.get('verts')
        if not cached_normal or not cached_verts:
            buckets['cached_missing'].add(face)
            continue

        if (face.normal - cached_normal).length >= 0.01:
            buckets['cached_normal_changed'].add(face)
            continue

        current_verts = [v.co for v in face.verts]
        same_vertex_count = len(current_verts) == len(cached_verts)
        if not same_vertex_count:
            buckets['cached_vertex_count_changed'].add(face)

        is_exact = False
        if same_vertex_count:
            is_exact = all((current_vert - cached_vert).length < 0.0001
                           for current_vert, cached_vert in zip(current_verts,
                                                                cached_verts))
            if is_exact:
                buckets['cached_exact'].add(face)

            if _is_translated_from_cache(face, cached):
                buckets['cached_translated'].add(face)
            elif not is_exact:
                buckets['cached_same_normal_shape_changed'].add(face)

        cached_center = cached.get('center')
        if cached_center is None:
            continue

        dist_to_plane = abs(cached_normal.dot(
            face.calc_center_median() - cached_center))
        if dist_to_plane < 0.01:
            if is_exact:
                buckets['cached_same_plane_exact'].add(face)
            else:
                buckets['cached_same_plane_changed'].add(face)
        else:
            buckets['cached_off_plane'].add(face)

    return buckets


def _linked_faces(face):
    linked_faces = set()
    for edge in face.edges:
        for linked_face in edge.link_faces:
            if linked_face != face and linked_face.is_valid:
                linked_faces.add(linked_face)
    return linked_faces


def _has_cached_non_cap_neighbor(face, cap_candidates, topology_new_faces,
                                 id_layer):
    for linked_face in _linked_faces(face):
        if linked_face in cap_candidates:
            continue
        if linked_face in topology_new_faces:
            continue
        linked_id = linked_face[id_layer]
        if linked_id != 0 and linked_id in face_data_cache:
            return True
    return False


def _find_collapsed_extrude_material_index(face, topology_new_faces,
                                           cap_candidates, id_layer):
    if face.material_index != 0:
        return None
    if face.calc_area() >= 1e-8:
        return None

    cluster_candidates = topology_new_faces | cap_candidates
    cluster = set()
    queue = [face]
    while queue:
        current = queue.pop(0)
        if current in cluster:
            continue
        if current not in cluster_candidates:
            continue
        cluster.add(current)
        for linked_face in _linked_faces(current):
            if linked_face in cluster_candidates and linked_face not in cluster:
                queue.append(linked_face)

    source_material_indices = set()
    for cluster_face in cluster:
        if cluster_face in cap_candidates or cluster_face.material_index != 0:
            source_material_indices.add(cluster_face.material_index)

        for linked_face in _linked_faces(cluster_face):
            if linked_face in cluster_candidates:
                continue
            linked_id = linked_face[id_layer]
            if linked_id != 0 and linked_id in face_data_cache:
                source_material_indices.add(linked_face.material_index)

    if len(source_material_indices) != 1:
        return None

    source_material_index = next(iter(source_material_indices))
    if source_material_index == face.material_index:
        return None
    return source_material_index


def _repair_donorless_extrude_side_face_materials(topology_new_faces,
                                                  cap_candidates,
                                                  id_layer):
    """Copy cap material onto extrusion side faces with no cached donor.

    Blender's connected-region extrude can delete the originating face before
    assigning side-wall materials. If no existing radial donor face is found,
    those side walls keep BMesh's default material index 0. Repair either from
    a single cap candidate, or from an unambiguous collapsed zero-area extrude
    cluster before Blender has moved the cap away from its source face.
    """
    repaired_count = 0
    for face in topology_new_faces:
        if not face.is_valid:
            continue

        adjacent_caps = [
            linked_face for linked_face in _linked_faces(face)
            if linked_face in cap_candidates
        ]
        source_material_index = None
        if (len(adjacent_caps) == 1 and
                not _has_cached_non_cap_neighbor(
                    face, cap_candidates, topology_new_faces, id_layer)):
            source_material_index = adjacent_caps[0].material_index

        if source_material_index is None:
            source_material_index = _find_collapsed_extrude_material_index(
                face, topology_new_faces, cap_candidates, id_layer)

        if source_material_index is None:
            continue

        if face.material_index == source_material_index:
            continue

        debug_log(
            f"[ProjectNewFaces] Repaired donorless extrude side material: "
            f"face {face.index} mat={source_material_index}"
        )
        face.material_index = source_material_index
        repaired_count += 1

    return repaired_count


def _find_spin_rotated_faces(bm, id_layer):
    """Identify faces that are precise rotations of a cached face around the
    active spin operator's axis.

    Rationale: bmesh.ops.spin copies custom-data face layers (including
    anvil_face_id) from source faces onto the new rotated copies AND onto
    unrelated wall/wedge faces. We can't tell caps from walls by fid alone,
    but we can verify geometrically: a face that is a source's vertices
    rotated by k·step_angle around the spin axis IS a cap (Blender already
    transported UVs onto it); anything else with the same fid is not.

    Returns the set of faces to treat as "already has correct UVs". Only
    fires when the active operator is MESH_OT_spin.
    """
    active_op = bpy.context.active_operator
    if active_op is None or active_op.bl_idname != "MESH_OT_spin":
        return set()

    try:
        center = Vector(active_op.properties.center)
        axis = Vector(active_op.properties.axis)
        steps = int(active_op.properties.steps)
        angle = float(active_op.properties.angle)
    except (AttributeError, TypeError):
        return set()

    if steps <= 0 or axis.length < 1e-8:
        return set()
    axis = axis.normalized()
    step_angle = angle / steps

    result = set()
    pos_eps = 1e-3
    for face in bm.faces:
        if not face.is_valid:
            continue
        fid = face[id_layer]
        if fid == 0:
            continue
        cached = face_data_cache.get(fid)
        if not cached:
            continue
        cached_verts = cached.get('verts')
        if not cached_verts:
            continue
        current_verts = [v.co for v in face.verts]
        if len(current_verts) != len(cached_verts):
            continue

        # Try both rotation directions. Blender's spin angle sign vs CW/CCW
        # interpretation can vary with the gizmo's viewport-drag direction, so
        # we don't assume a sign.
        for k in list(range(1, steps + 1)) + list(range(-steps, 0)):
            rot = Matrix.Rotation(k * step_angle, 4, axis)
            matched = True
            for cv, cached_v in zip(current_verts, cached_verts):
                expected = rot @ (cached_v - center) + center
                if (expected - cv).length > pos_eps:
                    matched = False
                    break
            if matched:
                result.add(face)
                break
    return result


def get_best_neighbor_face(face, excluded_faces, id_layer, allow_fallback=True):
    """Find the best neighboring face to use as UV source.

    Priority 1: Prefer neighbors facing a similar direction (positive normal dot product).
    Priority 2: Among those, prefer sideways (wall-like) faces over floor/ceiling.
    Priority 3: Among coplanar neighbors with the same sideways score, prefer
                 the one whose cached center is closest to the target face
                 (i.e. the face this one was most likely split from).

    Falls back to negative-dot-product neighbors (with sideways scoring) if
    no similar-facing neighbor exists and allow_fallback is True.
    """
    best_similar = None
    best_similar_score = -1
    best_similar_dist = float('inf')
    best_fallback = None
    best_fallback_score = -1

    face_center = face.calc_center_median()

    for edge in face.edges:
        for linked_face in edge.link_faces:
            if linked_face == face or not linked_face.is_valid:
                continue
            if linked_face in excluded_faces:
                continue

            sideways_score = 1.0 - abs(linked_face.normal.z)

            if face.normal.dot(linked_face.normal) > 0:
                # For coplanar faces tying on sideways_score, prefer the one
                # whose cached center is closest (most likely the parent face)
                is_coplanar = (face.normal - linked_face.normal).length < 0.01
                if is_coplanar:
                    cached = face_data_cache.get(linked_face[id_layer])
                    if cached and cached.get('center'):
                        dist = (cached['center'] - face_center).length
                    else:
                        dist = (linked_face.calc_center_median() - face_center).length
                else:
                    dist = float('inf')

                if (sideways_score > best_similar_score or
                        (sideways_score == best_similar_score and dist < best_similar_dist)):
                    best_similar_score = sideways_score
                    best_similar_dist = dist
                    best_similar = linked_face
            else:
                if sideways_score > best_fallback_score:
                    best_fallback_score = sideways_score
                    best_fallback = linked_face

    if best_similar:
        return best_similar
    return best_fallback if allow_fallback else None


def project_new_faces(context, bm):
    """Apply UV projection to newly created faces after topology changes.

    Uses a BFS approach to handle both new faces and existing faces displaced
    by topology operations (e.g. the original face pushed aside during extrude):
    1. Collect "displaced normals" from cached faces whose normals changed
    2. Identify new faces, skipping those whose normals match a displaced normal
       (these are extruded copies where Blender already set correct UVs)
    3. Seed from new faces that border at least one cached face
    4. BFS expand through adjacency to cached faces with moved vertices
    5. Project all affected faces using unchanged neighbors as UV source
    """
    from ..operators.texture_apply import set_uv_from_other_face, set_uv_from_source_params

    obj = context.object
    me = obj.data
    unlocked_layers = get_unlocked_uv_layers(bm, obj, me)
    if not unlocked_layers:
        return

    props = context.scene.level_design_props
    ppm = props.pixels_per_meter

    id_layer = get_face_id_layer(bm)

    # Faces that are exact rotations (around the active spin axis) of their
    # cached source. Treated as "preserved" throughout because Blender has
    # already placed correct UVs on them. Populated only for MESH_OT_spin.
    spin_rotated = _find_spin_rotated_faces(bm, id_layer)
    face_buckets = _classify_faces_by_cache_state(
        bm, me, id_layer, spin_rotated)

    # Duplicate IDs are a symptom of Blender copying custom face data during
    # topology operations. These role sets are derived from factual buckets.
    duplicate_faces = face_buckets['id_duplicate']
    classifiable_duplicates = (
        duplicate_faces -
        face_buckets['spin_rotated'] -
        face_buckets['hotspot']
    )
    dupe_exact = classifiable_duplicates & face_buckets['cached_same_plane_exact']
    dupe_coplanar = (
        classifiable_duplicates &
        face_buckets['cached_same_plane_changed']
    )
    dupe_extrusions = classifiable_duplicates & face_buckets['cached_off_plane']
    dupe_other = duplicate_faces - (
        dupe_exact | dupe_coplanar | dupe_extrusions | spin_rotated
    )

    # --- Identify new faces and build the affected set ---
    topology_new_faces = (
        face_buckets['id_zero'] |
        dupe_other
    ) - spin_rotated

    new_faces = set()
    for f in topology_new_faces:
        if face_has_hotspot_material(f, me):
            continue
        new_faces.add(f)

    translated_cached_faces = face_buckets['cached_translated']
    cap_candidates = translated_cached_faces | dupe_exact
    repaired_materials = _repair_donorless_extrude_side_face_materials(
        topology_new_faces, cap_candidates, id_layer)

    if not new_faces and not dupe_coplanar:
        if repaired_materials > 0:
            bmesh.update_edit_mesh(me)
        return

    affected = set(new_faces)
    queue = list(affected)
    visited = set(affected) | dupe_extrusions | dupe_exact | spin_rotated
    while queue:
        current = queue.pop(0)
        for edge in current.edges:
            for neighbor in edge.link_faces:
                if neighbor in visited or not neighbor.is_valid:
                    continue
                visited.add(neighbor)

                if face_has_hotspot_material(neighbor, me):
                    continue

                neighbor_id = neighbor[id_layer]
                cached = face_data_cache.get(neighbor_id) if neighbor_id != 0 else None
                if not cached:
                    continue

                cached_verts = cached['verts']
                current_verts = [v.co for v in neighbor.verts]
                if len(current_verts) != len(cached_verts):
                    affected.add(neighbor)
                    queue.append(neighbor)
                    continue

                has_moved = False
                for cv, cached_v in zip(current_verts, cached_verts):
                    if (cv - cached_v).length > 0.0001:
                        has_moved = True
                        break
                if has_moved:
                    affected.add(neighbor)
                    queue.append(neighbor)

    # Step 1: Re-project coplanar faces whose geometry changed from cache.
    translated = affected & face_buckets['cached_translated']
    coplanar_modified = (
        affected &
        face_buckets['cached_same_normal_shape_changed']
    )
    coplanar_reproject = dupe_coplanar | coplanar_modified

    for face in coplanar_reproject:
        if face.calc_area() < 1e-8:
            continue
        face_id = face[id_layer]
        cached = face_data_cache.get(face_id)
        if not cached:
            continue
        cached_normal = cached['normal']
        cached_verts = cached['verts']

        source_local_x = get_local_x_from_verts_3d(cached_verts)
        if not source_local_x:
            continue
        source_local_y = cached_normal.cross(source_local_x).normalized()

        for uv_layer in unlocked_layers:
            layer_data = get_cached_layer_data(face_id, uv_layer.name)
            if layer_data:
                scale_u = layer_data.get('scale_u', 1.0)
                scale_v = layer_data.get('scale_v', 1.0)
                rotation = layer_data.get('rotation', 0.0)
                cached_uvs = layer_data.get('uvs')
            else:
                scale_u = cached.get('scale_u', 1.0)
                scale_v = cached.get('scale_v', 1.0)
                rotation = cached.get('rotation', 0.0)
                cached_uvs = None

            if abs(scale_u) < 1e-8 or abs(scale_v) < 1e-8:
                continue

            ref_point_co = cached_verts[0]
            if cached_uvs and len(cached_uvs) > 0:
                ref_point_uv = cached_uvs[0]
            else:
                continue

            set_uv_from_source_params(
                face, uv_layer, ppm, me, obj.matrix_world,
                scale_u, scale_v, rotation,
                cached_normal, source_local_x, source_local_y,
                ref_point_co, ref_point_uv,
            )
    affected -= coplanar_reproject
    affected -= translated

    # Step 2: Wavefront projection of remaining affected faces.
    excluded = (new_faces | dupe_extrusions | spin_rotated) - coplanar_reproject
    projected_count = len(coplanar_reproject)
    remaining = sorted(affected,
                       key=lambda f: f.calc_center_median().length_squared)
    allow_fallback = False
    made_progress = True
    while made_progress:
        made_progress = False
        still_remaining = []
        for face in remaining:
            if face.calc_area() < 1e-8:
                continue

            source_face = get_best_neighbor_face(face, excluded, id_layer,
                                                 allow_fallback)

            if not source_face:
                still_remaining.append(face)
                continue

            if face.material_index != source_face.material_index:
                debug_log(
                    f"[ProjectNewFaces] Propagated material: "
                    f"face {face.index} mat={source_face.material_index} "
                    f"(source=face {source_face.index})"
                )
                face.material_index = source_face.material_index

            # Check if this face already has non-zero UVs (e.g. set by box builder)
            _had_uvs = False
            if unlocked_layers:
                _check_layer = unlocked_layers[0]
                _check_uvs = [loop[_check_layer].uv.copy() for loop in face.loops]
                _uv_area = 0.0
                for _i in range(1, len(_check_uvs) - 1):
                    _ea = _check_uvs[_i] - _check_uvs[0]
                    _eb = _check_uvs[_i + 1] - _check_uvs[0]
                    _uv_area += abs(_ea.x * _eb.y - _ea.y * _eb.x)
                _had_uvs = _uv_area > 1e-8

            for uv_layer in unlocked_layers:
                set_uv_from_other_face(source_face, face, uv_layer, ppm, me, obj.matrix_world)

            if _had_uvs:
                debug_log(f"[ProjectNewFaces] Re-projected face {face.index} that already had UVs "
                          f"(source=face {source_face.index})")

            excluded.discard(face)
            projected_count += 1
            made_progress = True

        remaining = still_remaining
        if not made_progress and remaining and not allow_fallback:
            allow_fallback = True
            made_progress = True

    if projected_count > 0 or repaired_materials > 0:
        bmesh.update_edit_mesh(me)
