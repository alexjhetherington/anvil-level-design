"""UV projection for newly created faces after topology changes."""

import math

import bmesh
import bpy
from mathutils import Matrix, Vector

from ..core.logging import debug_log
from ..core.face_id import get_face_id_layer
from ..core.geometry import get_local_x_from_verts_3d, normalize_offset
from ..core.uv_layers import get_unlocked_uv_layers
from ..core.uv_projection import (
    apply_uv_to_face,
    derive_transform_from_uvs,
    get_face_local_axes,
)
from ..core.materials import get_texture_dimensions_from_material
from ..core.hotspot_queries import face_has_hotspot_material

from .face_cache import face_data_cache, get_cached_layer_data


_NORMAL_SIMILARITY_THRESHOLD = 0.0
_AXIS_EPSILON = 1e-4
_AXIS_PARALLEL_DOT = 0.99


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


def _face_geometry_key(face):
    """Return a deterministic geometry-first ordering key for a BMesh face."""
    center = face.calc_center_median()
    normal = face.normal
    vertices = tuple(sorted(
        (
            round(vertex.co.x, 6),
            round(vertex.co.y, 6),
            round(vertex.co.z, 6),
        )
        for vertex in face.verts
    ))
    return (
        round(center.length_squared, 8),
        round(center.x, 8),
        round(center.y, 8),
        round(center.z, 8),
        round(normal.x, 6),
        round(normal.y, 6),
        round(normal.z, 6),
        vertices,
        face.index,
    )


def _face_set_geometry_key(faces):
    """Return a deterministic key for a non-empty face collection."""
    return min(_face_geometry_key(face) for face in faces)


def _face_has_valid_uvs(face, uv_layer):
    """Return whether a face has a non-collapsed UV polygon on one layer."""
    uvs = [loop[uv_layer].uv for loop in face.loops]
    if len(uvs) < 3:
        return False

    uv_area = 0.0
    for index in range(1, len(uvs) - 1):
        edge_a = uvs[index] - uvs[0]
        edge_b = uvs[index + 1] - uvs[0]
        uv_area += abs(edge_a.x * edge_b.y - edge_a.y * edge_b.x)
    return uv_area > 1e-8


def _build_face_components(faces, require_similar_normals):
    """Build deterministic adjacency components from a set of new faces."""
    face_set = set(faces)
    remaining = set(face_set)
    components = []

    while remaining:
        start = min(remaining, key=_face_geometry_key)
        component = set()
        queue = [start]
        remaining.remove(start)

        while queue:
            current = queue.pop(0)
            component.add(current)
            neighbors = sorted(
                (
                    linked_face
                    for linked_face in _linked_faces(current)
                    if linked_face in remaining
                    and (
                        not require_similar_normals
                        or current.normal.dot(linked_face.normal)
                        > _NORMAL_SIMILARITY_THRESHOLD
                    )
                ),
                key=_face_geometry_key,
            )
            for neighbor in neighbors:
                remaining.remove(neighbor)
                queue.append(neighbor)

        components.append(component)

    return sorted(components, key=_face_set_geometry_key)


def _build_similar_normal_groups(new_faces):
    """Partition every disconnected new-face graph by positive normal links."""
    groups = []
    topology_components = _build_face_components(new_faces, False)
    for topology_component in topology_components:
        groups.extend(_build_face_components(topology_component, True))
    return topology_components, sorted(groups, key=_face_set_geometry_key)


def _canonicalize_direction(direction):
    """Give an unoriented axis a stable sign based on its dominant component."""
    values = (direction.x, direction.y, direction.z)
    dominant_index = max(range(3), key=lambda index: abs(values[index]))
    if values[dominant_index] < 0:
        return -direction
    return direction


def _infer_group_depth_axis(group):
    """Infer a shared cut-depth axis from connected, non-coplanar face normals."""
    group_set = set(group)
    pair_axes = []
    ordered_faces = sorted(group, key=_face_geometry_key)

    for face in ordered_faces:
        for neighbor in sorted(_linked_faces(face), key=_face_geometry_key):
            if neighbor not in group_set:
                continue
            if _face_geometry_key(neighbor) <= _face_geometry_key(face):
                continue
            if face.normal.dot(neighbor.normal) <= _NORMAL_SIMILARITY_THRESHOLD:
                continue

            axis = face.normal.cross(neighbor.normal)
            if axis.length < _AXIS_EPSILON:
                continue
            pair_axes.append(_canonicalize_direction(axis.normalized()))

    if not pair_axes:
        return None

    reference = pair_axes[0]
    for axis in pair_axes[1:]:
        if abs(reference.dot(axis)) < _AXIS_PARALLEL_DOT:
            return None

    combined = Vector((0.0, 0.0, 0.0))
    for axis in pair_axes:
        if reference.dot(axis) < 0:
            axis = -axis
        combined += axis
    if combined.length < _AXIS_EPSILON:
        return None
    combined.normalize()
    combined = _canonicalize_direction(combined)

    for face in group:
        if abs(face.normal.normalized().dot(combined)) > 0.01:
            return None
    return combined


def _root_candidate_score(target_face, source_face, all_graph_faces,
                          id_layer):
    """Rank one external UV donor using the established neighbor priorities."""
    dot = target_face.normal.dot(source_face.normal)
    similar_rank = 0 if dot > _NORMAL_SIMILARITY_THRESHOLD else 1
    existing_rank = 0 if source_face not in all_graph_faces else 1
    sideways_rank = -(1.0 - abs(source_face.normal.z))

    coplanar_distance = float('inf')
    if (target_face.normal - source_face.normal).length < 0.01:
        cached = face_data_cache.get(source_face[id_layer])
        if cached and cached.get('center'):
            source_center = cached['center']
        else:
            source_center = source_face.calc_center_median()
        coplanar_distance = (
            source_center - target_face.calc_center_median()
        ).length

    return (
        similar_rank,
        existing_rank,
        sideways_rank,
        coplanar_distance,
        _face_geometry_key(target_face),
        _face_geometry_key(source_face),
    )


def _get_group_root_candidates(group, all_graph_faces,
                               projected_graph_faces, excluded_sources,
                               uv_layer, id_layer):
    """Find deterministic root/donor candidates for one normal-connected group."""
    candidates = []
    for target_face in sorted(group, key=_face_geometry_key):
        for source_face in sorted(_linked_faces(target_face),
                                  key=_face_geometry_key):
            if source_face in group or source_face in excluded_sources:
                continue
            if source_face in all_graph_faces and source_face not in projected_graph_faces:
                continue
            if not _face_has_valid_uvs(source_face, uv_layer):
                continue
            candidates.append((
                _root_candidate_score(
                    target_face, source_face, all_graph_faces, id_layer,
                ),
                target_face,
                source_face,
            ))
    candidates.sort(key=lambda item: item[0])
    return [(target, source) for _score, target, source in candidates]


def _get_shared_anchor_vert(first_face, second_face):
    """Choose a stable shared vertex for UV phase anchoring."""
    shared_verts = set(first_face.verts) & set(second_face.verts)
    if not shared_verts:
        return None
    return min(
        shared_verts,
        key=lambda vertex: (
            round(vertex.co.x, 8),
            round(vertex.co.y, 8),
            round(vertex.co.z, 8),
            vertex.index,
        ),
    )


def _align_face_uv_to_depth_axis(face, source_face, uv_layer, depth_axis,
                                 ppm, me):
    """Rotate one projected root while preserving scale and shared-vertex phase."""
    transform = derive_transform_from_uvs(face, uv_layer, ppm, me)
    if not transform:
        return False
    scale_u = transform['scale_u']
    scale_v = transform['scale_v']
    if abs(scale_u) < 1e-8 or abs(scale_v) < 1e-8:
        return False

    face_axes = get_face_local_axes(face)
    if not face_axes:
        return False
    face_local_x, face_local_y = face_axes

    normal = face.normal.normalized()
    aligned_v = depth_axis - normal * depth_axis.dot(normal)
    if aligned_v.length < _AXIS_EPSILON:
        return False
    aligned_v.normalize()
    rotation = math.degrees(math.atan2(
        aligned_v.dot(face_local_x),
        aligned_v.dot(face_local_y),
    ))

    anchor_vert = _get_shared_anchor_vert(face, source_face)
    if anchor_vert is None:
        anchor_loop = list(face.loops)[0]
    else:
        anchor_loop = next(
            loop for loop in face.loops if loop.vert == anchor_vert
        )
    anchor_co = anchor_loop.vert.co.copy()
    anchor_uv = anchor_loop[uv_layer].uv.copy()

    first_vert = list(face.loops)[0].vert.co
    rotation_rad = math.radians(rotation)
    cos_rotation = math.cos(rotation_rad)
    sin_rotation = math.sin(rotation_rad)
    projection_x = (
        face_local_x * cos_rotation - face_local_y * sin_rotation
    )
    projection_y = (
        face_local_x * sin_rotation + face_local_y * cos_rotation
    )
    delta = anchor_co - first_vert

    material = (
        me.materials[face.material_index]
        if face.material_index < len(me.materials)
        else None
    )
    texture_meters_u, texture_meters_v = get_texture_dimensions_from_material(
        material, ppm,
    )
    projected_u = delta.dot(projection_x) / (scale_u * texture_meters_u)
    projected_v = delta.dot(projection_y) / (scale_v * texture_meters_v)
    offset_x = normalize_offset(anchor_uv.x - projected_u)
    offset_y = normalize_offset(anchor_uv.y - projected_v)

    apply_uv_to_face(
        face, uv_layer, scale_u, scale_v, rotation,
        offset_x, offset_y, material, ppm, me,
    )
    return True


def _project_face_from_source(source_face, target_face, unlocked_layers,
                              ppm, me, obj_matrix, depth_axis,
                              set_uv_from_other_face):
    """Project one face from one chosen source, optionally aligning a root."""
    if target_face.material_index != source_face.material_index:
        debug_log(
            f"[ProjectNewFaces] Propagated material: "
            f"face {target_face.index} mat={source_face.material_index} "
            f"(source=face {source_face.index})"
        )
        target_face.material_index = source_face.material_index

    had_uvs = _face_has_valid_uvs(target_face, unlocked_layers[0])
    projected_any_layer = False
    for uv_layer in unlocked_layers:
        projected = set_uv_from_other_face(
            source_face, target_face, uv_layer, ppm, me, obj_matrix,
        )
        if not projected:
            continue
        projected_any_layer = True
        if (
                depth_axis is not None
                and source_face.normal.dot(target_face.normal)
                <= _NORMAL_SIMILARITY_THRESHOLD):
            _align_face_uv_to_depth_axis(
                target_face, source_face, uv_layer, depth_axis, ppm, me,
            )

    if had_uvs and projected_any_layer:
        debug_log(
            f"[ProjectNewFaces] Re-projected face {target_face.index} that "
            f"already had UVs (source=face {source_face.index})"
        )
    return projected_any_layer


def _propagate_similar_group(group, root_face, root_source, depth_axis,
                             unlocked_layers, ppm, me, obj_matrix,
                             set_uv_from_other_face):
    """Project one positive-normal component from exactly one chosen root."""
    projected = set()
    if not _project_face_from_source(
            root_source, root_face, unlocked_layers, ppm, me, obj_matrix,
            depth_axis, set_uv_from_other_face):
        return projected
    projected.add(root_face)

    remaining = set(group) - projected
    while remaining:
        frontier = []
        for target_face in sorted(remaining, key=_face_geometry_key):
            for source_face in sorted(
                    _linked_faces(target_face) & projected,
                    key=_face_geometry_key):
                normal_dot = target_face.normal.dot(source_face.normal)
                if normal_dot <= _NORMAL_SIMILARITY_THRESHOLD:
                    continue
                frontier.append((
                    (
                        -normal_dot,
                        -(1.0 - abs(source_face.normal.z)),
                        _face_geometry_key(target_face),
                        _face_geometry_key(source_face),
                    ),
                    source_face,
                    target_face,
                ))

        if not frontier:
            break
        frontier.sort(key=lambda item: item[0])

        made_progress = False
        for _score, source_face, target_face in frontier:
            if target_face not in remaining:
                continue
            if not _project_face_from_source(
                    source_face, target_face, unlocked_layers, ppm, me,
                    obj_matrix, None, set_uv_from_other_face):
                continue
            remaining.remove(target_face)
            projected.add(target_face)
            made_progress = True
            break
        if not made_progress:
            break

    return projected


def _project_normal_connected_groups(new_faces, excluded_sources,
                                     unlocked_layers, ppm, me, obj_matrix,
                                     id_layer, set_uv_from_other_face):
    """Project one root per positive-normal group across all topology graphs."""
    if not new_faces:
        return set()

    topology_components, groups = _build_similar_normal_groups(new_faces)
    debug_log(
        f"[ProjectNewFaces] Graphs={len(topology_components)} "
        f"normal_groups={len(groups)} new_faces={len(new_faces)}"
    )

    all_graph_faces = set(new_faces)
    projected_graph_faces = set()
    pending_groups = list(groups)

    made_progress = True
    while pending_groups and made_progress:
        made_progress = False
        still_pending = []
        for group in pending_groups:
            candidates = _get_group_root_candidates(
                group, all_graph_faces, projected_graph_faces,
                excluded_sources, unlocked_layers[0], id_layer,
            )
            if not candidates:
                still_pending.append(group)
                continue

            depth_axis = _infer_group_depth_axis(group)
            projected_group = set()
            for root_face, root_source in candidates:
                projected_group = _propagate_similar_group(
                    group, root_face, root_source, depth_axis,
                    unlocked_layers, ppm, me, obj_matrix,
                    set_uv_from_other_face,
                )
                if projected_group:
                    debug_log(
                        f"[ProjectNewFaces] Root face {root_face.index} from "
                        f"face {root_source.index}; group={len(group)} "
                        f"projected={len(projected_group)} "
                        f"axis={tuple(depth_axis) if depth_axis else None}"
                    )
                    break

            if not projected_group:
                still_pending.append(group)
                continue

            projected_graph_faces.update(projected_group)
            made_progress = True
            if len(projected_group) != len(group):
                debug_log(
                    f"[ProjectNewFaces] Group incomplete: "
                    f"projected={len(projected_group)} total={len(group)}"
                )

        pending_groups = still_pending

    for group in pending_groups:
        debug_log(
            f"[ProjectNewFaces] No UV-connected root for normal group of "
            f"{len(group)} face(s)"
        )
    return projected_graph_faces


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
    best_similar_key = None
    best_fallback = None
    best_fallback_key = None

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

                candidate_key = (
                    -sideways_score,
                    dist,
                    _face_geometry_key(linked_face),
                )
                if best_similar_key is None or candidate_key < best_similar_key:
                    best_similar_key = candidate_key
                    best_similar = linked_face
            else:
                candidate_key = (
                    -sideways_score,
                    _face_geometry_key(linked_face),
                )
                if best_fallback_key is None or candidate_key < best_fallback_key:
                    best_fallback_key = candidate_key
                    best_fallback = linked_face

    if best_similar:
        return best_similar
    return best_fallback if allow_fallback else None


def project_new_faces(context, bm):
    """Compatibility entry point for existing handler and operator callers."""
    project_new_faces_for_object(
        context.object,
        context.scene.level_design_props.pixels_per_meter,
        bm,
    )


def project_new_faces_for_object(obj, ppm, bm):
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

    me = obj.data
    unlocked_layers = get_unlocked_uv_layers(bm, obj, me)
    if not unlocked_layers:
        return

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
    cached_rebuilt_coplanar = (
        face_buckets['cached_vertex_count_changed'] &
        face_buckets['cached_same_plane_changed']
    ) - duplicate_faces - spin_rotated - face_buckets['hotspot']

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

    if not new_faces and not dupe_coplanar and not cached_rebuilt_coplanar:
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
    coplanar_reproject = (
        dupe_coplanar |
        coplanar_modified |
        cached_rebuilt_coplanar
    )

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

    # Step 2: Partition new faces by positive-normal adjacency. Each group gets
    # exactly one UV-connected root, then propagates only across its similar
    # internal edges. Orthogonal cube walls therefore remain separate roots,
    # while a faceted cylinder becomes one group even when its graph is cyclic.
    graph_faces = affected & new_faces
    projected_graph_faces = _project_normal_connected_groups(
        graph_faces,
        dupe_extrusions | spin_rotated,
        unlocked_layers,
        ppm,
        me,
        obj.matrix_world,
        id_layer,
        set_uv_from_other_face,
    )
    affected -= graph_faces

    # Step 3: Preserve the existing generic wavefront for cached faces whose
    # geometry was affected by the topology operation. Successfully projected
    # graph faces may now act as their sources; failed graph faces remain
    # excluded so they cannot accidentally create extra roots.
    excluded = (
        (new_faces | dupe_extrusions | spin_rotated)
        - coplanar_reproject
        - projected_graph_faces
    )
    projected_count = len(coplanar_reproject) + len(projected_graph_faces)
    remaining = sorted(affected, key=_face_geometry_key)
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

            if not _project_face_from_source(
                    source_face, face, unlocked_layers, ppm, me,
                    obj.matrix_world, None, set_uv_from_other_face):
                still_remaining.append(face)
                continue

            excluded.discard(face)
            projected_count += 1
            made_progress = True

        remaining = still_remaining
        if not made_progress and remaining and not allow_fallback:
            allow_fallback = True
            made_progress = True

    if projected_count > 0 or repaired_materials > 0:
        bmesh.update_edit_mesh(me)
