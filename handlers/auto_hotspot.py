"""Auto-hotspot projection and new-face UV projection."""

import bmesh
import bpy

from ..core.logging import debug_log
from ..core.face_id import (
    get_face_id_layer, save_face_selection, restore_face_selection,
)
from ..core.geometry import get_local_x_from_verts_3d
from ..core.uv_layers import get_unlocked_uv_layers
from ..core.hotspot_queries import face_has_hotspot_material

from .face_cache import (
    face_data_cache, cache_single_face, cache_face_data,
    get_cached_layer_data,
)


_auto_hotspot_pending = False
_force_auto_hotspot = False

# Track undo operations to skip depsgraph handling during undo
_undo_in_progress = False


def get_undo_in_progress():
    return _undo_in_progress


def set_undo_in_progress(value):
    global _undo_in_progress
    _undo_in_progress = value


def get_force_auto_hotspot():
    return _force_auto_hotspot


def set_force_auto_hotspot(value):
    global _force_auto_hotspot
    _force_auto_hotspot = value


def get_auto_hotspot_pending():
    return _auto_hotspot_pending


def set_auto_hotspot_pending(value):
    global _auto_hotspot_pending
    _auto_hotspot_pending = value


def _any_hotspot_geometry_changed(bm, me):
    """Check if any hotspot face has geometry that differs from cache.

    Returns True if any hotspot face is new or has moved vertices.
    Returns False if all hotspot faces match their cached geometry (e.g., after undo).
    """
    id_layer = get_face_id_layer(bm)
    for face in bm.faces:
        if not face.is_valid or not face_has_hotspot_material(face, me):
            continue

        face_id = face[id_layer]
        if face_id == 0 or face_id not in face_data_cache:
            # New face - geometry changed
            return True

        cached = face_data_cache[face_id]
        cached_verts = cached.get('verts', [])
        current_verts = [v.co.copy() for v in face.verts]

        if len(current_verts) != len(cached_verts):
            return True

        for current, cached_v in zip(current_verts, cached_verts):
            if (current - cached_v).length > 0.0001:
                return True

    return False


def _apply_auto_hotspots_deferred():
    """Deferred auto-hotspot application (runs from timer for proper context).

    Applies hotspots to ALL faces with hotspot materials, treating the entire
    object shape as relevant for island detection.
    """
    global _auto_hotspot_pending, _force_auto_hotspot

    # Early return if cancelled (e.g., by undo)
    if not _auto_hotspot_pending:
        return None

    from ..operators.hotspot_apply import apply_hotspots_to_mesh

    try:
        context = bpy.context
        if context.mode != 'EDIT_MESH':
            _auto_hotspot_pending = False
            _force_auto_hotspot = False
            return None

        # Skip if modal operator is running (e.g., extrude, grab)
        # Keep pending so we re-check after modal ends
        window = context.window
        if window and window.modal_operators:
            return 0.1  # Re-check in 0.1s

        force = _force_auto_hotspot
        _auto_hotspot_pending = False
        _force_auto_hotspot = False

        obj = context.object
        if not obj or obj.type != 'MESH':
            return None

        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        if not bm.is_valid:
            return None

        bm.faces.ensure_lookup_table()

        # Check if any hotspot geometry actually changed (skip if undo restored to cached state)
        # Topology changes (new faces) force re-application since cache_face_data runs first
        if not force and not _any_hotspot_geometry_changed(bm, me):
            return None

        # Get ALL faces with hotspot materials, excluding fixed faces
        fixed_layer = bm.faces.layers.int.get("anvil_fixed_hotspot")
        all_hotspot_faces = [
            f for f in bm.faces
            if f.is_valid and face_has_hotspot_material(f, me)
            and (fixed_layer is None or f[fixed_layer] == 0)
        ]

        if not all_hotspot_faces:
            return None

        # Save selection state
        id_layer = get_face_id_layer(bm)
        selected_ids, active_id = save_face_selection(bm, id_layer)

        allow_combined_faces = obj.anvil_allow_combined_faces
        size_weight = obj.anvil_hotspot_size_weight


        props = context.scene.level_design_props
        debug_log(f"[AutoHotspot] Processing {len(all_hotspot_faces)} hotspot faces")
        result = apply_hotspots_to_mesh(
            bm, me, all_hotspot_faces, allow_combined_faces,
            obj.matrix_world, props.pixels_per_meter, size_weight
        )
        debug_log(f"[AutoHotspot] Applied: {result}")

        # Restore selection state
        restore_face_selection(bm, id_layer, selected_ids, active_id)

        # Copy hotspot UVs from primary unlocked layer to other unlocked layers
        unlocked_layers = get_unlocked_uv_layers(bm, obj, me)
        if len(unlocked_layers) > 1:
            primary_layer = unlocked_layers[0]
            for face in all_hotspot_faces:
                if not face.is_valid:
                    continue
                for other_layer in unlocked_layers[1:]:
                    for loop in face.loops:
                        loop[other_layer].uv = loop[primary_layer].uv.copy()

        # Update cache for processed faces
        ppm = props.pixels_per_meter
        for face in all_hotspot_faces:
            if face.is_valid:
                cache_single_face(face, bm, ppm, me)

        bmesh.update_edit_mesh(me)

    except Exception as e:
        print(f"Anvil Level Design: Auto-hotspot error: {e}", flush=True)
        import traceback
        traceback.print_exc()

    return None


def apply_auto_hotspots():
    """Schedule auto-hotspot application via timer for proper operator context."""
    global _auto_hotspot_pending
    if _auto_hotspot_pending:
        return
    _auto_hotspot_pending = True
    bpy.app.timers.register(_apply_auto_hotspots_deferred, first_interval=0.1)


def _get_best_neighbor_face(face, excluded_faces, id_layer, allow_fallback=True):
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


def _project_new_faces(context, bm):
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

    # --- Handle duplicate face IDs ---
    seen_ids = set()
    duplicated_ids = set()
    for face in bm.faces:
        fid = face[id_layer]
        if fid == 0:
            continue
        if fid in seen_ids:
            duplicated_ids.add(fid)
        seen_ids.add(fid)

    dupe_exact = set()
    dupe_coplanar = set()
    dupe_extrusions = set()
    dupe_other = set()

    if duplicated_ids:
        id_to_coplanar = {}
        for face in bm.faces:
            fid = face[id_layer]
            if fid not in duplicated_ids or not face.is_valid:
                continue
            if face_has_hotspot_material(face, me):
                continue

            cached = face_data_cache.get(fid)
            if not cached or not cached.get('normal'):
                continue

            cached_normal = cached['normal']
            if (face.normal - cached_normal).length < 0.01:
                cached_center = cached.get('center')
                if cached_center is not None:
                    dist_to_plane = abs(cached_normal.dot(
                        face.calc_center_median() - cached_center))
                    if dist_to_plane < 0.01:
                        cached_verts = cached['verts']
                        current_verts = [v.co for v in face.verts]
                        if (len(current_verts) == len(cached_verts)
                                and all((cv - cav).length < 0.0001
                                        for cv, cav in zip(current_verts,
                                                           cached_verts))):
                            id_to_coplanar.setdefault(fid, []).append(('exact', face))
                        else:
                            id_to_coplanar.setdefault(fid, []).append(('coplanar', face))
                    else:
                        dupe_extrusions.add(face)

        for fid, entries in id_to_coplanar.items():
            for category, face in entries:
                if category == 'exact':
                    dupe_exact.add(face)
                else:
                    dupe_coplanar.add(face)

        all_categorized = dupe_exact | dupe_coplanar | dupe_extrusions
        for face in bm.faces:
            fid = face[id_layer]
            if fid not in duplicated_ids:
                continue
            if face not in all_categorized:
                dupe_other.add(face)

    # --- Identify new faces and build the affected set ---
    new_faces = set()
    for f in bm.faces:
        if not f.is_valid or face_has_hotspot_material(f, me):
            continue
        if f[id_layer] == 0 or f in dupe_other:
            new_faces.add(f)

    if not new_faces and not dupe_coplanar:
        return

    affected = set(new_faces)
    queue = list(affected)
    visited = set(affected) | dupe_extrusions | dupe_exact
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
    coplanar_modified = set()
    translated = set()
    for face in affected:
        face_id = face[id_layer]
        if face_id == 0:
            continue
        cached = face_data_cache.get(face_id)
        if not cached:
            continue
        cached_normal = cached.get('normal')
        if not cached_normal or (face.normal - cached_normal).length >= 0.01:
            continue

        cached_verts = cached['verts']
        current_verts = [v.co for v in face.verts]
        if len(current_verts) != len(cached_verts):
            continue

        if all((cv - cav).length < 0.0001
               for cv, cav in zip(current_verts, cached_verts)):
            continue

        offset = current_verts[0] - cached_verts[0]
        if all((cv - cav - offset).length < 0.0001
               for cv, cav in zip(current_verts[1:], cached_verts[1:])):
            translated.add(face)
        else:
            coplanar_modified.add(face)

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
    excluded = (new_faces | dupe_extrusions) - coplanar_reproject
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

            source_face = _get_best_neighbor_face(face, excluded, id_layer,
                                                  allow_fallback)

            if not source_face:
                still_remaining.append(face)
                continue

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

    if projected_count > 0:
        bmesh.update_edit_mesh(me)


def reset():
    """Reset auto-hotspot state."""
    global _auto_hotspot_pending, _force_auto_hotspot, _undo_in_progress
    _auto_hotspot_pending = False
    _force_auto_hotspot = False
    _undo_in_progress = False
