import os
import re
import ctypes

import bpy
import bmesh
import math
from bpy.app.handlers import persistent

# Memory offset of uvcalc_flag in ToolSettings struct (found experimentally for Blender 5.0)
_UVCALC_FLAG_OFFSET = 100
_UVCALC_TRANSFORM_CORRECT_SLIDE = 4  # 1 << 2

# HACK of the century. We need to turn off correct_uv slide but blender doesn't expose it
# So we do direct memory management. See /scripts for how to find the offset
# This is brittle as anything :)
def set_correct_uv_slide(enabled: bool):
    """Directly set the UVCALC_TRANSFORM_CORRECT_SLIDE flag in tool settings.

    This flag controls whether edge/vertex slide operations correct UVs.
    Blender's Python API doesn't expose this flag, so we modify it directly in memory.
    """
    try:
        ts = bpy.context.scene.tool_settings
        ts_ptr = ts.as_pointer()
        flag_ptr = ctypes.cast(ts_ptr + _UVCALC_FLAG_OFFSET, ctypes.POINTER(ctypes.c_ushort))

        if enabled:
            flag_ptr.contents.value |= _UVCALC_TRANSFORM_CORRECT_SLIDE
        else:
            flag_ptr.contents.value &= ~_UVCALC_TRANSFORM_CORRECT_SLIDE
    except Exception as e:
        print(f"Anvil Level Design: Failed to set correct_uv_slide — offset {_UVCALC_FLAG_OFFSET} may be wrong for this platform/Blender version: {e}", flush=True)

from .utils import (
    get_image_from_material, derive_transform_from_uvs,
    get_selected_image_path, find_material_with_image, create_material_with_image,
    get_texture_dimensions_from_material, get_face_local_axes, normalize_offset,
    get_local_x_from_verts_3d, debug_log, face_has_hotspot_material,
    any_connected_face_has_hotspot, get_all_hotspot_faces,
    get_face_id_layer, ensure_face_ids, assign_face_id,
    reindex_face_ids,
    save_face_selection, restore_face_selection,
    is_level_design_workspace, face_aligned_project,
    get_render_active_uv_layer, get_unlocked_uv_layers, get_locked_uv_layers,
    get_all_uv_layers, sync_uv_map_settings,
)
from .properties import set_updating_from_selection, sync_scale_tracking, apply_uv_to_face
from .workspace import setup_addon_workspaces, subscribe_splash_watcher, reset_specialized_template_flag


_auto_hotspot_pending = False
_force_auto_hotspot = False

# Face orientation overlay tracking for vertex paint / sculpt modes
_last_tracked_mode = None
_FACE_ORIENTATION_MODES = {'PAINT_VERTEX', 'SCULPT'}
_face_orientation_msgbus_owner = object()


def _get_saved_face_orientation():
    """Read persisted face orientation state. Returns bool or None."""
    val = bpy.context.scene.level_design_props.saved_face_orientation
    if val == -1:
        return None
    return bool(val)


def _set_saved_face_orientation(value):
    """Write persisted face orientation state. Accepts bool or None."""
    if value is None:
        bpy.context.scene.level_design_props.saved_face_orientation = -1
    else:
        bpy.context.scene.level_design_props.saved_face_orientation = int(value)


def _on_object_mode_changed():
    """Msgbus callback when object mode changes. Toggles face orientation overlay."""
    global _last_tracked_mode
    try:
        if not is_level_design_workspace():
            return
        current_mode = bpy.context.mode
        if current_mode == _last_tracked_mode:
            return
        was_in = _last_tracked_mode in _FACE_ORIENTATION_MODES if _last_tracked_mode else False
        now_in = current_mode in _FACE_ORIENTATION_MODES
        _last_tracked_mode = current_mode
        debug_log(f"[FaceOrientation] Mode changed: was_in={was_in} now_in={now_in}")
        saved = _get_saved_face_orientation()
        if now_in and not was_in:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                if saved is None:
                                    _set_saved_face_orientation(space.overlay.show_face_orientation)
                                space.overlay.show_face_orientation = True
                        area.tag_redraw()
            debug_log(f"[FaceOrientation] Enabled (saved={_get_saved_face_orientation()})")
        elif was_in and not now_in:
            if saved is not None:
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D':
                            for space in area.spaces:
                                if space.type == 'VIEW_3D':
                                    space.overlay.show_face_orientation = saved
                            area.tag_redraw()
                debug_log(f"[FaceOrientation] Restored (was={saved})")
                _set_saved_face_orientation(None)
    except Exception:
        pass


def _subscribe_object_mode():
    """Subscribe to object mode changes via msgbus."""
    try:
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.Object, "mode"),
            owner=_face_orientation_msgbus_owner,
            args=(),
            notify=_on_object_mode_changed,
            options={'PERSISTENT'},
        )
        debug_log("[FaceOrientation] Subscribed to object mode changes")
    except Exception as e:
        print(f"Anvil Level Design: Failed to subscribe to object mode changes: {e}", flush=True)


# Msgbus subscription owner for unit setting changes
_msgbus_owner = object()

# Multi-face selection state tracking
_multi_face_mode = False
_multi_face_unset_scale = False
_multi_face_unset_rotation = False
_multi_face_unset_offset = False
_all_selected_hotspot = False


def get_all_selected_hotspot():
    return _all_selected_hotspot


def get_multi_face_mode():
    return _multi_face_mode


def is_multi_face_unset_scale():
    return _multi_face_unset_scale


def is_multi_face_unset_rotation():
    return _multi_face_unset_rotation


def is_multi_face_unset_offset():
    return _multi_face_unset_offset


def mark_multi_face_set_scale():
    global _multi_face_unset_scale
    _multi_face_unset_scale = False


def mark_multi_face_set_rotation():
    global _multi_face_unset_rotation
    _multi_face_unset_rotation = False


def mark_multi_face_set_offset():
    global _multi_face_unset_offset
    _multi_face_unset_offset = False

# Track undo operations to skip depsgraph handling during undo
_undo_in_progress = False


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

    from .operators.uv_tools import apply_hotspots_to_mesh

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

        # Get ALL faces with hotspot materials
        all_hotspot_faces = [f for f in bm.faces if f.is_valid and face_has_hotspot_material(f, me)]

        if not all_hotspot_faces:
            return None

        # Save selection state
        id_layer = get_face_id_layer(bm)
        selected_ids, active_id = save_face_selection(bm, id_layer)

        props = context.scene.level_design_props
        seam_mode = props.hotspot_seam_mode
        allow_combined_faces = obj.anvil_allow_combined_faces
        size_weight = obj.anvil_hotspot_size_weight
        seam_angle = obj.anvil_hotspot_seam_angle

        debug_log(f"[AutoHotspot] Processing {len(all_hotspot_faces)} hotspot faces")
        result = apply_hotspots_to_mesh(
            bm, me, all_hotspot_faces, seam_mode, allow_combined_faces,
            obj.matrix_world, props.pixels_per_meter, size_weight, seam_angle
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


def _apply_auto_hotspots():
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
    from .operators.texture_apply import set_uv_from_other_face, set_uv_from_source_params

    obj = context.object
    me = obj.data
    unlocked_layers = get_unlocked_uv_layers(bm, obj, me)
    if not unlocked_layers:
        return

    props = context.scene.level_design_props
    ppm = props.pixels_per_meter

    id_layer = get_face_id_layer(bm)

    # --- Handle duplicate face IDs ---
    # BMesh operations (extrude, bevel, etc.) copy all custom data from
    # source faces to new faces, including our managed face ID.
    #
    # Categorize faces with duplicated IDs:
    #   coplanar: same normal & same plane as cache → UV from cached transform
    #   extrusion: same normal but different plane → skip (Blender set UVs)
    #   other (different normal): treat as new, UV from neighbors
    seen_ids = set()
    duplicated_ids = set()
    for face in bm.faces:
        fid = face[id_layer]
        if fid == 0:
            continue
        if fid in seen_ids:
            duplicated_ids.add(fid)
        seen_ids.add(fid)

    # For each duplicated ID group, categorize faces by their relationship
    # to the cached version. IDs are NOT reset here — reindex_face_ids()
    # assigns fresh unique IDs at the end of cache_face_data().
    #
    # Categories:
    #   exact: same normal, same plane, same verts as cache → skip
    #          (likely extrusion with no movement yet)
    #   coplanar: same normal & same plane but verts changed → UV from cache
    #   extrusion: same normal but different plane → skip (Blender set UVs)
    #   other: different normal or uncategorized → treat as new face
    dupe_exact = set()       # geometry unchanged from cache — skip entirely
    dupe_coplanar = set()    # faces to UV from cache (keep their ID)
    dupe_extrusions = set()  # faces to skip entirely
    dupe_other = set()       # different normal or uncategorized duplicates (treat as new)

    if duplicated_ids:
        # For each duplicated ID, find coplanar faces (may be multiple,
        # e.g. when a quad is triangulated into two coplanar triangles).
        id_to_coplanar = {}  # fid -> list of (category, face)
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
                        # Coplanar — check if verts are exactly unchanged
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

        # All remaining duplicate faces that aren't categorized
        all_categorized = dupe_exact | dupe_coplanar | dupe_extrusions
        for face in bm.faces:
            fid = face[id_layer]
            if fid not in duplicated_ids:
                continue
            if face not in all_categorized:
                dupe_other.add(face)

    # --- Identify new faces and build the affected set ---

    # New faces: ID 0 (freshly created) or dupe_other (duplicate ID but
    # different normal — not categorized as exact/coplanar/extrusion).
    new_faces = set()
    for f in bm.faces:
        if not f.is_valid or face_has_hotspot_material(f, me):
            continue
        if f[id_layer] == 0 or f in dupe_other:
            new_faces.add(f)

    if not new_faces and not dupe_coplanar:
        return

    # Start with all new faces, then BFS outward into adjacent cached faces
    # whose vertices have moved. Those may need re-projection too (e.g. bevel
    # shrinks a face, weld translates a face changing its world-space offset).
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

                # Check if any vertex has moved from cached position
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
    # Two sources feed into this:
    #   - dupe_coplanar: duplicate-ID faces on the same plane (verts changed)
    #   - coplanar_modified: unique-ID faces from the BFS that are still
    #     coplanar with cache but have moved vertices (e.g. subdivide/loop-cut)
    # Re-projecting these first makes them valid neighbor sources for step 2.
    # Categorize affected faces that have a cached ID (BFS-expanded existing
    # faces, not new/ID-0 faces):
    #   coplanar_modified: same normal, same plane, verts changed → re-project
    #                      from cache (e.g. subdivide/loop-cut shrinks a face)
    #   translated: same normal, all verts moved by the same offset (pure
    #               translation) → skip, Blender already handled the UVs
    #   (everything else falls through to Step 2 wavefront projection)
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

        # Check if pure translation (all verts moved by the same offset)
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
    # Process in BFS order outward from cached/coplanar faces: each projected
    # face becomes a valid source for its neighbors in the next wave. This
    # handles interior faces (e.g. bevel segments) that don't directly border
    # cached geometry.
    # Skip zero-area faces — they can't be projected meaningfully (e.g. during
    # modal extrude, new faces start collapsed). They'll be handled later when
    # the modal moves them and they gain area (see _was_zero_area check in main loop).
    excluded = (new_faces | dupe_extrusions) - coplanar_reproject
    projected_count = len(coplanar_reproject)
    remaining = sorted(affected,
                       key=lambda f: f.calc_center_median().length_squared)
    # Two-phase wavefront: first only use similar-direction sources (positive
    # dot product), then allow fallback sources.  This prevents interior faces
    # (e.g. middle bevel segments) from picking a perpendicular end-cap as
    # source when a better source would become available after an outer
    # segment is projected from an adjacent wall/floor.
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

            # Project on all unlocked layers
            for uv_layer in unlocked_layers:
                set_uv_from_other_face(source_face, face, uv_layer, ppm, me, obj.matrix_world)

            if _had_uvs:
                debug_log(f"[ProjectNewFaces] Re-projected face {face.index} that already had UVs "
                          f"(source=face {source_face.index})")

            # This face is now a valid source for its neighbors
            excluded.discard(face)
            projected_count += 1
            made_progress = True

        remaining = still_remaining
        # If no progress was made without fallbacks, enable them and retry
        if not made_progress and remaining and not allow_fallback:
            allow_fallback = True
            made_progress = True

    if projected_count > 0:
        bmesh.update_edit_mesh(me)


# Cache for face data (UV lock functionality)
face_data_cache = {}
last_face_count = 0
last_vertex_count = 0

# Track if this is the first save (for making paths relative after save)
_was_first_save = False


# Cache for material deduplication
_last_material_count = 0

# Cache for detecting selection changes
_last_selected_face_indices = set()
_last_active_face_index = -1
# Track which object we're editing to detect fresh edit sessions
_last_edit_object_name = None
# Set True on file load to allow first depsgraph to sync active image from selected face
_file_loaded_into_edit_depsgraph = False
# Track modal operators for UV world-scale baseline
_tracked_modal_operators = set()
# Modal operators that restore-from-snapshot and re-apply on each mouse move.
# During these modals, cache rebuild is deferred to preserve the pre-operation
# baseline for correct UV projection across restore/re-apply cycles.
_TOPOLOGY_MODAL_OPS = {
    'MESH_OT_bevel',
}
# Slide operators that should use grid snapping instead of incremental
_SLIDE_MODAL_OPS = {
    'MESH_OT_loopcut_slide',
    'TRANSFORM_OT_vert_slide',
    'TRANSFORM_OT_edge_slide',
}
_slide_snap_active = False
# Track the file browser watcher modal operator
_file_browser_watcher_running = False
# Track the previously selected file browser path (to avoid reapplying same image)
_last_file_browser_path = None

# The currently active image for texture operations.
# Updated by: file browser selection, user clicking a face
# Used by: Alt+Click apply, UI panel preview
_active_image = None
# The previously active image name and filepath, shown as a disabled preview when
# no texture is selected. Stored as strings (not a reference) so they survive undo/redo.
# The filepath allows re-loading the image if undo removes it from bpy.data.images.
_previous_image_name = None
_previous_image_filepath = None
# Specifically for the case where faces start as selected e.g. initial cube on file creation.
# Guard flag: when True, depsgraph should not overwrite _active_image
# (set by apply_texture_from_file_browser which runs in a timer context
# where context.tool_settings may not reflect the 3D viewport state)
_active_image_just_set = False


def get_active_image():
    """Get the currently active image for texture operations.

    Returns None if the stored reference has been invalidated (e.g. by undo).
    """
    global _active_image
    if _active_image is None:
        return None
    try:
        _active_image.name
        return _active_image
    except ReferenceError:
        _active_image = None
        return None


def get_previous_image():
    """Get the previously active image for display when no texture is selected.

    Looks up the image by name each time so the reference survives undo/redo.
    If undo removed the image from bpy.data.images, re-loads it from the filepath.
    """
    if _previous_image_name is None:
        return None
    img = bpy.data.images.get(_previous_image_name)
    if img is None and _previous_image_filepath:
        try:
            img = bpy.data.images.load(_previous_image_filepath, check_existing=True)
        except RuntimeError:
            pass
    return img


def set_active_image(image):
    """Set the currently active image for texture operations.

    When a non-None image is set, it is also saved as the previous image
    so the panel can show a disabled preview when no texture is selected.

    Note: Does not call redraw_ui_panels here to avoid requiring a context parameter.
    Callers should call redraw_ui_panels(context) if an immediate UI update is needed.
    """
    global _active_image
    if image is not None:
        set_previous_image(image)
    _active_image = image


def set_previous_image(image):
    """Set the previous image for display when no texture is selected."""
    global _previous_image_name, _previous_image_filepath
    if image is not None:
        _previous_image_name = image.name
        _previous_image_filepath = image.filepath_from_user()
    else:
        _previous_image_name = None
        _previous_image_filepath = None



def cache_single_face(face, bm, ppm=None, me=None):
    """Cache vertex positions, per-layer UVs, and transform for a single face.

    Updates the face_data_cache entry for this face without clearing the cache.
    Caches UV data for ALL UV layers in a 'uv_layers' dict keyed by layer name.
    Assigns a managed face ID if the face doesn't have one yet.

    Args:
        face: BMesh face to cache
        bm: BMesh instance (used to access all UV layers)
        ppm: Pixels per meter (optional, for deriving transform)
        me: Mesh data (optional, for deriving transform)
    """
    if face is None or not face.is_valid:
        return

    # Ensure the face has a managed ID
    id_layer = get_face_id_layer(bm)
    face_id = assign_face_id(face, id_layer)

    cache_entry = {
        'verts': [v.co.copy() for v in face.verts],
        'normal': face.normal.copy(),
        'center': face.calc_center_median().copy(),
        'uv_layers': {},
    }

    # Cache UVs for each UV layer, keyed by layer name
    for layer_idx in range(len(bm.loops.layers.uv)):
        uv_layer = bm.loops.layers.uv[layer_idx]
        layer_data = {
            'uvs': [loop[uv_layer].uv.copy() for loop in face.loops],
        }
        # Cache the derived transform for this layer
        if ppm is not None and me is not None:
            transform = derive_transform_from_uvs(face, uv_layer, ppm, me)
            if transform:
                layer_data['scale_u'] = transform['scale_u']
                layer_data['scale_v'] = transform['scale_v']
                layer_data['rotation'] = transform['rotation']
                layer_data['offset_x'] = transform['offset_x']
                layer_data['offset_y'] = transform['offset_y']
        cache_entry['uv_layers'][uv_layer.name] = layer_data

    # Also store render-active layer transform at top level for backwards compat
    if ppm is not None and me is not None:
        render_layer = get_render_active_uv_layer(bm, me) if me else None
        if render_layer is not None:
            transform = derive_transform_from_uvs(face, render_layer, ppm, me)
            if transform:
                cache_entry['scale_u'] = transform['scale_u']
                cache_entry['scale_v'] = transform['scale_v']
                cache_entry['rotation'] = transform['rotation']
                cache_entry['offset_x'] = transform['offset_x']
                cache_entry['offset_y'] = transform['offset_y']

    face_data_cache[face_id] = cache_entry


def get_cached_layer_data(face_id, layer_name):
    """Get cached UV data for a specific face and UV layer.

    Args:
        face_id: Managed face ID (from the anvil_face_id layer)
        layer_name: UV layer name

    Returns:
        Dict with 'uvs', 'scale_u', 'scale_v', 'rotation', 'offset_x', 'offset_y'
        or None if not cached.
    """
    cached = face_data_cache.get(face_id)
    if not cached:
        return None
    return cached.get('uv_layers', {}).get(layer_name)


def cache_face_data(context):
    """Cache vertex positions and UVs for all faces in the mesh.

    Clears and rebuilds the entire face_data_cache. Used when UV lock is toggled
    or when the mesh topology changes. Assigns managed face IDs to any faces
    that don't have one yet.
    """
    global last_face_count, last_vertex_count

    if context.mode != 'EDIT_MESH':
        return

    obj = context.object
    if not obj or obj.type != 'MESH':
        return

    # Safety check for properties
    if not hasattr(context.scene, 'level_design_props'):
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    ppm = context.scene.level_design_props.pixels_per_meter

    face_data_cache.clear()

    # Assign fresh unique IDs to all faces before caching.
    # This guarantees no duplicates from BMesh operations that copy custom data.
    id_layer = get_face_id_layer(bm)
    reindex_face_ids(bm, id_layer)

    for face in bm.faces:
        cache_single_face(face, bm, ppm, me)

    last_face_count = len(bm.faces)
    last_vertex_count = len(bm.verts)


def _check_multi_face_consistency(selected_faces, uv_layer, ppm, me, first_transform):
    """Check if all selected faces share the same transform values.

    Clears the unset flags for any property group where all faces agree.
    """
    global _multi_face_unset_scale, _multi_face_unset_rotation, _multi_face_unset_offset

    TOLERANCE = 0.01

    scale_consistent = True
    rotation_consistent = True
    offset_consistent = True

    for face in selected_faces[1:]:
        transform = derive_transform_from_uvs(face, uv_layer, ppm, me)
        if not transform:
            continue

        if scale_consistent:
            if (abs(transform['scale_u'] - first_transform['scale_u']) > TOLERANCE or
                    abs(transform['scale_v'] - first_transform['scale_v']) > TOLERANCE):
                scale_consistent = False

        if rotation_consistent:
            if abs(transform['rotation'] - first_transform['rotation']) > TOLERANCE:
                rotation_consistent = False

        if offset_consistent:
            if (abs(transform['offset_x'] - first_transform['offset_x']) > TOLERANCE or
                    abs(transform['offset_y'] - first_transform['offset_y']) > TOLERANCE):
                offset_consistent = False

        # Early out if all are inconsistent
        if not scale_consistent and not rotation_consistent and not offset_consistent:
            break

    if scale_consistent:
        _multi_face_unset_scale = False
    if rotation_consistent:
        _multi_face_unset_rotation = False
    if offset_consistent:
        _multi_face_unset_offset = False


def update_ui_from_selection(context):
    """Update UI properties when selection changes"""
    global _multi_face_mode, _multi_face_unset_scale, _multi_face_unset_rotation, _multi_face_unset_offset, _all_selected_hotspot

    if context.mode != 'EDIT_MESH':
        return

    obj = context.object
    if not obj or obj.type != 'MESH':
        return

    # Safety check
    if not hasattr(context.scene, 'level_design_props'):
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    uv_layer = get_render_active_uv_layer(bm, me)
    if uv_layer is None:
        return

    props = context.scene.level_design_props

    selected_faces = [f for f in bm.faces if f.select]

    # Check if all selected faces have hotspot materials
    if selected_faces:
        _all_selected_hotspot = all(
            face_has_hotspot_material(f, me) for f in selected_faces
        )
    else:
        _all_selected_hotspot = False

    if len(selected_faces) > 1:
        _multi_face_mode = True
        _multi_face_unset_scale = True
        _multi_face_unset_rotation = True
        _multi_face_unset_offset = True
    else:
        _multi_face_mode = False
        _multi_face_unset_scale = False
        _multi_face_unset_rotation = False
        _multi_face_unset_offset = False

    set_updating_from_selection(True)

    try:
        if selected_faces:
            # Get values from first selected face
            face = selected_faces[0]

            # Check if face has a material with an image texture
            mat_index = face.material_index
            mat = obj.data.materials[mat_index] if mat_index < len(obj.data.materials) else None
            has_image = get_image_from_material(mat) is not None

            if has_image:
                # Derive all transform values from current UVs
                ppm = props.pixels_per_meter
                transform = derive_transform_from_uvs(face, uv_layer, ppm, me)

                if transform:
                    props.texture_scale_u = transform['scale_u']
                    props.texture_scale_v = transform['scale_v']
                    props.texture_rotation = transform['rotation']
                    props.texture_offset_x = transform['offset_x']
                    props.texture_offset_y = transform['offset_y']

                    # For multi-face: check if all faces share the same values
                    if _multi_face_mode and transform:
                        _check_multi_face_consistency(selected_faces, uv_layer, ppm, me, transform)
            else:
                # Default material - show neutral values
                props.texture_scale_u = 1.0
                props.texture_scale_v = 1.0
                props.texture_rotation = 0.0
                props.texture_offset_x = 0.0
                props.texture_offset_y = 0.0
    finally:
        set_updating_from_selection(False)
        # Sync scale tracking after properties are updated from selection
        sync_scale_tracking(context)


def apply_world_scale_uvs(obj, scene):
    """Apply world-scale UVs when UV lock is disabled.

    When a face is transformed, UVs are re-projected while preserving the rotation.
    The texture appears stationary in world space - making a face larger reveals more texture.

    Uses modal operator tracking to maintain a stable baseline during operations:
    - While a modal operator is running, the cache is NOT updated (baseline preserved)
    - When the modal operator ends, the cache is refreshed with the final state
    This ensures that moving geometry back to its original position restores original UVs.
    """
    global _tracked_modal_operators

    me = obj.data

    # Skip if mesh data is not available or being modified
    if me is None or not me.is_editmode:
        debug_log("[WorldScale] Skip: mesh not available or not in edit mode")
        return

    # Track modal operators to detect operation boundaries
    window = bpy.context.window
    current_modals = set(op.bl_idname for op in window.modal_operators) if window else set()

    # Detect if modal operation just ended (before updating tracking)
    modal_just_ended = bool(_tracked_modal_operators) and not bool(current_modals)

    # Paint modals (alt+click texture apply, paint select) modify UVs or
    # selection without moving geometry. The cache-restore logic below is
    # designed for transform modals (grab, rotate, scale, extrude, shear, etc.)
    # where returning geometry to its original position should restore the
    # original UVs. Paint modals intentionally change UVs on unmoved faces,
    # so restoring from cache would revert those changes.
    #
    # A whitelist of transform modals (e.g. TRANSFORM_OT_translate,
    # TRANSFORM_OT_rotate, TRANSFORM_OT_resize, MESH_OT_extrude_region_move,
    # TRANSFORM_OT_shrink_fatten, TRANSFORM_OT_shear, etc.) might be more
    # robust long-term, but we don't know the full scope of Blender's
    # transform modals, so for now we blacklist our own paint modals.
    _PAINT_MODALS = {
        'LEVELDESIGN_OT_apply_image_to_face',
        'LEVELDESIGN_OT_pick_image_from_face',
        'LEVELDESIGN_OT_stretch_apply_image_to_face',
        'LEVELDESIGN_OT_apply_uv_transform_to_face',
        'LEVELDESIGN_OT_backface_paint_select',
    }
    is_paint_modal = bool(current_modals & _PAINT_MODALS)

    # Update tracking
    in_modal_operation = bool(current_modals) and not is_paint_modal
    _tracked_modal_operators = current_modals

    if current_modals:
        debug_log(f"[WorldScale] Active modals: {current_modals}")
    if modal_just_ended:
        debug_log("[WorldScale] Modal just ended")

    try:
        bm = bmesh.from_edit_mesh(me)
    except (ReferenceError, RuntimeError):
        # BMesh is invalid or being modified
        debug_log("[WorldScale] Skip: BMesh invalid or being modified")
        return

    # Validate BMesh state
    if not bm.is_valid:
        debug_log("[WorldScale] Skip: BMesh not valid")
        return

    # Ensure lookup tables are valid before accessing faces
    try:
        bm.faces.ensure_lookup_table()
    except (ReferenceError, RuntimeError):
        debug_log("[WorldScale] Skip: lookup table failed")
        return

    unlocked_layers = get_unlocked_uv_layers(bm, obj, me)
    if not unlocked_layers:
        debug_log("[WorldScale] Skip: no unlocked UV layers")
        return

    props = scene.level_design_props
    ppm = props.pixels_per_meter

    id_layer = get_face_id_layer(bm)

    # Iterate using indices to be more resilient during topology changes
    face_indices = list(range(len(bm.faces)))
    uv_applied_count = 0
    uv_restored_count = 0
    skipped_no_cache = 0
    skipped_hotspot = 0
    skipped_no_selection = 0
    skipped_vert_mismatch = 0
    skipped_no_move = 0
    for face_idx in face_indices:
        # Wrap each face access in try/except to handle race conditions
        # during modal operators like loop cut
        try:
            # Re-validate after potential changes
            if not bm.is_valid or face_idx >= len(bm.faces):
                debug_log(f"[WorldScale] Abort: BMesh invalidated at face {face_idx}")
                return

            face = bm.faces[face_idx]
            face_id = face[id_layer]

            if face_id == 0 or face_id not in face_data_cache:
                skipped_no_cache += 1
                continue

            # Skip faces with hotspottable materials
            if face_has_hotspot_material(face, me):
                skipped_hotspot += 1
                continue

            cached = face_data_cache[face_id]
            cached_verts = cached['verts']

            # HACK - premature optimisation? This cuts down a lot of faces but means we need to deal with Blender's own
            # UV correction on other faces which is difficult, but done successfully - see set_correct_uv_slide
            # Skip faces not relevant to current operation
            # A face is relevant if any of its vertices are selected
            if not any(v.select for v in face.verts):
                skipped_no_selection += 1
                continue

            current_verts = [v.co.copy() for v in face.verts]
            if len(current_verts) != len(cached_verts):
                skipped_vert_mismatch += 1
                continue

            # Check if vertices actually moved - skip if geometry unchanged
            has_moved = False
            for current_vert, cached_vert in zip(current_verts, cached_verts):
                if (current_vert - cached_vert).length > 0.0001:
                    has_moved = True
                    break

            if not has_moved:
                # During modal operations, the cache baseline is preserved (not updated).
                # If geometry returns to the original position, we must restore the cached
                # UVs. Otherwise, UVs computed for an intermediate geometry state remain
                # on the face, causing warping when the geometry is back at the baseline.
                if in_modal_operation:
                    restored = False
                    for uv_layer in unlocked_layers:
                        layer_data = get_cached_layer_data(face_id, uv_layer.name)
                        if layer_data:
                            cached_uvs = layer_data.get('uvs')
                            if cached_uvs and len(cached_uvs) == len(face.loops):
                                for loop, cached_uv in zip(face.loops, cached_uvs):
                                    loop[uv_layer].uv = cached_uv.copy()
                                restored = True
                    if restored:
                        bmesh.update_edit_mesh(me)
                        uv_restored_count += 1
                else:
                    skipped_no_move += 1
                continue

            # Pre-compute shared geometry for this face (used by all layers)
            old_edge = get_local_x_from_verts_3d(cached_verts)
            new_edge = get_local_x_from_verts_3d(current_verts)

            edge_rotation = 0.0
            if old_edge is not None and new_edge is not None:
                cross = old_edge.cross(new_edge)
                dot_val = old_edge.dot(new_edge)
                dot_val = max(-1.0, min(1.0, dot_val))
                edge_rotation = math.degrees(math.atan2(cross.dot(face.normal), dot_val))

            translation = current_verts[0] - cached_verts[0]
            face_axes = get_face_local_axes(face)
            mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None

            # Apply to each unlocked layer
            for uv_layer in unlocked_layers:
                layer_data = get_cached_layer_data(face_id, uv_layer.name)

                if layer_data:
                    scale_u = layer_data.get('scale_u', 1.0)
                    scale_v = layer_data.get('scale_v', 1.0)
                    rotation = layer_data.get('rotation', 0.0)
                    offset_x = layer_data.get('offset_x', 0.0)
                    offset_y = layer_data.get('offset_y', 0.0)
                else:
                    # Fallback to top-level cached transform
                    scale_u = cached.get('scale_u', 1.0)
                    scale_v = cached.get('scale_v', 1.0)
                    rotation = cached.get('rotation', 0.0)
                    offset_x = cached.get('offset_x', 0.0)
                    offset_y = cached.get('offset_y', 0.0)

                # Face was zero-area when cached — project from a neighbor
                _was_zero_area = abs(scale_u) < 1e-8 and abs(scale_v) < 1e-8
                if _was_zero_area and face.calc_area() > 1e-8:
                    from .operators.texture_apply import set_uv_from_other_face
                    excluded = {f for f in bm.faces
                                if f.is_valid and f[id_layer] != 0
                                and f[id_layer] in face_data_cache
                                and abs(face_data_cache[f[id_layer]].get('scale_u', 1.0)) < 1e-8
                                and abs(face_data_cache[f[id_layer]].get('scale_v', 1.0)) < 1e-8}
                    source_face = _get_best_neighbor_face(face, excluded, id_layer)
                    if source_face:
                        set_uv_from_other_face(source_face, face, uv_layer, ppm, me, obj.matrix_world)
                    else:
                        apply_uv_to_face(face, uv_layer, 1.0, 1.0, 0.0, 0.0, 0.0, mat, ppm, me)
                    continue

                # Apply edge rotation compensation
                rotation = rotation + edge_rotation

                # Compensate offset for translation
                if face_axes and abs(scale_u) > 1e-8 and abs(scale_v) > 1e-8:
                    face_local_x, face_local_y = face_axes
                    rot_rad = math.radians(rotation)
                    cos_rot = math.cos(rot_rad)
                    sin_rot = math.sin(rot_rad)
                    proj_x = face_local_x * cos_rot - face_local_y * sin_rot
                    proj_y = face_local_x * sin_rot + face_local_y * cos_rot

                    move_x = translation.dot(proj_x)
                    move_y = translation.dot(proj_y)

                    tex_meters_u, tex_meters_v = get_texture_dimensions_from_material(mat, ppm)

                    offset_x = normalize_offset(offset_x + move_x / (scale_u * tex_meters_u))
                    offset_y = normalize_offset(offset_y + move_y / (scale_v * tex_meters_v))

                debug_log(f"[WorldScale] Face {face.index} layer '{uv_layer.name}': "
                         f"scale=({scale_u:.4f}, {scale_v:.4f}) rotation={rotation:.2f} "
                         f"offset=({offset_x:.4f}, {offset_y:.4f})")
                apply_uv_to_face(face, uv_layer, scale_u, scale_v, rotation, offset_x, offset_y,
                                 mat, ppm, me)

            uv_applied_count += 1

            # Only update cache when NOT in a modal operation
            if not in_modal_operation:
                cache_single_face(face, bm, ppm, me)
                debug_log(f"[WorldScale] Face {face.index}: applied UVs + cached")
            else:
                debug_log(f"[WorldScale] Face {face.index}: applied UVs (modal, no cache)")

        except (ReferenceError, RuntimeError, OSError):
            # BMesh data became invalid during iteration (e.g., during loop cut)
            debug_log("[WorldScale] Abort: BMesh exception during face iteration")
            return

    debug_log(f"[WorldScale] Done: applied={uv_applied_count} restored={uv_restored_count} | skipped: no_cache={skipped_no_cache} hotspot={skipped_hotspot} no_sel={skipped_no_selection} vert_mismatch={skipped_vert_mismatch} no_move={skipped_no_move} | modal={in_modal_operation} modal_just_ended={modal_just_ended}")

    # Refresh cache when modal operation just ended
    if modal_just_ended:
        debug_log("[WorldScale] Refreshing full cache (modal just ended)")
        cache_face_data(bpy.context)


def apply_uv_lock(obj, scene):
    """UV lock: texture stays locked to geometry on locked UV layers.

    When a UV layer is locked, the texture moves/scales with the face like a sticker.
    We don't need to modify UVs - they naturally stay attached to vertices.
    Just update the cache to track the new vertex positions.

    Only runs if there are locked UV layers.
    """
    me = obj.data

    # Skip if mesh data is not available or being modified
    if me is None or not me.is_editmode:
        return

    try:
        bm = bmesh.from_edit_mesh(me)
    except (ReferenceError, RuntimeError):
        return

    if not bm.is_valid:
        return

    # Only process if there are locked layers
    locked_layers = get_locked_uv_layers(bm, obj, me)
    if not locked_layers:
        return

    # Ensure lookup tables are valid before accessing faces
    try:
        bm.faces.ensure_lookup_table()
    except (ReferenceError, RuntimeError):
        return

    # Check if any faces have moved
    id_layer = get_face_id_layer(bm)
    has_moved = False
    face_indices = list(range(len(bm.faces)))
    for face_idx in face_indices:
        try:
            if not bm.is_valid or face_idx >= len(bm.faces):
                return

            face = bm.faces[face_idx]
            face_id = face[id_layer]

            if face_id == 0 or face_id not in face_data_cache:
                continue

            cached = face_data_cache[face_id]
            current_verts = [v.co.copy() for v in face.verts]
            if len(current_verts) != len(cached['verts']):
                continue

            for current, original in zip(current_verts, cached['verts']):
                if (current - original).length > 0.0001:
                    has_moved = True
                    break
            if has_moved:
                break
        except (ReferenceError, RuntimeError, OSError):
            # BMesh data became invalid during iteration
            return

    # Only update cache if geometry changed (no UV modification needed)
    if has_moved:
        cache_face_data(bpy.context)


def redraw_ui_panels(context):
    """Force redraw of UI panels to update texture preview"""
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def check_selection_changed(bm):
    """Check if face selection has changed. Returns True if selection changed."""
    global _last_selected_face_indices, _last_active_face_index

    id_layer = get_face_id_layer(bm)
    current_selected = {f[id_layer] for f in bm.faces if f.select}
    current_active = bm.faces.active[id_layer] if bm.faces.active else -1

    if current_selected != _last_selected_face_indices or current_active != _last_active_face_index:
        _last_selected_face_indices = current_selected
        _last_active_face_index = current_active
        return True
    return False


def update_active_image_from_face(context):
    """Update the active image based on the active face's material.

    Clears the active image if not in edit mode, no faces are selected,
    or the active face has no image material.
    """
    global _active_image_just_set
    if _active_image_just_set:
        _active_image_just_set = False
        return
    try:
        obj = context.object
        if not obj or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
            set_active_image(None)
            return

        # Require face select mode
        if not context.tool_settings.mesh_select_mode[2]:
            set_active_image(None)
            return

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        # Check if any faces are selected
        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            set_active_image(None)
            return

        active_face = bm.faces.active
        if not active_face:
            set_active_image(None)
            return

        # Get the material on this face
        mat_index = active_face.material_index
        mat = obj.data.materials[mat_index] if mat_index < len(obj.data.materials) else None

        if mat:
            image = get_image_from_material(mat)
            if image:
                set_active_image(image)
            else:
                set_active_image(None)
        else:
            set_active_image(None)
    except Exception:
        pass  # Silently fail to avoid disrupting user workflow


def get_selected_faces_share_image(obj, bm, me):
    """Check if all selected faces share the same image texture.

    Returns (shared, image) where shared is True if all selected faces
    have the same image, and image is that shared image (or None).
    """
    selected_faces = [f for f in bm.faces if f.select]
    if not selected_faces:
        return False, None

    first_image = None
    for face in selected_faces:
        mat_index = face.material_index
        mat = me.materials[mat_index] if mat_index < len(me.materials) else None
        image = get_image_from_material(mat) if mat else None

        if first_image is None:
            first_image = image
        elif image != first_image:
            return False, None

    return True, first_image


def apply_texture_from_file_browser():
    """Apply texture from current file browser selection to selected faces.

    Called when user clicks in the file browser. Loads the selected image,
    sets it as active, and applies it to any selected faces in edit mode.
    For hotspottable textures, applies hotspot UVs instead of regular projection.
    """
    from .hotspot_mapping.json_storage import is_texture_hotspottable
    from .operators.uv_tools import apply_hotspots_to_mesh

    try:
        context = bpy.context
        obj = context.object

        # Get current file browser selection
        current_path = get_selected_image_path(context)

        if not current_path:
            return

        # Skip if it's a directory (not a file)
        if not os.path.isfile(current_path):
            return

        # Load the image early so we can update the previous image preview
        # even when faces aren't selected
        try:
            image = bpy.data.images.load(current_path, check_existing=True)
        except RuntimeError:
            return

        # Always update previous image when a file is selected in the browser
        set_previous_image(image)
        redraw_ui_panels(context)

        # Only apply to faces if on a mesh in edit or object mode
        if not obj or obj.type != 'MESH':
            return

        in_edit_mode = (context.mode == 'EDIT_MESH')
        in_object_mode = (context.mode == 'OBJECT')

        if not in_edit_mode and not in_object_mode:
            return

        # In object mode, only apply if the object is actually selected
        if in_object_mode and not obj.select_get():
            return

        if in_edit_mode:
            bm = bmesh.from_edit_mesh(obj.data)
        else:
            bm = bmesh.new()
            bm.from_mesh(obj.data)

        # Ensure all custom data layers exist BEFORE storing face references,
        # because adding new BMesh layers invalidates existing element refs.
        uv_layer = get_render_active_uv_layer(bm, obj.data)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()
        get_face_id_layer(bm)

        bm.faces.ensure_lookup_table()

        if in_edit_mode:
            selected_faces = [f for f in bm.faces if f.select]
            if not selected_faces:
                return
        else:
            selected_faces = list(bm.faces)
        props = context.scene.level_design_props
        ppm = props.pixels_per_meter

        # Check which selected faces previously had hotspottable textures (before changing material)
        faces_with_previous_hotspot = [f for f in selected_faces if face_has_hotspot_material(f, obj.data)]
        any_previous_was_hotspottable = len(faces_with_previous_hotspot) > 0

        # Check if any selected face has connected faces with hotspot textures
        any_connected_has_hotspot = False
        for f in selected_faces:
            if any_connected_face_has_hotspot(f, obj.data):
                any_connected_has_hotspot = True
                break

        # Get or create material
        mat = find_material_with_image(image)
        if mat is None:
            mat = create_material_with_image(image)

        # Capture old material info BEFORE adding the new material slot.
        # If captured after append, objects with zero materials would see
        # face.material_index 0 resolve to the NEW material, incorrectly
        # reporting has_image=True and preventing the scale/rotation/offset reset.
        fb_id_layer = get_face_id_layer(bm)
        face_old_info = {}
        for f in selected_faces:
            f_mat_idx = f.material_index
            f_mat = obj.data.materials[f_mat_idx] if f_mat_idx < len(obj.data.materials) else None
            f_img = get_image_from_material(f_mat)
            face_old_info[f[fb_id_layer]] = {
                'mat': f_mat,
                'has_image': f_img is not None,
                'tex_dims': get_texture_dimensions_from_material(f_mat, ppm),
            }

        # Ensure material slot exists (after capturing old info)
        if mat.name not in obj.data.materials:
            obj.data.materials.append(mat)

        mat_index = obj.data.materials.find(mat.name)

        # Assign material to all selected faces
        for target_face in selected_faces:
            target_face.material_index = mat_index

        new_is_hotspottable = is_texture_hotspottable(image.name)

        # Determine if we should apply hotspot logic
        # Only apply hotspots if auto_hotspot is enabled
        if props.auto_hotspot and new_is_hotspottable:
            # New texture is hotspottable - apply hotspots to ALL faces with hotspot materials
            # (entire object shape is relevant when finding hotspot islands)
            all_hotspot_faces = get_all_hotspot_faces(bm, obj.data)

            if all_hotspot_faces:
                # Save selection state (apply_hotspots_to_mesh modifies selection)
                id_layer = get_face_id_layer(bm)
                selected_ids, active_id = save_face_selection(bm, id_layer)

                seam_mode = props.hotspot_seam_mode
                allow_combined_faces = obj.anvil_allow_combined_faces
                size_weight = obj.anvil_hotspot_size_weight
                seam_angle = obj.anvil_hotspot_seam_angle

                debug_log(f"[FileBrowser] Applying hotspots to {len(all_hotspot_faces)} faces (all hotspot faces)")
                apply_hotspots_to_mesh(
                    bm, obj.data, all_hotspot_faces, seam_mode, allow_combined_faces,
                    obj.matrix_world, ppm, size_weight, seam_angle
                )

                # Restore selection state
                restore_face_selection(bm, id_layer, selected_ids, active_id)

                # Cache all hotspot faces after application
                for face in all_hotspot_faces:
                    if face.is_valid:
                        cache_single_face(face, bm, ppm, obj.data)
        elif props.auto_hotspot and not new_is_hotspottable and any_previous_was_hotspottable and any_connected_has_hotspot:
            # New texture is NOT hotspottable, but some selected faces previously had hotspot
            # AND some connected faces have hotspot textures - re-hotspot all to recalculate islands
            all_hotspot_faces = get_all_hotspot_faces(bm, obj.data)

            if all_hotspot_faces:
                # Save selection state
                id_layer = get_face_id_layer(bm)
                selected_ids, active_id = save_face_selection(bm, id_layer)

                seam_mode = props.hotspot_seam_mode
                allow_combined_faces = obj.anvil_allow_combined_faces
                size_weight = obj.anvil_hotspot_size_weight
                seam_angle = obj.anvil_hotspot_seam_angle

                debug_log(f"[FileBrowser] Re-hotspotting {len(all_hotspot_faces)} faces (island structure changed)")
                apply_hotspots_to_mesh(
                    bm, obj.data, all_hotspot_faces, seam_mode, allow_combined_faces,
                    obj.matrix_world, ppm, size_weight, seam_angle
                )

                # Restore selection state
                restore_face_selection(bm, id_layer, selected_ids, active_id)

                # Cache all hotspot faces
                for face in all_hotspot_faces:
                    if face.is_valid:
                        cache_single_face(face, bm, ppm, obj.data)

            # Apply regular UV projection to the selected faces (non-hotspot texture)
            _apply_regular_uv_projection(selected_faces, uv_layer, mat, ppm, obj.data, face_old_info, bm)
        else:
            # Either auto_hotspot is off, or it's a non-hotspot texture without hotspot neighbors
            # Regular UV projection
            _apply_regular_uv_projection(selected_faces, uv_layer, mat, ppm, obj.data, face_old_info, bm)

        if in_edit_mode:
            bmesh.update_edit_mesh(obj.data)
        else:
            bm.to_mesh(obj.data)
            bm.free()
            obj.data.update()

        # Update UI to reflect the new UVs
        update_ui_from_selection(context)
        # Set the active image directly since we know what was just applied
        # (update_active_image_from_face relies on bm.faces.active and
        # context.tool_settings which may be unreliable in timer context)
        global _active_image_just_set
        set_active_image(image)
        _active_image_just_set = True
        redraw_ui_panels(context)

        # Push an undo step so Ctrl+Z undoes the texture application
        # (timer callbacks run outside the operator/undo system)
        bpy.ops.ed.undo_push(message="Apply Texture from File Browser")

    except Exception as e:
        print(f"Anvil Level Design: Error applying texture from file browser: {e}", flush=True)


def _apply_regular_uv_projection(selected_faces, uv_layer, mat, ppm, me, face_old_info, bm=None):
    """Apply regular UV projection to selected faces, preserving transform where possible.

    Args:
        selected_faces: List of BMesh faces to apply UVs to
        uv_layer: BMesh UV layer
        mat: Material to use for texture dimensions
        ppm: Pixels per meter setting
        me: Mesh data
        face_old_info: Dict mapping managed face ID to old material info (captured before
            material assignment). Each entry has 'mat', 'has_image', 'tex_dims'.
        bm: BMesh instance (optional, for cache_single_face per-layer caching)
    """
    id_layer = get_face_id_layer(bm) if bm is not None else None
    for target_face in selected_faces:
        # Get current transform to preserve it
        current_transform = derive_transform_from_uvs(target_face, uv_layer, ppm, me)

        # Get old material info
        face_key = target_face[id_layer] if id_layer is not None else target_face.index
        old_info = face_old_info[face_key]
        old_has_image = old_info['has_image']
        old_tex_dims = old_info['tex_dims']

        # Get new texture dimensions
        new_tex_dims = get_texture_dimensions_from_material(mat, ppm)

        # Reapply the preserved transform with the new texture
        if current_transform and old_has_image:
            # Had a previous image - preserve or reset transform
            if old_tex_dims != new_tex_dims:
                # Texture dimensions changed - reset scale but keep rotation/offset
                scale_u, scale_v = 1.0, 1.0
            else:
                # Same texture dimensions - preserve everything
                scale_u = current_transform['scale_u']
                scale_v = current_transform['scale_v']

            apply_uv_to_face(
                target_face, uv_layer,
                scale_u, scale_v,
                current_transform['rotation'],
                current_transform['offset_x'], current_transform['offset_y'],
                mat, ppm, me
            )
            if bm is not None:
                cache_single_face(target_face, bm, ppm, me)
        else:
            # Blank face (no previous image) or transform can't be derived
            # - use face-aligned world-axis projection
            face_aligned_project(target_face, uv_layer, mat, ppm)
            if bm is not None:
                cache_single_face(target_face, bm, ppm, me)


def _file_browser_watcher_timer():
    """Timer-based file browser watcher. Polls for selection changes.

    Returns the interval to keep running, or None to stop.
    """
    global _file_browser_watcher_running, _last_file_browser_path

    # Stop if addon is being unregistered
    if not _file_browser_watcher_running:
        return None

    try:
        context = bpy.context

        # Check if any file browser exists
        has_file_browser = False
        if context.window and context.window.screen:
            for area in context.window.screen.areas:
                if area.type == 'FILE_BROWSER':
                    has_file_browser = True
                    break

        if not has_file_browser:
            # No file browser open, keep polling but less frequently
            return 0.5

        # Get current selection
        current_path = get_selected_image_path(context)

        # Apply if selection changed
        if current_path and current_path != _last_file_browser_path:
            apply_texture_from_file_browser()
            _last_file_browser_path = current_path

    except Exception:
        # Context may not be ready, keep trying
        pass

    # Poll every 100ms when file browser is open
    return 0.1


def start_file_browser_watcher():
    """Start the file browser watcher timer."""
    global _file_browser_watcher_running, _last_file_browser_path

    if _file_browser_watcher_running:
        return

    _file_browser_watcher_running = True
    # Initialize to current state so we only react to changes after startup
    _last_file_browser_path = get_selected_image_path(bpy.context)
    bpy.app.timers.register(_file_browser_watcher_timer, first_interval=0.1)


class LEVELDESIGN_OT_force_apply_texture(bpy.types.Operator):
    """Force apply texture from file browser (Alt+Click)"""
    bl_idname = "leveldesign.force_apply_texture"
    bl_label = "Repick File Browser Selected Texture"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        apply_texture_from_file_browser()
        return {'FINISHED'}


# Store keymap items for cleanup
_addon_keymaps = []


def consolidate_duplicate_materials():
    """Find and merge duplicate IMG_ materials created by copy/paste.

    When objects are duplicated, Blender creates copies of materials with
    suffixes like .001, .002, etc. This function finds these duplicates
    and consolidates them to the base material name.
    """
    global _last_material_count

    current_count = len(bpy.data.materials)

    # Only run if material count has increased
    if current_count <= _last_material_count:
        _last_material_count = current_count
        return

    _last_material_count = current_count

    # Pattern to match IMG_ materials with .001, .002, etc. suffixes
    duplicate_pattern = re.compile(r'^(IMG_.+)\.(\d{3,})$')

    # Group duplicates by base name: {base_name: [(suffix_num, material), ...]}
    material_groups = {}

    for mat in bpy.data.materials:
        match = duplicate_pattern.match(mat.name)
        if match:
            base_name = match.group(1)
            suffix_num = int(match.group(2))
            if base_name not in material_groups:
                material_groups[base_name] = []
            material_groups[base_name].append((suffix_num, mat))

    if not material_groups:
        return

    # Ensure base material exists for each group (rename lowest duplicate if needed)
    for base_name, duplicates in material_groups.items():
        if base_name not in bpy.data.materials:
            # Rename lowest-numbered duplicate to base name
            duplicates.sort(key=lambda x: x[0])
            duplicates[0][1].name = base_name

    # Build mapping of duplicate -> base
    replacements = {}
    for base_name, duplicates in material_groups.items():
        canonical = bpy.data.materials[base_name]
        for suffix_num, mat in duplicates:
            if mat != canonical:
                replacements[mat] = canonical

    if not replacements:
        return

    # Replace materials on all objects
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or not obj.data:
            continue

        materials = obj.data.materials
        for i, mat in enumerate(materials):
            if mat in replacements:
                materials[i] = replacements[mat]

    # Remove the duplicate materials (they should now have 0 users)
    for dup_mat in replacements.keys():
        if dup_mat.users == 0:
            bpy.data.materials.remove(dup_mat)

    # Update count after removal
    _last_material_count = len(bpy.data.materials)


def set_all_grid_scales_to_default():
    """Set anvil grid scale to 1 and apply with unit conversion on all 3D views."""
    from .operators.grid_tools import apply_anvil_grid_scale

    scene = bpy.context.scene
    if hasattr(scene, 'level_design_props'):
        scene.level_design_props.anvil_grid_scale = 1.0
    unit_settings = scene.unit_settings
    apply_anvil_grid_scale(1.0, unit_settings.system, unit_settings.length_unit)


def disable_correct_uv_slide():
    """Disable the correct_uv flag for slide operations via direct memory access."""
    set_correct_uv_slide(False)


def _clear_file_loaded_flag():
    """Clear the file loaded flag after timeout."""
    global _file_loaded_into_edit_depsgraph
    _file_loaded_into_edit_depsgraph = False


def _clear_undo_flag():
    """Clear the undo flag after depsgraph has processed."""
    global _undo_in_progress
    _undo_in_progress = False


def _sync_weld_and_snapshot_selection():
    """Sync weld state from BMesh/Mesh to scene props after undo/redo.

    Also snapshots the current face selection so that the next depsgraph
    check_selection_changed call doesn't see a false change and incorrectly
    clear the restored weld state.
    """
    global _last_selected_face_indices, _last_active_face_index
    try:
        context = bpy.context
        if context.mode == 'EDIT_MESH':
            obj = context.active_object
            if obj and obj.type == 'MESH':
                import bmesh as _bm
                bm = _bm.from_edit_mesh(obj.data)
                from .operators.weld import sync_weld_props
                sync_weld_props(context, bm)
                # Snapshot selection to prevent false weld clearing
                id_layer = get_face_id_layer(bm)
                _last_selected_face_indices = {f[id_layer] for f in bm.faces if f.select}
                _last_active_face_index = bm.faces.active[id_layer] if bm.faces.active else -1
        else:
            from .operators.weld import sync_weld_props
            sync_weld_props(context, None)
            _last_selected_face_indices = set()
            _last_active_face_index = -1
    except Exception:
        _last_selected_face_indices = set()
        _last_active_face_index = -1


@persistent
def on_undo_pre(scene):
    """Handler called before an undo operation."""
    global _undo_in_progress, _auto_hotspot_pending
    _undo_in_progress = True
    _auto_hotspot_pending = False


@persistent
def on_undo_post(scene):
    """Handler called after an undo operation.

    Clears the undo flag via timer to ensure depsgraph update has completed.
    Invalidates face caches since geometry state has changed.
    """
    global last_face_count, last_vertex_count, _last_selected_face_indices, _last_active_face_index
    # Rebuild face cache from the undo-restored mesh state so that a
    # subsequent operation (e.g. bridge) in the same depsgraph cycle can
    # correctly identify which faces are genuinely new vs undo-restored.
    face_data_cache.clear()
    try:
        context = bpy.context
        if context.mode == 'EDIT_MESH':
            cache_face_data(context)
        else:
            last_face_count = 0
            last_vertex_count = 0
    except Exception:
        last_face_count = 0
        last_vertex_count = 0
    # Sync weld state and snapshot selection (shared with redo handler)
    _sync_weld_and_snapshot_selection()
    # Use a short timer to ensure the depsgraph update triggered by undo
    # has completed before we clear the flag
    bpy.app.timers.register(_clear_undo_flag, first_interval=0.05)
    # Refresh UI panels immediately
    try:
        context = bpy.context
        saved_prev_name = _previous_image_name
        saved_prev_path = _previous_image_filepath
        update_ui_from_selection(context)
        update_active_image_from_face(context)
        # Only restore previous image if undo left no active face;
        # if a face is now selected, let the normal update take effect.
        if _active_image is None:
            _previous_image_name = saved_prev_name
            _previous_image_filepath = saved_prev_path
        redraw_ui_panels(context)
    except Exception:
        pass


@persistent
def on_redo_pre(scene):
    """Handler called before a redo operation."""
    global _undo_in_progress, _auto_hotspot_pending
    _undo_in_progress = True
    _auto_hotspot_pending = False


@persistent
def on_redo_post(scene):
    """Handler called after a redo operation.

    Clears the undo flag via timer to ensure depsgraph update has completed.
    Invalidates face caches since geometry state has changed.
    """
    global last_face_count, last_vertex_count, _last_selected_face_indices, _last_active_face_index
    # Rebuild face cache from the redo-restored mesh state so that a
    # subsequent operation in the same depsgraph cycle can correctly
    # identify which faces are genuinely new vs redo-restored.
    face_data_cache.clear()
    try:
        context = bpy.context
        if context.mode == 'EDIT_MESH':
            cache_face_data(context)
        else:
            last_face_count = 0
            last_vertex_count = 0
    except Exception:
        last_face_count = 0
        last_vertex_count = 0
    # Sync weld state and snapshot selection (shared with undo handler)
    _sync_weld_and_snapshot_selection()
    bpy.app.timers.register(_clear_undo_flag, first_interval=0.05)
    # Refresh UI panels immediately
    try:
        context = bpy.context
        saved_prev_name = _previous_image_name
        saved_prev_path = _previous_image_filepath
        update_ui_from_selection(context)
        update_active_image_from_face(context)
        if _active_image is None:
            _previous_image_name = saved_prev_name
            _previous_image_filepath = saved_prev_path
        redraw_ui_panels(context)
    except Exception:
        pass


@persistent
def on_save_pre(dummy):
    """Handler called before saving a .blend file.

    Makes all paths relative if the file has been saved before.
    For first-time saves, we wait until save_post to make paths relative.
    """
    global _was_first_save
    _was_first_save = not bpy.data.filepath

    if bpy.data.filepath:
        print("Anvil Level Design: Making all paths relative (pre save)", flush=True)
        bpy.ops.file.make_paths_relative()


@persistent
def on_save_post(dummy):
    """Handler called after saving a .blend file.

    If this was the first save, makes paths relative and triggers a second save.
    """
    global _was_first_save

    if _was_first_save:
        _was_first_save = False
        print("Anvil Level Design: Making all paths relative (post save, for first save)", flush=True)
        bpy.ops.file.make_paths_relative()
        print("Anvil Level Design: Triggering second save to apply relative paths", flush=True)
        bpy.ops.wm.save_mainfile()


def _migrate_legacy_uv_lock():
    """Migrate old per-object anvil_uv_lock boolean to per-UV-map settings.

    Old files have obj.anvil_uv_lock (BoolProperty). New files use
    obj.anvil_uv_map_settings (CollectionProperty). If the old property
    exists and is True, lock all UV maps on that object.
    """
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        # Check for legacy property (it may still exist on objects from old files)
        legacy_lock = obj.get("anvil_uv_lock", False)
        if not legacy_lock:
            continue

        me = obj.data
        if not me.uv_layers:
            continue

        # Sync and lock all UV maps
        sync_uv_map_settings(obj)
        for setting in obj.anvil_uv_map_settings:
            setting.locked = True

        # Clear the legacy property
        if "anvil_uv_lock" in obj:
            del obj["anvil_uv_lock"]

        debug_log(f"[Migration] Migrated UV lock for '{obj.name}': locked all {len(me.uv_layers)} UV maps")


@persistent
def on_load_post(dummy):
    """Handler called after a .blend file is loaded."""
    global _file_browser_watcher_running, _last_file_browser_path, _file_loaded_into_edit_depsgraph
    global last_face_count, last_vertex_count, _last_selected_face_indices, _last_active_face_index
    global _last_tracked_mode

    # Set up workspaces and scene settings for new (unsaved) files
    reset_specialized_template_flag()
    if not bpy.data.filepath:
        setup_addon_workspaces()
        subscribe_splash_watcher()

    # Invalidate face caches - file state has changed
    face_data_cache.clear()
    last_face_count = 0
    last_vertex_count = 0
    # Reset selection tracking so the next depsgraph update detects the current selection
    _last_selected_face_indices = set()
    _last_active_face_index = -1
    # Reset watcher state on file load (modal was killed when file loaded)
    _file_browser_watcher_running = False
    # Reset last file browser path so first click applies regardless of previous session
    _last_file_browser_path = None
    # Allow first depsgraph update to set active image from selected face
    _file_loaded_into_edit_depsgraph = True
    # Restore face orientation mode tracking so transitions are detected after load
    if _get_saved_face_orientation() is not None:
        _last_tracked_mode = bpy.context.mode
    else:
        _last_tracked_mode = None
    # Use a timer to ensure all UI is ready
    bpy.app.timers.register(set_all_grid_scales_to_default, first_interval=0.1)
    # Restart the file browser watcher
    bpy.app.timers.register(start_file_browser_watcher, first_interval=0.2)
    # Disable correct_uv for slide operations
    bpy.app.timers.register(disable_correct_uv_slide, first_interval=0.1)
    # Re-subscribe to msgbus (subscriptions are lost on file load)
    bpy.app.timers.register(_subscribe_unit_settings, first_interval=0.1)
    bpy.app.timers.register(_subscribe_object_mode, first_interval=0.1)
    # Clear the file loaded flag after 1 second (fallback if depsgraph doesn't fire)
    bpy.app.timers.register(_clear_file_loaded_flag, first_interval=1.0)
    # Migrate legacy per-object UV lock to per-UV-map settings
    _migrate_legacy_uv_lock()


def _poll_slide_snap_end():
    """Timer that polls until the slide modal exits, then restores snap to INCREMENT."""
    global _slide_snap_active
    window = bpy.context.window
    if not window:
        _slide_snap_active = False
        return None
    current_modals = set(op.bl_idname for op in window.modal_operators)
    if bool(current_modals & _SLIDE_MODAL_OPS):
        return 0.05
    bpy.context.scene.tool_settings.snap_elements = {'INCREMENT'}
    _slide_snap_active = False
    debug_log("[SnapMode] Slide modal ended, snap set to INCREMENT")
    return None


def _update_slide_snap_mode():
    global _slide_snap_active
    if _slide_snap_active:
        return
    window = bpy.context.window
    if not window:
        return
    current_modals = set(op.bl_idname for op in window.modal_operators)
    if bool(current_modals & _SLIDE_MODAL_OPS):
        bpy.context.scene.tool_settings.snap_elements = {'GRID'}
        _slide_snap_active = True
        bpy.app.timers.register(_poll_slide_snap_end, first_interval=0.05)
        debug_log("[SnapMode] Slide modal entered, snap set to GRID")


@persistent
def on_depsgraph_update(scene, depsgraph):
    """Consolidated depsgraph update handler"""
    global last_face_count, _file_loaded_into_edit_depsgraph, _force_auto_hotspot, _last_selected_face_indices, _last_active_face_index, _last_edit_object_name

    # Skip all depsgraph handling during undo operations
    if _undo_in_progress:
        return

    # Only process in the Level Design workspace
    if not is_level_design_workspace():
        return

    # Auto-switch snap mode for slide operators (loop cut, vertex slide, edge slide)
    _update_slide_snap_mode()

    try:
        # Check for duplicate materials (from copy/paste operations)
        consolidate_duplicate_materials()

        # Ensure all updated mesh objects have a color attribute
        for update in depsgraph.updates:
            if isinstance(update.id, bpy.types.Object):
                obj = update.id
                if obj.type == 'MESH' and obj.data and len(obj.data.color_attributes) == 0:
                    # Use original data (not evaluated copy) so changes persist
                    orig_mesh = bpy.data.meshes.get(obj.data.name)
                    if orig_mesh and len(orig_mesh.color_attributes) == 0:
                        orig_mesh.color_attributes.new("Color", 'BYTE_COLOR', 'CORNER')
                        orig_mesh.color_attributes.active_color_index = 0
                        orig_mesh.color_attributes.render_color_index = 0

        context = bpy.context

        # Safety check - ensure properties are registered
        if not hasattr(scene, 'level_design_props'):
            return

        props = scene.level_design_props

        # Clear edit session tracking when not in edit mode so that
        # re-entering edit mode (even on the same object) is detected as fresh
        if context.mode != 'EDIT_MESH' and _last_edit_object_name is not None:
            _last_edit_object_name = None

        # In object mode, sync weld display from active object's mesh
        if context.mode != 'EDIT_MESH':
            from .operators.weld import sync_weld_props
            sync_weld_props(context, None)

        # Handle mesh updates (UV lock, world-scale UVs)
        for update in depsgraph.updates:
            if isinstance(update.id, bpy.types.Object):
                obj = update.id
                if obj.type == 'MESH' and obj.mode == 'EDIT':
                    me = obj.data

                    # Skip if mesh data is not available
                    if me is None or not me.is_editmode:
                        debug_log(f"[Depsgraph] Skip: mesh not available or not in edit mode")
                        continue

                    # Track if this is an actual geometry update (not just tab switch, etc.)
                    is_geometry_update = update.is_updated_geometry
                    is_transform_update = getattr(update, 'is_updated_transform', False)
                    debug_log(f"[Depsgraph] Update for '{obj.name}': geometry={is_geometry_update} transform={is_transform_update}")

                    try:
                        bm = bmesh.from_edit_mesh(me)
                    except (ReferenceError, RuntimeError):
                        # BMesh is invalid or being modified by modal operator
                        debug_log(f"[Depsgraph] Skip: BMesh invalid/being modified")
                        continue

                    # Validate BMesh state
                    if not bm.is_valid:
                        debug_log(f"[Depsgraph] Skip: BMesh not valid")
                        continue

                    current_face_count = len(bm.faces)
                    current_vertex_count = len(bm.verts)

                    # Detect fresh edit session (entering edit mode or switching objects)
                    # This must happen before topology/selection checks to prevent
                    # updating active image when user didn't explicitly click a face
                    is_fresh_start = (obj.name != _last_edit_object_name)
                    _last_edit_object_name = obj.name

                    if is_fresh_start:
                        debug_log(f"[Depsgraph] Fresh edit session for '{obj.name}'")
                        # Sync per-UV-map settings for this object
                        sync_uv_map_settings(obj)
                        # Rebuild cache for the new object and reset selection tracking
                        # so check_selection_changed detects the current selection as new
                        cache_face_data(context)
                        _last_selected_face_indices = set()
                        _last_active_face_index = -1
                        debug_log(f"[Depsgraph] Cache rebuilt for new object ({len(face_data_cache)} faces)")

                    # On file load, allow active image sync even if fresh start
                    # (to restore state from previous session)
                    allow_active_image_update = not is_fresh_start or _file_loaded_into_edit_depsgraph

                    # Check if topology changed (subdivision, extrusion, etc.)
                    topology_changed = current_face_count != last_face_count or current_vertex_count != last_vertex_count

                    # Also detect when a modal operator (e.g. bevel) restored
                    # from its internal snapshot and re-applied, which recreates
                    # the BMesh.  The BMesh bevel op copies custom data from
                    # source faces to new faces, producing duplicate face IDs.
                    # Face/vert counts stay the same, so the normal topology
                    # check misses this — but the stale IDs cause
                    # apply_world_scale_uvs to apply wrong cached transforms.
                    if not topology_changed and not is_fresh_start:
                        id_layer_check = get_face_id_layer(bm)
                        seen_ids = set()
                        for face in bm.faces:
                            fid = face[id_layer_check]
                            if fid in seen_ids:
                                topology_changed = True
                                debug_log("[Depsgraph] Duplicate face IDs detected (modal restore/re-apply)")
                                break
                            seen_ids.add(fid)

                    if topology_changed:
                        debug_log(f"[Depsgraph] Topology changed: faces {last_face_count}->{current_face_count} verts {last_vertex_count}->{current_vertex_count}")

                        # Project new faces when faces were added or modal
                        # re-applied (duplicate IDs mean the mesh was rebuilt)
                        if not is_fresh_start:
                            _project_new_faces(context, bm)

                        # Check if a topology-changing modal (e.g. bevel) is
                        # active.  These modals restore from a pre-op snapshot
                        # and re-apply on every mouse move, so the cache must
                        # stay at the pre-op baseline for correct projection.
                        # Cache rebuild is deferred until the modal ends (next
                        # topology-changed fire with no active modal).
                        topo_window = bpy.context.window
                        topo_modals = set(op.bl_idname for op in topo_window.modal_operators) if topo_window else set()
                        in_topology_modal = bool(topo_modals & _TOPOLOGY_MODAL_OPS)

                        if in_topology_modal:
                            debug_log("[Depsgraph] Skipping cache rebuild (topology modal active)")
                        else:
                            cache_face_data(context)
                            debug_log(f"[Depsgraph] Cache rebuilt ({len(face_data_cache)} faces)")
                            update_ui_from_selection(context)
                            # Only update active image if allowed
                            if allow_active_image_update:
                                update_active_image_from_face(context)
                            _file_loaded_into_edit_depsgraph = False
                            # Snapshot selection so next check_selection_changed
                            # doesn't see a false change from the topology update
                            id_layer = get_face_id_layer(bm)
                            _last_selected_face_indices = {f[id_layer] for f in bm.faces if f.select}
                            _last_active_face_index = bm.faces.active[id_layer] if bm.faces.active else -1
                            # Consume the weld-just-stored flag so it doesn't
                            # incorrectly block the next genuine selection change
                            from .operators import weld as _weld_mod
                            _weld_mod._weld_just_stored = False
                            if not is_fresh_start and props.auto_hotspot:
                                _force_auto_hotspot = True
                                _apply_auto_hotspots()
                        return

                    # Check if selection changed
                    selection_changed = check_selection_changed(bm)
                    if selection_changed:
                        debug_log(f"[Depsgraph] Selection changed")
                        update_ui_from_selection(context)
                        # Only update active image if allowed
                        if allow_active_image_update:
                            update_active_image_from_face(context)
                        # Clear weld state when user changes selection,
                        # but skip if weld was just stored (the operation
                        # that set the weld also changed selection).
                        from .operators import weld as _weld_mod
                        if _weld_mod._weld_just_stored:
                            _weld_mod._weld_just_stored = False
                        else:
                            _weld_mod.clear_weld_on_bmesh(bm)

                    # Keep weld scene properties in sync with BMesh layers.
                    from .operators.weld import sync_weld_props
                    sync_weld_props(context, bm)

                    # Store data before any transform if cache is empty
                    if not face_data_cache and context.mode == 'EDIT_MESH':
                        debug_log(f"[Depsgraph] Cache empty, rebuilding")
                        cache_face_data(context)

                    # Always run both: world-scale for unlocked layers, uv-lock for locked layers
                    debug_log(f"[Depsgraph] Applying world-scale UVs (cache size={len(face_data_cache)})")
                    apply_world_scale_uvs(obj, scene)
                    apply_uv_lock(obj, scene)

                    if not is_fresh_start and props.auto_hotspot:
                        _apply_auto_hotspots()

                    _file_loaded_into_edit_depsgraph = False
                    break
    except Exception as e:
        print(f"Anvil Level Design: Error in depsgraph handler: {e}", flush=True)


def _on_unit_settings_changed():
    """Msgbus callback: re-apply anvil grid scale when unit settings change."""
    from .operators.grid_tools import apply_anvil_grid_scale

    try:
        scene = bpy.context.scene
        if not hasattr(scene, 'level_design_props'):
            return
        anvil_scale = scene.level_design_props.anvil_grid_scale
        if anvil_scale == 0.0:
            anvil_scale = 1.0
        unit_settings = scene.unit_settings
        apply_anvil_grid_scale(anvil_scale, unit_settings.system, unit_settings.length_unit)
    except Exception:
        pass


def _subscribe_unit_settings():
    """Subscribe to unit setting changes via bpy.msgbus."""
    try:
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.UnitSettings, "system"),
            owner=_msgbus_owner,
            args=(),
            notify=_on_unit_settings_changed,
        )
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.UnitSettings, "length_unit"),
            owner=_msgbus_owner,
            args=(),
            notify=_on_unit_settings_changed,
        )
    except Exception:
        pass


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_force_apply_texture)

    # Subscribe to object mode changes for face orientation overlay
    _subscribe_object_mode()

    if on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)
    if on_save_pre not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(on_save_pre)
    if on_save_post not in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.append(on_save_post)
    if on_undo_pre not in bpy.app.handlers.undo_pre:
        bpy.app.handlers.undo_pre.append(on_undo_pre)
    if on_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(on_undo_post)
    if on_redo_pre not in bpy.app.handlers.redo_pre:
        bpy.app.handlers.redo_pre.append(on_redo_pre)
    if on_redo_post not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(on_redo_post)
    # Also set grid on addon enable
    bpy.app.timers.register(set_all_grid_scales_to_default, first_interval=0.1)
    # Start file browser watcher
    bpy.app.timers.register(start_file_browser_watcher, first_interval=0.2)

    # Register keymaps
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        # Alt+Click in file browser
        km = kc.keymaps.new(name='File Browser', space_type='FILE_BROWSER')
        kmi = km.keymap_items.new(
            'leveldesign.force_apply_texture',
            'LEFTMOUSE', 'CLICK',
            alt=True
        )
        _addon_keymaps.append((km, kmi))

    # Disable correct_uv for slide operations via direct memory access
    bpy.app.timers.register(disable_correct_uv_slide, first_interval=0.1)

    # Subscribe to unit setting changes (via timer for context availability)
    bpy.app.timers.register(_subscribe_unit_settings, first_interval=0.1)


def unregister():
    global last_face_count, last_vertex_count, _last_selected_face_indices, _last_active_face_index, _last_edit_object_name, _last_material_count, _active_image, _active_image_just_set, _file_browser_watcher_running, _last_file_browser_path, _file_loaded_into_edit_depsgraph, _was_first_save, _auto_hotspot_pending, _undo_in_progress, _multi_face_mode, _multi_face_unset_scale, _multi_face_unset_rotation, _multi_face_unset_offset, _all_selected_hotspot, _last_tracked_mode, _slide_snap_active

    _slide_snap_active = False

    # Clear face orientation msgbus subscription and restore state
    bpy.msgbus.clear_by_owner(_face_orientation_msgbus_owner)
    saved = _get_saved_face_orientation()
    if saved is not None:
        try:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                space.overlay.show_face_orientation = saved
        except Exception:
            pass
        _set_saved_face_orientation(None)
    _last_tracked_mode = None

    # Clear msgbus subscriptions
    bpy.msgbus.clear_by_owner(_msgbus_owner)

    # Stop the file browser watcher timer
    _file_browser_watcher_running = False
    _last_file_browser_path = None
    _auto_hotspot_pending = False
    _undo_in_progress = False

    # Remove keymaps
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
    if on_save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(on_save_pre)
    if on_save_post in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.remove(on_save_post)
    if on_undo_pre in bpy.app.handlers.undo_pre:
        bpy.app.handlers.undo_pre.remove(on_undo_pre)
    if on_undo_post in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(on_undo_post)
    if on_redo_pre in bpy.app.handlers.redo_pre:
        bpy.app.handlers.redo_pre.remove(on_redo_pre)
    if on_redo_post in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.remove(on_redo_post)

    bpy.utils.unregister_class(LEVELDESIGN_OT_force_apply_texture)

    face_data_cache.clear()
    last_face_count = 0
    last_vertex_count = 0
    _last_material_count = 0
    _last_selected_face_indices = set()
    _last_active_face_index = -1
    _last_edit_object_name = None
    _active_image = None
    _active_image_just_set = False
    _file_loaded_into_edit_depsgraph = False
    _was_first_save = False
    _multi_face_mode = False
    _multi_face_unset_scale = False
    _multi_face_unset_rotation = False
    _multi_face_unset_offset = False
    _all_selected_hotspot = False
