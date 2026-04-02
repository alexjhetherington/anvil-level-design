"""World-scale UV application and UV lock handling."""

import math

import bmesh
import bpy

from ..core.logging import debug_log
from ..core.face_id import get_face_id_layer
from ..core.geometry import get_local_x_from_verts_3d, normalize_offset
from ..core.materials import get_texture_dimensions_from_material
from ..core.uv_projection import get_face_local_axes, apply_uv_to_face
from ..core.uv_layers import get_unlocked_uv_layers, get_locked_uv_layers
from ..core.hotspot_queries import face_has_hotspot_material

from .face_cache import (
    face_data_cache, cache_single_face, cache_face_data,
    get_cached_layer_data,
)
from .auto_hotspot import _get_best_neighbor_face


# Modal operators that restore-from-snapshot and re-apply on each mouse move.
_TOPOLOGY_MODAL_OPS = {
    'MESH_OT_bevel',
}

# Track modal operators for UV world-scale baseline
_tracked_modal_operators = set()


def get_topology_modal_ops():
    return _TOPOLOGY_MODAL_OPS


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
        debug_log("[WorldScale] Skip: BMesh invalid or being modified")
        return

    if not bm.is_valid:
        debug_log("[WorldScale] Skip: BMesh not valid")
        return

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

    face_indices = list(range(len(bm.faces)))
    uv_applied_count = 0
    uv_restored_count = 0
    skipped_no_cache = 0
    skipped_hotspot = 0
    skipped_no_selection = 0
    skipped_vert_mismatch = 0
    skipped_no_move = 0
    for face_idx in face_indices:
        try:
            if not bm.is_valid or face_idx >= len(bm.faces):
                debug_log(f"[WorldScale] Abort: BMesh invalidated at face {face_idx}")
                return

            face = bm.faces[face_idx]
            face_id = face[id_layer]

            if face_id == 0 or face_id not in face_data_cache:
                skipped_no_cache += 1
                continue

            if face_has_hotspot_material(face, me):
                skipped_hotspot += 1
                continue

            cached = face_data_cache[face_id]
            cached_verts = cached['verts']

            if not any(v.select for v in face.verts):
                skipped_no_selection += 1
                continue

            current_verts = [v.co.copy() for v in face.verts]
            if len(current_verts) != len(cached_verts):
                skipped_vert_mismatch += 1
                continue

            has_moved = False
            for current_vert, cached_vert in zip(current_verts, cached_verts):
                if (current_vert - cached_vert).length > 0.0001:
                    has_moved = True
                    break

            if not has_moved:
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

            for uv_layer in unlocked_layers:
                layer_data = get_cached_layer_data(face_id, uv_layer.name)

                if layer_data:
                    scale_u = layer_data.get('scale_u', 1.0)
                    scale_v = layer_data.get('scale_v', 1.0)
                    rotation = layer_data.get('rotation', 0.0)
                    offset_x = layer_data.get('offset_x', 0.0)
                    offset_y = layer_data.get('offset_y', 0.0)
                else:
                    scale_u = cached.get('scale_u', 1.0)
                    scale_v = cached.get('scale_v', 1.0)
                    rotation = cached.get('rotation', 0.0)
                    offset_x = cached.get('offset_x', 0.0)
                    offset_y = cached.get('offset_y', 0.0)

                _was_zero_area = abs(scale_u) < 1e-8 and abs(scale_v) < 1e-8
                if _was_zero_area and face.calc_area() > 1e-8:
                    from ..operators.texture_apply import set_uv_from_other_face
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

                rotation = rotation + edge_rotation

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

            if not in_modal_operation:
                cache_single_face(face, bm, ppm, me)
                debug_log(f"[WorldScale] Face {face.index}: applied UVs + cached")
            else:
                debug_log(f"[WorldScale] Face {face.index}: applied UVs (modal, no cache)")

        except (ReferenceError, RuntimeError, OSError):
            debug_log("[WorldScale] Abort: BMesh exception during face iteration")
            return

    debug_log(f"[WorldScale] Done: applied={uv_applied_count} restored={uv_restored_count} | skipped: no_cache={skipped_no_cache} hotspot={skipped_hotspot} no_sel={skipped_no_selection} vert_mismatch={skipped_vert_mismatch} no_move={skipped_no_move} | modal={in_modal_operation} modal_just_ended={modal_just_ended}")

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

    if me is None or not me.is_editmode:
        return

    try:
        bm = bmesh.from_edit_mesh(me)
    except (ReferenceError, RuntimeError):
        return

    if not bm.is_valid:
        return

    locked_layers = get_locked_uv_layers(bm, obj, me)
    if not locked_layers:
        return

    try:
        bm.faces.ensure_lookup_table()
    except (ReferenceError, RuntimeError):
        return

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
            return

    if has_moved:
        cache_face_data(bpy.context)
