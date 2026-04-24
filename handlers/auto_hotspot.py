"""Auto-hotspot projection handler."""

import bmesh
import bpy

from ..core.logging import debug_log
from ..core.face_id import (
    get_face_id_layer, save_face_selection, restore_face_selection,
)
from ..core.uv_layers import get_unlocked_uv_layers
from ..core.hotspot_queries import face_has_hotspot_material

from .face_cache import face_data_cache, cache_single_face


_auto_hotspot_pending = False
_force_auto_hotspot = False


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
        seam_angle = obj.anvil_hotspot_seam_angle

        props = context.scene.level_design_props
        debug_log(f"[AutoHotspot] Processing {len(all_hotspot_faces)} hotspot faces")
        result = apply_hotspots_to_mesh(
            bm, me, all_hotspot_faces, allow_combined_faces,
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


def apply_auto_hotspots():
    """Schedule auto-hotspot application via timer for proper operator context."""
    global _auto_hotspot_pending
    if _auto_hotspot_pending:
        return
    _auto_hotspot_pending = True
    bpy.app.timers.register(_apply_auto_hotspots_deferred, first_interval=0.1)


def reset():
    """Reset auto-hotspot state."""
    global _auto_hotspot_pending, _force_auto_hotspot
    _auto_hotspot_pending = False
    _force_auto_hotspot = False
