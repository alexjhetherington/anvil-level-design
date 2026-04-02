"""Face data caching, multi-face state tracking, and selection change detection."""

import bmesh

from ..core.face_id import get_face_id_layer, assign_face_id, reindex_face_ids
from ..core.uv_projection import derive_transform_from_uvs
from ..core.uv_layers import get_render_active_uv_layer
from ..core.materials import get_image_from_material
from ..core.hotspot_queries import face_has_hotspot_material
from ..properties import set_updating_from_selection, sync_scale_tracking


# Cache for face data (UV lock functionality)
face_data_cache = {}
last_face_count = 0
last_vertex_count = 0

# Multi-face selection state tracking
_multi_face_mode = False
_multi_face_unset_scale = False
_multi_face_unset_rotation = False
_multi_face_unset_offset = False
_all_selected_hotspot = False
_any_selected_hotspot = False
_any_selected_fixed_hotspot = False

# Cache for detecting selection changes
_last_selected_face_indices = set()
_last_active_face_index = -1
# Track which object we're editing to detect fresh edit sessions
_last_edit_object_name = None


def get_all_selected_hotspot():
    return _all_selected_hotspot


def get_any_selected_hotspot():
    return _any_selected_hotspot


def get_any_selected_fixed_hotspot():
    return _any_selected_fixed_hotspot


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


def get_last_edit_object_name():
    return _last_edit_object_name


def set_last_edit_object_name(name):
    global _last_edit_object_name
    _last_edit_object_name = name


def get_last_selected_face_indices():
    return _last_selected_face_indices


def set_last_selected_face_indices(indices):
    global _last_selected_face_indices
    _last_selected_face_indices = indices


def get_last_active_face_index():
    return _last_active_face_index


def set_last_active_face_index(index):
    global _last_active_face_index
    _last_active_face_index = index


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
    global _multi_face_mode, _multi_face_unset_scale, _multi_face_unset_rotation, _multi_face_unset_offset, _all_selected_hotspot, _any_selected_hotspot, _any_selected_fixed_hotspot

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

    # Check if all/any selected faces have hotspot materials
    if selected_faces:
        hotspot_flags = [face_has_hotspot_material(f, me) for f in selected_faces]
        _all_selected_hotspot = all(hotspot_flags)
        _any_selected_hotspot = any(hotspot_flags)

        # Check if any selected faces have the fixed hotspot flag
        fixed_layer = bm.faces.layers.int.get("anvil_fixed_hotspot")
        if fixed_layer is not None:
            _any_selected_fixed_hotspot = any(f[fixed_layer] != 0 for f in selected_faces)
        else:
            _any_selected_fixed_hotspot = False
    else:
        _all_selected_hotspot = False
        _any_selected_hotspot = False
        _any_selected_fixed_hotspot = False

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


def snapshot_selection(bm):
    """Snapshot the current face selection so next check_selection_changed works correctly."""
    global _last_selected_face_indices, _last_active_face_index
    id_layer = get_face_id_layer(bm)
    _last_selected_face_indices = {f[id_layer] for f in bm.faces if f.select}
    _last_active_face_index = bm.faces.active[id_layer] if bm.faces.active else -1


def reset():
    """Reset all face cache state."""
    global last_face_count, last_vertex_count, _last_selected_face_indices, _last_active_face_index
    global _last_edit_object_name, _multi_face_mode, _multi_face_unset_scale
    global _multi_face_unset_rotation, _multi_face_unset_offset
    global _all_selected_hotspot, _any_selected_hotspot, _any_selected_fixed_hotspot

    face_data_cache.clear()
    last_face_count = 0
    last_vertex_count = 0
    _last_selected_face_indices = set()
    _last_active_face_index = -1
    _last_edit_object_name = None
    _multi_face_mode = False
    _multi_face_unset_scale = False
    _multi_face_unset_rotation = False
    _multi_face_unset_offset = False
    _all_selected_hotspot = False
    _any_selected_hotspot = False
    _any_selected_fixed_hotspot = False
