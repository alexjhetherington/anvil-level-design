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
        print(f"Anvil Level Design: Failed to set correct_uv_slide (see method comments): {e}")

from .utils import (
    get_image_from_material, derive_transform_from_uvs,
    get_selected_image_path, find_material_with_image, create_material_with_image,
    get_texture_dimensions_from_material, get_face_local_axes, normalize_offset,
    get_local_x_from_verts_3d, debug_log, face_has_hotspot_material,
    any_connected_face_has_hotspot, get_all_hotspot_faces,
    is_level_design_workspace,
)
from .properties import set_updating_from_selection, sync_scale_tracking, apply_uv_to_face


_auto_hotspot_pending = False
_force_auto_hotspot = False

# Track undo operations to skip depsgraph handling during undo
_undo_in_progress = False


def _any_hotspot_geometry_changed(bm, me):
    """Check if any hotspot face has geometry that differs from cache.

    Returns True if any hotspot face is new or has moved vertices.
    Returns False if all hotspot faces match their cached geometry (e.g., after undo).
    """
    for face in bm.faces:
        if not face.is_valid or not face_has_hotspot_material(face, me):
            continue

        if face.index not in face_data_cache:
            # New face - geometry changed
            return True

        cached = face_data_cache[face.index]
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
        selected_face_indices = {f.index for f in bm.faces if f.select}
        active_face_index = bm.faces.active.index if bm.faces.active else None

        props = context.scene.level_design_props
        seam_mode = props.hotspot_seam_mode
        allow_combined_faces = obj.anvil_allow_combined_faces
        size_weight = obj.anvil_hotspot_size_weight

        debug_log(f"[AutoHotspot] Processing {len(all_hotspot_faces)} hotspot faces")
        result = apply_hotspots_to_mesh(
            bm, me, all_hotspot_faces, seam_mode, allow_combined_faces,
            obj.matrix_world, props.pixels_per_meter, size_weight
        )
        debug_log(f"[AutoHotspot] Applied: {result}")

        # Restore selection state
        bm.faces.ensure_lookup_table()
        for face in bm.faces:
            face.select = face.index in selected_face_indices
        if active_face_index is not None and active_face_index < len(bm.faces):
            bm.faces.active = bm.faces[active_face_index]

        # Update cache for processed faces
        uv_layer = bm.loops.layers.uv.verify()
        ppm = props.pixels_per_meter
        for face in all_hotspot_faces:
            if face.is_valid:
                cache_single_face(face, uv_layer, ppm, me)

        bmesh.update_edit_mesh(me)

    except Exception as e:
        print(f"Anvil Level Design: Auto-hotspot error: {e}")
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


def _get_best_neighbor_face(face, selected_faces_set):
    """Find the best neighboring face to use as UV source.

    Prefers neighbors that are facing sideways (wall-like) over up/down (floor/ceiling).
    A sideways face has a more horizontal normal (small Z component).
    Only considers unselected neighbors (which have existing good UVs).
    """
    best_neighbor = None
    best_score = -1  # Higher = more sideways

    for edge in face.edges:
        # Get the neighbor face through this edge
        for linked_face in edge.link_faces:
            if linked_face == face or not linked_face.is_valid:
                continue
            # Skip selected faces (they're also new and don't have good UVs)
            if linked_face in selected_faces_set:
                continue

            # Score by how sideways the neighbor is facing
            # Walls have normals like (1,0,0) or (0,1,0) - Z near 0
            # Floors/ceilings have normals like (0,0,1) - Z near 1
            neighbor_normal = linked_face.normal
            sideways_score = 1.0 - abs(neighbor_normal.z)

            if sideways_score > best_score:
                best_score = sideways_score
                best_neighbor = linked_face

    return best_neighbor


def _project_selected_faces_on_topology_change(context, bm):
    """Apply UV projection to selected non-hotspot faces after topology change.

    Operations like Bridge Edge Loops, Fill, Grid Fill select the newly created faces.
    Uses neighboring faces as UV source for seamless tiling (like alt-click).
    Hotspot faces are handled by the deferred auto-hotspot system.
    """
    from .operators.texture_apply import set_uv_from_other_face

    obj = context.object
    me = obj.data
    uv_layer = bm.loops.layers.uv.verify()
    props = context.scene.level_design_props
    ppm = props.pixels_per_meter

    selected_faces = [f for f in bm.faces if f.select and f.is_valid]
    selected_faces_set = set(selected_faces)

    if not selected_faces:
        return

    # Only process non-hotspot faces - hotspot faces handled by deferred system
    normal_faces = [f for f in selected_faces if not face_has_hotspot_material(f, me)]

    # Apply normal UVs (world-space projection from neighbors)
    projected_count = 0
    for face in normal_faces:
        # Find best neighboring face to use as UV source
        source_face = _get_best_neighbor_face(face, selected_faces_set)

        if source_face:
            # Transfer UVs from neighbor for seamless tiling
            set_uv_from_other_face(source_face, face, uv_layer, ppm, me, obj.matrix_world)
            projected_count += 1
        else:
            # No valid neighbor - fall back to world-space projection
            mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
            apply_uv_to_face(face, uv_layer, 1.0, 1.0, 0.0, 0.0, 0.0, mat, ppm, me)
            projected_count += 1

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
# Track the file browser watcher modal operator
_file_browser_watcher_running = False
# Track the previously selected file browser path (to avoid reapplying same image)
_last_file_browser_path = None

# The currently active image for texture operations.
# Updated by: file browser selection, user clicking a face
# Used by: Alt+Click apply, UI panel preview
_active_image = None


def get_active_image():
    """Get the currently active image for texture operations."""
    return _active_image


def set_active_image(image):
    """Set the currently active image for texture operations.

    Note: Does not call redraw_ui_panels here to avoid requiring a context parameter.
    Callers should call redraw_ui_panels(context) if an immediate UI update is needed.
    """
    global _active_image
    _active_image = image


def cache_single_face(face, uv_layer, ppm=None, me=None):
    """Cache vertex positions, UVs, and transform for a single face.

    Updates the face_data_cache entry for this face without clearing the cache.
    Used by apply_uv_to_face after modifying a face's UVs.
    """
    if face is None or not face.is_valid:
        return

    cache_entry = {
        'verts': [v.co.copy() for v in face.verts],
        'uvs': [loop[uv_layer].uv.copy() for loop in face.loops],
        'normal': face.normal.copy(),
        'center': face.calc_center_median().copy()
    }

    # Cache the derived transform (scale, rotation, offset) if we have the required data
    if ppm is not None and me is not None:
        transform = derive_transform_from_uvs(face, uv_layer, ppm, me)
        if transform:
            cache_entry['scale_u'] = transform['scale_u']
            cache_entry['scale_v'] = transform['scale_v']
            cache_entry['rotation'] = transform['rotation']
            cache_entry['offset_x'] = transform['offset_x']
            cache_entry['offset_y'] = transform['offset_y']
            
    face_data_cache[face.index] = cache_entry


def cache_face_data(context):
    """Cache vertex positions and UVs for all faces in the mesh.

    Clears and rebuilds the entire face_data_cache. Used when UV lock is toggled
    or when the mesh topology changes.
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

    uv_layer = bm.loops.layers.uv.verify()
    ppm = context.scene.level_design_props.pixels_per_meter

    face_data_cache.clear()

    for face in bm.faces:
        cache_single_face(face, uv_layer, ppm, me)

    last_face_count = len(bm.faces)
    last_vertex_count = len(bm.verts)


def update_ui_from_selection(context):
    """Update UI properties when selection changes"""
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
    uv_layer = bm.loops.layers.uv.verify()

    props = context.scene.level_design_props

    selected_faces = [f for f in bm.faces if f.select]

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
        return

    # Track modal operators to detect operation boundaries
    window = bpy.context.window
    current_modals = set(op.bl_idname for op in window.modal_operators) if window else set()

    # Detect if modal operation just ended (before updating tracking)
    modal_just_ended = bool(_tracked_modal_operators) and not bool(current_modals)

    # Update tracking
    in_modal_operation = bool(current_modals)
    _tracked_modal_operators = current_modals

    try:
        bm = bmesh.from_edit_mesh(me)
    except (ReferenceError, RuntimeError):
        # BMesh is invalid or being modified
        return

    # Validate BMesh state
    if not bm.is_valid:
        return

    # Ensure lookup tables are valid before accessing faces
    try:
        bm.faces.ensure_lookup_table()
    except (ReferenceError, RuntimeError):
        return

    uv_layer = bm.loops.layers.uv.verify()
    props = scene.level_design_props
    ppm = props.pixels_per_meter

    # Iterate using indices to be more resilient during topology changes
    face_indices = list(range(len(bm.faces)))
    for face_idx in face_indices:
        # Wrap each face access in try/except to handle race conditions
        # during modal operators like loop cut
        try:
            # Re-validate after potential changes
            if not bm.is_valid or face_idx >= len(bm.faces):
                return

            face = bm.faces[face_idx]

            if face.index not in face_data_cache:
                continue

            # Skip faces with hotspottable materials
            if face_has_hotspot_material(face, me):
                continue

            cached = face_data_cache[face.index]
            cached_verts = cached['verts']

            # HACK - premature optimisation? This cuts down a lot of faces but means we need to deal with Blender's own
            # UV correction on other faces which is difficult, but done successfully - see set_correct_uv_slide
            # Skip faces not relevant to current operation
            # A face is relevant if any of its vertices are selected
            if not any(v.select for v in face.verts):
                continue

            current_verts = [v.co.copy() for v in face.verts]
            if len(current_verts) != len(cached_verts):
                continue

            # Check if vertices actually moved - skip if geometry unchanged
            # In general my preference is to always update selected + adjacent faces just to keep code simple
            # This logic is here because face aligned project does a different type of projection than the one in this method
            # i.e. If you face align an object and then edit it, unmoved faces will confusingly get overwritten with the projection used here
            # consider revising?
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
                    cached_uvs = cached.get('uvs')
                    if cached_uvs and len(cached_uvs) == len(face.loops):
                        for loop, cached_uv in zip(face.loops, cached_uvs):
                            loop[uv_layer].uv = cached_uv.copy()
                        bmesh.update_edit_mesh(me)
                continue

            # Get cached transform (defaults if not cached)
            scale_u = cached.get('scale_u', 1.0)
            scale_v = cached.get('scale_v', 1.0)
            rotation = cached.get('rotation', 0.0)
            offset_x = cached.get('offset_x', 0.0)
            offset_y = cached.get('offset_y', 0.0)

            # Compensate for first edge rotation to keep texture fixed in world space
            # The local coordinate system is based on the first non-zero edge, so if the face
            # rotates, we need to counter-rotate the texture rotation
            old_edge = get_local_x_from_verts_3d(cached_verts)
            new_edge = get_local_x_from_verts_3d(current_verts)
            if old_edge is not None and new_edge is not None:

                # Compute signed angle between old and new edge directions
                # using the face normal as the rotation axis
                cross = old_edge.cross(new_edge)
                dot = old_edge.dot(new_edge)
                dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability
                edge_rotation = math.degrees(math.atan2(cross.dot(face.normal), dot))

                # Counter-rotate texture rotation to keep it fixed in world space
                rotation = rotation + edge_rotation

                # Compensate offset for translation of first vertex
                # The texture should stay fixed in world space, so when the first
                # vertex moves, the offset must change to keep the same world position
                translation = current_verts[0] - cached_verts[0]

                # Get the world-aligned projection axes (after rotation compensation)
                face_axes = get_face_local_axes(face)
                if face_axes:
                    face_local_x, face_local_y = face_axes
                    rot_rad = math.radians(rotation)
                    cos_rot = math.cos(rot_rad)
                    sin_rot = math.sin(rot_rad)
                    proj_x = face_local_x * cos_rot - face_local_y * sin_rot
                    proj_y = face_local_x * sin_rot + face_local_y * cos_rot

                    # Project translation onto projection axes (in meters)
                    move_x = translation.dot(proj_x)
                    move_y = translation.dot(proj_y)

                    # Convert to UV and add to offset to compensate
                    mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
                    tex_meters_u, tex_meters_v = get_texture_dimensions_from_material(mat, ppm)

                    offset_x = normalize_offset(offset_x + move_x / (scale_u * tex_meters_u))
                    offset_y = normalize_offset(offset_y + move_y / (scale_v * tex_meters_v))

            # Get material for this face
            mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None

            # Re-project UVs using apply_uv_to_face which properly handles rotation
            apply_uv_to_face(face, uv_layer, scale_u, scale_v, rotation, offset_x, offset_y,
                             mat, ppm, me)

            # Only update cache when NOT in a modal operation
            # During modal ops, we keep the baseline stable so returning to original
            # position gives original UVs
            if not in_modal_operation:
                cache_single_face(face, uv_layer, ppm, me)

        except (ReferenceError, RuntimeError, OSError):
            # BMesh data became invalid during iteration (e.g., during loop cut)
            return

    # Refresh cache when modal operation just ended
    # This ensures the baseline is updated with final geometry state
    if modal_just_ended:
        cache_face_data(bpy.context)


def apply_uv_lock(obj, scene):
    """UV lock ON: texture stays locked to geometry, UVs should not change.

    When UV lock is enabled, the texture moves/scales with the face like a sticker.
    We don't need to modify UVs - they naturally stay attached to vertices.
    Just update the cache to track the new vertex positions.
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

    # Ensure lookup tables are valid before accessing faces
    try:
        bm.faces.ensure_lookup_table()
    except (ReferenceError, RuntimeError):
        return

    # Check if any faces have moved
    has_moved = False
    face_indices = list(range(len(bm.faces)))
    for face_idx in face_indices:
        try:
            if not bm.is_valid or face_idx >= len(bm.faces):
                return

            face = bm.faces[face_idx]

            if face.index not in face_data_cache:
                continue

            cached = face_data_cache[face.index]
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

    current_selected = {f.index for f in bm.faces if f.select}
    current_active = bm.faces.active.index if bm.faces.active else -1

    if current_selected != _last_selected_face_indices or current_active != _last_active_face_index:
        _last_selected_face_indices = current_selected
        _last_active_face_index = current_active
        return True
    return False


def update_active_image_from_face(context):
    """Update the active image based on the active face's material."""
    try:
        obj = context.object
        if not obj or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
            return

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        active_face = bm.faces.active
        if not active_face:
            return

        # Get the material on this face
        mat_index = active_face.material_index
        mat = obj.data.materials[mat_index] if mat_index < len(obj.data.materials) else None

        if mat:
            image = get_image_from_material(mat)
            if image:
                set_active_image(image)
    except Exception:
        pass  # Silently fail to avoid disrupting user workflow


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

        # Load the image and set as active
        try:
            image = bpy.data.images.load(current_path, check_existing=True)
            set_active_image(image)
        except RuntimeError:
            return

        # Only apply to faces if in edit mode on a mesh with selected faces
        if not obj or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
            return

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        selected_faces = [f for f in bm.faces if f.select]

        if not selected_faces:
            return

        uv_layer = bm.loops.layers.uv.verify()
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

        # Ensure material slot exists
        if mat.name not in obj.data.materials:
            obj.data.materials.append(mat)

        mat_index = obj.data.materials.find(mat.name)

        # Assign material to all selected faces first
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
                selected_face_indices = {f.index for f in bm.faces if f.select}
                active_face_index = bm.faces.active.index if bm.faces.active else None

                seam_mode = props.hotspot_seam_mode
                allow_combined_faces = obj.anvil_allow_combined_faces
                size_weight = obj.anvil_hotspot_size_weight

                debug_log(f"[FileBrowser] Applying hotspots to {len(all_hotspot_faces)} faces (all hotspot faces)")
                apply_hotspots_to_mesh(
                    bm, obj.data, all_hotspot_faces, seam_mode, allow_combined_faces,
                    obj.matrix_world, ppm, size_weight
                )

                # Restore selection state
                bm.faces.ensure_lookup_table()
                for face in bm.faces:
                    face.select = face.index in selected_face_indices
                if active_face_index is not None and active_face_index < len(bm.faces):
                    bm.faces.active = bm.faces[active_face_index]

                # Cache all hotspot faces after application
                for face in all_hotspot_faces:
                    if face.is_valid:
                        cache_single_face(face, uv_layer, ppm, obj.data)
        elif props.auto_hotspot and not new_is_hotspottable and any_previous_was_hotspottable and any_connected_has_hotspot:
            # New texture is NOT hotspottable, but some selected faces previously had hotspot
            # AND some connected faces have hotspot textures - re-hotspot all to recalculate islands
            all_hotspot_faces = get_all_hotspot_faces(bm, obj.data)

            if all_hotspot_faces:
                # Save selection state
                selected_face_indices = {f.index for f in bm.faces if f.select}
                active_face_index = bm.faces.active.index if bm.faces.active else None

                seam_mode = props.hotspot_seam_mode
                allow_combined_faces = obj.anvil_allow_combined_faces
                size_weight = obj.anvil_hotspot_size_weight

                debug_log(f"[FileBrowser] Re-hotspotting {len(all_hotspot_faces)} faces (island structure changed)")
                apply_hotspots_to_mesh(
                    bm, obj.data, all_hotspot_faces, seam_mode, allow_combined_faces,
                    obj.matrix_world, ppm, size_weight
                )

                # Restore selection state
                bm.faces.ensure_lookup_table()
                for face in bm.faces:
                    face.select = face.index in selected_face_indices
                if active_face_index is not None and active_face_index < len(bm.faces):
                    bm.faces.active = bm.faces[active_face_index]

                # Cache all hotspot faces
                for face in all_hotspot_faces:
                    if face.is_valid:
                        cache_single_face(face, uv_layer, ppm, obj.data)

            # Apply regular UV projection to the selected faces (non-hotspot texture)
            _apply_regular_uv_projection(selected_faces, uv_layer, mat, ppm, obj.data)
        else:
            # Either auto_hotspot is off, or it's a non-hotspot texture without hotspot neighbors
            # Regular UV projection
            _apply_regular_uv_projection(selected_faces, uv_layer, mat, ppm, obj.data)

        bmesh.update_edit_mesh(obj.data)

        # Update UI to reflect the new UVs
        update_ui_from_selection(context)

    except Exception as e:
        print(f"Anvil Level Design: Error applying texture from file browser: {e}")


def _apply_regular_uv_projection(selected_faces, uv_layer, mat, ppm, me):
    """Apply regular UV projection to selected faces, preserving transform where possible.

    Args:
        selected_faces: List of BMesh faces to apply UVs to
        uv_layer: BMesh UV layer
        mat: Material to use for texture dimensions
        ppm: Pixels per meter setting
        me: Mesh data
    """
    # Capture old material info for transform preservation
    face_old_info = {}
    for f in selected_faces:
        f_mat_idx = f.material_index
        f_mat = me.materials[f_mat_idx] if f_mat_idx < len(me.materials) else None
        f_img = get_image_from_material(f_mat)
        face_old_info[f.index] = {
            'mat': f_mat,
            'has_image': f_img is not None,
            'tex_dims': get_texture_dimensions_from_material(f_mat, ppm),
        }

    for target_face in selected_faces:
        # Get current transform to preserve it
        current_transform = derive_transform_from_uvs(target_face, uv_layer, ppm, me)

        # Get old material info
        old_info = face_old_info[target_face.index]
        old_has_image = old_info['has_image']
        old_tex_dims = old_info['tex_dims']

        # Get new texture dimensions
        new_tex_dims = get_texture_dimensions_from_material(mat, ppm)

        # Reapply the preserved transform with the new texture
        if current_transform:
            # Reset scale to 1,1 if old material had no image (default material)
            # or if texture dimensions changed
            if not old_has_image or old_tex_dims != new_tex_dims:
                scale_u, scale_v = 1.0, 1.0
            else:
                scale_u = current_transform['scale_u']
                scale_v = current_transform['scale_v']

            apply_uv_to_face(
                target_face, uv_layer,
                scale_u, scale_v,
                current_transform['rotation'],
                current_transform['offset_x'], current_transform['offset_y'],
                mat, ppm, me
            )
            cache_single_face(target_face, uv_layer, ppm, me)
        else:
            # Use default values when transform can't be derived
            apply_uv_to_face(
                target_face, uv_layer,
                1.0, 1.0,  # scale
                0.0,       # rotation
                0.0, 0.0,  # offset
                mat, ppm, me
            )
            cache_single_face(target_face, uv_layer, ppm, me)


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
    """Set grid scale to 1 on all 3D views."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.overlay.grid_scale = 1.0


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
    global last_face_count, last_vertex_count
    # Invalidate face caches - geometry state has changed
    face_data_cache.clear()
    last_face_count = 0
    last_vertex_count = 0
    # Use a short timer to ensure the depsgraph update triggered by undo
    # has completed before we clear the flag
    bpy.app.timers.register(_clear_undo_flag, first_interval=0.05)


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
    global last_face_count, last_vertex_count
    # Invalidate face caches - geometry state has changed
    face_data_cache.clear()
    last_face_count = 0
    last_vertex_count = 0
    bpy.app.timers.register(_clear_undo_flag, first_interval=0.05)


@persistent
def on_save_pre(dummy):
    """Handler called before saving a .blend file.

    Makes all paths relative if the file has been saved before.
    For first-time saves, we wait until save_post to make paths relative.
    """
    global _was_first_save
    _was_first_save = not bpy.data.filepath

    if bpy.data.filepath:
        print("Anvil Level Design: Making all paths relative (pre save)")
        bpy.ops.file.make_paths_relative()


@persistent
def on_save_post(dummy):
    """Handler called after saving a .blend file.

    If this was the first save, makes paths relative and triggers a second save.
    """
    global _was_first_save

    if _was_first_save:
        _was_first_save = False
        print("Anvil Level Design: Making all paths relative (post save, for first save)")
        bpy.ops.file.make_paths_relative()
        print("Anvil Level Design: Triggering second save to apply relative paths")
        bpy.ops.wm.save_mainfile()


@persistent
def on_load_post(dummy):
    """Handler called after a .blend file is loaded."""
    global _file_browser_watcher_running, _last_file_browser_path, _file_loaded_into_edit_depsgraph
    # Reset watcher state on file load (modal was killed when file loaded)
    _file_browser_watcher_running = False
    # Reset last file browser path so first click applies regardless of previous session
    _last_file_browser_path = None
    # Allow first depsgraph update to set active image from selected face
    _file_loaded_into_edit_depsgraph = True
    # Use a timer to ensure all UI is ready
    bpy.app.timers.register(set_all_grid_scales_to_default, first_interval=0.1)
    # Restart the file browser watcher
    bpy.app.timers.register(start_file_browser_watcher, first_interval=0.2)
    # Disable correct_uv for slide operations
    bpy.app.timers.register(disable_correct_uv_slide, first_interval=0.1)
    # Clear the file loaded flag after 1 second (fallback if depsgraph doesn't fire)
    bpy.app.timers.register(_clear_file_loaded_flag, first_interval=1.0)


@persistent
def on_depsgraph_update(scene, depsgraph):
    """Consolidated depsgraph update handler"""
    global last_face_count, _file_loaded_into_edit_depsgraph, _force_auto_hotspot

    # Skip all depsgraph handling during undo operations
    if _undo_in_progress:
        return

    # Only process in the Level Design workspace
    if not is_level_design_workspace():
        return

    try:
        # Check for duplicate materials (from copy/paste operations)
        consolidate_duplicate_materials()

        context = bpy.context

        # Safety check - ensure properties are registered
        if not hasattr(scene, 'level_design_props'):
            return

        props = scene.level_design_props

        # Handle mesh updates (UV lock, world-scale UVs)
        for update in depsgraph.updates:
            if isinstance(update.id, bpy.types.Object):
                obj = update.id
                if obj.type == 'MESH' and obj.mode == 'EDIT':
                    me = obj.data

                    # Skip if mesh data is not available
                    if me is None or not me.is_editmode:
                        continue

                    # Track if this is an actual geometry update (not just tab switch, etc.)
                    is_geometry_update = update.is_updated_geometry

                    try:
                        bm = bmesh.from_edit_mesh(me)
                    except (ReferenceError, RuntimeError):
                        # BMesh is invalid or being modified by modal operator
                        continue

                    # Validate BMesh state
                    if not bm.is_valid:
                        continue

                    current_face_count = len(bm.faces)
                    current_vertex_count = len(bm.verts)

                    # Detect fresh edit session (entering edit mode or switching objects)
                    # This must happen before topology/selection checks to prevent
                    # updating active image when user didn't explicitly click a face
                    global _last_edit_object_name
                    is_fresh_start = (obj.name != _last_edit_object_name)
                    _last_edit_object_name = obj.name

                    # On file load, allow active image sync even if fresh start
                    # (to restore state from previous session)
                    allow_active_image_update = not is_fresh_start or _file_loaded_into_edit_depsgraph

                    # Check if topology changed (subdivision, extrusion, etc.)
                    if current_face_count != last_face_count or current_vertex_count != last_vertex_count:
                        # Topology changed - but skip projection if this is just switching objects
                        if not is_fresh_start:
                            _project_selected_faces_on_topology_change(context, bm)
                        cache_face_data(context)
                        update_ui_from_selection(context)
                        # Only update active image if allowed
                        if allow_active_image_update:
                            update_active_image_from_face(context)
                        _file_loaded_into_edit_depsgraph = False
                        # Apply auto-hotspotting if enabled (after cache is updated)
                        # Force because cache_face_data already cached the new faces,
                        # so the geometry-changed check would incorrectly skip them
                        if not is_fresh_start and props.auto_hotspot:
                            _force_auto_hotspot = True
                            _apply_auto_hotspots()
                        return

                    # Check if selection changed
                    if check_selection_changed(bm):
                        update_ui_from_selection(context)
                        # Only update active image if allowed
                        if allow_active_image_update:
                            update_active_image_from_face(context)

                    # Store data before any transform if cache is empty
                    if not face_data_cache and context.mode == 'EDIT_MESH':
                        cache_face_data(context)

                    if obj.anvil_uv_lock:
                        apply_uv_lock(obj, scene)
                    else:
                        apply_world_scale_uvs(obj, scene)

                    if not is_fresh_start and props.auto_hotspot:
                        _apply_auto_hotspots()

                    _file_loaded_into_edit_depsgraph = False
                    break
    except Exception as e:
        print(f"Anvil Level Design: Error in depsgraph handler: {e}")


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_force_apply_texture)

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


def unregister():
    global last_face_count, last_vertex_count, _last_selected_face_indices, _last_active_face_index, _last_edit_object_name, _last_material_count, _active_image, _file_browser_watcher_running, _last_file_browser_path, _file_loaded_into_edit_depsgraph, _was_first_save, _auto_hotspot_pending, _undo_in_progress

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
    _file_loaded_into_edit_depsgraph = False
    _was_first_save = False
