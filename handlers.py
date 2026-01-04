import os
import re

import bpy
import bmesh
import math
from mathutils import Vector
from bpy.app.handlers import persistent

from .utils import (
    get_image_from_material, derive_transform_from_uvs,
    get_selected_image_path, find_material_with_image, create_material_with_image,
    get_texture_dimensions_from_material
)
from .properties import set_updating_from_selection, sync_scale_tracking, apply_uv_to_face


# Cache for face data (UV lock functionality)
face_data_cache = {}
last_face_count = 0

# Cache for material deduplication
_last_material_count = 0

# Cache for detecting selection changes
_last_selected_face_indices = set()
_last_active_face_index = -1
# Track which object we're editing to detect fresh edit sessions
_last_edit_object_name = None
# Track modal operators for UV world-scale baseline
_tracked_modal_operators = set()
# Track the file browser watcher modal operator
_file_browser_watcher_running = False

# The currently active image for texture operations.
# Updated by: file browser selection, user clicking a face
# Used by: Alt+Click apply, UI panel preview
_active_image = None


def get_active_image():
    """Get the currently active image for texture operations."""
    return _active_image


def set_active_image(image):
    """Set the currently active image for texture operations."""
    global _active_image
    _active_image = image


def cache_single_face(face, uv_layer, ppm=None, me=None):
    """Cache vertex positions, UVs, and transform for a single face.

    Updates the face_data_cache entry for this face without clearing the cache.
    Used by apply_uv_to_face after modifying a face's UVs.
    """
    cache_entry = {
        'verts': [v.co.copy() for v in face.verts],
        'uvs': [loop[uv_layer].uv.copy() for loop in face.loops],
        'normal': face.normal.copy(),
        'center': face.calc_center_median().copy()
    }

    # Cache the derived transform (rotation, offset) if we have the required data
    if ppm is not None and me is not None:
        transform = derive_transform_from_uvs(face, uv_layer, ppm, me)
        if transform:
            cache_entry['rotation'] = transform['rotation']
            cache_entry['offset_x'] = transform['offset_x']
            cache_entry['offset_y'] = transform['offset_y']

    face_data_cache[face.index] = cache_entry


def cache_face_data(context):
    """Cache vertex positions and UVs for all faces in the mesh.

    Clears and rebuilds the entire face_data_cache. Used when UV lock is toggled
    or when the mesh topology changes.
    """
    global last_face_count

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

            # Derive all transform values from current UVs
            ppm = props.pixels_per_meter
            transform = derive_transform_from_uvs(face, uv_layer, ppm, me)

            if transform:
                props.texture_scale_u = transform['scale_u']
                props.texture_scale_v = transform['scale_v']
                props.texture_rotation = transform['rotation']
                props.texture_offset_x = transform['offset_x']
                props.texture_offset_y = transform['offset_y']
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

    # Detect modal operation just ended
    if _tracked_modal_operators and not current_modals:
        # Operation finished - refresh cache with final geometry state
        _tracked_modal_operators = current_modals
        cache_face_data(bpy.context)
        return

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
    scale_u = props.texture_scale_u
    scale_v = props.texture_scale_v
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

            cached = face_data_cache[face.index]

            # Check if face has been transformed
            current_verts = [v.co.copy() for v in face.verts]
            if len(current_verts) != len(cached['verts']):
                continue

            # Calculate if vertices have moved
            has_moved = False
            for i, (current, original) in enumerate(zip(current_verts, cached['verts'])):
                if (current - original).length > 0.0001:
                    has_moved = True
                    break

            if not has_moved:
                continue

            # Get cached rotation (default to 0 if not cached)
            rotation = cached.get('rotation', 0.0)
            offset_x = cached.get('offset_x', 0.0)
            offset_y = cached.get('offset_y', 0.0)

            # Compensate for first edge rotation to keep texture fixed in world space
            # The local coordinate system is based on the first edge, so if the face
            # rotates, we need to counter-rotate the texture rotation
            if len(cached['verts']) >= 2 and len(current_verts) >= 2:
                old_edge = (cached['verts'][1] - cached['verts'][0]).normalized()
                new_edge = (current_verts[1] - current_verts[0]).normalized()

                # Compute signed angle between old and new edge directions
                # using the face normal as the rotation axis
                cross = old_edge.cross(new_edge)
                dot = old_edge.dot(new_edge)
                dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability
                edge_rotation = math.degrees(math.atan2(cross.dot(face.normal), dot))

                # Subtract edge rotation to keep texture fixed in world space
                rotation = rotation + edge_rotation

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
    """
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

        # Get or create material
        mat = find_material_with_image(image)
        if mat is None:
            mat = create_material_with_image(image)

        # Ensure material slot exists
        if mat.name not in obj.data.materials:
            obj.data.materials.append(mat)

        mat_index = obj.data.materials.find(mat.name)
        ppm = context.scene.level_design_props.pixels_per_meter

        for target_face in selected_faces:
            # Get current transform to preserve it
            current_transform = derive_transform_from_uvs(target_face, uv_layer, ppm, obj.data)

            # Get old texture dimensions before changing material
            old_mat = obj.data.materials[target_face.material_index] if target_face.material_index < len(obj.data.materials) else None
            old_tex_dims = get_texture_dimensions_from_material(old_mat, ppm)

            target_face.material_index = mat_index

            # Get new texture dimensions
            new_tex_dims = get_texture_dimensions_from_material(mat, ppm)

            # Reapply the preserved transform with the new texture
            if current_transform:
                # Reset scale to 1,1 if texture dimensions changed
                if old_tex_dims != new_tex_dims:
                    scale_u, scale_v = 1.0, 1.0
                else:
                    scale_u = current_transform['scale_u']
                    scale_v = current_transform['scale_v']

                apply_uv_to_face(
                    target_face, uv_layer,
                    scale_u, scale_v,
                    current_transform['rotation'],
                    current_transform['offset_x'], current_transform['offset_y'],
                    mat, ppm, obj.data
                )
                cache_single_face(target_face, uv_layer, ppm, obj.data)
            else:
                # Use default values when transform can't be derived
                apply_uv_to_face(
                    target_face, uv_layer,
                    1.0, 1.0,  # scale
                    0.0,       # rotation
                    0.0, 0.0,  # offset
                    mat, ppm, obj.data
                )
                cache_single_face(target_face, uv_layer, ppm, obj.data)

    except Exception as e:
        print(f"Level Design Tools: Error applying texture from file browser: {e}")


class LEVELDESIGN_OT_file_browser_watcher(bpy.types.Operator):
    """Watch for clicks in the file browser and apply textures"""
    bl_idname = "leveldesign.file_browser_watcher"
    bl_label = "File Browser Watcher"
    bl_options = {'INTERNAL'}

    def modal(self, context, event):
        global _file_browser_watcher_running

        # Check if we should stop (addon being unregistered)
        if not _file_browser_watcher_running:
            return {'CANCELLED'}

        # Detect left-click release in file browser area
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            # Check if mouse is over a file browser area
            for area in context.screen.areas:
                if area.type == 'FILE_BROWSER':
                    if (area.x <= event.mouse_x <= area.x + area.width and
                            area.y <= event.mouse_y <= area.y + area.height):
                        # Small delay to let file browser update its selection
                        # bpy.app.timers.register(
                        #    apply_texture_from_file_browser,
                        #    first_interval=0.05
                        # )
                        apply_texture_from_file_browser()
                        break

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        global _file_browser_watcher_running

        # Don't start if already running
        if _file_browser_watcher_running:
            return {'CANCELLED'}

        _file_browser_watcher_running = True
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


def start_file_browser_watcher():
    """Start the file browser watcher modal."""
    global _file_browser_watcher_running

    if _file_browser_watcher_running:
        return

    try:
        bpy.ops.leveldesign.file_browser_watcher('INVOKE_DEFAULT')
    except RuntimeError:
        # Context may not be ready yet
        pass


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


@persistent
def on_load_post(dummy):
    """Handler called after a .blend file is loaded."""
    global _file_browser_watcher_running
    # Reset watcher state on file load (modal was killed when file loaded)
    _file_browser_watcher_running = False
    # Use a timer to ensure all UI is ready
    bpy.app.timers.register(set_all_grid_scales_to_default, first_interval=0.1)
    # Restart the file browser watcher
    bpy.app.timers.register(start_file_browser_watcher, first_interval=0.2)


@persistent
def on_depsgraph_update(scene, depsgraph):
    """Consolidated depsgraph update handler"""
    global last_face_count

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

                    try:
                        bm = bmesh.from_edit_mesh(me)
                    except (ReferenceError, RuntimeError):
                        # BMesh is invalid or being modified by modal operator
                        continue

                    # Validate BMesh state
                    if not bm.is_valid:
                        continue

                    current_face_count = len(bm.faces)

                    # Detect fresh edit session (entering edit mode or switching objects)
                    # This must happen before topology/selection checks to prevent
                    # updating active image when user didn't explicitly click a face
                    global _last_edit_object_name
                    is_fresh_start = (obj.name != _last_edit_object_name)
                    _last_edit_object_name = obj.name

                    # Check if topology changed (subdivision, extrusion, etc.)
                    if current_face_count != last_face_count:
                        # Topology changed - refresh cache
                        cache_face_data(context)
                        update_ui_from_selection(context)
                        # Only update active image if not a fresh start
                        if not is_fresh_start:
                            update_active_image_from_face(context)
                        return

                    # Check if selection changed
                    if check_selection_changed(bm):
                        update_ui_from_selection(context)
                        # Only update active image if not a fresh start
                        if not is_fresh_start:
                            update_active_image_from_face(context)

                    # Store data before any transform if cache is empty
                    if not face_data_cache and context.mode == 'EDIT_MESH':
                        cache_face_data(context)

                    if props.uv_lock:
                        apply_uv_lock(obj, scene)
                    else:
                        apply_world_scale_uvs(obj, scene)

                    break
    except Exception as e:
        print(f"Level Design Tools: Error in depsgraph handler: {e}")


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_file_browser_watcher)
    if on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)
    # Also set grid on addon enable
    bpy.app.timers.register(set_all_grid_scales_to_default, first_interval=0.1)
    # Start file browser watcher
    bpy.app.timers.register(start_file_browser_watcher, first_interval=0.2)


def unregister():
    global last_face_count, _last_selected_face_indices, _last_active_face_index, _last_edit_object_name, _last_material_count, _active_image, _file_browser_watcher_running

    # Stop the file browser watcher modal
    _file_browser_watcher_running = False

    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)

    bpy.utils.unregister_class(LEVELDESIGN_OT_file_browser_watcher)

    face_data_cache.clear()
    last_face_count = 0
    _last_material_count = 0
    _last_selected_face_indices = set()
    _last_active_face_index = -1
    _last_edit_object_name = None
    _active_image = None
