"""File browser watcher and texture application from file browser."""

import os
import re

import bmesh
import bpy

from ..core.logging import debug_log
from ..core.face_id import get_face_id_layer, save_face_selection, restore_face_selection
from ..core.materials import (
    get_image_from_material, get_selected_image_path,
    find_material_with_image, create_material_with_image,
    get_texture_dimensions_from_material,
)
from ..core.uv_projection import face_aligned_project, apply_uv_to_face, derive_transform_from_uvs
from ..core.uv_layers import get_render_active_uv_layer
from ..core.hotspot_queries import (
    face_has_hotspot_material, any_connected_face_has_hotspot, get_all_hotspot_faces,
)
from ..core.workspace_check import is_level_design_workspace

from .face_cache import face_data_cache, cache_single_face, cache_face_data
from .active_image import (
    set_active_image, set_previous_image, set_active_image_just_set,
    redraw_ui_panels,
)
from .face_cache import update_ui_from_selection


# Track the file browser watcher modal operator
_file_browser_watcher_running = False
# Track the previously selected file browser path (to avoid reapplying same image)
_last_file_browser_path = None

# Cache for material deduplication
_last_material_count = 0

# Track if this is the first save (for making paths relative after save)
_was_first_save = False


def get_was_first_save():
    return _was_first_save


def set_was_first_save(value):
    global _was_first_save
    _was_first_save = value


def consolidate_duplicate_materials():
    """Find and merge duplicate IMG_ materials created by copy/paste.

    When objects are duplicated, Blender creates copies of materials with
    suffixes like .001, .002, etc. This function finds these duplicates
    and consolidates them to the base material name.
    """
    global _last_material_count

    current_count = len(bpy.data.materials)

    if current_count <= _last_material_count:
        _last_material_count = current_count
        return

    _last_material_count = current_count

    duplicate_pattern = re.compile(r'^(IMG_.+)\.(\d{3,})$')

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

    for base_name, duplicates in material_groups.items():
        if base_name not in bpy.data.materials:
            duplicates.sort(key=lambda x: x[0])
            duplicates[0][1].name = base_name

    replacements = {}
    for base_name, duplicates in material_groups.items():
        canonical = bpy.data.materials[base_name]
        for suffix_num, mat in duplicates:
            if mat != canonical:
                replacements[mat] = canonical

    if not replacements:
        return

    for obj in bpy.data.objects:
        if obj.type != 'MESH' or not obj.data:
            continue

        materials = obj.data.materials
        for i, mat in enumerate(materials):
            if mat in replacements:
                materials[i] = replacements[mat]

    for dup_mat in replacements.keys():
        if dup_mat.users == 0:
            bpy.data.materials.remove(dup_mat)

    _last_material_count = len(bpy.data.materials)


def apply_texture_from_file_browser():
    """Apply texture from current file browser selection to selected faces.

    Called when user clicks in the file browser. Loads the selected image,
    sets it as active, and applies it to any selected faces in edit mode.
    For hotspottable textures, applies hotspot UVs instead of regular projection.
    """
    from ..hotspot_mapping.json_storage import is_texture_hotspottable
    from ..operators.hotspot_apply import apply_hotspots_to_mesh

    try:
        context = bpy.context
        obj = context.object

        current_path = get_selected_image_path(context)

        if not current_path:
            return

        if not os.path.isfile(current_path):
            return

        try:
            image = bpy.data.images.load(current_path, check_existing=True)
        except RuntimeError:
            return

        set_previous_image(image)
        redraw_ui_panels(context)

        if not obj or obj.type != 'MESH':
            return

        in_edit_mode = (context.mode == 'EDIT_MESH')
        in_object_mode = (context.mode == 'OBJECT')

        if not in_edit_mode and not in_object_mode:
            return

        if in_object_mode and not obj.select_get():
            return

        if in_edit_mode:
            bm = bmesh.from_edit_mesh(obj.data)
        else:
            bm = bmesh.new()
            bm.from_mesh(obj.data)

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

        faces_with_previous_hotspot = [f for f in selected_faces if face_has_hotspot_material(f, obj.data)]
        any_previous_was_hotspottable = len(faces_with_previous_hotspot) > 0

        any_connected_has_hotspot = False
        for f in selected_faces:
            if any_connected_face_has_hotspot(f, obj.data):
                any_connected_has_hotspot = True
                break

        mat = find_material_with_image(image)
        if mat is None:
            mat = create_material_with_image(image)

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

        if mat.name not in obj.data.materials:
            obj.data.materials.append(mat)

        mat_index = obj.data.materials.find(mat.name)

        for target_face in selected_faces:
            target_face.material_index = mat_index

        new_is_hotspottable = is_texture_hotspottable(image.name)

        if obj.anvil_auto_hotspot and new_is_hotspottable:
            all_hotspot_faces = get_all_hotspot_faces(bm, obj.data)

            if all_hotspot_faces:
                id_layer = get_face_id_layer(bm)
                selected_ids, active_id = save_face_selection(bm, id_layer)

                allow_combined_faces = obj.anvil_allow_combined_faces
                size_weight = obj.anvil_hotspot_size_weight
                seam_angle = obj.anvil_hotspot_seam_angle

                debug_log(f"[FileBrowser] Applying hotspots to {len(all_hotspot_faces)} faces (all hotspot faces)")
                apply_hotspots_to_mesh(
                    bm, obj.data, all_hotspot_faces, allow_combined_faces,
                    obj.matrix_world, ppm, size_weight, seam_angle
                )

                restore_face_selection(bm, id_layer, selected_ids, active_id)

                for face in all_hotspot_faces:
                    if face.is_valid:
                        cache_single_face(face, bm, ppm, obj.data)
        elif obj.anvil_auto_hotspot and not new_is_hotspottable and any_previous_was_hotspottable and any_connected_has_hotspot:
            all_hotspot_faces = get_all_hotspot_faces(bm, obj.data)

            if all_hotspot_faces:
                id_layer = get_face_id_layer(bm)
                selected_ids, active_id = save_face_selection(bm, id_layer)

                allow_combined_faces = obj.anvil_allow_combined_faces
                size_weight = obj.anvil_hotspot_size_weight
                seam_angle = obj.anvil_hotspot_seam_angle

                debug_log(f"[FileBrowser] Re-hotspotting {len(all_hotspot_faces)} faces (island structure changed)")
                apply_hotspots_to_mesh(
                    bm, obj.data, all_hotspot_faces, allow_combined_faces,
                    obj.matrix_world, ppm, size_weight, seam_angle
                )

                restore_face_selection(bm, id_layer, selected_ids, active_id)

                for face in all_hotspot_faces:
                    if face.is_valid:
                        cache_single_face(face, bm, ppm, obj.data)

            _apply_regular_uv_projection(selected_faces, uv_layer, mat, ppm, obj.data, face_old_info, bm)
        else:
            _apply_regular_uv_projection(selected_faces, uv_layer, mat, ppm, obj.data, face_old_info, bm)

        if in_edit_mode:
            bmesh.update_edit_mesh(obj.data)
        else:
            bm.to_mesh(obj.data)
            bm.free()
            obj.data.update()

        update_ui_from_selection(context)
        set_active_image(image)
        set_active_image_just_set(True)
        redraw_ui_panels(context)

        bpy.ops.ed.undo_push(message="Apply Texture from File Browser")

    except Exception as e:
        print(f"Anvil Level Design: Error applying texture from file browser: {e}", flush=True)


def _apply_regular_uv_projection(selected_faces, uv_layer, mat, ppm, me, face_old_info, bm=None):
    """Apply regular UV projection to selected faces, preserving transform where possible."""
    id_layer = get_face_id_layer(bm) if bm is not None else None
    for target_face in selected_faces:
        current_transform = derive_transform_from_uvs(target_face, uv_layer, ppm, me)

        face_key = target_face[id_layer] if id_layer is not None else target_face.index
        old_info = face_old_info[face_key]
        old_has_image = old_info['has_image']
        old_tex_dims = old_info['tex_dims']

        new_tex_dims = get_texture_dimensions_from_material(mat, ppm)

        if current_transform and old_has_image:
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
                mat, ppm, me
            )
            if bm is not None:
                cache_single_face(target_face, bm, ppm, me)
        else:
            face_aligned_project(target_face, uv_layer, mat, ppm)
            if bm is not None:
                cache_single_face(target_face, bm, ppm, me)


def _file_browser_watcher_timer():
    """Timer-based file browser watcher. Polls for selection changes."""
    global _file_browser_watcher_running, _last_file_browser_path

    if not _file_browser_watcher_running:
        return None

    try:
        context = bpy.context

        has_file_browser = False
        if context.window and context.window.screen:
            for area in context.window.screen.areas:
                if area.type == 'FILE_BROWSER':
                    has_file_browser = True
                    break

        if not has_file_browser:
            return 0.5

        current_path = get_selected_image_path(context)

        if current_path and current_path != _last_file_browser_path:
            apply_texture_from_file_browser()
            _last_file_browser_path = current_path

    except Exception:
        pass

    return 0.1


def start_file_browser_watcher():
    """Start the file browser watcher timer."""
    global _file_browser_watcher_running, _last_file_browser_path

    if _file_browser_watcher_running:
        return

    _file_browser_watcher_running = True
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


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_force_apply_texture)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='File Browser', space_type='FILE_BROWSER')
        kmi = km.keymap_items.new(
            'leveldesign.force_apply_texture',
            'LEFTMOUSE', 'CLICK',
            alt=True
        )
        _addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    bpy.utils.unregister_class(LEVELDESIGN_OT_force_apply_texture)


def reset():
    """Reset file browser state."""
    global _file_browser_watcher_running, _last_file_browser_path, _last_material_count, _was_first_save
    _file_browser_watcher_running = False
    _last_file_browser_path = None
    _last_material_count = 0
    _was_first_save = False
