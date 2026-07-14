"""Texture application operations owned by the texture browser."""

import os

import bmesh
import bpy

from ..core.face_id import get_face_id_layer, save_face_selection, restore_face_selection
from ..core.hotspot_queries import (
    face_has_hotspot_material, any_connected_face_has_hotspot, get_all_hotspot_faces,
)
from ..core.library import is_library_object
from ..core.logging import debug_log
from ..core.materials import (
    get_image_from_material,
    find_material_with_image,
    create_material_with_image,
    get_unassigned_material,
    ensure_material_slot,
)
from ..core.uv_layers import get_render_active_uv_layer
from ..core.uv_projection import face_aligned_project, apply_uv_to_face, derive_transform_from_uvs
from ..handlers.active_image import set_previous_image
from ..handlers.face_cache import cache_single_face


def apply_texture_path_to_selection(filepath, obj, mode, scene):
    """Apply an image in Edit Mode using the caller's original mode as scope."""
    from ..hotspot_mapping.json_storage import is_texture_hotspottable
    from ..operators.hotspot_apply import apply_hotspots_to_mesh

    if not filepath:
        return None, False

    if not os.path.isfile(filepath):
        return None, False

    try:
        image = bpy.data.images.load(filepath, check_existing=True)
    except RuntimeError:
        return None, False

    set_previous_image(image)

    if not obj or obj.type != 'MESH':
        return image, False

    if is_library_object(obj):
        debug_log(f"[TextureBrowser] skipped library object: {obj.name}")
        return image, False

    in_edit_mode = (mode == 'EDIT_MESH')
    in_object_mode = (mode == 'OBJECT')

    if not in_edit_mode and not in_object_mode:
        return image, False

    if in_object_mode and not obj.select_get():
        return image, False

    bm = bmesh.from_edit_mesh(obj.data)

    uv_layer = get_render_active_uv_layer(bm, obj.data)
    if uv_layer is None:
        uv_layer = bm.loops.layers.uv.verify()
    get_face_id_layer(bm)

    bm.faces.ensure_lookup_table()

    if in_edit_mode:
        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            return image, False
    else:
        selected_faces = list(bm.faces)

    props = scene.level_design_props
    ppm = props.pixels_per_meter

    any_previous_was_hotspottable = any(
        face_has_hotspot_material(f, obj.data)
        for f in selected_faces
    )

    any_connected_has_hotspot = False
    for f in selected_faces:
        if any_connected_face_has_hotspot(f, obj.data):
            any_connected_has_hotspot = True
            break

    mat = find_material_with_image(image)
    if mat is None:
        mat = create_material_with_image(image)

    if in_edit_mode and len(obj.data.materials) == 0 and len(selected_faces) < len(bm.faces):
        obj.data.materials.append(get_unassigned_material())

    face_id_layer = get_face_id_layer(bm)
    face_old_info = {}
    for f in selected_faces:
        f_mat_idx = f.material_index
        f_mat = obj.data.materials[f_mat_idx] if f_mat_idx < len(obj.data.materials) else None
        f_img = get_image_from_material(f_mat)
        face_old_info[f[face_id_layer]] = {
            'has_image': f_img is not None,
            'transform': derive_transform_from_uvs(f, uv_layer, ppm, obj.data),
        }

    mat_index = ensure_material_slot(obj.data, mat)

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

            debug_log(f"[TextureBrowser] Applying hotspots to {len(all_hotspot_faces)} faces (all hotspot faces)")
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

            debug_log(f"[TextureBrowser] Re-hotspotting {len(all_hotspot_faces)} faces (island structure changed)")
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

    bmesh.update_edit_mesh(obj.data)

    return image, True


def _apply_regular_uv_projection(selected_faces, uv_layer, mat, ppm, me, face_old_info, bm):
    """Apply regular UV projection to selected faces, preserving transform where possible."""
    id_layer = get_face_id_layer(bm)
    for target_face in selected_faces:
        face_key = target_face[id_layer]
        old_info = face_old_info[face_key]
        old_has_image = old_info['has_image']
        current_transform = old_info['transform']

        if current_transform and old_has_image:
            apply_uv_to_face(
                target_face, uv_layer,
                current_transform['scale_u'], current_transform['scale_v'],
                current_transform['rotation'],
                current_transform['offset_x'], current_transform['offset_y'],
                mat, ppm, me
            )
            cache_single_face(target_face, bm, ppm, me)
        else:
            face_aligned_project(target_face, uv_layer, mat, ppm)
            cache_single_face(target_face, bm, ppm, me)
