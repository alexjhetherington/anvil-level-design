import bpy
import bmesh
import math
from bpy.types import Operator
from bpy_extras import view3d_utils

from ..utils import is_level_design_workspace, debug_log
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..utils import (
    find_material_with_image,
    create_material_with_image,
    get_texture_dimensions_from_material,
    get_face_local_axes,
    derive_transform_from_uvs,
    normalize_offset,
    get_image_from_material,
    face_has_hotspot_material,
    any_connected_face_has_hotspot,
    get_all_hotspot_faces,
    get_face_id_layer,
    save_face_selection,
    restore_face_selection,
)
from ..properties import apply_uv_to_face
from ..handlers import (
    cache_single_face, get_active_image, set_active_image, redraw_ui_panels,
    update_ui_from_selection, update_active_image_from_face,
)
from .backface_select.paint_base import ModalPaintBase
from .backface_select.raycast import (
    raycast_bvh_skip_backfaces,
    raycast_bvh_skip_backfaces_polys,
)


def set_uv_from_source_params(target_face, uv_layer, ppm, me, obj_matrix,
                              scale_u, scale_v, source_rotation,
                              source_normal, source_u, source_v,
                              ref_point_co, ref_point_uv,
                              bm=None):
    """Apply UV to target_face using explicit source transform parameters.

    Instead of reading from a live source BMFace, this accepts the source's
    derived transform, axes, and a reference point (position + UV) to compute
    the correct rotation and offset for the target face.

    Args:
        target_face: BMesh face to apply UVs to
        uv_layer: BMesh UV layer
        ppm: Pixels per meter setting
        me: Mesh data for target face
        obj_matrix: Target object world matrix (for parallel plane reference)
        scale_u, scale_v: Source texture scale factors
        source_rotation: Source rotation in degrees
        source_normal: Source face normal (Vector, already in target space)
        source_u, source_v: Source face local axes (Vectors, already in target space)
        ref_point_co: 3D position of a reference point (already in target space)
        ref_point_uv: UV coordinates of that reference point (Vector2D)
        bm: BMesh instance (optional, for cache_single_face per-layer caching)
    """
    target_normal = target_face.normal.normalized()
    target_axes = get_face_local_axes(target_face)
    if not target_axes:
        return False

    target_u, target_v = target_axes

    # Compute reference axis for rotation calculation
    # For intersecting planes: use the intersection line
    # For parallel planes: use "most upward" direction on the plane
    intersection = source_normal.cross(target_normal)
    if intersection.length < 0.0001:
        # Parallel planes - compute "most upward" vector on plane as reference
        local_up = (obj_matrix.inverted().to_3x3() @ Vector((0, 0, 1))).normalized()
        reference = local_up - source_normal.dot(local_up) * source_normal
        if reference.length < 0.0001:
            # Plane is horizontal, use texture V direction as reference
            rot_rad = math.radians(source_rotation)
            reference = source_u * math.sin(rot_rad) + source_v * math.cos(rot_rad)
        reference = reference.normalized()
    else:
        reference = intersection.normalized()

    # Compute angle from reference to each face's U direction
    angle_ref_to_source_u = math.atan2(
        reference.cross(source_u).dot(source_normal),
        reference.dot(source_u)
    )
    angle_ref_to_target_u = math.atan2(
        reference.cross(target_u).dot(target_normal),
        reference.dot(target_u)
    )

    # Source's texture U direction relative to reference
    texture_angle_from_ref = angle_ref_to_source_u - math.radians(source_rotation)

    # For anti-parallel faces (opposite normals), mirror the rotation
    if source_normal.dot(target_normal) < -0.9999:
        texture_angle_from_ref = -texture_angle_from_ref + math.radians(180)

    # Target rotation needed to achieve the same texture angle from reference
    target_rotation = math.degrees(angle_ref_to_target_u - texture_angle_from_ref)

    # Check for opposite winding (u→v rotation direction differs between faces)
    source_handedness = source_u.cross(source_v).dot(source_normal)
    target_handedness = target_u.cross(target_v).dot(target_normal)
    if source_handedness * target_handedness < 0:
        target_rotation += 180

    # Compute offset: project the reference point onto the target face and
    # figure out what offset is needed to match its known UV.
    target_plane_point = list(target_face.loops)[0].vert.co
    dist_to_plane = (ref_point_co - target_plane_point).dot(target_normal)
    projected_point = ref_point_co - dist_to_plane * target_normal

    target_mat = me.materials[target_face.material_index] if target_face.material_index < len(me.materials) else None
    tex_meters_u, tex_meters_v = get_texture_dimensions_from_material(target_mat, ppm)

    rot_rad = math.radians(target_rotation)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    proj_x = target_u * cos_r - target_v * sin_r
    proj_y = target_u * sin_r + target_v * cos_r

    delta = projected_point - target_plane_point
    x = delta.dot(proj_x)
    y = delta.dot(proj_y)

    u = x / (scale_u * tex_meters_u)
    v = y / (scale_v * tex_meters_v)

    target_offset_x = normalize_offset(ref_point_uv.x - u)
    target_offset_y = normalize_offset(ref_point_uv.y - v)

    # Apply to target face
    debug_log(f"[ApplyImage] set_uv_from_source_params: face {target_face.index} | "
              f"scale=({scale_u:.4f}, {scale_v:.4f}) rotation={target_rotation:.2f} offset=({target_offset_x:.4f}, {target_offset_y:.4f})")
    apply_uv_to_face(target_face, uv_layer, scale_u, scale_v, target_rotation,
                     target_offset_x, target_offset_y, target_mat, ppm, me)
    if bm is not None:
        cache_single_face(target_face, bm, ppm, me)

    return True


def set_uv_from_other_face(source_face, target_face, uv_layer, ppm, me, obj_matrix, bm=None,
                            source_uv_layer=None, source_me=None, source_to_target=None):
    """Copy UV settings from source face to target face with proper rotation/offset handling.

    Extracts transform parameters from the source face and delegates to
    set_uv_from_source_params().

    Args:
        source_face: BMesh face to copy UV settings from
        target_face: BMesh face to apply UV settings to
        uv_layer: BMesh UV layer (target face's layer)
        ppm: Pixels per meter setting
        me: Mesh data for target face
        obj_matrix: Target object world matrix (for parallel plane reference calculation)
        bm: BMesh instance (optional, for cache_single_face per-layer caching)
        source_uv_layer: UV layer for source face (cross-object only; defaults to uv_layer)
        source_me: Mesh data for source face (cross-object only; defaults to me)
        source_to_target: 4x4 matrix from source object space to target object space
    """
    # ---- Cross-object: resolve source data references ----
    src_uv = source_uv_layer if source_uv_layer is not None else uv_layer
    src_me = source_me if source_me is not None else me

    source_transform = derive_transform_from_uvs(source_face, src_uv, ppm, src_me)
    if not source_transform:
        return False

    scale_u = source_transform['scale_u']
    scale_v = source_transform['scale_v']

    # Zero scale means source has collapsed/zero-area UVs — can't derive settings
    if abs(scale_u) < 1e-8 or abs(scale_v) < 1e-8:
        return False
    source_rotation = source_transform['rotation']

    source_normal = source_face.normal.normalized()
    source_axes = get_face_local_axes(source_face)
    if not source_axes:
        return False
    source_u, source_v = source_axes

    # Choose the best reference point for offset calculation:
    # prefer a shared vertex (exact match), fall back to projecting source's first vert
    source_verts = set(source_face.verts)
    target_verts = set(target_face.verts)
    shared_verts = source_verts & target_verts

    if len(shared_verts) >= 1:
        shared_vert = list(shared_verts)[0]
        ref_point_co = shared_vert.co.copy()
        # Find UV of shared vert in source
        ref_point_uv = None
        for loop in source_face.loops:
            if loop.vert == shared_vert:
                ref_point_uv = loop[src_uv].uv.copy()
                break
        if ref_point_uv is None:
            ref_point_uv = list(source_face.loops)[0][src_uv].uv.copy()
            ref_point_co = list(source_face.loops)[0].vert.co.copy()
    else:
        source_loop_0 = list(source_face.loops)[0]
        ref_point_co = source_loop_0.vert.co.copy()
        ref_point_uv = source_loop_0[src_uv].uv.copy()

    # ---- Cross-object: transform source geometry to target's local space ----
    if source_to_target is not None:
        rot = source_to_target.to_3x3()
        source_normal = (rot @ source_normal).normalized()
        source_u = (rot @ source_u).normalized()
        source_v = (rot @ source_v).normalized()
        ref_point_co = source_to_target @ ref_point_co

    return set_uv_from_source_params(
        target_face, uv_layer, ppm, me, obj_matrix,
        scale_u, scale_v, source_rotation,
        source_normal, source_u, source_v,
        ref_point_co, ref_point_uv,
        bm=bm,
    )


class apply_image_to_face(ModalPaintBase, Operator):
    """Apply selected File Browser image to hovered face (drag to paint)"""
    bl_idname = "leveldesign.apply_image_to_face"
    bl_label = "Apply Image to Face"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        obj = context.object
        if not obj or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
            debug_log("[ApplyImage] PASS_THROUGH: no mesh object or not in EDIT_MESH mode")
            return {'PASS_THROUGH'}

        # Only work in face select mode
        if not context.tool_settings.mesh_select_mode[2]:
            debug_log("[ApplyImage] PASS_THROUGH: not in face select mode")
            return {'PASS_THROUGH'}

        image = get_active_image()
        if not image:
            debug_log("[ApplyImage] PASS_THROUGH: no active image in file browser")
            return {'PASS_THROUGH'}

        # Require exactly 1 face selected
        bm_check = bmesh.from_edit_mesh(obj.data)
        bm_check.faces.ensure_lookup_table()
        selected_count = sum(1 for f in bm_check.faces if f.select)
        if selected_count != 1:
            debug_log(f"[ApplyImage] PASS_THROUGH: need exactly 1 face selected, got {selected_count}")
            return {'PASS_THROUGH'}

        source_face = bm_check.faces.active
        if source_face is None or not source_face.select:
            debug_log(f"[ApplyImage] PASS_THROUGH: active face is None ({source_face is None}) or not selected")
            return {'PASS_THROUGH'}

        # Store state for paint session
        self._source_face_index = source_face.index
        self._image = image
        self._obj_matrix = obj.matrix_world.copy()

        props = context.scene.level_design_props
        self._ppm = props.pixels_per_meter
        self._auto_hotspot = props.auto_hotspot
        self._hotspot_seam_mode = props.hotspot_seam_mode
        self._allow_combined_faces = obj.anvil_allow_combined_faces
        self._size_weight = obj.anvil_hotspot_size_weight
        self._painted_face_indices = set()
        self._faces_previously_hotspottable = set()

        # Get or create material
        mat = find_material_with_image(image)
        if mat is None:
            mat = create_material_with_image(image)
        self._mat = mat
        if mat.name not in obj.data.materials:
            obj.data.materials.append(mat)
        self._mat_index = obj.data.materials.find(mat.name)

        # ---- Cross-object: pre-build BVH trees for other visible meshes ----
        self._other_objects_info = []
        self._other_bmeshes = {}
        self._paint_visited_other = set()
        for other_obj in context.view_layer.objects:
            if other_obj == obj or other_obj.type != 'MESH' or not other_obj.visible_get():
                continue
            other_me = other_obj.data
            if other_me is None or len(other_me.polygons) == 0:
                continue
            bvh = BVHTree.FromPolygons(
                [v.co for v in other_me.vertices],
                [p.vertices for p in other_me.polygons]
            )
            self._other_objects_info.append({
                'obj': other_obj,
                'bvh': bvh,
                'polygons': other_me.polygons,
                'materials': other_me.materials,
            })

        debug_log(f"[ApplyImage] invoke OK: source_face={self._source_face_index}, image={image.name}, mat={mat.name}")
        return self._invoke_paint(context, event)

    def modal(self, context, event):
        return self._modal_paint(context, event)

    def paint_begin(self, context, event):
        return True

    def paint_cancel(self, context):
        self._discard_other_bmeshes()

    def paint_sample(self, context, mouse_2d, region, rv3d):
        obj = self._paint_obj
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()

        origin_local, dir_local = self._paint_ray_local(region, rv3d, mouse_2d)

        location, normal, face_index, distance = raycast_bvh_skip_backfaces(
            self._paint_bvh, origin_local, dir_local,
            bm, me.materials, max_iterations=64
        )

        # ---- Cross-object: check other objects for a closer hit ----
        hit_other_obj, hit_other_face_index = self._raycast_other_objects(
            mouse_2d, region, rv3d, face_index, distance
        )
        if hit_other_obj is not None:
            self._apply_to_other_object(hit_other_obj, hit_other_face_index, bm, me)
            return

        # ---- Apply to active object ----
        if face_index is None:
            debug_log(f"[ApplyImage] raycast miss at mouse ({mouse_2d.x:.0f}, {mouse_2d.y:.0f}) - no face hit")
            return

        debug_log(f"[ApplyImage] raycast hit face {face_index} at distance {distance:.4f}, mouse ({mouse_2d.x:.0f}, {mouse_2d.y:.0f})")
        if face_index in self._paint_visited:
            return
        self._paint_visited.add(face_index)

        target_face = bm.faces[face_index]
        source_face = bm.faces[self._source_face_index]

        from ..utils import get_render_active_uv_layer
        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

        # Track if face previously had hotspot material
        if face_has_hotspot_material(target_face, me):
            self._faces_previously_hotspottable.add(face_index)

        # Assign material
        target_face.material_index = self._mat_index

        # Apply UV from source face
        set_uv_from_other_face(
            source_face, target_face, uv_layer,
            self._ppm, me, self._obj_matrix
        )

        self._painted_face_indices.add(face_index)

    # ---- Cross-object helper methods ----

    def _raycast_other_objects(self, mouse_2d, region, rv3d, active_face_index, active_distance):
        """Raycast other visible mesh objects. Returns (obj, face_index) if a closer hit found."""
        if not self._other_objects_info:
            return None, None

        best_distance = active_distance if active_face_index is not None else float('inf')
        best_obj = None
        best_face_index = None

        coord = (mouse_2d.x, mouse_2d.y)
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        for info in self._other_objects_info:
            other_obj = info['obj']
            matrix_inv = other_obj.matrix_world.inverted()
            origin_other = matrix_inv @ ray_origin
            dir_other = (matrix_inv.to_3x3() @ view_vector).normalized()

            loc, norm, fidx, dist = raycast_bvh_skip_backfaces_polys(
                info['bvh'], origin_other, dir_other,
                info['polygons'], info['materials'], max_iterations=64
            )

            if fidx is not None and dist < best_distance:
                best_distance = dist
                best_obj = other_obj
                best_face_index = fidx

        return best_obj, best_face_index

    def _apply_to_other_object(self, other_obj, face_index, source_bm, source_me):
        """Apply texture from source face to a face on another object."""
        visit_key = (id(other_obj), face_index)
        if visit_key in self._paint_visited_other:
            return
        self._paint_visited_other.add(visit_key)

        other_me = other_obj.data
        obj_id = id(other_obj)

        # Get or create bmesh for this object
        if obj_id not in self._other_bmeshes:
            other_bm = bmesh.new()
            other_bm.from_mesh(other_me)
            self._other_bmeshes[obj_id] = {
                'bm': other_bm,
                'obj': other_obj,
            }
        other_data = self._other_bmeshes[obj_id]
        other_bm = other_data['bm']
        other_bm.faces.ensure_lookup_table()

        # Ensure material exists on target object
        if self._mat.name not in other_me.materials:
            other_me.materials.append(self._mat)
        other_mat_index = other_me.materials.find(self._mat.name)

        target_face = other_bm.faces[face_index]
        source_face = source_bm.faces[self._source_face_index]

        target_face.material_index = other_mat_index

        # Get UV layers for source and target
        from ..utils import get_render_active_uv_layer
        source_uv = get_render_active_uv_layer(source_bm, source_me)
        if source_uv is None:
            source_uv = source_bm.loops.layers.uv.verify()
        target_uv = other_bm.loops.layers.uv.verify()

        source_to_target = other_obj.matrix_world.inverted() @ self._paint_obj.matrix_world

        set_uv_from_other_face(
            source_face, target_face, target_uv,
            self._ppm, other_me, other_obj.matrix_world,
            source_uv_layer=source_uv, source_me=source_me,
            source_to_target=source_to_target,
        )

        debug_log(f"[ApplyImage] cross-object hit: {other_obj.name} face {face_index}")

    def _flush_other_bmeshes(self):
        """Write back and free all cross-object bmeshes."""
        for data in self._other_bmeshes.values():
            data['bm'].to_mesh(data['obj'].data)
            data['bm'].free()
        self._other_bmeshes.clear()

    def _discard_other_bmeshes(self):
        """Free all cross-object bmeshes without writing back."""
        for data in self._other_bmeshes.values():
            data['bm'].free()
        self._other_bmeshes.clear()

    # ---- End cross-object helper methods ----

    def paint_finish(self, context):
        # Flush cross-object changes
        self._flush_other_bmeshes()

        if not self._painted_face_indices:
            return

        from ..hotspot_mapping.json_storage import is_texture_hotspottable
        from .uv_tools import apply_hotspots_to_mesh

        obj = self._paint_obj
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()

        from ..utils import get_render_active_uv_layer
        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

        new_is_hotspottable = is_texture_hotspottable(self._image.name)

        if self._auto_hotspot and new_is_hotspottable:
            all_hotspot_faces = get_all_hotspot_faces(bm, me)

            if all_hotspot_faces:
                id_layer = get_face_id_layer(bm)
                selected_ids, active_id = save_face_selection(bm, id_layer)

                apply_hotspots_to_mesh(
                    bm, me, all_hotspot_faces, self._hotspot_seam_mode,
                    self._allow_combined_faces, self._obj_matrix,
                    self._ppm, self._size_weight
                )

                restore_face_selection(bm, id_layer, selected_ids, active_id)

                for face in all_hotspot_faces:
                    if face.is_valid:
                        cache_single_face(face, bm, self._ppm, me)
        elif (self._auto_hotspot and not new_is_hotspottable
              and self._faces_previously_hotspottable):
            # Check if any painted face has connected hotspot faces
            has_connected = False
            for fi in self._painted_face_indices:
                if fi < len(bm.faces) and any_connected_face_has_hotspot(bm.faces[fi], me):
                    has_connected = True
                    break

            if has_connected:
                all_hotspot_faces = get_all_hotspot_faces(bm, me)

                if all_hotspot_faces:
                    id_layer = get_face_id_layer(bm)
                    selected_ids, active_id = save_face_selection(bm, id_layer)

                    apply_hotspots_to_mesh(
                        bm, me, all_hotspot_faces, self._hotspot_seam_mode,
                        self._allow_combined_faces, self._obj_matrix,
                        self._ppm, self._size_weight
                    )

                    restore_face_selection(bm, id_layer, selected_ids, active_id)

                    for face in all_hotspot_faces:
                        if face.is_valid:
                            cache_single_face(face, bm, self._ppm, me)

        update_ui_from_selection(context)
        update_active_image_from_face(context)
        redraw_ui_panels(context)


class pick_image_from_face(Operator):
    """Pick texture from hovered face and apply to selected faces"""
    bl_idname = "leveldesign.pick_image_from_face"
    bl_label = "Pick and Apply Texture"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        from ..hotspot_mapping.json_storage import is_texture_hotspottable
        from .uv_tools import apply_hotspots_to_mesh

        # Require edit mode with faces selected
        edit_obj = context.object
        if not edit_obj or edit_obj.type != 'MESH' or context.mode != 'EDIT_MESH':
            return {'PASS_THROUGH'}

        bm_edit = bmesh.from_edit_mesh(edit_obj.data)
        bm_edit.faces.ensure_lookup_table()
        selected_faces = [f for f in bm_edit.faces if f.select]
        if not selected_faces:
            return {'PASS_THROUGH'}

        # Get ray from mouse position
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)

        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        # Track closest hit across all objects
        closest_distance = float('inf')
        hit_obj = None
        hit_mat_index = None
        hit_face_index = None

        # Raycast against all visible mesh objects
        for obj in context.view_layer.objects:
            if obj.type != 'MESH' or not obj.visible_get():
                continue

            # Skip objects with no mesh data
            me = obj.data
            if me is None:
                continue

            # Transform ray to object local space
            matrix_inv = obj.matrix_world.inverted()
            ray_origin_local = matrix_inv @ ray_origin
            ray_direction_local = (matrix_inv.to_3x3() @ view_vector).normalized()

            # Create BVHTree from the mesh
            # For objects in edit mode, use bmesh; otherwise use evaluated mesh
            if obj.mode == 'EDIT' and obj == context.object:
                bm = bmesh.from_edit_mesh(me)
                bm.faces.ensure_lookup_table()
                bvh = BVHTree.FromBMesh(bm)

                # Raycast, skipping backface-culled faces
                location, normal, face_index, distance = raycast_bvh_skip_backfaces(
                    bvh, ray_origin_local, ray_direction_local,
                    bm, me.materials, max_iterations=64
                )

                if face_index is not None and distance < closest_distance:
                    closest_distance = distance
                    hit_obj = obj
                    hit_mat_index = bm.faces[face_index].material_index
                    hit_face_index = face_index
            else:
                # Use evaluated mesh for object mode or non-active edit objects
                depsgraph = context.evaluated_depsgraph_get()
                obj_eval = obj.evaluated_get(depsgraph)
                me_eval = obj_eval.to_mesh()
                if me_eval is None or len(me_eval.polygons) == 0:
                    if me_eval:
                        obj_eval.to_mesh_clear()
                    continue

                bvh = BVHTree.FromPolygons(
                    [v.co for v in me_eval.vertices],
                    [p.vertices for p in me_eval.polygons]
                )

                # Raycast, skipping backface-culled faces
                location, normal, face_index, distance = raycast_bvh_skip_backfaces_polys(
                    bvh, ray_origin_local, ray_direction_local,
                    me_eval.polygons, me_eval.materials, max_iterations=64
                )

                # Get material index before clearing eval mesh
                if face_index is not None and distance < closest_distance:
                    closest_distance = distance
                    hit_obj = obj
                    hit_mat_index = me_eval.polygons[face_index].material_index
                    hit_face_index = face_index

                # Clean up evaluated mesh
                obj_eval.to_mesh_clear()

        if hit_obj is None:
            self.report({'WARNING'}, "No face under cursor")
            return {'CANCELLED'}

        # Get material and extract image
        hit_me = hit_obj.data
        hit_mat = hit_me.materials[hit_mat_index] if hit_mat_index < len(hit_me.materials) else None
        if not hit_mat:
            self.report({'WARNING'}, "Face has no material")
            return {'CANCELLED'}

        image = get_image_from_material(hit_mat)
        if not image:
            self.report({'WARNING'}, "Material has no image texture")
            return {'CANCELLED'}

        # Apply the picked texture to all selected faces on the edit object
        me = edit_obj.data
        from ..utils import get_render_active_uv_layer
        uv_layer = get_render_active_uv_layer(bm_edit, me)
        if uv_layer is None:
            uv_layer = bm_edit.loops.layers.uv.verify()
        props = context.scene.level_design_props
        ppm = props.pixels_per_meter

        # Check which selected faces previously had hotspottable textures
        faces_with_previous_hotspot = [f for f in selected_faces if face_has_hotspot_material(f, me)]
        any_previous_was_hotspottable = len(faces_with_previous_hotspot) > 0

        any_connected_has_hotspot = False
        for f in selected_faces:
            if any_connected_face_has_hotspot(f, me):
                any_connected_has_hotspot = True
                break

        # Get or create material
        mat = find_material_with_image(image)
        if mat is None:
            mat = create_material_with_image(image)

        # Ensure material slot exists
        if mat.name not in me.materials:
            me.materials.append(mat)

        mat_index = me.materials.find(mat.name)

        # Assign material to all selected faces
        for face in selected_faces:
            face.material_index = mat_index

        # Build source face reference for UV transfer from hovered face
        is_same_object = (hit_obj == edit_obj)
        _source_bm = None
        if is_same_object:
            _source_face = bm_edit.faces[hit_face_index]
            _source_uv = uv_layer
            _source_me = me
            _source_to_target = None
        else:
            _source_bm = bmesh.new()
            _source_bm.from_mesh(hit_obj.data)
            _source_bm.faces.ensure_lookup_table()
            _source_face = _source_bm.faces[hit_face_index]
            _source_uv = _source_bm.loops.layers.uv.verify()
            _source_me = hit_obj.data
            _source_to_target = edit_obj.matrix_world.inverted() @ hit_obj.matrix_world

        new_is_hotspottable = is_texture_hotspottable(image.name)

        if props.auto_hotspot and new_is_hotspottable:
            all_hotspot_faces = get_all_hotspot_faces(bm_edit, me)

            if all_hotspot_faces:
                id_layer = get_face_id_layer(bm_edit)
                selected_ids, active_id = save_face_selection(bm_edit, id_layer)

                seam_mode = props.hotspot_seam_mode
                allow_combined_faces = edit_obj.anvil_allow_combined_faces
                size_weight = edit_obj.anvil_hotspot_size_weight

                apply_hotspots_to_mesh(
                    bm_edit, me, all_hotspot_faces, seam_mode, allow_combined_faces,
                    edit_obj.matrix_world, ppm, size_weight
                )

                restore_face_selection(bm_edit, id_layer, selected_ids, active_id)

                for face in all_hotspot_faces:
                    if face.is_valid:
                        cache_single_face(face, bm_edit, ppm, me)
        else:
            if (props.auto_hotspot and not new_is_hotspottable
                    and any_previous_was_hotspottable and any_connected_has_hotspot):
                all_hotspot_faces = get_all_hotspot_faces(bm_edit, me)

                if all_hotspot_faces:
                    id_layer = get_face_id_layer(bm_edit)
                    selected_ids, active_id = save_face_selection(bm_edit, id_layer)

                    seam_mode = props.hotspot_seam_mode
                    allow_combined_faces = edit_obj.anvil_allow_combined_faces
                    size_weight = edit_obj.anvil_hotspot_size_weight

                    apply_hotspots_to_mesh(
                        bm_edit, me, all_hotspot_faces, seam_mode, allow_combined_faces,
                        edit_obj.matrix_world, ppm, size_weight
                    )

                    restore_face_selection(bm_edit, id_layer, selected_ids, active_id)

                    for face in all_hotspot_faces:
                        if face.is_valid:
                            cache_single_face(face, bm_edit, ppm, me)

            # Transfer UV from hovered (source) face to each selected (target) face
            for target_face in selected_faces:
                set_uv_from_other_face(
                    _source_face, target_face, uv_layer,
                    ppm, me, edit_obj.matrix_world,
                    bm=bm_edit,
                    source_uv_layer=_source_uv,
                    source_me=_source_me,
                    source_to_target=_source_to_target,
                )

        if _source_bm is not None:
            _source_bm.free()

        bmesh.update_edit_mesh(me)

        # Update UI state
        update_ui_from_selection(context)
        update_active_image_from_face(context)
        redraw_ui_panels(context)
        self.report({'INFO'}, f"Applied: {image.name}")

        return {'FINISHED'}


classes = (
    apply_image_to_face,
    pick_image_from_face,
)

addon_keymaps = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Register Alt+Click keymap
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    # Register in Mesh keymap for edit mode
    km = kc.keymaps.new(name='Mesh', space_type='EMPTY')

    # Alt+Left Click to apply image to face
    kmi = km.keymap_items.new(
        apply_image_to_face.bl_idname,
        'LEFTMOUSE', 'PRESS',
        alt=True,
        head=True
    )
    addon_keymaps.append((km, kmi))

    # Alt+Right Click to pick image from face (edit mode)
    kmi = km.keymap_items.new(
        pick_image_from_face.bl_idname,
        'RIGHTMOUSE', 'PRESS',
        alt=True,
        head=True
    )
    addon_keymaps.append((km, kmi))

    # Register Alt+Right Click in 3D View for object mode as well
    km = kc.keymaps.new(name='3D View', space_type='VIEW_3D')
    kmi = km.keymap_items.new(
        pick_image_from_face.bl_idname,
        'RIGHTMOUSE', 'PRESS',
        alt=True,
        head=True
    )
    addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
