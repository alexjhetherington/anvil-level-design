import bpy
import bmesh
import math
from bpy.types import Operator
from bpy_extras import view3d_utils
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
)
from ..properties import apply_uv_to_face
from ..handlers import cache_single_face, get_active_image, set_active_image


def set_uv_from_other_face(source_face, target_face, uv_layer, ppm, me):
    """Copy UV settings from source face to target face with proper rotation/offset handling.

    - Scale is copied directly from source
    - Rotation uses plane intersection logic:
      - If planes intersect: rotation is relative to the intersection axis
      - If planes are parallel: rotation adjusted for different first-edge directions
    - Offset uses face intersection:
      - If faces share an edge: offset relative to shared edge midpoint
      - Otherwise: offset copied directly

    Args:
        source_face: BMesh face to copy UV settings from
        target_face: BMesh face to apply UV settings to
        uv_layer: BMesh UV layer
        ppm: Pixels per meter setting
        me: Mesh data (for bmesh.update_edit_mesh)
    """
    # Get source transform
    source_transform = derive_transform_from_uvs(source_face, uv_layer, ppm, me)
    if not source_transform:
        return False

    scale_u = source_transform['scale_u']
    scale_v = source_transform['scale_v']
    source_rotation = source_transform['rotation']
    source_offset_x = source_transform['offset_x']
    source_offset_y = source_transform['offset_y']

    # Get face normals and local axes
    source_normal = source_face.normal.normalized()
    target_normal = target_face.normal.normalized()

    source_axes = get_face_local_axes(source_face)
    target_axes = get_face_local_axes(target_face)
    if not source_axes or not target_axes:
        return False

    source_u, source_v = source_axes  # source_u is first edge direction
    target_u, target_v = target_axes  # target_u is first edge direction

    # Compute plane intersection axis
    intersection = source_normal.cross(target_normal)
    intersection_length = intersection.length

    if intersection_length < 0.0001:
        # Planes are parallel - adjust rotation for different first-edge directions
        # Find angle between source_u and target_u projected onto the same plane
        dot = source_u.dot(target_u)
        cross = source_u.cross(target_u)
        # Sign of angle determined by which way cross points relative to normal
        angle_diff = math.atan2(cross.dot(source_normal), dot)

        # Check if faces are opposite-facing (parallel but normals point opposite directions)
        if source_normal.dot(target_normal) < 0:
            # Opposite faces
            target_rotation = source_rotation + math.degrees(angle_diff) + 180
            if (target_rotation > 360):
                target_rotation = target_rotation - 360
        else:
            target_rotation = source_rotation + math.degrees(angle_diff)
    else:
        # Planes intersect - use intersection axis as reference
        intersection = intersection.normalized()

        # Angle from intersection axis to source's first edge (source_u), in source's plane
        # Using signed angle with source_normal as reference
        dot_source = intersection.dot(source_u)
        cross_source = intersection.cross(source_u)
        angle_intersection_to_source_u = math.atan2(cross_source.dot(source_normal), dot_source)

        # Angle from intersection axis to target's first edge (target_u), in target's plane
        dot_target = intersection.dot(target_u)
        cross_target = intersection.cross(target_u)
        angle_intersection_to_target_u = math.atan2(cross_target.dot(target_normal), dot_target)

        # Source's texture U direction relative to intersection:
        # texture_u is source_u rotated by -source_rotation in the face plane
        # So angle from intersection to texture_u = angle_intersection_to_source_u - source_rotation
        texture_angle_from_intersection = angle_intersection_to_source_u - math.radians(source_rotation)

        # On target, we want the same angle from intersection to texture_u
        # angle_intersection_to_target_u - target_rotation = texture_angle_from_intersection
        target_rotation = math.degrees(angle_intersection_to_target_u - texture_angle_from_intersection)

    # Handle offset based on face intersection
    source_verts = set(source_face.verts)
    target_verts = set(target_face.verts)
    shared_verts = source_verts & target_verts

    if len(shared_verts) >= 2:
        # Faces share an edge - compute offset so shared edge midpoint has same UV
        shared_list = list(shared_verts)[:2]
        midpoint = (shared_list[0].co + shared_list[1].co) / 2

        # Find UV of midpoint in source face (interpolate between the two shared verts)
        source_uv1 = source_uv2 = None
        for loop in source_face.loops:
            if loop.vert == shared_list[0]:
                source_uv1 = loop[uv_layer].uv.copy()
            elif loop.vert == shared_list[1]:
                source_uv2 = loop[uv_layer].uv.copy()

        if source_uv1 and source_uv2:
            source_midpoint_uv_x = (source_uv1.x + source_uv2.x) / 2
            source_midpoint_uv_y = (source_uv1.y + source_uv2.y) / 2

            # Compute what UV the midpoint would have on target with offset=0
            target_mat = me.materials[target_face.material_index] if target_face.material_index < len(me.materials) else None
            tex_meters_u, tex_meters_v = get_texture_dimensions_from_material(target_mat, ppm)

            # Rotate projection axes in face space
            rot_rad = math.radians(target_rotation)
            cos_r = math.cos(rot_rad)
            sin_r = math.sin(rot_rad)
            proj_x = target_u * cos_r - target_v * sin_r
            proj_y = target_u * sin_r + target_v * cos_r

            # Project midpoint onto rotated axes
            first_vert_target = list(target_face.loops)[0].vert.co
            delta = midpoint - first_vert_target
            x = delta.dot(proj_x)
            y = delta.dot(proj_y)

            # Apply scale
            u = x / (scale_u * tex_meters_u)
            v = y / (scale_v * tex_meters_v)

            # Offset needed so midpoint UV matches source
            target_offset_x = normalize_offset(source_midpoint_uv_x - u)
            target_offset_y = normalize_offset(source_midpoint_uv_y - v)
        else:
            target_offset_x = normalize_offset(source_offset_x)
            target_offset_y = normalize_offset(source_offset_y)
    elif len(shared_verts) == 1:
        # Faces share a single vertex - use that vertex as reference
        shared_vert = list(shared_verts)[0]

        # Find UV of shared vert in source
        source_shared_uv = None
        for loop in source_face.loops:
            if loop.vert == shared_vert:
                source_shared_uv = loop[uv_layer].uv.copy()
                break

        if source_shared_uv:
            # Compute what UV the shared vert would have on target with offset=0
            target_mat = me.materials[target_face.material_index] if target_face.material_index < len(me.materials) else None
            tex_meters_u, tex_meters_v = get_texture_dimensions_from_material(target_mat, ppm)

            # Rotate projection axes in face space
            rot_rad = math.radians(target_rotation)
            cos_r = math.cos(rot_rad)
            sin_r = math.sin(rot_rad)
            proj_x = target_u * cos_r - target_v * sin_r
            proj_y = target_u * sin_r + target_v * cos_r

            # Project shared vertex onto rotated axes
            first_vert_target = list(target_face.loops)[0].vert.co
            delta = shared_vert.co - first_vert_target
            x = delta.dot(proj_x)
            y = delta.dot(proj_y)

            # Apply scale
            u = x / (scale_u * tex_meters_u)
            v = y / (scale_v * tex_meters_v)

            target_offset_x = normalize_offset(source_shared_uv.x - u)
            target_offset_y = normalize_offset(source_shared_uv.y - v)
        else:
            target_offset_x = normalize_offset(source_offset_x)
            target_offset_y = normalize_offset(source_offset_y)
    else:
        # No shared vertices - just copy offset
        target_offset_x = normalize_offset(source_offset_x)
        target_offset_y = normalize_offset(source_offset_y)

    # Apply to target face
    target_mat = me.materials[target_face.material_index] if target_face.material_index < len(me.materials) else None
    apply_uv_to_face(target_face, uv_layer, scale_u, scale_v, target_rotation,
                     target_offset_x, target_offset_y, target_mat, ppm, me)
    cache_single_face(target_face, uv_layer, ppm, me)

    return True


class apply_image_to_face(Operator):
    """Apply selected File Browser image to hovered face"""
    bl_idname = "leveldesign.apply_image_to_face"
    bl_label = "Apply Image to Face"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        obj = context.object
        if not obj or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
            return {'CANCELLED'}

        image = get_active_image()
        if not image:
            self.report({'WARNING'}, "No active texture (select a face or image in File Browser)")
            return {'CANCELLED'}

        # Raycast using BVHTree on the bmesh (scene.ray_cast returns wrong face indices in edit mode)
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)

        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        # Transform ray to object local space
        matrix_inv = obj.matrix_world.inverted()
        ray_origin_local = matrix_inv @ ray_origin
        ray_direction_local = (matrix_inv.to_3x3() @ view_vector).normalized()

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        # Raycast against bmesh BVHTree
        bvh = BVHTree.FromBMesh(bm)
        location, normal, face_index, distance = bvh.ray_cast(ray_origin_local, ray_direction_local)

        if face_index is None:
            return {'CANCELLED'}

        target_face = bm.faces[face_index]

        # Check if there's an active face that could be the source
        source_face = bm.faces.active
        uv_layer = bm.loops.layers.uv.verify()

        ppm = context.scene.level_design_props.pixels_per_meter

        # Get or create material
        mat = find_material_with_image(image)
        if mat is None:
            mat = create_material_with_image(image)

        # Ensure material slot exists
        if mat.name not in obj.data.materials:
            obj.data.materials.append(mat)

        mat_index = obj.data.materials.find(mat.name)
        target_face.material_index = mat_index

        set_uv_from_other_face(source_face, target_face, uv_layer, ppm, obj.data)

        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}


class pick_image_from_face(Operator):
    """Pick active image from hovered face (works across all visible objects)"""
    bl_idname = "leveldesign.pick_image_from_face"
    bl_label = "Pick Image from Face"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
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

                # Raycast
                location, normal, face_index, distance = bvh.ray_cast(
                    ray_origin_local, ray_direction_local
                )

                if face_index is not None and distance < closest_distance:
                    closest_distance = distance
                    hit_obj = obj
                    hit_mat_index = bm.faces[face_index].material_index
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

                # Raycast
                location, normal, face_index, distance = bvh.ray_cast(
                    ray_origin_local, ray_direction_local
                )

                # Get material index before clearing eval mesh
                if face_index is not None and distance < closest_distance:
                    closest_distance = distance
                    hit_obj = obj
                    hit_mat_index = me_eval.polygons[face_index].material_index

                # Clean up evaluated mesh
                obj_eval.to_mesh_clear()

        if hit_obj is None:
            self.report({'WARNING'}, "No face under cursor")
            return {'CANCELLED'}

        # Get material and extract image
        me = hit_obj.data
        mat = me.materials[hit_mat_index] if hit_mat_index < len(me.materials) else None
        if not mat:
            self.report({'WARNING'}, "Face has no material")
            return {'CANCELLED'}

        image = get_image_from_material(mat)
        if not image:
            self.report({'WARNING'}, "Material has no image texture")
            return {'CANCELLED'}

        # Set as active image
        set_active_image(image)
        self.report({'INFO'}, f"Active image: {image.name}")

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
