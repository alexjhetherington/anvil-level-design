import math

import bpy
import bmesh
from bpy.types import Operator
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ...utils import is_level_design_workspace
from .raycast import (
    raycast_bvh_skip_backfaces,
    is_face_backfacing,
    has_backface_culling_enabled,
)

# Screen-space pixel threshold for picking edges/verts on culled faces
_PICK_THRESHOLD_PX = 15
# Number of rays in the fan around the center ray
_FAN_RAY_COUNT = 8


def _nearest_vert_on_face(hit_point, face):
    """Find the vertex on a face nearest to the hit point."""
    best_vert = None
    best_dist = float('inf')
    for vert in face.verts:
        dist = (vert.co - hit_point).length_squared
        if dist < best_dist:
            best_dist = dist
            best_vert = vert
    return best_vert


def _point_to_segment_dist_sq(point, seg_a, seg_b):
    """Squared distance from a point to a line segment."""
    ab = seg_b - seg_a
    ab_sq = ab.length_squared
    if ab_sq < 1e-12:
        return (point - seg_a).length_squared
    t = max(0.0, min(1.0, (point - seg_a).dot(ab) / ab_sq))
    proj = seg_a + ab * t
    return (point - proj).length_squared


def _nearest_edge_on_face(hit_point, face):
    """Find the edge on a face nearest to the hit point."""
    best_edge = None
    best_dist = float('inf')
    for edge in face.edges:
        dist = _point_to_segment_dist_sq(hit_point, edge.verts[0].co, edge.verts[1].co)
        if dist < best_dist:
            best_dist = dist
            best_edge = edge
    return best_edge


def _point_to_segment_dist_2d(point, seg_a, seg_b):
    """Distance from a 2D point to a 2D line segment."""
    ab = seg_b - seg_a
    ab_sq = ab.length_squared
    if ab_sq < 1e-12:
        return (point - seg_a).length
    t = max(0.0, min(1.0, (point - seg_a).dot(ab) / ab_sq))
    proj = seg_a + ab * t
    return (point - proj).length


def _screen_nearest_vert_on_face(face, region, rv3d, obj_matrix, mouse_2d):
    """Find the vertex on a face closest to the mouse in screen space.

    Returns (vert, screen_distance) or (None, inf).
    """
    best_vert = None
    best_dist = float('inf')
    for vert in face.verts:
        screen_co = view3d_utils.location_3d_to_region_2d(
            region, rv3d, obj_matrix @ vert.co
        )
        if screen_co is None:
            continue
        dist = (mouse_2d - screen_co).length
        if dist < best_dist:
            best_dist = dist
            best_vert = vert
    return best_vert, best_dist


def _screen_nearest_edge_on_face(face, region, rv3d, obj_matrix, mouse_2d):
    """Find the edge on a face closest to the mouse in screen space.

    Returns (edge, screen_distance) or (None, inf).
    """
    best_edge = None
    best_dist = float('inf')
    for edge in face.edges:
        sa = view3d_utils.location_3d_to_region_2d(
            region, rv3d, obj_matrix @ edge.verts[0].co
        )
        sb = view3d_utils.location_3d_to_region_2d(
            region, rv3d, obj_matrix @ edge.verts[1].co
        )
        if sa is None or sb is None:
            continue
        dist = _point_to_segment_dist_2d(mouse_2d, sa, sb)
        if dist < best_dist:
            best_dist = dist
            best_edge = edge
    return best_edge, best_dist


def _check_culled_face_element(face, is_edge_mode, region, rv3d, obj_matrix, mouse_2d):
    """Check if cursor is near an edge/vert on a culled face in screen space.

    Returns (element, screen_distance) or (None, inf).
    """
    if is_edge_mode:
        return _screen_nearest_edge_on_face(face, region, rv3d, obj_matrix, mouse_2d)
    return _screen_nearest_vert_on_face(face, region, rv3d, obj_matrix, mouse_2d)


def _is_culled_backface(face, ray_direction_local, materials):
    """Check if a face is a backface with culling enabled."""
    return (is_face_backfacing(face.normal, ray_direction_local)
            and has_backface_culling_enabled(face.material_index, materials))


def _collect_fan_faces(bvh, bm, materials, region, rv3d, obj_matrix,
                       mouse_2d, max_iterations):
    """Cast fan rays around the cursor to find nearby faces.

    Returns (culled_faces, front_faces) — two sets of face indices hit by
    fan rays. Culled backfaces are walked through; the first front face on
    each ray is also collected so edges/verts near the cursor but off the
    face can still be picked.
    """
    culled_faces = set()
    front_faces = set()
    epsilon = 0.0001
    matrix_inv = obj_matrix.inverted()
    rot_inv = matrix_inv.to_3x3()

    for i in range(_FAN_RAY_COUNT):
        angle = (2.0 * math.pi * i) / _FAN_RAY_COUNT
        offset_x = math.cos(angle) * _PICK_THRESHOLD_PX
        offset_y = math.sin(angle) * _PICK_THRESHOLD_PX
        fan_coord = (mouse_2d.x + offset_x, mouse_2d.y + offset_y)

        fan_view = view3d_utils.region_2d_to_vector_3d(region, rv3d, fan_coord)
        fan_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, fan_coord)

        origin_local = matrix_inv @ fan_origin
        dir_local = (rot_inv @ fan_view).normalized()

        # Walk through hits on this fan ray
        origin = origin_local.copy()
        for _ in range(max_iterations):
            location, normal, face_index, distance = bvh.ray_cast(origin, dir_local)
            if face_index is None:
                break

            face = bm.faces[face_index]
            if _is_culled_backface(face, dir_local, materials):
                culled_faces.add(face_index)
                origin = origin + dir_local * (distance + epsilon)
                continue
            # Front face — collect it, then stop this fan ray
            front_faces.add(face_index)
            break

    return culled_faces, front_faces


def _raycast_element_aware(bvh, ray_origin_local, ray_direction_local,
                            bm, materials, region, rv3d, obj_matrix, mouse_2d,
                            is_edge_mode, max_iterations):
    """Raycast that skips culled backfaces but catches nearby edges/verts.

    First casts the center ray, checking culled faces along the way. Then casts
    fan rays around the cursor to find faces the center ray missed (e.g. when
    the cursor is just past the edge of a face). The best element across all
    discovered faces is selected if within screen threshold.

    Returns (face, location, element) where element is the nearby BMEdge/BMVert
    found on a nearby face, or None if the final hit was a front face.
    """
    origin = ray_origin_local.copy()
    epsilon = 0.0001

    # Track the best culled-face element found across center + fan rays
    best_culled_elem = None
    best_culled_dist = float('inf')
    best_culled_face = None

    # Track culled face indices seen by the center ray
    center_culled_faces = set()

    # Front-face result from center ray (if any)
    front_face = None
    front_location = None

    # Phase 1: center ray — walk through hits
    for _ in range(max_iterations):
        location, normal, face_index, distance = bvh.ray_cast(origin, ray_direction_local)

        if face_index is None:
            break

        face = bm.faces[face_index]

        if _is_culled_backface(face, ray_direction_local, materials):
            center_culled_faces.add(face_index)

            elem, screen_dist = _check_culled_face_element(
                face, is_edge_mode, region, rv3d, obj_matrix, mouse_2d
            )
            if elem is not None and screen_dist < best_culled_dist:
                best_culled_dist = screen_dist
                best_culled_elem = elem
                best_culled_face = face

            origin = origin + ray_direction_local * (distance + epsilon)
            continue

        # Front-facing hit
        front_face = face
        front_location = location
        break

    # Phase 2: fan rays — find faces the center ray missed
    fan_culled, fan_front = _collect_fan_faces(
        bvh, bm, materials, region, rv3d, obj_matrix,
        mouse_2d, max_iterations=8
    )
    # Combine new culled and front faces not already handled by center ray
    new_faces = (fan_culled - center_culled_faces) | fan_front
    if front_face is not None:
        new_faces.discard(front_face.index)

    for face_index in new_faces:
        face = bm.faces[face_index]
        elem, screen_dist = _check_culled_face_element(
            face, is_edge_mode, region, rv3d, obj_matrix, mouse_2d
        )
        if elem is not None and screen_dist < best_culled_dist:
            best_culled_dist = screen_dist
            best_culled_elem = elem
            best_culled_face = face

    # Decide: use nearby element if within threshold, otherwise front face
    if best_culled_elem is not None and best_culled_dist <= _PICK_THRESHOLD_PX:
        return best_culled_face, None, best_culled_elem

    if front_face is not None:
        return front_face, front_location, None

    return None, None, None


def _do_edge_loop_select(bm, me, face, hit_point, extend, target_edge_override):
    """Perform edge loop selection from the nearest edge on the hit face."""
    if target_edge_override is not None:
        target_edge = target_edge_override
    else:
        target_edge = _nearest_edge_on_face(hit_point, face)
    if target_edge is None:
        return

    saved_vert_sel = None
    saved_edge_sel = None
    saved_face_sel = None
    if extend:
        saved_vert_sel = {v.index for v in bm.verts if v.select}
        saved_edge_sel = {e.index for e in bm.edges if e.select}
        saved_face_sel = {f.index for f in bm.faces if f.select}

    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False

    target_edge.select = True
    for v in target_edge.verts:
        v.select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(me)

    bpy.ops.mesh.select_edge_loop_multi()

    if extend:
        bm = bmesh.from_edit_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        for v in bm.verts:
            if v.index in saved_vert_sel:
                v.select = True
        for e in bm.edges:
            if e.index in saved_edge_sel:
                e.select = True
        for f in bm.faces:
            if f.index in saved_face_sel:
                f.select = True

        bm.select_flush_mode()
        bmesh.update_edit_mesh(me)


def _do_face_loop_select(bm, me, face, hit_point, extend, target_edge_override):
    """Perform face loop selection from the nearest edge on the hit face.

    A face loop is the strip of faces along an edge ring. We find the edge ring
    from the nearest edge, then select faces that have 2+ edges in the ring.
    """
    if target_edge_override is not None:
        target_edge = target_edge_override
    else:
        target_edge = _nearest_edge_on_face(hit_point, face)
    if target_edge is None:
        return

    saved_face_sel = None
    if extend:
        saved_face_sel = {f.index for f in bm.faces if f.select}

    # Deselect all, select target edge, get edge ring
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False

    target_edge.select = True
    for v in target_edge.verts:
        v.select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(me)

    # Get edge ring (face loops correspond to edge rings)
    bpy.ops.mesh.select_edge_ring_multi()

    # Re-fetch bmesh, collect ring edge indices
    bm = bmesh.from_edit_mesh(me)
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    ring_edges = {e.index for e in bm.edges if e.select}

    # Deselect all edges/verts
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False

    # Select faces that have 2+ edges in the ring (the face loop)
    for f in bm.faces:
        edge_count = sum(1 for e in f.edges if e.index in ring_edges)
        if edge_count >= 2:
            f.select = True

    if extend:
        for f in bm.faces:
            if f.index in saved_face_sel:
                f.select = True

    bm.select_flush_mode()
    bmesh.update_edit_mesh(me)


class LEVELDESIGN_OT_backface_select(Operator):
    """Select through backface-culled faces"""
    bl_idname = "leveldesign.backface_select"
    bl_label = "Backface-Aware Select"
    bl_options = {'REGISTER', 'UNDO'}

    extend: bpy.props.BoolProperty()
    loop: bpy.props.BoolProperty()

    @classmethod
    def poll(cls, context):
        if not is_level_design_workspace():
            return False
        obj = context.object
        return (obj is not None
                and obj.type == 'MESH'
                and context.mode == 'EDIT_MESH')

    def invoke(self, context, event):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()

        # Build ray from mouse position
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        mouse_2d = Vector((float(coord[0]), float(coord[1])))

        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        # Transform to object local space
        matrix_inv = obj.matrix_world.inverted()
        ray_origin_local = matrix_inv @ ray_origin
        ray_direction_local = (matrix_inv.to_3x3() @ view_vector).normalized()

        select_mode = context.tool_settings.mesh_select_mode
        is_vert_mode = select_mode[0]
        is_edge_mode = select_mode[1]
        is_face_mode = select_mode[2]

        bvh = BVHTree.FromBMesh(bm)
        culled_element = None

        if is_face_mode:
            # Face mode: skip culled backfaces entirely
            location, normal, face_index, distance = raycast_bvh_skip_backfaces(
                bvh, ray_origin_local, ray_direction_local,
                bm, me.materials, max_iterations=64
            )
            face = bm.faces[face_index] if face_index is not None else None
        else:
            # Edge/vert mode: skip culled backfaces but catch nearby
            # edges/verts on them in screen space
            face, location, culled_element = _raycast_element_aware(
                bvh, ray_origin_local, ray_direction_local,
                bm, me.materials, region, rv3d, obj.matrix_world, mouse_2d,
                is_edge_mode, max_iterations=64
            )

        if face is None:
            if not self.extend:
                bpy.ops.mesh.select_all(action='DESELECT')
            return {'FINISHED'}

        hit_point = location

        # Alt+click: loop select
        if self.loop:
            # Determine the target edge for loop select
            if culled_element is not None:
                if is_edge_mode:
                    loop_edge = culled_element
                else:
                    # For face/vert mode, culled_element isn't an edge —
                    # find the nearest edge on the face instead
                    loop_edge, _ = _screen_nearest_edge_on_face(
                        face, region, rv3d, obj.matrix_world, mouse_2d
                    )
            else:
                loop_edge = None

            if is_face_mode:
                _do_face_loop_select(bm, me, face, hit_point, self.extend, loop_edge)
            else:
                _do_edge_loop_select(bm, me, face, hit_point, self.extend, loop_edge)
            return {'FINISHED'}

        # Plain or Shift click
        if not self.extend:
            bpy.ops.mesh.select_all(action='DESELECT')
            # Re-fetch bmesh after operator call
            bm = bmesh.from_edit_mesh(me)
            bm.faces.ensure_lookup_table()
            face = bm.faces[face.index]
            # Re-lookup culled_element after bmesh refresh
            if culled_element is not None:
                if is_edge_mode:
                    bm.edges.ensure_lookup_table()
                    culled_element = bm.edges[culled_element.index]
                elif is_vert_mode:
                    bm.verts.ensure_lookup_table()
                    culled_element = bm.verts[culled_element.index]

        if is_face_mode:
            face.select = not face.select if self.extend else True
            bm.faces.active = face
        elif culled_element is not None:
            # Picked an edge/vert from a culled backface
            if is_edge_mode:
                new_state = not culled_element.select if self.extend else True
                culled_element.select = new_state
                for v in culled_element.verts:
                    v.select = new_state
            else:
                culled_element.select = not culled_element.select if self.extend else True
        elif is_edge_mode:
            edge = _nearest_edge_on_face(hit_point, face)
            if edge is not None:
                new_state = not edge.select if self.extend else True
                edge.select = new_state
                for v in edge.verts:
                    v.select = new_state
        elif is_vert_mode:
            vert = _nearest_vert_on_face(hit_point, face)
            if vert is not None:
                vert.select = not vert.select if self.extend else True

        bm.select_flush_mode()
        bmesh.update_edit_mesh(me)

        return {'FINISHED'}


def _resolve_select_target(depsgraph, hit_obj, hit_matrix):
    """Resolve a raycast hit to the object that should be selected.

    If the hit object is part of a collection instance, returns the
    instancing empty. Otherwise returns the original object.
    Uses the hit matrix from scene.ray_cast() to distinguish between
    multiple instances of the same collection.
    """
    hit_original = hit_obj.original
    for instance in depsgraph.object_instances:
        if not instance.is_instance:
            continue
        if instance.object.original != hit_original:
            continue
        # Compare the instance matrix to the hit matrix to identify
        # which specific collection instance was clicked
        if _matrices_equal(instance.matrix_world, hit_matrix):
            return instance.parent.original
    return hit_original


def _matrices_equal(a, b, epsilon=1e-4):
    """Compare two matrices for approximate equality."""
    for i in range(4):
        for j in range(4):
            if abs(a[i][j] - b[i][j]) > epsilon:
                return False
    return True


class LEVELDESIGN_OT_backface_object_select(Operator):
    """Select objects through backface-culled faces"""
    bl_idname = "leveldesign.backface_object_select"
    bl_label = "Backface-Aware Object Select"
    bl_options = {'REGISTER', 'UNDO'}

    extend: bpy.props.BoolProperty()

    @classmethod
    def poll(cls, context):
        if not is_level_design_workspace():
            return False
        return context.mode == 'OBJECT'

    def invoke(self, context, event):
        from .raycast import raycast_scene_skip_backfaces

        region = context.region
        rv3d = context.region_data
        if rv3d is None:
            return {'PASS_THROUGH'}

        coord = (event.mouse_region_x, event.mouse_region_y)
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        depsgraph = context.evaluated_depsgraph_get()
        hit, location, normal, face_index, obj, matrix = raycast_scene_skip_backfaces(
            depsgraph, context.scene, ray_origin, view_vector, max_iterations=64
        )

        if not hit:
            if not self.extend:
                bpy.ops.object.select_all(action='DESELECT')
            return {'FINISHED'}

        select_obj = _resolve_select_target(depsgraph, obj, matrix)

        if not self.extend:
            bpy.ops.object.select_all(action='DESELECT')

        select_obj.select_set(not select_obj.select_get() if self.extend else True)
        context.view_layer.objects.active = select_obj

        return {'FINISHED'}


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_backface_select)
    bpy.utils.register_class(LEVELDESIGN_OT_backface_object_select)


def unregister():
    bpy.utils.unregister_class(LEVELDESIGN_OT_backface_object_select)
    bpy.utils.unregister_class(LEVELDESIGN_OT_backface_select)
