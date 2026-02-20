import math
from collections import deque

import bpy
import bmesh
from bpy.types import Operator
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..utils import is_level_design_workspace
from .backface_select.raycast import (
    raycast_bvh_skip_backfaces,
    is_face_backfacing,
    has_backface_culling_enabled,
)


# Screen-space pixel threshold for picking edges/verts on culled faces
_PICK_THRESHOLD_PX = 15
# Number of rays in the fan around the center ray
_FAN_RAY_COUNT = 8

# Small epsilon added to threshold angles so the boundary face is included
_ANGLE_EPSILON = 0.01

# Module-level state for Ctrl+L / Ctrl+Shift+L angle tracking
_linked_state = {
    'object_name': None,
    'seed_face': None,      # face index under cursor when Ctrl+L started
    'seed_normal': None,     # normal of seed face
    'angle': 0.1,            # current angle threshold
    'angle_history': [],     # stack of previous angles for shrink
    'result': set(),         # face indices after last operation
}

_addon_keymaps = []


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
    """Find the vertex on a face closest to the mouse in screen space."""
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
    """Find the edge on a face closest to the mouse in screen space."""
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
    """Check if cursor is near an edge/vert on a culled face in screen space."""
    if is_edge_mode:
        return _screen_nearest_edge_on_face(face, region, rv3d, obj_matrix, mouse_2d)
    return _screen_nearest_vert_on_face(face, region, rv3d, obj_matrix, mouse_2d)


def _is_culled_backface(face, ray_direction_local, materials):
    """Check if a face is a backface with culling enabled."""
    return (is_face_backfacing(face.normal, ray_direction_local)
            and has_backface_culling_enabled(face.material_index, materials))


def _collect_fan_culled_faces(bvh, bm, materials, region, rv3d, obj_matrix,
                               mouse_2d, max_iterations):
    """Cast fan rays around the cursor to find nearby culled backfaces."""
    culled_faces = set()
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
            break

    return culled_faces


def _raycast_element_aware(bvh, ray_origin_local, ray_direction_local,
                            bm, materials, region, rv3d, obj_matrix, mouse_2d,
                            is_edge_mode, max_iterations):
    """Raycast that skips culled backfaces but catches nearby edges/verts on them.

    Returns (face, location, element) where element is the nearby BMEdge/BMVert
    found on a culled face, or None if the final hit was a front face.
    """
    origin = ray_origin_local.copy()
    epsilon = 0.0001

    best_culled_elem = None
    best_culled_dist = float('inf')
    best_culled_face = None
    center_culled_faces = set()
    front_face = None
    front_location = None

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

        front_face = face
        front_location = location
        break

    fan_culled = _collect_fan_culled_faces(
        bvh, bm, materials, region, rv3d, obj_matrix,
        mouse_2d, max_iterations=8
    )
    new_culled = fan_culled - center_culled_faces

    for face_index in new_culled:
        face = bm.faces[face_index]
        elem, screen_dist = _check_culled_face_element(
            face, is_edge_mode, region, rv3d, obj_matrix, mouse_2d
        )
        if elem is not None and screen_dist < best_culled_dist:
            best_culled_dist = screen_dist
            best_culled_elem = elem
            best_culled_face = face

    if best_culled_elem is not None and best_culled_dist <= _PICK_THRESHOLD_PX:
        return best_culled_face, None, best_culled_elem

    if front_face is not None:
        return front_face, front_location, None

    return None, None, None


def _flood_fill_faces(bm, seed_indices):
    """BFS flood-fill connected faces from seed face indices. Returns set of face indices."""
    visited = set()
    queue = deque()
    for idx in seed_indices:
        if idx not in visited:
            visited.add(idx)
            queue.append(idx)

    while queue:
        fi = queue.popleft()
        face = bm.faces[fi]
        for edge in face.edges:
            for neighbor in edge.link_faces:
                if neighbor.index not in visited:
                    visited.add(neighbor.index)
                    queue.append(neighbor.index)
    return visited


def _flood_fill_edges(bm, seed_indices):
    """BFS flood-fill connected edges from seed edge indices. Returns set of edge indices."""
    visited = set()
    queue = deque()
    for idx in seed_indices:
        if idx not in visited:
            visited.add(idx)
            queue.append(idx)

    while queue:
        ei = queue.popleft()
        edge = bm.edges[ei]
        for vert in edge.verts:
            for neighbor in vert.link_edges:
                if neighbor.index not in visited:
                    visited.add(neighbor.index)
                    queue.append(neighbor.index)
    return visited


def _flood_fill_verts(bm, seed_indices):
    """BFS flood-fill connected verts from seed vert indices. Returns set of vert indices."""
    visited = set()
    queue = deque()
    for idx in seed_indices:
        if idx not in visited:
            visited.add(idx)
            queue.append(idx)

    while queue:
        vi = queue.popleft()
        vert = bm.verts[vi]
        for edge in vert.link_edges:
            other = edge.other_vert(vert)
            if other.index not in visited:
                visited.add(other.index)
                queue.append(other.index)
    return visited


def _flood_fill_faces_by_normal(bm, seeds, angle_deg):
    """BFS flood-fill faces from seeds, constrained by normal angle.

    seeds: dict of face_index -> normal Vector
    angle_deg: maximum angle in degrees between seed normal and neighbor normal

    Returns set of face indices.
    """
    cos_limit = math.cos(math.radians(angle_deg))
    visited = set()
    queue = deque()

    for idx, normal in seeds.items():
        visited.add(idx)
        queue.append((idx, normal))

    while queue:
        fi, seed_normal = queue.popleft()
        face = bm.faces[fi]
        for edge in face.edges:
            for neighbor in edge.link_faces:
                if neighbor.index not in visited:
                    if neighbor.normal.dot(seed_normal) >= cos_limit:
                        visited.add(neighbor.index)
                        queue.append((neighbor.index, seed_normal))
    return visited


def _reset_linked_state():
    """Reset the module-level linked state."""
    _linked_state['object_name'] = None
    _linked_state['seed_face'] = None
    _linked_state['seed_normal'] = None
    _linked_state['angle'] = 0.1
    _linked_state['angle_history'] = []
    _linked_state['result'] = set()


def _find_next_expand_angle(bm, current_result, seed_normal):
    """Find the smallest angle that would include at least one more face.

    Scans the boundary of the current result for the nearest neighbor face
    (by normal angle to seed_normal) that isn't already selected.

    Returns the angle in degrees, or None if no more faces to expand to.
    """
    min_angle = None
    for fi in current_result:
        face = bm.faces[fi]
        for edge in face.edges:
            for neighbor in edge.link_faces:
                if neighbor.index not in current_result:
                    dot = neighbor.normal.dot(seed_normal)
                    dot = max(-1.0, min(1.0, dot))
                    angle = math.degrees(math.acos(dot))
                    if min_angle is None or angle < min_angle:
                        min_angle = angle
    return min_angle


def _select_face_indices(bm, indices):
    """Select exactly the given face indices, deselecting all others."""
    for f in bm.faces:
        f.select = (f.index in indices)
    bm.select_flush_mode()


def _get_seed_face(bvh, ray_origin_local, ray_direction_local, bm, materials):
    """Get a seed face from raycast when nothing is selected."""
    location, normal, face_index, distance = raycast_bvh_skip_backfaces(
        bvh, ray_origin_local, ray_direction_local,
        bm, materials, max_iterations=64
    )
    return face_index


def _get_seed_edge(bvh, ray_origin_local, ray_direction_local, bm, materials,
                   region, rv3d, obj_matrix, mouse_2d):
    """Get a seed edge from raycast when nothing is selected."""
    face, location, culled_element = _raycast_element_aware(
        bvh, ray_origin_local, ray_direction_local,
        bm, materials, region, rv3d, obj_matrix, mouse_2d,
        True, max_iterations=64
    )
    if culled_element is not None:
        return culled_element.index
    if face is not None and location is not None:
        edge = _nearest_edge_on_face(location, face)
        if edge is not None:
            return edge.index
    return None


def _get_seed_vert(bvh, ray_origin_local, ray_direction_local, bm, materials,
                   region, rv3d, obj_matrix, mouse_2d):
    """Get a seed vert from raycast when nothing is selected."""
    face, location, culled_element = _raycast_element_aware(
        bvh, ray_origin_local, ray_direction_local,
        bm, materials, region, rv3d, obj_matrix, mouse_2d,
        False, max_iterations=64
    )
    if culled_element is not None:
        return culled_element.index
    if face is not None and location is not None:
        vert = _nearest_vert_on_face(location, face)
        if vert is not None:
            return vert.index
    return None


class LEVELDESIGN_OT_select_linked(Operator):
    """Select linked geometry with backface-aware picking"""
    bl_idname = "leveldesign.select_linked"
    bl_label = "Select Linked"
    bl_options = {'REGISTER', 'UNDO'}

    normal_mode: bpy.props.EnumProperty(
        items=[
            ('NONE', "None", "Plain select linked"),
            ('EXPAND', "Expand", "Select linked by normal (expand)"),
            ('SHRINK', "Shrink", "Select linked by normal (shrink)"),
        ],
    )

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
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        select_mode = context.tool_settings.mesh_select_mode
        is_vert_mode = select_mode[0]
        is_edge_mode = select_mode[1]

        # Build raycast data
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        mouse_2d = Vector((float(coord[0]), float(coord[1])))
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        matrix_inv = obj.matrix_world.inverted()
        ray_origin_local = matrix_inv @ ray_origin
        ray_direction_local = (matrix_inv.to_3x3() @ view_vector).normalized()

        bvh = BVHTree.FromBMesh(bm)

        if self.normal_mode == 'NONE':
            # L key: plain linked flood-fill, mode-specific
            if is_edge_mode:
                return self._do_edge_linked(bm, me, bvh, ray_origin_local,
                                            ray_direction_local, region, rv3d,
                                            obj.matrix_world, mouse_2d, me.materials)
            if is_vert_mode:
                return self._do_vert_linked(bm, me, bvh, ray_origin_local,
                                            ray_direction_local, region, rv3d,
                                            obj.matrix_world, mouse_2d, me.materials)
            return self._do_face_linked(bm, me, bvh, ray_origin_local,
                                        ray_direction_local, obj.name, me.materials)

        # Ctrl+L / Ctrl+Shift+L: normal-constrained, works in all modes
        if self.normal_mode == 'EXPAND':
            return self._do_normal_expand(bm, me, bvh, ray_origin_local,
                                          ray_direction_local, obj.name,
                                          me.materials, select_mode)
        return self._do_normal_shrink(bm, me, bvh, ray_origin_local,
                                      ray_direction_local, obj.name,
                                      me.materials, select_mode)

    def _do_face_linked(self, bm, me, bvh, ray_origin_local, ray_direction_local,
                        obj_name, materials):
        """L key in face mode: deselect all, raycast for seed, flood-fill."""
        _reset_linked_state()

        # Always deselect and pick from raycast
        for f in bm.faces:
            f.select = False
        bm.select_flush_mode()

        seed = _get_seed_face(bvh, ray_origin_local, ray_direction_local,
                              bm, materials)
        if seed is None:
            bmesh.update_edit_mesh(me)
            return {'FINISHED'}

        result = _flood_fill_faces(bm, {seed})
        _select_face_indices(bm, result)
        bmesh.update_edit_mesh(me)
        return {'FINISHED'}

    def _is_seed_face_selected(self, bm, select_mode):
        """Check if the stored seed face is effectively selected in current mode."""
        seed = _linked_state['seed_face']
        if seed is None:
            return False
        face = bm.faces[seed]
        if select_mode[2]:  # face mode
            return face.select
        # Edge/vert mode: seed face is "selected" if all its verts are selected
        return all(v.select for v in face.verts)

    def _apply_normal_selection(self, bm, face_indices, select_mode):
        """Apply face-based normal result to the current selection mode."""
        if select_mode[2]:  # face mode
            _select_face_indices(bm, face_indices)
        elif select_mode[1]:  # edge mode
            edge_indices = set()
            for fi in face_indices:
                for edge in bm.faces[fi].edges:
                    edge_indices.add(edge.index)
            for e in bm.edges:
                e.select = (e.index in edge_indices)
            bm.select_flush_mode()
        else:  # vert mode
            vert_indices = set()
            for fi in face_indices:
                for vert in bm.faces[fi].verts:
                    vert_indices.add(vert.index)
            for v in bm.verts:
                v.select = (v.index in vert_indices)
            bm.select_flush_mode()

    def _deselect_all(self, bm):
        """Deselect all geometry."""
        for v in bm.verts:
            v.select = False
        for e in bm.edges:
            e.select = False
        for f in bm.faces:
            f.select = False
        bm.select_flush_mode()

    def _do_normal_expand(self, bm, me, bvh, ray_origin_local,
                           ray_direction_local, obj_name, materials, select_mode):
        """Ctrl+L: select linked by normal, expanding angle. Works in all modes."""
        hover_face = _get_seed_face(bvh, ray_origin_local, ray_direction_local,
                                    bm, materials)
        if hover_face is None:
            _reset_linked_state()
            self._deselect_all(bm)
            bmesh.update_edit_mesh(me)
            return {'FINISHED'}

        # Check if hovering over the stored seed face and it's still selected
        is_continuation = (
            _linked_state['object_name'] == obj_name
            and _linked_state['seed_face'] == hover_face
            and self._is_seed_face_selected(bm, select_mode)
        )

        if is_continuation:
            # Find the exact angle to the next unselected neighbor
            next_angle = _find_next_expand_angle(
                bm, _linked_state['result'], _linked_state['seed_normal']
            )
            if next_angle is None:
                return {'FINISHED'}
            _linked_state['angle_history'].append(_linked_state['angle'])
            _linked_state['angle'] = next_angle + _ANGLE_EPSILON
        else:
            # New operation from hovered face
            _linked_state['object_name'] = obj_name
            _linked_state['seed_face'] = hover_face
            _linked_state['seed_normal'] = bm.faces[hover_face].normal.copy()
            _linked_state['angle'] = 0.1
            _linked_state['angle_history'] = []

        seeds = {_linked_state['seed_face']: _linked_state['seed_normal']}
        result = _flood_fill_faces_by_normal(bm, seeds, _linked_state['angle'])
        _linked_state['result'] = result
        self._apply_normal_selection(bm, result, select_mode)
        bmesh.update_edit_mesh(me)
        return {'FINISHED'}

    def _do_normal_shrink(self, bm, me, bvh, ray_origin_local,
                           ray_direction_local, obj_name, materials, select_mode):
        """Ctrl+Shift+L: select linked by normal, shrinking angle. Works in all modes."""
        hover_face = _get_seed_face(bvh, ray_origin_local, ray_direction_local,
                                    bm, materials)
        if hover_face is None:
            return {'FINISHED'}

        is_continuation = (
            _linked_state['object_name'] == obj_name
            and _linked_state['seed_face'] == hover_face
            and self._is_seed_face_selected(bm, select_mode)
        )

        if not is_continuation or not _linked_state['angle_history']:
            return {'FINISHED'}

        # Pop back to previous angle
        _linked_state['angle'] = _linked_state['angle_history'].pop()

        seeds = {_linked_state['seed_face']: _linked_state['seed_normal']}
        result = _flood_fill_faces_by_normal(bm, seeds, _linked_state['angle'])
        _linked_state['result'] = result
        self._apply_normal_selection(bm, result, select_mode)
        bmesh.update_edit_mesh(me)
        return {'FINISHED'}

    def _do_edge_linked(self, bm, me, bvh, ray_origin_local, ray_direction_local,
                        region, rv3d, obj_matrix, mouse_2d, materials):
        """Edge mode: deselect all, raycast for seed, flood-fill linked edges."""
        _reset_linked_state()

        # Always deselect and pick from raycast
        for e in bm.edges:
            e.select = False
        for v in bm.verts:
            v.select = False
        bm.select_flush_mode()

        seed = _get_seed_edge(bvh, ray_origin_local, ray_direction_local,
                              bm, materials, region, rv3d, obj_matrix, mouse_2d)
        if seed is None:
            bmesh.update_edit_mesh(me)
            return {'FINISHED'}

        result = _flood_fill_edges(bm, {seed})
        for e in bm.edges:
            e.select = (e.index in result)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(me)
        return {'FINISHED'}

    def _do_vert_linked(self, bm, me, bvh, ray_origin_local, ray_direction_local,
                        region, rv3d, obj_matrix, mouse_2d, materials):
        """Vert mode: deselect all, raycast for seed, flood-fill linked verts."""
        _reset_linked_state()

        # Always deselect and pick from raycast
        for v in bm.verts:
            v.select = False
        for e in bm.edges:
            e.select = False
        bm.select_flush_mode()

        seed = _get_seed_vert(bvh, ray_origin_local, ray_direction_local,
                              bm, materials, region, rv3d, obj_matrix, mouse_2d)
        if seed is None:
            bmesh.update_edit_mesh(me)
            return {'FINISHED'}

        result = _flood_fill_verts(bm, {seed})
        for v in bm.verts:
            v.select = (v.index in result)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(me)
        return {'FINISHED'}


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_select_linked)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    km = kc.keymaps.new(name='Mesh', space_type='EMPTY')

    # L — plain select linked
    kmi = km.keymap_items.new(
        LEVELDESIGN_OT_select_linked.bl_idname,
        'L', 'PRESS',
        head=True
    )
    kmi.properties.normal_mode = 'NONE'
    _addon_keymaps.append((km, kmi))

    # Ctrl+L — select linked by normal (expand)
    kmi = km.keymap_items.new(
        LEVELDESIGN_OT_select_linked.bl_idname,
        'L', 'PRESS',
        ctrl=True,
        head=True
    )
    kmi.properties.normal_mode = 'EXPAND'
    _addon_keymaps.append((km, kmi))

    # Ctrl+Shift+L — select linked by normal (shrink)
    kmi = km.keymap_items.new(
        LEVELDESIGN_OT_select_linked.bl_idname,
        'L', 'PRESS',
        ctrl=True,
        shift=True,
        head=True
    )
    kmi.properties.normal_mode = 'SHRINK'
    _addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    bpy.utils.unregister_class(LEVELDESIGN_OT_select_linked)
