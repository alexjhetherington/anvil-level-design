import math

import bpy
import bmesh
from bpy.types import Operator
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ...core.workspace_check import is_level_design_workspace
from .raycast import (
    is_face_backfacing,
    has_backface_culling_enabled,
)

# Screen-space pixel threshold for picking edges/verts on culled faces
_PICK_THRESHOLD_PX = 40
# Fan ray rings: (ray_count, radius_in_pixels)
_FAN_RINGS = [(8, 20), (16, 40)]
# Minimum cursor movement before Shift+click becomes paint selection.
_PAINT_DRAG_THRESHOLD_PX = 4
_PAINT_SAMPLE_SPACING_PX = 10


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


def _compute_front_face_plane(face, region, rv3d, obj_matrix):
    """Compute a clipping plane from a front face.

    Uses 3 vertices to define the plane.  When the face has more than 3
    vertices (quads / ngons that may not be planar), the 3 vertices closest
    to the camera are chosen so the plane best represents what the viewer
    actually sees.

    Returns (plane_point_world, plane_normal_world) with the normal pointing
    toward the camera.
    """
    verts_world = [(v, obj_matrix @ v.co) for v in face.verts]

    if len(verts_world) <= 3:
        points = [vw for _, vw in verts_world]
    else:
        view_origin = view3d_utils.region_2d_to_origin_3d(
            region, rv3d,
            (region.width / 2, region.height / 2)
        )
        verts_world.sort(key=lambda pair: (pair[1] - view_origin).length_squared)
        points = [verts_world[i][1] for i in range(3)]

    edge1 = points[1] - points[0]
    edge2 = points[2] - points[0]
    plane_normal = edge1.cross(edge2).normalized()

    # Ensure the normal points toward the camera (same side as the view)
    view_origin = view3d_utils.region_2d_to_origin_3d(
        region, rv3d,
        (region.width / 2, region.height / 2)
    )
    if (view_origin - points[0]).dot(plane_normal) < 0:
        plane_normal = -plane_normal

    return points[0], plane_normal


def _is_element_behind_plane(elem, is_edge_mode, obj_matrix, plane_point, plane_normal):
    """Check if an edge/vert is behind a plane.

    For vertices: behind if the world-space position is on the negative side.
    For edges: behind if *both* verts are on the negative side.
    """
    if is_edge_mode:
        for v in elem.verts:
            if (obj_matrix @ v.co - plane_point).dot(plane_normal) >= 0:
                return False
        return True
    else:
        return (obj_matrix @ elem.co - plane_point).dot(plane_normal) < 0


def _screen_nearest_loose_vert(bm, region, rv3d, obj_matrix, mouse_2d,
                               ray_origin_local, ray_direction_local, clip_plane):
    """Find the closest vertex that is not attached to any face."""
    best_vert = None
    best_screen_dist = float('inf')
    best_depth = float('inf')

    bm.verts.ensure_lookup_table()

    for vert in bm.verts:
        if vert.hide:
            continue
        if len(vert.link_faces) > 0:
            continue

        screen_co = view3d_utils.location_3d_to_region_2d(
            region, rv3d, obj_matrix @ vert.co
        )
        if screen_co is None:
            continue

        screen_dist = (mouse_2d - screen_co).length
        if screen_dist > _PICK_THRESHOLD_PX:
            continue

        depth = (vert.co - ray_origin_local).dot(ray_direction_local)
        if depth < 0:
            continue

        if clip_plane is not None:
            if _is_element_behind_plane(
                    vert, False, obj_matrix, clip_plane[0], clip_plane[1]
            ):
                continue

        if (screen_dist < best_screen_dist
                or (abs(screen_dist - best_screen_dist) < 1e-6
                    and depth < best_depth)):
            best_screen_dist = screen_dist
            best_depth = depth
            best_vert = vert

    return best_vert, best_screen_dist


def _is_culled_backface(face, ray_direction_local, materials):
    """Check if a face is a backface with culling enabled."""
    return (is_face_backfacing(face.normal, ray_direction_local)
            and has_backface_culling_enabled(face.material_index, materials))


def _remove_from_select_history(bm, element):
    """Remove an element from selection history if it is present."""
    try:
        bm.select_history.remove(element)
    except (ReferenceError, ValueError):
        pass


def _activate_selection_element(bm, element):
    """Make an explicitly selected element Blender's active edit selection."""
    _remove_from_select_history(bm, element)
    bm.select_history.add(element)


def _select_active_element(bm, element, selected):
    """Select/deselect one element and keep Blender's active element in sync."""
    element.select = selected
    if selected:
        _activate_selection_element(bm, element)
    else:
        _remove_from_select_history(bm, element)


def _selected_elements_for_mode(bm, select_mode):
    """Return selected elements for the active mesh select mode."""
    if select_mode[2]:
        return [face for face in bm.faces if face.select]
    if select_mode[1]:
        return [edge for edge in bm.edges if edge.select]
    if select_mode[0]:
        return [vert for vert in bm.verts if vert.select]
    return []


def _activate_single_remaining_selected_element(bm, select_mode):
    """Promote the sole remaining selected edit element to active/primary."""
    selected = _selected_elements_for_mode(bm, select_mode)
    if len(selected) != 1:
        return False

    element = selected[0]
    _activate_selection_element(bm, element)
    if isinstance(element, bmesh.types.BMFace):
        bm.faces.active = element
    return True


def _activate_single_remaining_selected_object(view_layer):
    """Promote the sole remaining selected object to active/primary."""
    selected = [obj for obj in view_layer.objects if obj.select_get()]
    if len(selected) != 1:
        return False

    view_layer.objects.active = selected[0]
    return True


def _edit_selection_has_any(bm):
    return (
        any(vert.select for vert in bm.verts)
        or any(edge.select for edge in bm.edges)
        or any(face.select for face in bm.faces)
    )


def _object_selection_has_any(view_layer):
    return any(obj.select_get() for obj in view_layer.objects)


def _clear_edit_selection(bm):
    """Clear BMesh edit selection without invoking Blender's select operator."""
    for vert in bm.verts:
        vert.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False
    bm.select_history.clear()


def _clear_object_selection(view_layer):
    """Clear object selection without invoking Blender's select operator."""
    for obj in view_layer.objects:
        if obj.select_get():
            obj.select_set(False)


class _VisiblePick:
    """Visible mesh pick result for the current select mode."""

    def __init__(self, me, bm, region, rv3d, mouse_2d,
                 select_mode, face, location, culled_element, hidden_face_hit):
        self.me = me
        self.bm = bm
        self.region = region
        self.rv3d = rv3d
        self.mouse_2d = mouse_2d
        self.select_mode = select_mode
        self.face = face
        self.location = location
        self.culled_element = culled_element
        self.hidden_face_hit = hidden_face_hit

    @property
    def is_vert_mode(self):
        return self.select_mode[0]

    @property
    def is_edge_mode(self):
        return self.select_mode[1]

    @property
    def is_face_mode(self):
        return self.select_mode[2]

    def target_index(self):
        element = self.target_element()
        return element.index if element is not None else -1

    def target_element(self):
        if self.is_face_mode:
            return self.face
        if self.culled_element is not None:
            return self.culled_element
        if self.face is None:
            return None
        if self.is_edge_mode:
            return _nearest_edge_on_face(self.location, self.face)
        if self.is_vert_mode:
            return _nearest_vert_on_face(self.location, self.face)
        return None

    def select_target_for_shortest_path(self):
        target = self.target_element()
        if target is None:
            return False

        _select_active_element(self.bm, target, True)
        if isinstance(target, bmesh.types.BMEdge):
            for vert in target.verts:
                vert.select = True
        elif isinstance(target, bmesh.types.BMFace):
            self.bm.faces.active = target

        self.bm.select_flush_mode()
        bmesh.update_edit_mesh(self.me)
        return True


class _MappedBVH:
    """BVH wrapper that translates polygon hits back to BMesh face indices."""

    def __init__(self, bvh, face_indices):
        self._bvh = bvh
        self._face_indices = face_indices

    def ray_cast(self, origin, direction):
        location, normal, face_index, distance = self._bvh.ray_cast(origin, direction)
        if face_index is not None:
            face_index = self._face_indices[face_index]
        return location, normal, face_index, distance


def _build_visible_face_bvh(bm):
    """Build a BVH from edit faces that are visible to selection."""
    bm.faces.ensure_lookup_table()

    visible_faces = []
    face_indices = []
    for face_index, face in enumerate(bm.faces):
        if face.hide:
            continue
        visible_faces.append(face)
        face_indices.append(face_index)

    vertices = []
    vert_indices = {}
    polygons = []
    for face in visible_faces:
        polygon = []
        for vert in face.verts:
            vert_index = vert_indices.get(vert)
            if vert_index is None:
                vert_index = len(vertices)
                vert_indices[vert] = vert_index
                vertices.append(vert.co.copy())
            polygon.append(vert_index)
        polygons.append(polygon)

    if not polygons:
        return None

    return _MappedBVH(BVHTree.FromPolygons(vertices, polygons), face_indices)


def _mouse_2d_from_event(event):
    return Vector((float(event.mouse_region_x), float(event.mouse_region_y)))


def _ray_local_from_mouse(obj, region, rv3d, mouse_2d):
    coord = (mouse_2d.x, mouse_2d.y)
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

    matrix_inv = obj.matrix_world.inverted()
    ray_origin_local = matrix_inv @ ray_origin
    ray_direction_local = (matrix_inv.to_3x3() @ view_vector).normalized()

    return ray_origin_local, ray_direction_local


def _resolve_visible_pick(obj, select_mode, region, rv3d, mouse_2d):
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()

    bvh = BVHTree.FromBMesh(bm)

    pick = _resolve_visible_pick_from_bvh(
        obj, me, bm, bvh, select_mode, region, rv3d, mouse_2d
    )
    if not pick.hidden_face_hit:
        return pick

    filtered_bvh = _build_visible_face_bvh(bm)
    return _resolve_visible_pick_from_bvh(
        obj, me, bm, filtered_bvh, select_mode, region, rv3d, mouse_2d
    )


def _raycast_bvh_skip_hidden_and_backfaces(
        bvh, ray_origin_local, ray_direction_local, bm, materials, max_iterations):
    """Raycast while reporting whether a hidden edit face blocked the path."""
    if bvh is None:
        return None, None, None, None, False

    origin = ray_origin_local.copy()
    total_distance = 0.0
    epsilon = 0.0001
    hidden_face_hit = False

    for _ in range(max_iterations):
        location, normal, face_index, distance = bvh.ray_cast(
            origin, ray_direction_local
        )

        if face_index is None:
            return None, None, None, None, hidden_face_hit

        face = bm.faces[face_index]

        if face.hide:
            hidden_face_hit = True
            total_distance += distance + epsilon
            origin = origin + ray_direction_local * (distance + epsilon)
            continue

        if _is_culled_backface(face, ray_direction_local, materials):
            total_distance += distance + epsilon
            origin = origin + ray_direction_local * (distance + epsilon)
            continue

        return location, normal, face_index, total_distance + distance, hidden_face_hit

    return None, None, None, None, hidden_face_hit


def _resolve_visible_pick_from_bvh(
        obj, me, bm, bvh, select_mode, region, rv3d, mouse_2d):
    ray_origin_local, ray_direction_local = _ray_local_from_mouse(
        obj, region, rv3d, mouse_2d
    )

    is_edge_mode = select_mode[1]
    is_face_mode = select_mode[2]

    culled_element = None
    hidden_face_hit = False

    if is_face_mode:
        location, normal, face_index, distance, hidden_face_hit = (
            _raycast_bvh_skip_hidden_and_backfaces(
                bvh, ray_origin_local, ray_direction_local,
                bm, me.materials, max_iterations=64
            )
        )
        face = bm.faces[face_index] if face_index is not None else None
    else:
        face, location, culled_element, hidden_face_hit = _raycast_element_aware(
            bvh, ray_origin_local, ray_direction_local,
            bm, me.materials, region, rv3d, obj.matrix_world, mouse_2d,
            is_edge_mode, max_iterations=64
        )

    return _VisiblePick(
        me, bm, region, rv3d, mouse_2d,
        select_mode, face, location, culled_element, hidden_face_hit
    )


def _selection_history_key(element):
    """Return a stable key for a bmesh selection-history element."""
    if isinstance(element, bmesh.types.BMVert):
        return ('VERT', element.index)
    if isinstance(element, bmesh.types.BMEdge):
        return ('EDGE', element.index)
    if isinstance(element, bmesh.types.BMFace):
        return ('FACE', element.index)
    return None


def _selection_element_from_key(bm, key):
    """Resolve a saved selection-history key back to its current element."""
    if key is None:
        return None
    elem_type, index = key
    if elem_type == 'VERT':
        bm.verts.ensure_lookup_table()
        if 0 <= index < len(bm.verts):
            return bm.verts[index]
    elif elem_type == 'EDGE':
        bm.edges.ensure_lookup_table()
        if 0 <= index < len(bm.edges):
            return bm.edges[index]
    elif elem_type == 'FACE':
        bm.faces.ensure_lookup_table()
        if 0 <= index < len(bm.faces):
            return bm.faces[index]
    return None


def _selection_history_keys(bm):
    """Save bmesh selection history by element type and index."""
    return [
        key for key in (_selection_history_key(element) for element in bm.select_history)
        if key is not None
    ]


def _restore_selection_history(bm, history_keys):
    """Restore bmesh selection history from saved element keys."""
    bm.select_history.clear()
    for key in history_keys:
        element = _selection_element_from_key(bm, key)
        if element is not None and element.select:
            bm.select_history.add(element)


def _snapshot_edit_selection(bm):
    """Save edit selection state for cancelling a paint drag."""
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    active_face_index = -1
    if bm.faces.active is not None and bm.faces.active.is_valid:
        active_face_index = bm.faces.active.index

    return (
        {vert.index for vert in bm.verts if vert.select},
        {edge.index for edge in bm.edges if edge.select},
        {face.index for face in bm.faces if face.select},
        _selection_history_keys(bm),
        active_face_index,
    )


def _restore_edit_selection(bm, snapshot):
    """Restore edit selection state saved by _snapshot_edit_selection()."""
    selected_verts, selected_edges, selected_faces, history_keys, active_face_index = snapshot

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    for vert in bm.verts:
        vert.select = vert.index in selected_verts
    for edge in bm.edges:
        edge.select = edge.index in selected_edges
    for face in bm.faces:
        face.select = face.index in selected_faces

    bm.select_flush_mode()
    _restore_selection_history(bm, history_keys)

    if 0 <= active_face_index < len(bm.faces):
        active_face = bm.faces[active_face_index]
        if active_face.select:
            bm.faces.active = active_face


def _collect_fan_faces(bvh, bm, materials, region, rv3d, obj_matrix,
                       mouse_2d, max_iterations):
    """Cast fan rays around the cursor to find nearby faces.

    Returns (culled_faces, front_faces) — two sets of face indices.
    """
    culled_faces = set()
    front_faces = set()
    hidden_face_hit = False
    epsilon = 0.0001
    matrix_inv = obj_matrix.inverted()
    rot_inv = matrix_inv.to_3x3()

    for ray_count, radius in _FAN_RINGS:
        for i in range(ray_count):
            angle = (2.0 * math.pi * i) / ray_count
            offset_x = math.cos(angle) * radius
            offset_y = math.sin(angle) * radius
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
                if face.hide:
                    hidden_face_hit = True
                    origin = origin + dir_local * (distance + epsilon)
                    continue

                if _is_culled_backface(face, dir_local, materials):
                    culled_faces.add(face_index)
                    origin = origin + dir_local * (distance + epsilon)
                    continue
                # Front face — collect it, then stop this fan ray
                front_faces.add(face_index)
                break

    return culled_faces, front_faces, hidden_face_hit


def _raycast_element_aware(bvh, ray_origin_local, ray_direction_local,
                            bm, materials, region, rv3d, obj_matrix, mouse_2d,
                            is_edge_mode, max_iterations):
    """Raycast that skips culled backfaces but catches nearby edges/verts.

    First casts the center ray, checking culled faces along the way. Then casts
    fan rays around the cursor to find faces the center ray missed (e.g. when
    the cursor is just past the edge of a face). The best element across all
    discovered faces is selected if within screen threshold.

    Returns (face, location, element) where element is the nearby BMEdge/BMVert
    found in screen space, or None if the final hit was a front face.
    """
    origin = ray_origin_local.copy()
    epsilon = 0.0001

    # Track the best element found across center + fan rays
    best_culled_elem = None
    best_culled_dist = float('inf')
    best_culled_face = None
    hidden_face_hit = False

    # Track culled face indices seen by the center ray
    center_culled_faces = set()

    # Front-face result from center ray (if any)
    front_face = None
    front_location = None

    if bvh is None:
        if not is_edge_mode:
            loose_vert, screen_dist = _screen_nearest_loose_vert(
                bm, region, rv3d, obj_matrix, mouse_2d,
                ray_origin_local, ray_direction_local, None
            )
            if loose_vert is not None and screen_dist <= _PICK_THRESHOLD_PX:
                return None, None, loose_vert, hidden_face_hit
        return None, None, None, hidden_face_hit

    # Phase 1: center ray — walk through hits
    for _ in range(max_iterations):
        location, normal, face_index, distance = bvh.ray_cast(origin, ray_direction_local)

        if face_index is None:
            break

        face = bm.faces[face_index]

        if face.hide:
            hidden_face_hit = True
            origin = origin + ray_direction_local * (distance + epsilon)
            continue

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

        # Check elements on the front face too so it competes fairly
        elem, screen_dist = _check_culled_face_element(
            face, is_edge_mode, region, rv3d, obj_matrix, mouse_2d
        )
        if elem is not None and screen_dist < best_culled_dist:
            best_culled_dist = screen_dist
            best_culled_elem = elem
            best_culled_face = face
        break

    # Compute clipping plane from front face to filter elements behind it
    clip_plane = None
    if front_face is not None:
        plane_point, plane_normal = _compute_front_face_plane(
            front_face, region, rv3d, obj_matrix
        )
        clip_plane = (plane_point, plane_normal)

    # Phase 2: fan rays — find faces the center ray missed
    fan_culled, fan_front, fan_hidden_face_hit = _collect_fan_faces(
        bvh, bm, materials, region, rv3d, obj_matrix,
        mouse_2d, max_iterations=8
    )
    hidden_face_hit = hidden_face_hit or fan_hidden_face_hit

    # Combine new culled and front faces not already handled by center ray
    new_faces = (fan_culled | fan_front) - center_culled_faces
    if front_face is not None:
        new_faces.discard(front_face.index)

    for face_index in new_faces:
        face = bm.faces[face_index]
        elem, screen_dist = _check_culled_face_element(
            face, is_edge_mode, region, rv3d, obj_matrix, mouse_2d
        )
        if elem is not None:
            if clip_plane is not None:
                if _is_element_behind_plane(elem, is_edge_mode, obj_matrix, clip_plane[0], clip_plane[1]):
                    continue
            if screen_dist < best_culled_dist:
                best_culled_dist = screen_dist
                best_culled_elem = elem
                best_culled_face = face

    if not is_edge_mode:
        loose_vert, screen_dist = _screen_nearest_loose_vert(
            bm, region, rv3d, obj_matrix, mouse_2d,
            ray_origin_local, ray_direction_local, clip_plane
        )
        if loose_vert is not None and screen_dist < best_culled_dist:
            best_culled_dist = screen_dist
            best_culled_elem = loose_vert
            best_culled_face = None

    # Decide: use nearby element if within threshold, otherwise front face
    if best_culled_elem is not None and best_culled_dist <= _PICK_THRESHOLD_PX:
        return best_culled_face, None, best_culled_elem, hidden_face_hit

    if front_face is not None:
        return front_face, front_location, None, hidden_face_hit

    return None, None, None, hidden_face_hit


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


class LEVELDESIGN_OT_visible_select(Operator):
    """Select visible edit mesh elements through culled surfaces"""
    bl_idname = "leveldesign.visible_select"
    bl_label = "Visible Select (Edit Mode)"
    bl_options = {'REGISTER', 'UNDO'}

    extend: bpy.props.BoolProperty()
    loop: bpy.props.BoolProperty()

    _paint_obj = None
    _paint_bvh = None
    _paint_filtered_bvh = None
    _paint_select_mode = None
    _paint_start_mouse = None
    _paint_prev_mouse = None
    _paint_started = False
    _paint_visited = None
    _paint_selection_snapshot = None

    @classmethod
    def poll(cls, context):
        if not is_level_design_workspace():
            return False
        obj = context.object
        return (obj is not None
                and obj.type == 'MESH'
                and context.mode == 'EDIT_MESH')

    def invoke(self, context, event):
        if self.extend and not self.loop:
            return self._invoke_paint_select(context, event)

        return self._do_click_select(
            context, _mouse_2d_from_event(event)
        )

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            return self._modal_paint_select_move(context, event)

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            return self._modal_paint_select_finish(context)

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return self._modal_paint_select_cancel(context)

        return {'RUNNING_MODAL'}

    def _do_click_select(self, context, mouse_2d):
        obj = context.object
        pick = _resolve_visible_pick(
            obj, tuple(context.tool_settings.mesh_select_mode),
            context.region, context.region_data, mouse_2d
        )
        me = pick.me
        bm = pick.bm
        face = pick.face
        culled_element = pick.culled_element

        if face is None and culled_element is None:
            if self.extend or not _edit_selection_has_any(bm):
                return {'CANCELLED'}
            _clear_edit_selection(bm)
            bm.select_flush_mode()
            bmesh.update_edit_mesh(me)
            return {'FINISHED'}

        hit_point = pick.location

        # Alt+click: loop select
        if self.loop:
            if face is None:
                return {'CANCELLED'}

            # Determine the target edge for loop select
            if culled_element is not None:
                if pick.is_edge_mode:
                    loop_edge = culled_element
                else:
                    # For face/vert mode, culled_element isn't an edge -
                    # find the nearest edge on the face instead
                    loop_edge, _ = _screen_nearest_edge_on_face(
                        face, pick.region, pick.rv3d,
                        obj.matrix_world, pick.mouse_2d
                    )
            else:
                loop_edge = None

            if pick.is_face_mode:
                _do_face_loop_select(bm, me, face, hit_point, self.extend, loop_edge)
            else:
                _do_edge_loop_select(bm, me, face, hit_point, self.extend, loop_edge)
            return {'FINISHED'}

        # Plain or Shift click
        if not self.extend:
            _clear_edit_selection(bm)

        unselected_target = False

        if pick.is_face_mode:
            new_state = not face.select if self.extend else True
            _select_active_element(bm, face, new_state)
            if new_state:
                bm.faces.active = face
            else:
                unselected_target = True
        elif culled_element is not None:
            # Picked an edge/vert from a screen-space element candidate.
            if pick.is_edge_mode:
                new_state = not culled_element.select if self.extend else True
                _select_active_element(bm, culled_element, new_state)
                for v in culled_element.verts:
                    v.select = new_state
                if not new_state:
                    unselected_target = True
            else:
                new_state = not culled_element.select if self.extend else True
                _select_active_element(bm, culled_element, new_state)
                if not new_state:
                    unselected_target = True
        elif pick.is_edge_mode:
            edge = _nearest_edge_on_face(hit_point, face)
            if edge is not None:
                new_state = not edge.select if self.extend else True
                _select_active_element(bm, edge, new_state)
                for v in edge.verts:
                    v.select = new_state
                if not new_state:
                    unselected_target = True
        elif pick.is_vert_mode:
            vert = _nearest_vert_on_face(hit_point, face)
            if vert is not None:
                new_state = not vert.select if self.extend else True
                _select_active_element(bm, vert, new_state)
                if not new_state:
                    unselected_target = True

        bm.select_flush_mode()
        if unselected_target:
            _activate_single_remaining_selected_element(bm, pick.select_mode)
        bmesh.update_edit_mesh(me)

        return {'FINISHED'}

    def _invoke_paint_select(self, context, event):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()

        self._paint_obj = obj
        self._paint_bvh = BVHTree.FromBMesh(bm)
        self._paint_filtered_bvh = None
        self._paint_select_mode = tuple(context.tool_settings.mesh_select_mode)
        self._paint_start_mouse = _mouse_2d_from_event(event)
        self._paint_prev_mouse = self._paint_start_mouse.copy()
        self._paint_started = False
        self._paint_visited = set()
        self._paint_selection_snapshot = _snapshot_edit_selection(bm)

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _modal_paint_select_move(self, context, event):
        curr_mouse = _mouse_2d_from_event(event)
        start_delta = curr_mouse - self._paint_start_mouse

        if (not self._paint_started
                and start_delta.length < _PAINT_DRAG_THRESHOLD_PX):
            self._paint_prev_mouse = curr_mouse
            return {'RUNNING_MODAL'}

        if not self._paint_started:
            self._paint_started = True
            self._paint_visible_selection_sample(
                context.region, context.region_data, self._paint_start_mouse
            )

        self._paint_visible_selection_segment(
            context.region, context.region_data, self._paint_prev_mouse, curr_mouse
        )
        self._paint_prev_mouse = curr_mouse
        self._flush_paint_selection(context)

        return {'RUNNING_MODAL'}

    def _modal_paint_select_finish(self, context):
        if not self._paint_started:
            result = self._do_click_select(context, self._paint_start_mouse)
            self._clear_paint_select_state()
            return result

        self._flush_paint_selection(context)
        self._clear_paint_select_state()
        return {'FINISHED'}

    def _modal_paint_select_cancel(self, context):
        if self._paint_selection_snapshot is not None:
            me = self._paint_obj.data
            bm = bmesh.from_edit_mesh(me)
            _restore_edit_selection(bm, self._paint_selection_snapshot)
            bmesh.update_edit_mesh(me)

        self._clear_paint_select_state()
        return {'CANCELLED'}

    def _paint_visible_selection_segment(self, region, rv3d, prev_mouse, curr_mouse):
        delta = curr_mouse - prev_mouse
        dist = delta.length

        if dist <= 0:
            return

        steps = max(1, int(dist / _PAINT_SAMPLE_SPACING_PX))
        for i in range(1, steps + 1):
            t = i / steps
            self._paint_visible_selection_sample(region, rv3d, prev_mouse + delta * t)

    def _paint_visible_selection_sample(self, region, rv3d, mouse_2d):
        me = self._paint_obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()

        pick = _resolve_visible_pick_from_bvh(
            self._paint_obj, me, bm, self._paint_bvh,
            self._paint_select_mode, region, rv3d, mouse_2d
        )
        if pick.hidden_face_hit:
            if self._paint_filtered_bvh is None:
                self._paint_filtered_bvh = _build_visible_face_bvh(bm)
            pick = _resolve_visible_pick_from_bvh(
                self._paint_obj, me, bm, self._paint_filtered_bvh,
                self._paint_select_mode, region, rv3d, mouse_2d
            )
        target = pick.target_element()
        if target is None:
            return

        target_key = _selection_history_key(target)
        if target_key is not None:
            if target_key in self._paint_visited:
                return
            self._paint_visited.add(target_key)

        _select_active_element(bm, target, True)
        if isinstance(target, bmesh.types.BMEdge):
            for vert in target.verts:
                vert.select = True
        elif isinstance(target, bmesh.types.BMFace):
            bm.faces.active = target

    def _flush_paint_selection(self, context):
        me = self._paint_obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(me)
        context.area.tag_redraw()

    def _clear_paint_select_state(self):
        self._paint_obj = None
        self._paint_bvh = None
        self._paint_filtered_bvh = None
        self._paint_select_mode = None
        self._paint_start_mouse = None
        self._paint_prev_mouse = None
        self._paint_started = False
        self._paint_visited = None
        self._paint_selection_snapshot = None


class LEVELDESIGN_OT_visible_shortest_path_pick(Operator):
    """Shortest path select visible edit mesh elements through culled surfaces"""
    bl_idname = "leveldesign.visible_shortest_path_pick"
    bl_label = "Visible Shortest Path Pick"
    bl_options = {'REGISTER', 'UNDO'}

    use_fill: bpy.props.BoolProperty()

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
        pick = _resolve_visible_pick(
            obj, tuple(context.tool_settings.mesh_select_mode),
            context.region, context.region_data, _mouse_2d_from_event(event)
        )

        if pick.is_vert_mode:
            target_index = pick.target_index()
            if target_index < 0:
                return {'CANCELLED'}
            bpy.ops.mesh.shortest_path_pick(
                index=target_index,
                use_fill=self.use_fill
            )
            return {'FINISHED'}

        if not pick.select_target_for_shortest_path():
            return {'CANCELLED'}

        bpy.ops.mesh.shortest_path_select(
            use_fill=self.use_fill
        )
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


# Screen-space pixel threshold for clicking origin-picked objects
_OBJECT_PICK_THRESHOLD_PX = 20


def _mesh_has_raycastable_surface(obj, depsgraph):
    obj_eval = obj.evaluated_get(depsgraph)
    me_eval = obj_eval.to_mesh()
    if me_eval is None:
        return False

    try:
        return len(me_eval.polygons) > 0
    finally:
        obj_eval.to_mesh_clear()


def _is_origin_pickable_object(obj, depsgraph):
    if obj.type != 'MESH':
        return True

    # Iteras Tools 3 Vertex Light objects are mesh-based handlers with one
    # vertex and no faces, so scene.ray_cast() cannot hit them like normal mesh.
    return not _mesh_has_raycastable_surface(obj, depsgraph)


def _find_origin_pick_object_at_cursor(
        view_layer_objects, depsgraph, region, rv3d, mouse_2d, ray_origin, view_vector):
    """Find the nearest visible origin-picked object close to the cursor.

    Projects each object's world origin to screen space and picks the
    closest one within _OBJECT_PICK_THRESHOLD_PX.

    Returns (object, depth) or (None, float('inf')).
    Depth is distance along the view ray from the camera.
    """
    best_obj = None
    best_screen_dist = float('inf')
    best_depth = float('inf')

    for obj in view_layer_objects:
        if obj.hide_get():
            continue
        if not obj.visible_get():
            continue

        screen_co = view3d_utils.location_3d_to_region_2d(region, rv3d, obj.matrix_world.translation)
        if screen_co is None:
            continue

        screen_dist = (mouse_2d - screen_co).length
        if screen_dist > _OBJECT_PICK_THRESHOLD_PX:
            continue

        to_obj = obj.matrix_world.translation - ray_origin
        depth = to_obj.dot(view_vector)
        if depth < 0:
            continue

        if not _is_origin_pickable_object(obj, depsgraph):
            continue

        if screen_dist < best_screen_dist:
            best_screen_dist = screen_dist
            best_obj = obj
            best_depth = depth

    return best_obj, best_depth


class LEVELDESIGN_OT_visible_object_select(Operator):
    """Select visible objects through culled surfaces"""
    bl_idname = "leveldesign.visible_object_select"
    bl_label = "Visible Select (Object Mode)"
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
        mouse_2d = Vector((float(coord[0]), float(coord[1])))
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        depsgraph = context.evaluated_depsgraph_get()
        hit, location, normal, face_index, obj, matrix = raycast_scene_skip_backfaces(
            depsgraph, context.scene, ray_origin, view_vector, max_iterations=64
        )

        # Also check objects selected by origin proximity (lights, cameras,
        # empties, and mesh handlers without raycastable surfaces).
        origin_pick_obj, origin_pick_depth = _find_origin_pick_object_at_cursor(
            context.view_layer.objects, depsgraph, region, rv3d,
            mouse_2d, ray_origin, view_vector
        )

        # Decide which hit wins
        select_obj = None
        if hit and origin_pick_obj is not None:
            mesh_depth = (location - ray_origin).dot(view_vector)
            if origin_pick_depth < mesh_depth:
                select_obj = origin_pick_obj
            else:
                select_obj = _resolve_select_target(depsgraph, obj, matrix)
        elif hit:
            select_obj = _resolve_select_target(depsgraph, obj, matrix)
        elif origin_pick_obj is not None:
            select_obj = origin_pick_obj

        if select_obj is None:
            if self.extend or not _object_selection_has_any(context.view_layer):
                return {'CANCELLED'}
            _clear_object_selection(context.view_layer)
            return {'FINISHED'}

        if not self.extend:
            _clear_object_selection(context.view_layer)

        new_state = not select_obj.select_get() if self.extend else True
        select_obj.select_set(new_state)
        if new_state:
            context.view_layer.objects.active = select_obj
        else:
            _activate_single_remaining_selected_object(context.view_layer)

        return {'FINISHED'}


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_visible_select)
    bpy.utils.register_class(LEVELDESIGN_OT_visible_shortest_path_pick)
    bpy.utils.register_class(LEVELDESIGN_OT_visible_object_select)


def unregister():
    bpy.utils.unregister_class(LEVELDESIGN_OT_visible_object_select)
    bpy.utils.unregister_class(LEVELDESIGN_OT_visible_shortest_path_pick)
    bpy.utils.unregister_class(LEVELDESIGN_OT_visible_select)
