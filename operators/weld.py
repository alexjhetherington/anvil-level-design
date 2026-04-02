"""
Context Aware Weld - Operator and state management

Performs contextual weld actions based on the state left by previous operations
(e.g., cube cut, box builder). The weld action depends on context:
- Two distinct edge groups → Bridge Edge Loops
- One distinct edge group → Corridor (fill face + extrude)
- Box builder with no coplanar inward-facing overlap → Invert (flip normals)

Edit mode: weld state is stored in BMesh custom data layers, which participate
in Blender's edit-mode undo system.  On undo/redo the layers are restored
automatically — no geometry derivation needed.

Object mode: weld state is stored as Mesh custom properties (obj.data), which
participate in object-mode undo.
"""

import bpy
import bmesh
from mathutils import Vector

from ..core.logging import debug_log
from ..core.workspace_check import is_level_design_workspace
from ..core.geometry import are_verts_coplanar, compute_normal_from_verts
from ..core.uv_layers import get_render_active_uv_layer

from .texture_apply import set_uv_from_other_face


# BMesh layer names (all stored on vert 0)
_MODE_LAYER = "_aw_mode"    # int on verts — mode enum
_DEPTH_LAYER = "_aw_depth"  # float on verts
_DX_LAYER = "_aw_dx"        # float on verts — direction X
_DY_LAYER = "_aw_dy"        # float on verts — direction Y
_DZ_LAYER = "_aw_dz"        # float on verts — direction Z
_BPO_LAYER = "_aw_bpo"      # float on verts — back plane offset
_FACE_LAYER = "_aw_face"    # int on faces — 1 = box face for INVERT

# Cuboid params layers (FOLDED_PLANE only, all floats on vert 0, object local space)
_CUBOID_LAYERS = [
    "_aw_cox", "_aw_coy", "_aw_coz",   # cuboid origin
    "_aw_lxx", "_aw_lxy", "_aw_lxz",   # cuboid local_x
    "_aw_lyx", "_aw_lyy", "_aw_lyz",   # cuboid local_y
    "_aw_cdx", "_aw_cdy",              # cuboid dx, dy
]
_COPLANAR_LAYER = "_aw_copl"  # int on verts — bitmask of blocked side planes

# Mesh custom property prefix (object mode only)
_AW = "_aw_"

# Guard flag: True while the weld operator is executing, prevents the
# depsgraph handler from running redundant sync mid-operator.
_weld_op_running = False

# Guard flag: True after weld state is stored, consumed on the first
# depsgraph selection-change so that the very selection change caused by
# the operation that SET the weld doesn't immediately clear it.
_weld_just_stored = False

# Mode ↔ int mapping for the BMesh int layer
_MODE_TO_STR = {0: 'NONE', 1: 'BRIDGE', 2: 'CORRIDOR', 3: 'INVERT', 4: 'FOLDED_PLANE'}
_STR_TO_MODE = {'NONE': 0, 'BRIDGE': 1, 'CORRIDOR': 2, 'INVERT': 3, 'FOLDED_PLANE': 4}


# ---------------------------------------------------------------------------
# BMesh layer helpers (edit mode)
# ---------------------------------------------------------------------------

def _set_weld_on_bmesh(bm, mode, depth, direction, back_plane_offset,
                       box_faces, cuboid_params, coplanar_blocked):
    """Store weld state in BMesh custom data layers."""
    # Collect indices from saved refs BEFORE creating layers, because
    # adding a new layer can invalidate existing BMFace/BMEdge pointers.
    box_indices = set(f.index for f in box_faces) if box_faces else None

    # Ensure all layers exist before accessing elements
    mode_layer = bm.verts.layers.int.get(_MODE_LAYER) or bm.verts.layers.int.new(_MODE_LAYER)
    depth_layer = bm.verts.layers.float.get(_DEPTH_LAYER) or bm.verts.layers.float.new(_DEPTH_LAYER)
    dx_layer = bm.verts.layers.float.get(_DX_LAYER) or bm.verts.layers.float.new(_DX_LAYER)
    dy_layer = bm.verts.layers.float.get(_DY_LAYER) or bm.verts.layers.float.new(_DY_LAYER)
    dz_layer = bm.verts.layers.float.get(_DZ_LAYER) or bm.verts.layers.float.new(_DZ_LAYER)
    bpo_layer = bm.verts.layers.float.get(_BPO_LAYER) or bm.verts.layers.float.new(_BPO_LAYER)

    bm.verts.ensure_lookup_table()
    v0 = bm.verts[0]
    v0[mode_layer] = _STR_TO_MODE.get(mode, 0)
    v0[depth_layer] = depth
    v0[dx_layer] = direction[0]
    v0[dy_layer] = direction[1]
    v0[dz_layer] = direction[2]
    v0[bpo_layer] = back_plane_offset

    # Mark box faces (INVERT mode)
    if box_indices is not None:
        face_layer = bm.faces.layers.int.get(_FACE_LAYER) or bm.faces.layers.int.new(_FACE_LAYER)
        bm.faces.ensure_lookup_table()
        for f in bm.faces:
            f[face_layer] = 1 if f.index in box_indices else 0

    # Store cuboid params (FOLDED_PLANE mode, object local space)
    if cuboid_params is not None:
        origin, local_x, local_y, cdx, cdy = cuboid_params
        values = [
            origin[0], origin[1], origin[2],
            local_x[0], local_x[1], local_x[2],
            local_y[0], local_y[1], local_y[2],
            cdx, cdy,
        ]
        cuboid_layers = []
        for name in _CUBOID_LAYERS:
            layer = bm.verts.layers.float.get(name) or bm.verts.layers.float.new(name)
            cuboid_layers.append(layer)
        bm.verts.ensure_lookup_table()
        v0 = bm.verts[0]
        for layer, val in zip(cuboid_layers, values):
            v0[layer] = val

    # Store coplanar blocked bitmask (FOLDED_PLANE mode)
    if coplanar_blocked is not None:
        copl_layer = (bm.verts.layers.int.get(_COPLANAR_LAYER)
                      or bm.verts.layers.int.new(_COPLANAR_LAYER))
        bm.verts.ensure_lookup_table()
        bm.verts[0][copl_layer] = coplanar_blocked


def _get_weld_from_bmesh(bm):
    """Read weld parameters from BMesh layers.

    Returns (mode_str, depth, direction_tuple, back_plane_offset, cuboid_params,
    coplanar_blocked).
    cuboid_params is (origin, local_x, local_y, cdx, cdy) or None.
    coplanar_blocked is an int bitmask or 0.
    """
    mode_layer = bm.verts.layers.int.get(_MODE_LAYER)
    if mode_layer is None:
        return 'NONE', 0.0, (0.0, 0.0, 0.0), 0.0, None, 0

    bm.verts.ensure_lookup_table()
    v0 = bm.verts[0]
    mode = _MODE_TO_STR.get(v0[mode_layer], 'NONE')

    depth_layer = bm.verts.layers.float.get(_DEPTH_LAYER)
    dx_layer = bm.verts.layers.float.get(_DX_LAYER)
    dy_layer = bm.verts.layers.float.get(_DY_LAYER)
    dz_layer = bm.verts.layers.float.get(_DZ_LAYER)
    bpo_layer = bm.verts.layers.float.get(_BPO_LAYER)

    depth = v0[depth_layer] if depth_layer is not None else 0.0
    direction = (
        v0[dx_layer] if dx_layer is not None else 0.0,
        v0[dy_layer] if dy_layer is not None else 0.0,
        v0[dz_layer] if dz_layer is not None else 0.0,
    )
    bpo = v0[bpo_layer] if bpo_layer is not None else 0.0

    # Read cuboid params (FOLDED_PLANE mode)
    cuboid_params = None
    if mode == 'FOLDED_PLANE':
        first_layer = bm.verts.layers.float.get(_CUBOID_LAYERS[0])
        if first_layer is not None:
            values = []
            for name in _CUBOID_LAYERS:
                layer = bm.verts.layers.float.get(name)
                values.append(v0[layer] if layer is not None else 0.0)
            cuboid_params = (
                Vector((values[0], values[1], values[2])),   # origin
                Vector((values[3], values[4], values[5])),   # local_x
                Vector((values[6], values[7], values[8])),   # local_y
                values[9],                                    # cdx
                values[10],                                   # cdy
            )

    # Read coplanar blocked bitmask
    coplanar_blocked = 0
    copl_layer = bm.verts.layers.int.get(_COPLANAR_LAYER)
    if copl_layer is not None:
        coplanar_blocked = v0[copl_layer]

    return mode, depth, direction, bpo, cuboid_params, coplanar_blocked


def clear_weld_on_bmesh(bm):
    """Clear all weld BMesh layers (set to zero)."""
    mode_layer = bm.verts.layers.int.get(_MODE_LAYER)
    if mode_layer is not None:
        bm.verts.ensure_lookup_table()
        v0 = bm.verts[0]
        v0[mode_layer] = 0
        for name in (_DEPTH_LAYER, _DX_LAYER, _DY_LAYER, _DZ_LAYER, _BPO_LAYER):
            layer = bm.verts.layers.float.get(name)
            if layer is not None:
                v0[layer] = 0.0
        for name in _CUBOID_LAYERS:
            layer = bm.verts.layers.float.get(name)
            if layer is not None:
                v0[layer] = 0.0

    face_layer = bm.faces.layers.int.get(_FACE_LAYER)
    if face_layer is not None:
        for f in bm.faces:
            f[face_layer] = 0


def _get_marked_faces(bm):
    """Return the list of BMFaces marked as box faces for INVERT."""
    face_layer = bm.faces.layers.int.get(_FACE_LAYER)
    if face_layer is None:
        return []
    bm.faces.ensure_lookup_table()
    return [f for f in bm.faces if f[face_layer] != 0]


# ---------------------------------------------------------------------------
# Mesh property helpers (object mode only)
# ---------------------------------------------------------------------------

def _set_weld_mode_on_mesh(me, mode):
    """Store weld mode as a Mesh custom property (object mode only)."""
    me[_AW + "mode"] = mode


def _get_weld_mode_from_mesh(me):
    """Read weld mode from a Mesh custom property (object mode only)."""
    return me.get(_AW + "mode", "NONE")


def _clear_weld_on_mesh(me):
    """Remove all weld custom properties from a Mesh datablock."""
    for key in [k for k in me.keys() if k.startswith(_AW)]:
        del me[key]


# ---------------------------------------------------------------------------
# Edge group counting
# ---------------------------------------------------------------------------

def _count_edge_groups(bm):
    """Count the number of connected components of selected edges."""
    bm.edges.ensure_lookup_table()
    selected_edges = [e for e in bm.edges if e.select]
    if not selected_edges:
        return 0

    visited = set()
    groups = 0

    edge_set = set(e.index for e in selected_edges)
    vert_to_edges = {}
    for e in selected_edges:
        for v in e.verts:
            vert_to_edges.setdefault(v.index, []).append(e.index)

    for e in selected_edges:
        if e.index in visited:
            continue
        groups += 1
        queue = [e.index]
        visited.add(e.index)
        while queue:
            current = queue.pop()
            current_edge = bm.edges[current]
            for v in current_edge.verts:
                for neighbor_idx in vert_to_edges.get(v.index, []):
                    if neighbor_idx not in visited and neighbor_idx in edge_set:
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)

    return groups


# ---------------------------------------------------------------------------
# Mode derivation
# ---------------------------------------------------------------------------

_FOLDED_EPSILON = 1e-4


def _ranges_overlap(a_min, a_max, b_min, b_max):
    """Return True if 1D ranges [a_min, a_max] and [b_min, b_max] overlap."""
    return a_min < b_max + _FOLDED_EPSILON and b_min < a_max + _FOLDED_EPSILON


def snapshot_coplanar_sides(bm, cuboid_params):
    """Snapshot which cuboid side planes have overlapping coplanar mesh faces.

    Must be called BEFORE the cube cut modifies geometry.
    Returns an int bitmask: bit i set means side i is blocked.

    Side indices match the order in _execute_folded_plane's side_faces_def:
      0: local_x = 0      (left)
      1: local_x = cdx    (right)
      2: local_y = 0      (bottom)
      3: local_y = cdy    (top)
    """
    origin, local_x, local_y, cdx, cdy = cuboid_params
    local_z = local_x.cross(local_y).normalized()

    # Side definitions: (filter_axis, offset, inward_normal, u_axis, w_axis)
    side_faces_def = [
        (local_x, 0.0, local_x, local_y, local_z),
        (local_x, cdx, -local_x, local_y, local_z),
        (local_y, 0.0, local_y, local_x, local_z),
        (local_y, cdy, -local_y, local_x, local_z),
    ]

    bm.faces.ensure_lookup_table()
    bm.normal_update()
    blocked = 0

    for side_idx, (filter_axis, offset_val, inward_normal, u_axis, w_axis) in enumerate(side_faces_def):
        # Compute the cuboid's footprint on this plane in (u, w) space
        # The cuboid spans [0, u_extent] x [0, w_extent] on each side plane
        # (u_axis and w_axis are the two remaining axes)
        u_extent_vec = u_axis * (cdx if u_axis == local_x or u_axis == -local_x else cdy)
        w_extent = None  # depth extent is unbounded for the snapshot

        # Actually we just need the cuboid u-range on this plane
        # u_axis is one of local_x or local_y, w_axis is local_z (depth)
        if abs(u_axis.dot(local_x)) > 0.5:
            cu_extent = cdx
        else:
            cu_extent = cdy

        cuboid_u_min = 0.0
        cuboid_u_max = cu_extent

        for f in bm.faces:
            if abs(abs(f.normal.dot(inward_normal)) - 1.0) >= _FOLDED_EPSILON:
                continue
            if not all(abs((v.co - origin).dot(filter_axis) - offset_val)
                       < _FOLDED_EPSILON for v in f.verts):
                continue

            # Face is coplanar with this side plane — check u-axis overlap
            face_us = [(v.co - origin).dot(u_axis) for v in f.verts]
            face_ws = [(v.co - origin).dot(w_axis) for v in f.verts]
            face_u_min, face_u_max = min(face_us), max(face_us)
            face_w_min, face_w_max = min(face_ws), max(face_ws)

            # The cuboid extends the full depth along w_axis (local_z),
            # so only check u-axis overlap
            if _ranges_overlap(cuboid_u_min, cuboid_u_max, face_u_min, face_u_max):
                blocked |= (1 << side_idx)
                debug_log(f"[FoldedPlane] Side {side_idx} blocked by pre-existing "
                          f"coplanar face (u={face_u_min:.3f}..{face_u_max:.3f})")
                break

    return blocked


def _check_folded_plane(selected_verts, cuboid_params):
    """Check if at least 2 selected vertices lie on any single depth edge.

    cuboid_params: (origin, local_x, local_y, cdx, cdy) in object local space.
    Depth edges are at the 4 corners of the cuboid rectangle, running along the
    depth direction (perpendicular to both local_x and local_y).
    """
    origin, local_x, local_y, cdx, cdy = cuboid_params
    corners = [
        origin,
        origin + local_x * cdx,
        origin + local_x * cdx + local_y * cdy,
        origin + local_y * cdy,
    ]
    for ci, corner in enumerate(corners):
        count = 0
        for v in selected_verts:
            offset = v.co - corner
            along_x = offset.dot(local_x)
            along_y = offset.dot(local_y)
            in_plane_vec = local_x * along_x + local_y * along_y
            if in_plane_vec.length < _FOLDED_EPSILON:
                count += 1
        if count >= 2:
            return True
    return False


def _derive_weld_mode(bm, depth, cuboid_params):
    """Derive the weld mode from the current edge selection and depth.

    Returns (mode, depth) tuple where mode is 'BRIDGE', 'CORRIDOR',
    'FOLDED_PLANE', or 'NONE'.
    """
    groups = _count_edge_groups(bm)

    if groups == 2:
        return 'BRIDGE', 0.0
    elif groups == 1:
        selected_verts = list({v for e in bm.edges if e.select for v in e.verts})
        if abs(depth) > 0 and are_verts_coplanar(selected_verts):
            return 'CORRIDOR', depth
        if cuboid_params is not None:
            if _check_folded_plane(selected_verts, cuboid_params):
                return 'FOLDED_PLANE', depth
        return 'NONE', 0.0
    return 'NONE', 0.0


# ---------------------------------------------------------------------------
# Public API — setting weld state
# ---------------------------------------------------------------------------

def set_weld_from_edge_selection(context, depth, direction, back_plane_offset,
                                 first_vertex, second_vertex, local_x, local_y,
                                 coplanar_blocked):
    """Analyze current edge selection and store weld state in BMesh layers.

    Called from cube_cut after successful execution.
    first_vertex, second_vertex, local_x, local_y are in world space.
    coplanar_blocked is an int bitmask from snapshot_coplanar_sides().
    """
    global _weld_just_stored

    obj = context.active_object
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # Transform cuboid params to object local space for detection and storage
    w2l = obj.matrix_world.inverted()
    w2l_rot = w2l.to_3x3()
    local_origin = w2l @ first_vertex
    local_lx = (w2l_rot @ Vector(local_x)).normalized()
    local_ly = (w2l_rot @ Vector(local_y)).normalized()

    diff_world = Vector(second_vertex) - Vector(first_vertex)
    cdx = abs(diff_world.dot(Vector(local_x)))
    cdy = abs(diff_world.dot(Vector(local_y)))

    # Flip axes if needed so dx/dy are positive
    local_diff = (w2l @ Vector(second_vertex)) - local_origin
    if local_diff.dot(local_lx) < 0:
        local_lx = -local_lx
    if local_diff.dot(local_ly) < 0:
        local_ly = -local_ly

    # Scale dx/dy to local space
    scale_x = (w2l_rot @ Vector(local_x)).length
    scale_y = (w2l_rot @ Vector(local_y)).length
    local_cdx = cdx * scale_x
    local_cdy = cdy * scale_y

    cuboid_params_local = (local_origin, local_lx, local_ly, local_cdx, local_cdy)
    debug_log(f"[Weld] Cuboid local: origin={local_origin}, lx={local_lx}, ly={local_ly}, "
              f"cdx={local_cdx:.4f}, cdy={local_cdy:.4f}")

    mode, effective_depth = _derive_weld_mode(bm, depth, cuboid_params_local)
    dir_tuple = tuple(direction)

    store_cuboid = cuboid_params_local if mode == 'FOLDED_PLANE' else None
    store_coplanar = coplanar_blocked if mode == 'FOLDED_PLANE' else None
    _set_weld_on_bmesh(bm, mode, effective_depth, dir_tuple, back_plane_offset,
                       None, store_cuboid, store_coplanar)
    _weld_just_stored = True
    bmesh.update_edit_mesh(me)

    # Sync to scene props for UI
    props = context.scene.level_design_props
    props.weld_mode = mode
    props.weld_depth = effective_depth
    props.weld_direction = dir_tuple
    props.weld_back_plane_offset = back_plane_offset
    debug_log(f"[Weld] Set weld mode: {mode} (depth={depth}, direction={dir_tuple})")


def set_weld_from_box_builder(context, new_face_vert_positions):
    """Analyze box builder output and set weld mode to INVERT or NONE.

    Called from the box builder operator after successful box creation in edit mode.
    """
    global _weld_just_stored

    obj = context.active_object
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.edges.index_update()
    bm.faces.ensure_lookup_table()
    bm.normal_update()

    # Find the box faces by matching vertex positions within the selection
    position_set = set(new_face_vert_positions)
    box_faces = []
    for f in bm.faces:
        if not f.select:
            continue
        face_verts = frozenset(tuple(v.co) for v in f.verts)
        if face_verts in position_set:
            box_faces.append(f)
    if not box_faces:
        return

    if _check_box_needs_invert(bm, box_faces):
        mode = 'INVERT'
    else:
        mode = 'NONE'

    _set_weld_on_bmesh(bm, mode, 0.0, (0.0, 0.0, 0.0), 0.0, box_faces, None, None)
    _weld_just_stored = True
    bmesh.update_edit_mesh(me)

    props = context.scene.level_design_props
    props.weld_mode = mode
    debug_log(f"[Weld] Box builder weld mode: {mode}")


def set_weld_from_box_builder_object_mode(obj):
    """Store INVERT weld on an object created by box builder in object mode.

    Uses Mesh custom properties (which participate in object-mode undo).
    """
    _set_weld_mode_on_mesh(obj.data, 'INVERT')


# ---------------------------------------------------------------------------
# Sync weld state to scene props
# ---------------------------------------------------------------------------

def sync_weld_props(context, bm):
    """Read weld state and sync to scene props for UI display.

    Edit mode (bm is not None): reads from BMesh layers.  BMesh layers
    participate in edit-mode undo, so weld data is restored automatically
    on undo — no geometry derivation needed.

    Object mode (bm is None): reads from Mesh custom properties.

    Args:
        context: Blender context
        bm: BMesh of the active object (edit mode) or None (object mode)
    """
    if _weld_op_running:
        return

    props = context.scene.level_design_props
    obj = context.active_object
    if not obj or obj.type != 'MESH':
        if props.weld_mode != 'NONE':
            props.weld_mode = 'NONE'
        return

    if bm is not None:
        # Edit mode: read from BMesh layers
        mode, depth, direction, bpo, _cuboid, _copl = _get_weld_from_bmesh(bm)

        if mode == 'NONE':
            if props.weld_mode != 'NONE':
                props.weld_mode = 'NONE'
            return
    else:
        # Object mode: read from Mesh custom properties
        mode = _get_weld_mode_from_mesh(obj.data)
        depth = 0.0
        direction = (0.0, 0.0, 0.0)
        bpo = 0.0

        if mode == 'NONE':
            if props.weld_mode != 'NONE':
                props.weld_mode = 'NONE'
            return

    # Update scene props for UI display
    if (mode != props.weld_mode or
            (mode != 'NONE' and abs(depth - props.weld_depth) > 0.001)):
        props.weld_mode = mode
        props.weld_depth = depth
        props.weld_direction = direction
        props.weld_back_plane_offset = bpo
        debug_log(f"[Weld] Sync props: {mode} (depth={depth}, direction={direction})")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def get_weld_display_name(weld_mode):
    """Get a human-readable name for the weld mode."""
    if weld_mode == 'BRIDGE':
        return "Bridge Edge Loops"
    elif weld_mode == 'CORRIDOR':
        return "Corridor"
    elif weld_mode == 'INVERT':
        return "Invert"
    elif weld_mode == 'FOLDED_PLANE':
        return "Folded Plane"
    return "None"


# ---------------------------------------------------------------------------
# Geometry helpers (invert mode — used at setup time, not for undo)
# ---------------------------------------------------------------------------

def _faces_coplanar_antiparallel(face_a, face_b, tolerance=0.001):
    """Check if two faces are on the same plane with opposite normals."""
    dot = face_a.normal.dot(face_b.normal)
    if dot > -0.99:
        return False
    dist = abs((face_b.verts[0].co - face_a.verts[0].co).dot(face_a.normal))
    return dist < tolerance


def _project_face_2d(face, axis_u, axis_v):
    """Project face vertices onto 2D coordinates using two plane axes."""
    return [(v.co.dot(axis_u), v.co.dot(axis_v)) for v in face.verts]


def _polygons_overlap_2d(poly_a, poly_b):
    """Check if two convex 2D polygons overlap using the separating axis theorem."""
    for poly in [poly_a, poly_b]:
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge_x = poly[j][0] - poly[i][0]
            edge_y = poly[j][1] - poly[i][1]
            axis = (-edge_y, edge_x)

            min_a = min(p[0] * axis[0] + p[1] * axis[1] for p in poly_a)
            max_a = max(p[0] * axis[0] + p[1] * axis[1] for p in poly_a)
            min_b = min(p[0] * axis[0] + p[1] * axis[1] for p in poly_b)
            max_b = max(p[0] * axis[0] + p[1] * axis[1] for p in poly_b)

            if max_a <= min_b + 1e-6 or max_b <= min_a + 1e-6:
                return False
    return True


def _faces_overlap(face_a, face_b):
    """Check if two coplanar faces overlap when projected onto their shared plane."""
    normal = face_a.normal
    if abs(normal.z) < 0.9:
        up = Vector((0, 0, 1))
    else:
        up = Vector((1, 0, 0))
    axis_u = normal.cross(up).normalized()
    axis_v = normal.cross(axis_u).normalized()

    poly_a = _project_face_2d(face_a, axis_u, axis_v)
    poly_b = _project_face_2d(face_b, axis_u, axis_v)
    return _polygons_overlap_2d(poly_a, poly_b)


def _check_box_needs_invert(bm, box_faces):
    """Check if a box builder result should use INVERT weld mode.

    Returns True if no non-box face is coplanar with a box face, has an
    anti-parallel normal (pointing into the box), and overlaps it in 2D.
    """
    box_face_set = set(f.index for f in box_faces)
    other_faces = [f for f in bm.faces if f.index not in box_face_set]

    if not other_faces:
        return True

    for box_face in box_faces:
        for other_face in other_faces:
            if _faces_coplanar_antiparallel(box_face, other_face):
                if _faces_overlap(box_face, other_face):
                    return False
    return True


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class MESH_OT_context_weld(bpy.types.Operator):
    """Perform a context-aware weld action based on recent operations"""
    bl_idname = "leveldesign.context_weld"
    bl_label = "Context Weld"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        global _weld_op_running

        if not hasattr(context.scene, 'level_design_props'):
            return {'CANCELLED'}

        props = context.scene.level_design_props
        weld_mode = props.weld_mode

        if weld_mode == 'NONE':
            return {'CANCELLED'}

        if not (context.active_object and
                context.active_object.type == 'MESH'):
            return {'CANCELLED'}

        # BRIDGE, CORRIDOR and FOLDED_PLANE require edit mode; INVERT works from object mode too
        if weld_mode in ('BRIDGE', 'CORRIDOR', 'FOLDED_PLANE') and context.mode != 'EDIT_MESH':
            return {'CANCELLED'}

        # Read marked box faces from BMesh layers for INVERT in edit mode
        invert_box_faces = []
        if weld_mode == 'INVERT' and context.mode == 'EDIT_MESH':
            bm = bmesh.from_edit_mesh(context.active_object.data)
            invert_box_faces = _get_marked_faces(bm)

        _weld_op_running = True
        try:
            if weld_mode == 'BRIDGE':
                return self._execute_bridge(context)
            elif weld_mode == 'CORRIDOR':
                return self._execute_corridor(
                    context, props.weld_depth,
                    Vector(props.weld_direction),
                    props.weld_back_plane_offset,
                )
            elif weld_mode == 'INVERT':
                return self._execute_invert(context, invert_box_faces)
            elif weld_mode == 'FOLDED_PLANE':
                bm = bmesh.from_edit_mesh(context.active_object.data)
                _mode, _depth, _dir, _bpo, cuboid_params, coplanar_blocked = _get_weld_from_bmesh(bm)
                if cuboid_params is None:
                    self.report({'ERROR'}, "Missing cuboid parameters")
                    return {'CANCELLED'}
                return self._execute_folded_plane(context, cuboid_params, coplanar_blocked)
        finally:
            _weld_op_running = False

        return {'CANCELLED'}

    def _execute_bridge(self, context):
        """Bridge two edge loops."""
        try:
            bpy.ops.mesh.bridge_edge_loops()
        except RuntimeError as e:
            self.report({'ERROR'}, f"Bridge failed: {e}")
            return {'CANCELLED'}

        # Clear BMesh weld layers (participates in edit-mode undo).
        me = context.active_object.data
        bm = bmesh.from_edit_mesh(me)
        clear_weld_on_bmesh(bm)

        # Deselect everything and switch to face select mode
        for f in bm.faces:
            f.select = False
        for e in bm.edges:
            e.select = False
        for v in bm.verts:
            v.select = False
        bm.select_flush(False)
        context.tool_settings.mesh_select_mode = (False, False, True)

        bmesh.update_edit_mesh(me)

        context.scene.level_design_props.weld_mode = 'NONE'
        self.report({'INFO'}, "Bridged edge loops")
        return {'FINISHED'}

    def _execute_invert(self, context, box_faces):
        """Flip normals on the mesh.

        In edit mode with box_faces: selects only those, flips, then restores
        the original selection.
        In object mode: enters edit mode, selects all, flips, returns to
        object mode.
        """
        entered_edit = False
        prev_selected_faces = None

        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')
            entered_edit = True
            bpy.ops.mesh.select_all(action='SELECT')
        elif box_faces:
            obj = context.active_object
            me = obj.data
            bm = bmesh.from_edit_mesh(me)
            bm.faces.ensure_lookup_table()

            # Save selected face indices, then select only the box faces
            prev_selected_faces = [f.index for f in bm.faces if f.select]

            for f in bm.faces:
                f.select = False
            for e in bm.edges:
                e.select = False
            for v in bm.verts:
                v.select = False
            for f in box_faces:
                if f.is_valid:
                    f.select = True
            bm.select_flush(True)
            bmesh.update_edit_mesh(me)

        try:
            bpy.ops.mesh.flip_normals()
        except RuntimeError as e:
            if entered_edit:
                bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'ERROR'}, f"Invert failed: {e}")
            return {'CANCELLED'}

        if entered_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
            # Object mode: clear Mesh custom properties (object-mode undo
            # restores them).
            _clear_weld_on_mesh(context.active_object.data)
        else:
            obj = context.active_object
            me = obj.data
            bm = bmesh.from_edit_mesh(me)
            bm.faces.ensure_lookup_table()

            # Restore the original selection
            if prev_selected_faces is not None:
                for f in bm.faces:
                    f.select = False
                for e in bm.edges:
                    e.select = False
                for v in bm.verts:
                    v.select = False
                for i in prev_selected_faces:
                    if i < len(bm.faces):
                        bm.faces[i].select = True
                bm.select_flush(True)

            # Clear BMesh weld layers (participates in edit-mode undo).
            clear_weld_on_bmesh(bm)
            bmesh.update_edit_mesh(me)

        context.scene.level_design_props.weld_mode = 'NONE'
        self.report({'INFO'}, "Normals inverted")
        return {'FINISHED'}

    def _execute_corridor(self, context, depth, direction, back_plane_offset):
        """Create a face from the edge loop, then extrude it to the back plane."""
        debug_log(f"[Corridor] Execute: depth={depth:.4f}, direction={direction}, "
                  f"back_plane_offset={back_plane_offset:.4f}")
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return {'CANCELLED'}

        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        # Step 1: Create face from selected edges using bmesh directly
        selected_edges = [e for e in bm.edges if e.select]
        if not selected_edges:
            self.report({'ERROR'}, "No edges selected")
            return {'CANCELLED'}

        try:
            result = bmesh.ops.contextual_create(bm, geom=selected_edges)
        except Exception as e:
            self.report({'ERROR'}, f"Fill failed: {e}")
            return {'CANCELLED'}

        new_faces = result.get('faces', [])
        if not new_faces:
            self.report({'ERROR'}, "No face created")
            return {'CANCELLED'}

        # Select the new face(s)
        for f in bm.faces:
            f.select = False
        for f in new_faces:
            f.select = True
        bm.select_flush_mode()

        # Find an adjacent frame face to use as UV source for the cap later.
        bm.normal_update()
        uv_source_face = None
        for filled_face in new_faces:
            for edge in filled_face.edges:
                for neighbor in edge.link_faces:
                    if neighbor != filled_face and neighbor.is_valid:
                        uv_source_face = neighbor
                        break
                if uv_source_face:
                    break
            if uv_source_face:
                break

        # Step 2: Extrude the new face along the cube cut direction
        selected_faces = new_faces

        bm.select_mode = {'FACE'}
        context.tool_settings.mesh_select_mode = (False, False, True)
        world_to_local_rot = obj.matrix_world.inverted().to_3x3()
        extrude_dir = (world_to_local_rot @ direction).normalized()
        origin_proj = obj.matrix_world.translation.dot(direction.normalized())
        local_back_plane_offset = back_plane_offset - origin_proj

        debug_log(f"[Corridor] Filled face center={selected_faces[0].calc_center_median()}, "
                  f"normal={selected_faces[0].normal}")
        debug_log(f"[Corridor] World->local: extrude_dir={extrude_dir}, "
                  f"local_back_plane_offset={local_back_plane_offset:.4f} "
                  f"(world={back_plane_offset:.4f} - origin_proj={origin_proj:.4f})")

        extrude_geom = list(selected_faces)
        for f in selected_faces:
            extrude_geom.extend(f.edges)
        result = bmesh.ops.extrude_face_region(bm, geom=extrude_geom)

        extruded_verts = [g for g in result['geom'] if isinstance(g, bmesh.types.BMVert)]
        extruded_faces = [g for g in result['geom'] if isinstance(g, bmesh.types.BMFace)]

        if not extruded_verts:
            self.report({'ERROR'}, "Extrude failed")
            return {'CANCELLED'}

        # Deselect old, select new
        for f in bm.faces:
            f.select = False
        for e in bm.edges:
            e.select = False
        for v in bm.verts:
            v.select = False
        for f in extruded_faces:
            f.select = True

        # Project extruded verts onto the cube cut's back plane
        debug_log(f"[Corridor] Projecting {len(extruded_verts)} verts onto back plane: "
                  f"extrude_dir={extrude_dir}, local_back_plane_offset={local_back_plane_offset:.4f}")
        for v in extruded_verts:
            _old_co = v.co.copy()
            _proj = v.co.dot(extrude_dir)
            _delta = local_back_plane_offset - _proj
            v.co += extrude_dir * _delta
            debug_log(f"[Corridor]   vert {v.index}: {_old_co} -> {v.co} "
                      f"(proj={_proj:.4f}, delta={_delta:.4f})")

        bm.normal_update()

        # Apply UVs to the extruded cap face from the adjacent frame face.
        if uv_source_face and uv_source_face.is_valid:
            uv_layer = get_render_active_uv_layer(bm, me)
            if uv_layer:
                ppm = context.scene.level_design_props.pixels_per_meter
                for cap_face in extruded_faces:
                    set_uv_from_other_face(
                        uv_source_face, cap_face, uv_layer, ppm, me,
                        obj.matrix_world,
                    )

        # Clear BMesh weld layers (participates in edit-mode undo).
        clear_weld_on_bmesh(bm)

        # Deselect everything and switch to face select mode
        for f in bm.faces:
            f.select = False
        for e in bm.edges:
            e.select = False
        for v in bm.verts:
            v.select = False
        bm.select_flush(False)
        context.tool_settings.mesh_select_mode = (False, False, True)

        bmesh.update_edit_mesh(me)

        context.scene.level_design_props.weld_mode = 'NONE'
        self.report({'INFO'}, f"Corridor created (depth: {depth:.3f})")
        return {'FINISHED'}


    def _execute_folded_plane(self, context, cuboid_params, coplanar_blocked):
        """Create faces on cuboid side planes from the selected boundary edges."""
        debug_log("[FoldedPlane] Execute folded plane weld")
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return {'CANCELLED'}

        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        origin, local_x, local_y, cdx, cdy = cuboid_params
        local_z = local_x.cross(local_y).normalized()

        # Collect selected boundary edges and their vertices
        selected_edges = [e for e in bm.edges if e.select]
        selected_verts = list({v for e in selected_edges for v in e.verts})

        if not selected_edges:
            self.report({'ERROR'}, "No edges selected")
            return {'CANCELLED'}

        # Step 1: Find vertices on each depth edge and create connecting edges
        corners = [
            origin,
            origin + local_x * cdx,
            origin + local_x * cdx + local_y * cdy,
            origin + local_y * cdy,
        ]

        depth_edge_verts = {}  # corner_idx -> list of BMVerts sorted along local_z
        for ci, corner in enumerate(corners):
            verts_on_edge = []
            for v in selected_verts:
                offset = v.co - corner
                along_x = offset.dot(local_x)
                along_y = offset.dot(local_y)
                in_plane_vec = local_x * along_x + local_y * along_y
                if in_plane_vec.length < _FOLDED_EPSILON:
                    verts_on_edge.append(v)
            if len(verts_on_edge) >= 2:
                verts_on_edge.sort(key=lambda v: (v.co - corner).dot(local_z))
                depth_edge_verts[ci] = verts_on_edge

        debug_log(f"[FoldedPlane] Found {len(depth_edge_verts)} depth edges with vertices")

        new_edges = []
        for ci, verts in depth_edge_verts.items():
            for i in range(len(verts) - 1):
                v1, v2 = verts[i], verts[i + 1]
                existing = None
                for e in v1.link_edges:
                    if e.other_vert(v1) == v2:
                        existing = e
                        break
                if existing is None:
                    edge = bm.edges.new((v1, v2))
                    new_edges.append(edge)
                    debug_log(f"[FoldedPlane] Created depth edge: {v1.co} -> {v2.co}")
                else:
                    new_edges.append(existing)

        # Step 2: Create faces on each cuboid side face (not front/back)
        # Side faces are: left (x=0), right (x=cdx), bottom (y=0), top (y=cdy)
        # Each entry: (filter_axis, offset, inward_normal, u_axis, w_axis)
        side_faces_def = [
            (local_x, 0.0, local_x, local_y, local_z),
            (local_x, cdx, -local_x, local_y, local_z),
            (local_y, 0.0, local_y, local_x, local_z),
            (local_y, cdy, -local_y, local_x, local_z),
        ]

        all_relevant_verts = set(selected_verts)
        for e in new_edges:
            all_relevant_verts.update(e.verts)
        created_faces = []

        for side_idx, (filter_axis, offset_val, inward_normal, u_axis, w_axis) in enumerate(side_faces_def):
            # Find vertices on this plane
            plane_verts = [v for v in all_relevant_verts
                          if abs((v.co - origin).dot(filter_axis) - offset_val)
                          < _FOLDED_EPSILON]

            if len(plane_verts) < 3:
                continue

            # Skip sides that had overlapping coplanar faces before the cut
            if coplanar_blocked & (1 << side_idx):
                debug_log(f"[FoldedPlane] Skipping side {side_idx} (axis={filter_axis}, "
                          f"offset={offset_val}): blocked by pre-cut coplanar face")
                continue

            # Project to 2D (u, w) on the plane
            vert_uw = {}
            for v in plane_verts:
                off = v.co - origin
                vert_uw[v] = (off.dot(u_axis), off.dot(w_axis))

            us = [c[0] for c in vert_uw.values()]
            ws = [c[1] for c in vert_uw.values()]
            u_min, u_max = min(us), max(us)
            w_min, w_max = min(ws), max(ws)

            eps = _FOLDED_EPSILON

            # Classify vertices onto rectangle perimeter edges and sort.
            # Walk order: bottom → right → top → left.
            bottom = sorted(
                [v for v in plane_verts if abs(vert_uw[v][1] - w_min) < eps],
                key=lambda v: vert_uw[v][0])
            right_side = sorted(
                [v for v in plane_verts if abs(vert_uw[v][0] - u_max) < eps],
                key=lambda v: vert_uw[v][1])
            top = sorted(
                [v for v in plane_verts if abs(vert_uw[v][1] - w_max) < eps],
                key=lambda v: -vert_uw[v][0])
            left_side = sorted(
                [v for v in plane_verts if abs(vert_uw[v][0] - u_min) < eps],
                key=lambda v: -vert_uw[v][1])

            # Walk perimeter, deduplicating at shared corners
            polygon = list(bottom)
            for side_verts in (right_side, top, left_side):
                for v in side_verts:
                    if v != polygon[-1]:
                        polygon.append(v)
            if len(polygon) > 1 and polygon[-1] == polygon[0]:
                polygon.pop()

            if len(polygon) < 3:
                continue

            # Check winding against inward normal
            poly_normal = compute_normal_from_verts([v.co for v in polygon])
            if poly_normal is not None and poly_normal.dot(inward_normal) < 0:
                polygon.reverse()

            try:
                new_face = bm.faces.new(polygon)
                created_faces.append(new_face)
                debug_log(f"[FoldedPlane] Created face with {len(polygon)} verts, "
                          f"normal target={inward_normal}")
            except ValueError as e:
                debug_log(f"[FoldedPlane] Failed to create face: {e}")

        debug_log(f"[FoldedPlane] Created {len(created_faces)} faces total")

        # Step 3: Triangulate faces with > 4 edges, then join to quads
        faces_to_tri = [f for f in created_faces if f.is_valid and len(f.verts) > 4]
        if faces_to_tri:
            tri_result = bmesh.ops.triangulate(bm, faces=faces_to_tri)
            tri_faces = tri_result.get('faces', [])
            # Collect all triangle faces (deduplicated)
            tri_set = set()
            for f in tri_faces:
                if f.is_valid:
                    tri_set.add(f)
            for f in created_faces:
                if f.is_valid and len(f.verts) == 3:
                    tri_set.add(f)
            all_tris = list(tri_set)
            if all_tris:
                bmesh.ops.join_triangles(
                    bm, faces=all_tris,
                    cmp_seam=False, cmp_sharp=False, cmp_uvs=False,
                    cmp_vcols=False, cmp_materials=False,
                    angle_face_threshold=3.14, angle_shape_threshold=3.14,
                )

        bm.normal_update()

        # Clear BMesh weld layers
        clear_weld_on_bmesh(bm)

        # Deselect everything and switch to face select mode
        for f in bm.faces:
            f.select = False
        for e in bm.edges:
            e.select = False
        for v in bm.verts:
            v.select = False
        bm.select_flush(False)
        context.tool_settings.mesh_select_mode = (False, False, True)

        bmesh.update_edit_mesh(me)

        context.scene.level_design_props.weld_mode = 'NONE'
        self.report({'INFO'}, "Folded plane weld completed")
        return {'FINISHED'}


_addon_keymaps = []


def register():
    bpy.utils.register_class(MESH_OT_context_weld)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    # Register W key in both Mesh (edit mode) and Object Mode keymaps
    for km_name, space_type in [('Mesh', 'EMPTY'), ('Object Mode', 'EMPTY')]:
        km = kc.keymaps.new(name=km_name, space_type=space_type)
        kmi = km.keymap_items.new(
            "leveldesign.context_weld", 'W', 'PRESS',
            head=True
        )
        _addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    bpy.utils.unregister_class(MESH_OT_context_weld)
