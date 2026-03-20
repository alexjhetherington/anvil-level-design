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

from ..utils import is_level_design_workspace, debug_log, are_verts_coplanar, get_render_active_uv_layer

from .texture_apply import set_uv_from_other_face


# BMesh layer names (all stored on vert 0)
_MODE_LAYER = "_aw_mode"    # int on verts — mode enum
_DEPTH_LAYER = "_aw_depth"  # float on verts
_DX_LAYER = "_aw_dx"        # float on verts — direction X
_DY_LAYER = "_aw_dy"        # float on verts — direction Y
_DZ_LAYER = "_aw_dz"        # float on verts — direction Z
_BPO_LAYER = "_aw_bpo"      # float on verts — back plane offset
_FACE_LAYER = "_aw_face"    # int on faces — 1 = box face for INVERT

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
_MODE_TO_STR = {0: 'NONE', 1: 'BRIDGE', 2: 'CORRIDOR', 3: 'INVERT'}
_STR_TO_MODE = {'NONE': 0, 'BRIDGE': 1, 'CORRIDOR': 2, 'INVERT': 3}


# ---------------------------------------------------------------------------
# BMesh layer helpers (edit mode)
# ---------------------------------------------------------------------------

def _set_weld_on_bmesh(bm, mode, depth=0.0, direction=(0.0, 0.0, 0.0),
                       back_plane_offset=0.0, box_faces=None):
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


def _get_weld_from_bmesh(bm):
    """Read weld parameters from BMesh layers.

    Returns (mode_str, depth, direction_tuple, back_plane_offset).
    """
    mode_layer = bm.verts.layers.int.get(_MODE_LAYER)
    if mode_layer is None:
        return 'NONE', 0.0, (0.0, 0.0, 0.0), 0.0

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

    return mode, depth, direction, bpo


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

def _derive_weld_mode(bm, depth):
    """Derive the weld mode from the current edge selection and depth.

    Returns (mode, depth) tuple where mode is 'BRIDGE', 'CORRIDOR', or 'NONE'.
    """
    groups = _count_edge_groups(bm)

    if groups == 2:
        return 'BRIDGE', 0.0
    elif groups == 1:
        selected_verts = list({v for e in bm.edges if e.select for v in e.verts})
        if abs(depth) > 0 and are_verts_coplanar(selected_verts):
            return 'CORRIDOR', depth
        return 'NONE', 0.0
    return 'NONE', 0.0


# ---------------------------------------------------------------------------
# Public API — setting weld state
# ---------------------------------------------------------------------------

def set_weld_from_edge_selection(context, depth, direction, back_plane_offset):
    """Analyze current edge selection and store weld state in BMesh layers.

    Called from cube_cut after successful execution.
    """
    global _weld_just_stored

    obj = context.active_object
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    mode, effective_depth = _derive_weld_mode(bm, depth)
    dir_tuple = tuple(direction)

    _set_weld_on_bmesh(bm, mode, effective_depth, dir_tuple, back_plane_offset)
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

    _set_weld_on_bmesh(bm, mode, box_faces=box_faces)
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
        mode, depth, direction, bpo = _get_weld_from_bmesh(bm)

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

        # BRIDGE and CORRIDOR require edit mode; INVERT works from object mode too
        if weld_mode in ('BRIDGE', 'CORRIDOR') and context.mode != 'EDIT_MESH':
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
