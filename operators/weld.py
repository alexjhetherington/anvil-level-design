"""
Context Aware Weld - Operator and state management

Performs contextual weld actions based on the state left by previous operations
(e.g., cube cut). The weld action depends on the edge selection topology:
- Two distinct edge groups → Bridge Edge Loops
- One distinct edge group → Corridor (fill face + extrude)

Weld state is tracked via a module-level stack that maps edge selections to
their weld parameters.  Because Blender's undo reliably restores *geometry*
(and therefore edge selections) but does NOT reliably restore scene properties
set by addon operators, the stack lets us re-derive the correct weld mode by
matching the current edge selection against stored entries.
"""

import bpy
import bmesh
from mathutils import Vector

from ..utils import is_level_design_workspace, debug_log, are_verts_coplanar, get_render_active_uv_layer
from ..handlers import cache_single_face, face_data_cache
from .texture_apply import set_uv_from_other_face


# Stack of weld entries: each is (depth, frozenset_of_edge_indices, direction).
# direction is a tuple of 3 floats (the cube cut local_z vector).
# Pushed by set_weld_from_edge_selection (cube cut), never popped — entries
# become inert once the edge selection no longer matches.  This survives
# undo/redo (module globals are outside Blender's undo system).
_weld_stack = []

# Guard flag: True while the weld operator is executing, prevents the
# depsgraph handler from running redundant derivation mid-operator.
_weld_op_running = False


def _count_edge_groups(bm):
    """Count the number of connected components of selected edges.

    Returns the number of distinct groups of selected edges, where edges
    are connected if they share a vertex.
    """
    bm.edges.ensure_lookup_table()
    selected_edges = [e for e in bm.edges if e.select]
    if not selected_edges:
        return 0

    visited = set()
    groups = 0

    edge_set = set(e.index for e in selected_edges)
    # Map vertex -> list of selected edges
    vert_to_edges = {}
    for e in selected_edges:
        for v in e.verts:
            vert_to_edges.setdefault(v.index, []).append(e.index)

    for e in selected_edges:
        if e.index in visited:
            continue
        groups += 1
        # BFS
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


def _direction_for_stack_entry(entry):
    """Extract direction tuple from a stack entry, handling old 2-tuple format."""
    if len(entry) >= 3:
        return entry[2]
    return (0.0, 0.0, 0.0)


def set_weld_from_edge_selection(context, depth, direction):
    """Analyze current edge selection and push a weld entry onto the stack.

    Called from cube_cut after successful execution.
    direction is a Vector or tuple of 3 floats (the cube cut local_z).
    """
    obj = context.active_object
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    mode, effective_depth = _derive_weld_mode(bm, depth)
    props = context.scene.level_design_props
    props.weld_mode = mode
    props.weld_depth = effective_depth
    dir_tuple = tuple(direction)
    props.weld_direction = dir_tuple
    debug_log(f"[Weld] Set weld mode: {mode} (depth={depth}, direction={dir_tuple})")

    edge_indices = frozenset(e.index for e in bm.edges if e.select)
    _weld_stack.append((depth, edge_indices, dir_tuple))


def derive_weld_from_stack(bm):
    """Derive weld mode by matching current edge selection against the stack.

    Returns (mode, depth, direction) tuple.  Checks from most-recent to oldest.
    direction is a tuple of 3 floats.
    """
    if not _weld_stack:
        return 'NONE', 0.0, (0.0, 0.0, 0.0)

    current = frozenset(e.index for e in bm.edges if e.select)
    if not current:
        return 'NONE', 0.0, (0.0, 0.0, 0.0)

    for entry in reversed(_weld_stack):
        depth = entry[0]
        edge_indices = entry[1]
        if current == edge_indices:
            mode, eff_depth = _derive_weld_mode(bm, depth)
            return mode, eff_depth, _direction_for_stack_entry(entry)

    return 'NONE', 0.0, (0.0, 0.0, 0.0)


def sync_weld_props(context, bm):
    """Re-derive weld state from the stack and update scene properties.

    Called from undo/redo handlers and the depsgraph handler to keep
    scene properties in sync with the authoritative stack state.
    Skipped while the weld operator is executing.
    """
    if _weld_op_running:
        return

    mode, depth, direction = derive_weld_from_stack(bm)
    props = context.scene.level_design_props
    if mode != props.weld_mode or (mode != 'NONE' and abs(depth - props.weld_depth) > 0.001):
        props.weld_mode = mode
        props.weld_depth = depth
        props.weld_direction = direction
        debug_log(f"[Weld] Sync props: {mode} (depth={depth}, direction={direction})")


def clear_weld_stack():
    """Clear the weld stack.  Called on test cleanup and mode changes."""
    _weld_stack.clear()


def get_weld_display_name(weld_mode):
    """Get a human-readable name for the weld mode."""
    if weld_mode == 'BRIDGE':
        return "Bridge Edge Loops"
    elif weld_mode == 'CORRIDOR':
        return "Corridor"
    return "None"


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
                context.active_object.type == 'MESH' and
                context.mode == 'EDIT_MESH'):
            return {'CANCELLED'}

        _weld_op_running = True
        try:
            if weld_mode == 'BRIDGE':
                return self._execute_bridge(context)
            elif weld_mode == 'CORRIDOR':
                return self._execute_corridor(
                    context, props.weld_depth,
                    Vector(props.weld_direction),
                )
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

        context.scene.level_design_props.weld_mode = 'NONE'
        self.report({'INFO'}, "Bridged edge loops")
        return {'FINISHED'}

    def _execute_corridor(self, context, depth, direction):
        """Create a face from the edge loop, then extrude it to the given depth."""
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return {'CANCELLED'}

        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        # Step 1: Create face from selected edges using bmesh directly
        # (avoids bpy.ops.mesh.edge_face_add which creates a separate undo
        # step, breaking Ctrl+Z restoration of the weld state)
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
        # Must be captured before extrusion since the filled face gets consumed.
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

        # Step 2: Extrude the new face along the cube cut direction by the given depth
        selected_faces = new_faces

        # Switch to face select mode for the extrude
        bm.select_mode = {'FACE'}
        context.tool_settings.mesh_select_mode = (False, False, True)
        # Use the cube cut direction (local_z) stored when the cut was made.
        # This ensures the corridor is axis-aligned even on sloped surfaces.
        extrude_dir = direction.normalized()

        # Include the face's edges in geom so extrude_face_region properly
        # consumes the original face instead of leaving it behind.
        extrude_geom = list(selected_faces)
        for f in selected_faces:
            extrude_geom.extend(f.edges)
        result = bmesh.ops.extrude_face_region(bm, geom=extrude_geom)

        # Get the new geometry and move it
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

        # Select extruded geometry
        for f in extruded_faces:
            f.select = True

        # Move extruded verts along the stored cube cut direction
        for v in extruded_verts:
            v.co += extrude_dir * depth

        bm.normal_update()

        # Apply UVs to the extruded cap face from the adjacent frame face.
        # Then cache the cap so the depsgraph handler doesn't overwrite it.
        if uv_source_face and uv_source_face.is_valid:
            uv_layer = get_render_active_uv_layer(bm, me)
            if uv_layer:
                ppm = context.scene.level_design_props.pixels_per_meter
                for cap_face in extruded_faces:
                    set_uv_from_other_face(
                        uv_source_face, cap_face, uv_layer, ppm, me,
                        obj.matrix_world,
                    )
                    cache_single_face(cap_face, bm, ppm, me)

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
