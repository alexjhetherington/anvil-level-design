"""
Context Aware Weld - Operator and state management

Performs contextual weld actions based on the state left by previous operations
(e.g., cube cut). The weld action depends on the edge selection topology:
- Two distinct edge groups → Bridge Edge Loops
- One distinct edge group → Corridor (fill face + extrude)
"""

import bpy
import bmesh

from ..utils import is_level_design_workspace, debug_log


# Track selected edge indices when weld was set, to detect selection changes
_weld_edge_selection = frozenset()

# After undo/redo, we need to re-sync edge tracking from the restored state
_weld_needs_resync = False


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


def set_weld_from_edge_selection(context, depth):
    """Analyze current edge selection and set the weld state on the scene.

    Called from cube_cut after successful execution.
    """
    global _weld_edge_selection

    obj = context.active_object
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    groups = _count_edge_groups(bm)
    props = context.scene.level_design_props

    if groups == 2:
        props.weld_mode = 'BRIDGE'
        props.weld_depth = 0.0
        debug_log(f"[Weld] Set weld mode: BRIDGE (2 edge groups)")
    elif groups == 1:
        props.weld_mode = 'CORRIDOR'
        props.weld_depth = depth
        debug_log(f"[Weld] Set weld mode: CORRIDOR (1 edge group, depth={depth})")
    else:
        props.weld_mode = 'NONE'
        props.weld_depth = 0.0
        debug_log(f"[Weld] No weld: {groups} edge groups")

    _weld_edge_selection = frozenset(e.index for e in bm.edges if e.select)


def check_weld_selection_changed(bm):
    """Check if edge selection has changed since weld was set.

    Returns True if selection differs from the stored weld selection.
    After undo/redo, re-syncs edge tracking from the restored state.
    """
    global _weld_edge_selection, _weld_needs_resync

    if _weld_needs_resync:
        # Undo/redo restored the weld state; re-sync edge tracking
        _weld_edge_selection = frozenset(e.index for e in bm.edges if e.select)
        _weld_needs_resync = False
        return False

    current = frozenset(e.index for e in bm.edges if e.select)
    return current != _weld_edge_selection


def clear_weld_state(context):
    """Clear the weld state."""
    props = context.scene.level_design_props
    if props.weld_mode != 'NONE':
        debug_log(f"[Weld] Cleared weld state (was {props.weld_mode})")
        props.weld_mode = 'NONE'
        props.weld_depth = 0.0


def reset_weld_edge_tracking():
    """Reset edge selection tracking. Called on undo/redo.

    Sets a resync flag so the next depsgraph check re-populates edge tracking
    from the undo-restored state rather than immediately clearing the weld.
    """
    global _weld_needs_resync
    _weld_needs_resync = True


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

        if weld_mode == 'BRIDGE':
            return self._execute_bridge(context)
        elif weld_mode == 'CORRIDOR':
            return self._execute_corridor(context, props.weld_depth)

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

    def _execute_corridor(self, context, depth):
        """Create a face from the edge loop, then extrude it to the given depth."""
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return {'CANCELLED'}

        # Step 1: Create face from selected edges (like pressing F)
        try:
            bpy.ops.mesh.edge_face_add()
        except RuntimeError as e:
            self.report({'ERROR'}, f"Fill failed: {e}")
            return {'CANCELLED'}

        # Step 2: Extrude the new face along its normal by the given depth
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        # The newly created face should be selected
        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            self.report({'ERROR'}, "No face created")
            return {'CANCELLED'}

        # Switch to face select mode for the extrude
        bm.select_mode = {'FACE'}
        context.tool_settings.mesh_select_mode = (False, False, True)

        # Extrude along normal
        result = bmesh.ops.extrude_face_region(bm, geom=selected_faces)

        # Get the new geometry and move it
        extruded_verts = [g for g in result['geom'] if isinstance(g, bmesh.types.BMVert)]

        if not extruded_verts:
            self.report({'ERROR'}, "Extrude failed")
            return {'CANCELLED'}

        # Use the face normal to determine extrude direction
        # The normal should point inward (into the void), so we extrude along it
        face_normal = selected_faces[0].normal.copy()

        # Deselect old, select new
        for f in bm.faces:
            f.select = False
        for e in bm.edges:
            e.select = False
        for v in bm.verts:
            v.select = False

        # Select extruded geometry
        extruded_faces = [g for g in result['geom'] if isinstance(g, bmesh.types.BMFace)]
        for f in extruded_faces:
            f.select = True

        # Move extruded verts along the face normal
        for v in extruded_verts:
            v.co += face_normal * depth

        bm.normal_update()
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
