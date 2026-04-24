"""Consolidated depsgraph update handler."""

import bmesh
import bpy
from bpy.app.handlers import persistent

from ..core.logging import debug_log
from ..core.face_id import get_face_id_layer
from ..core.uv_layers import sync_uv_map_settings
from ..core.workspace_check import is_level_design_workspace

from .face_cache import (
    face_data_cache, cache_face_data, update_ui_from_selection,
    check_selection_changed, snapshot_selection,
    get_last_edit_object_name, set_last_edit_object_name,
)
from .active_image import update_active_image_from_face
from .auto_hotspot import apply_auto_hotspots, set_force_auto_hotspot
from .new_face_projection import project_new_faces
from .uv_world_scale import apply_world_scale_uvs, apply_uv_lock, get_topology_modal_ops
from .file_browser import consolidate_duplicate_materials
from .lifecycle import (
    get_file_loaded_into_edit_depsgraph, set_file_loaded_into_edit_depsgraph,
    get_undo_in_progress,
)


def _verts_near_selected(bm, epsilon):
    """Return selected verts plus any un-selected verts within epsilon of one.

    Spin's axis-duplicate verts are coincident with the original axis verts
    but are not selected after the operator runs, so a merge limited to the
    selection misses them. Widen the net by proximity.
    """
    selected = [v for v in bm.verts if v.select]
    if not selected:
        return []
    eps_sq = epsilon * epsilon
    selected_set = set(selected)
    result = list(selected)
    for v in bm.verts:
        if v in selected_set:
            continue
        for sv in selected:
            if (v.co - sv.co).length_squared < eps_sq:
                result.append(v)
                break
    return result


# Fingerprint (center, axis, steps, angle) of the last MESH_OT_spin execution
# we've cleaned up. Cleared whenever the active operator is no longer spin.
# Drag-spin advances the angle on every step → fingerprint changes → cleanup
# fires again. E/G/R after a spin don't touch spin's properties → fingerprint
# stays equal → cleanup skipped, even though active_operator is still spin
# during the extrude macro's initial geometry burst (before transform.translate
# registers in window.modal_operators).
_last_cleaned_spin_fingerprint = None


def _spin_op_fingerprint(op):
    try:
        return (
            tuple(op.properties.center),
            tuple(op.properties.axis),
            int(op.properties.steps),
            float(op.properties.angle),
        )
    except (AttributeError, TypeError):
        return None


def _cleanup_spin_degenerate_faces(bm, me):
    """Weld axis duplicates and delete zero-area faces left behind by mesh.spin.

    When the spin axis coincides with an edge of the selection, each step emits
    a zero-area wall quad at the axis. use_auto_merge (when on) welds the
    duplicate axis verts but leaves the collapsed faces in place; when off,
    duplicates remain entirely.
    """
    merge_epsilon = 1e-5
    merge_verts = _verts_near_selected(bm, merge_epsilon)
    if merge_verts:
        bmesh.ops.remove_doubles(bm, verts=merge_verts, dist=merge_epsilon)

    zero_area_faces = [f for f in bm.faces if f.is_valid and f.calc_area() < 1e-8]
    if zero_area_faces:
        bmesh.ops.delete(bm, geom=zero_area_faces, context='FACES')

    if merge_verts or zero_area_faces:
        bmesh.update_edit_mesh(me)


@persistent
def on_depsgraph_update(scene, depsgraph):
    """Consolidated depsgraph update handler"""
    if get_undo_in_progress():
        return

    if not is_level_design_workspace():
        return

    try:
        consolidate_duplicate_materials()

        for update in depsgraph.updates:
            if isinstance(update.id, bpy.types.Object):
                obj = update.id
                if obj.type == 'MESH' and obj.data and len(obj.data.color_attributes) == 0:
                    orig_mesh = bpy.data.meshes.get(obj.data.name)
                    if orig_mesh and len(orig_mesh.color_attributes) == 0:
                        orig_mesh.color_attributes.new("Color", 'BYTE_COLOR', 'CORNER')
                        orig_mesh.color_attributes.active_color_index = 0
                        orig_mesh.color_attributes.render_color_index = 0

        context = bpy.context

        if not hasattr(scene, 'level_design_props'):
            return

        props = scene.level_design_props

        if context.mode != 'EDIT_MESH' and get_last_edit_object_name() is not None:
            set_last_edit_object_name(None)

        if context.mode != 'EDIT_MESH':
            from ..operators.weld import sync_weld_props
            sync_weld_props(context, None)

        for update in depsgraph.updates:
            if isinstance(update.id, bpy.types.Object):
                obj = update.id
                if obj.type == 'MESH' and obj.mode != 'EDIT':
                    is_transform = getattr(update, 'is_updated_transform', False)
                    if is_transform:
                        from ..operators.fixed_hotspot_overlay import invalidate_overlay as _invalidate_fixed_overlay
                        _invalidate_fixed_overlay()
                        break

        for update in depsgraph.updates:
            if isinstance(update.id, bpy.types.Object):
                obj = update.id
                if obj.type == 'MESH' and obj.mode == 'EDIT':
                    me = obj.data

                    if me is None or not me.is_editmode:
                        debug_log(f"[Depsgraph] Skip: mesh not available or not in edit mode")
                        continue

                    is_geometry_update = update.is_updated_geometry
                    is_transform_update = getattr(update, 'is_updated_transform', False)
                    debug_log(f"[Depsgraph] Update for '{obj.name}': geometry={is_geometry_update} transform={is_transform_update}")

                    try:
                        bm = bmesh.from_edit_mesh(me)
                    except (ReferenceError, RuntimeError):
                        debug_log(f"[Depsgraph] Skip: BMesh invalid/being modified")
                        continue

                    if not bm.is_valid:
                        debug_log(f"[Depsgraph] Skip: BMesh not valid")
                        continue

                    active_op = bpy.context.active_operator
                    global _last_cleaned_spin_fingerprint
                    if active_op is None or active_op.bl_idname != "MESH_OT_spin":
                        _last_cleaned_spin_fingerprint = None
                    elif is_geometry_update:
                        fp = _spin_op_fingerprint(active_op)
                        if fp is not None and fp != _last_cleaned_spin_fingerprint:
                            _cleanup_spin_degenerate_faces(bm, me)
                            _last_cleaned_spin_fingerprint = fp

                    from .face_cache import last_face_count, last_vertex_count
                    current_face_count = len(bm.faces)
                    current_vertex_count = len(bm.verts)

                    is_fresh_start = (obj.name != get_last_edit_object_name())
                    set_last_edit_object_name(obj.name)

                    if is_fresh_start:
                        debug_log(f"[Depsgraph] Fresh edit session for '{obj.name}'")
                        sync_uv_map_settings(obj)
                        cache_face_data(context)
                        from .face_cache import set_last_selected_face_indices, set_last_active_face_index
                        set_last_selected_face_indices(set())
                        set_last_active_face_index(-1)
                        debug_log(f"[Depsgraph] Cache rebuilt for new object ({len(face_data_cache)} faces)")

                    allow_active_image_update = not is_fresh_start or get_file_loaded_into_edit_depsgraph()

                    topology_changed = current_face_count != last_face_count or current_vertex_count != last_vertex_count

                    if not topology_changed and not is_fresh_start:
                        id_layer_check = get_face_id_layer(bm)
                        seen_ids = set()
                        for face in bm.faces:
                            fid = face[id_layer_check]
                            if fid in seen_ids:
                                topology_changed = True
                                debug_log("[Depsgraph] Duplicate face IDs detected (modal restore/re-apply)")
                                break
                            seen_ids.add(fid)

                    if topology_changed:
                        debug_log(f"[Depsgraph] Topology changed: faces {last_face_count}->{current_face_count} verts {last_vertex_count}->{current_vertex_count}")

                        from ..operators.fixed_hotspot_overlay import invalidate_overlay as _invalidate_fixed_overlay
                        _invalidate_fixed_overlay()

                        if not is_fresh_start:
                            project_new_faces(context, bm)

                        topo_window = bpy.context.window
                        topo_modals = set(op.bl_idname for op in topo_window.modal_operators) if topo_window else set()
                        in_topology_modal = bool(topo_modals & get_topology_modal_ops())

                        if in_topology_modal:
                            debug_log("[Depsgraph] Skipping cache rebuild (topology modal active)")
                        else:
                            cache_face_data(context)
                            debug_log(f"[Depsgraph] Cache rebuilt ({len(face_data_cache)} faces)")
                            update_ui_from_selection(context)
                            if allow_active_image_update:
                                update_active_image_from_face(context)
                            set_file_loaded_into_edit_depsgraph(False)
                            snapshot_selection(bm)
                            from ..operators import weld as _weld_mod
                            _weld_mod._weld_just_stored = False
                            if not is_fresh_start and obj.anvil_auto_hotspot:
                                set_force_auto_hotspot(True)
                                apply_auto_hotspots()
                        return

                    selection_changed = check_selection_changed(bm)
                    if selection_changed:
                        debug_log(f"[Depsgraph] Selection changed")
                        update_ui_from_selection(context)
                        if allow_active_image_update:
                            update_active_image_from_face(context)
                        from ..operators import weld as _weld_mod
                        if _weld_mod._weld_just_stored:
                            _weld_mod._weld_just_stored = False
                        else:
                            _weld_mod.clear_weld_on_bmesh(bm)

                    from ..operators.weld import sync_weld_props
                    sync_weld_props(context, bm)

                    if not face_data_cache and context.mode == 'EDIT_MESH':
                        debug_log(f"[Depsgraph] Cache empty, rebuilding")
                        cache_face_data(context)

                    if is_geometry_update or is_transform_update:
                        from ..operators.fixed_hotspot_overlay import invalidate_overlay as _invalidate_fixed_overlay
                        _invalidate_fixed_overlay()

                    debug_log(f"[Depsgraph] Applying world-scale UVs (cache size={len(face_data_cache)})")
                    apply_world_scale_uvs(obj, scene)
                    apply_uv_lock(obj, scene)

                    if not is_fresh_start and obj.anvil_auto_hotspot:
                        apply_auto_hotspots()

                    set_file_loaded_into_edit_depsgraph(False)
                    break
    except Exception as e:
        print(f"Anvil Level Design: Error in depsgraph handler: {e}", flush=True)
