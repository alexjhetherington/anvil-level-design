"""Consolidated depsgraph update handler."""

import bmesh
import bpy
from bpy.app.handlers import persistent

from ..core.logging import (
    add_performance_detail,
    begin_performance_mode_report,
    debug_log,
    finish_performance_mode_report,
    performance_stage,
)
from ..core.materials import consolidate_duplicate_materials
from ..core.face_id import get_face_id_layer
from ..core.uv_layers import sync_uv_map_settings
from ..core.workspace_check import is_level_design_workspace

from .face_cache import (
    face_data_cache, cache_face_data, update_ui_from_selection,
    check_selection_changed, snapshot_selection,
    get_last_edit_mesh_names, set_last_edit_mesh_names,
)
from .active_image import update_active_image_from_face
from .auto_hotspot import apply_auto_hotspots, set_force_auto_hotspot
from .new_face_projection import project_new_faces
from .uv_world_scale import apply_world_scale_uvs, apply_uv_lock, get_topology_modal_ops
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


def _invalidate_view_overlays():
    from ..operators.fixed_hotspot_overlay import invalidate_overlay as _invalidate_fixed_overlay
    from ..operators.library_object_overlay import invalidate_overlay as _invalidate_library_overlay
    _invalidate_fixed_overlay()
    _invalidate_library_overlay()


@persistent
def on_depsgraph_update(scene, depsgraph):
    """Consolidated depsgraph update handler"""
    if get_undo_in_progress():
        return

    if not is_level_design_workspace():
        return

    performance_report = None
    try:
        context = bpy.context
        transition_update_ready = False
        for update in depsgraph.updates:
            if not isinstance(update.id, bpy.types.Object):
                continue
            update_obj = update.id
            if update_obj.type != 'MESH':
                continue
            if context.mode == 'EDIT_MESH' and update_obj.mode == 'EDIT':
                transition_update_ready = True
                break
            if context.mode == 'OBJECT' and update_obj.mode != 'EDIT':
                transition_update_ready = True
                break
        performance_report = begin_performance_mode_report(
            context.mode,
            transition_update_ready,
        )

        active_obj = context.active_object
        add_performance_detail(
            performance_report,
            "Active object",
            active_obj.name if active_obj is not None else "None",
        )
        if active_obj is not None and active_obj.type == 'MESH' and active_obj.data is not None:
            active_me = active_obj.data
            add_performance_detail(
                performance_report,
                "Active mesh",
                f"{len(active_me.vertices)} verts, {len(active_me.edges)} edges, "
                f"{len(active_me.polygons)} faces, {len(active_me.uv_layers)} UV maps",
            )

        with performance_stage(performance_report, "Duplicate material consolidation"):
            consolidate_duplicate_materials()

        with performance_stage(performance_report, "Ensure mesh colour attributes"):
            for update in depsgraph.updates:
                if isinstance(update.id, bpy.types.Object):
                    obj = update.id
                    if obj.type == 'MESH' and obj.data and len(obj.data.color_attributes) == 0:
                        orig_mesh = bpy.data.meshes.get(obj.data.name)
                        if orig_mesh and len(orig_mesh.color_attributes) == 0:
                            orig_mesh.color_attributes.new("Color", 'BYTE_COLOR', 'CORNER')
                            orig_mesh.color_attributes.active_color_index = 0
                            orig_mesh.color_attributes.render_color_index = 0

        if not hasattr(scene, 'level_design_props'):
            return

        props = scene.level_design_props

        if context.mode != 'EDIT_MESH':
            with performance_stage(performance_report, "Exit Edit Mode state sync"):
                if get_last_edit_mesh_names():
                    set_last_edit_mesh_names(())
                from ..operators.weld import sync_weld_props
                sync_weld_props(context, None)

        edit_objects = []
        seen_edit_meshes = set()
        with performance_stage(performance_report, "Collect Edit Mode meshes"):
            if context.mode == 'EDIT_MESH':
                for edit_obj in context.view_layer.objects:
                    if edit_obj.type != 'MESH' or edit_obj.data is None or not edit_obj.data.is_editmode:
                        continue
                    if edit_obj.data.name in seen_edit_meshes:
                        continue
                    seen_edit_meshes.add(edit_obj.data.name)
                    edit_objects.append(edit_obj)

        add_performance_detail(performance_report, "Edit meshes", len(edit_objects))

        edit_mesh_names = tuple(edit_obj.data.name for edit_obj in edit_objects)

        with performance_stage(performance_report, "Object Mode overlay invalidation"):
            for update in depsgraph.updates:
                if isinstance(update.id, bpy.types.Object):
                    obj = update.id
                    if obj.type == 'MESH' and obj.mode != 'EDIT':
                        is_transform = getattr(update, 'is_updated_transform', False)
                        is_geometry = getattr(update, 'is_updated_geometry', False)
                        if is_transform or is_geometry:
                            _invalidate_view_overlays()
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

                    with performance_stage(performance_report, "Access active Edit Mode BMesh"):
                        try:
                            bm = bmesh.from_edit_mesh(me)
                        except (ReferenceError, RuntimeError):
                            debug_log(f"[Depsgraph] Skip: BMesh invalid/being modified")
                            continue

                    if not bm.is_valid:
                        debug_log(f"[Depsgraph] Skip: BMesh not valid")
                        continue

                    with performance_stage(performance_report, "Spin cleanup check"):
                        active_op = bpy.context.active_operator
                        global _last_cleaned_spin_fingerprint
                        if active_op is None or active_op.bl_idname != "MESH_OT_spin":
                            _last_cleaned_spin_fingerprint = None
                        elif is_geometry_update:
                            fp = _spin_op_fingerprint(active_op)
                            if fp is not None and fp != _last_cleaned_spin_fingerprint:
                                _cleanup_spin_degenerate_faces(bm, me)
                                _last_cleaned_spin_fingerprint = fp

                    current_face_count = 0
                    current_vertex_count = 0
                    edit_bmeshes = []
                    with performance_stage(performance_report, "Inspect Edit Mode mesh totals"):
                        for edit_obj in edit_objects:
                            try:
                                edit_bm = bmesh.from_edit_mesh(edit_obj.data)
                            except (ReferenceError, RuntimeError):
                                continue
                            if not edit_bm.is_valid:
                                continue
                            edit_bmeshes.append(edit_bm)
                            current_face_count += len(edit_bm.faces)
                            current_vertex_count += len(edit_bm.verts)

                    add_performance_detail(
                        performance_report,
                        "Edit geometry",
                        f"{current_vertex_count} verts, {current_face_count} faces",
                    )

                    is_fresh_start = edit_mesh_names != get_last_edit_mesh_names()
                    set_last_edit_mesh_names(edit_mesh_names)

                    if is_fresh_start:
                        debug_log(f"[Depsgraph] Fresh edit session for {edit_mesh_names}")
                        with performance_stage(performance_report, "Sync UV map settings"):
                            for edit_obj in edit_objects:
                                sync_uv_map_settings(edit_obj)
                        with performance_stage(performance_report, "Build fresh face/UV cache"):
                            cache_face_data(context)
                        from .face_cache import set_last_selected_face_indices, set_last_active_face_index
                        set_last_selected_face_indices(set())
                        set_last_active_face_index(-1)
                        debug_log(f"[Depsgraph] Cache rebuilt for Edit Mode meshes ({len(face_data_cache)} faces)")

                    allow_active_image_update = not is_fresh_start or get_file_loaded_into_edit_depsgraph()

                    from .face_cache import last_face_count, last_vertex_count
                    with performance_stage(performance_report, "Topology and face-ID change detection"):
                        topology_changed = current_face_count != last_face_count or current_vertex_count != last_vertex_count

                        if not topology_changed and not is_fresh_start:
                            seen_ids = set()
                            for edit_bm in edit_bmeshes:
                                id_layer_check = get_face_id_layer(edit_bm)
                                for face in edit_bm.faces:
                                    fid = face[id_layer_check]
                                    if fid in seen_ids:
                                        topology_changed = True
                                        debug_log("[Depsgraph] Duplicate face IDs detected (modal restore/re-apply)")
                                        break
                                    seen_ids.add(fid)
                                if topology_changed:
                                    break

                    if topology_changed:
                        debug_log(f"[Depsgraph] Topology changed: faces {last_face_count}->{current_face_count} verts {last_vertex_count}->{current_vertex_count}")

                        with performance_stage(performance_report, "Topology overlay invalidation"):
                            _invalidate_view_overlays()

                        if not is_fresh_start:
                            with performance_stage(performance_report, "Project new faces"):
                                project_new_faces(context, bm)

                        topo_window = bpy.context.window
                        topo_modals = set(op.bl_idname for op in topo_window.modal_operators) if topo_window else set()
                        in_topology_modal = bool(topo_modals & get_topology_modal_ops())

                        if in_topology_modal:
                            debug_log("[Depsgraph] Skipping cache rebuild (topology modal active)")
                        else:
                            with performance_stage(performance_report, "Rebuild face/UV cache after topology"):
                                cache_face_data(context)
                            debug_log(f"[Depsgraph] Cache rebuilt ({len(face_data_cache)} faces)")
                            with performance_stage(performance_report, "Topology selection and image UI sync"):
                                update_ui_from_selection(context)
                                if allow_active_image_update:
                                    update_active_image_from_face(context)
                            set_file_loaded_into_edit_depsgraph(False)
                            with performance_stage(performance_report, "Snapshot face selection"):
                                snapshot_selection(bm)
                            from ..operators import weld as _weld_mod
                            _weld_mod._weld_just_stored = False
                            if not is_fresh_start and obj.anvil_auto_hotspot:
                                with performance_stage(performance_report, "Schedule forced auto-hotspot"):
                                    set_force_auto_hotspot(True)
                                    apply_auto_hotspots()
                        return

                    with performance_stage(performance_report, "Face selection change detection"):
                        selection_changed = check_selection_changed(bm)
                    if selection_changed:
                        debug_log(f"[Depsgraph] Selection changed")
                        with performance_stage(performance_report, "Selection and active-image UI sync"):
                            update_ui_from_selection(context)
                            if allow_active_image_update:
                                update_active_image_from_face(context)
                            from ..operators import weld as _weld_mod
                            if _weld_mod._weld_just_stored:
                                _weld_mod._weld_just_stored = False
                            else:
                                _weld_mod.clear_weld_on_bmesh(bm)

                    with performance_stage(performance_report, "Sync weld properties"):
                        from ..operators.weld import sync_weld_props
                        sync_weld_props(context, bm)

                    if not face_data_cache and context.mode == 'EDIT_MESH':
                        debug_log(f"[Depsgraph] Cache empty, rebuilding")
                        with performance_stage(performance_report, "Emergency face/UV cache rebuild"):
                            cache_face_data(context)

                    if is_geometry_update or is_transform_update:
                        with performance_stage(performance_report, "Edit Mode overlay invalidation"):
                            _invalidate_view_overlays()

                    debug_log(f"[Depsgraph] Applying world-scale UVs (cache size={len(face_data_cache)})")
                    with performance_stage(performance_report, "Apply world-scale UVs"):
                        apply_world_scale_uvs(obj, scene)
                    with performance_stage(performance_report, "Apply UV lock"):
                        apply_uv_lock(obj, scene)

                    if not is_fresh_start and obj.anvil_auto_hotspot:
                        with performance_stage(performance_report, "Schedule auto-hotspot"):
                            apply_auto_hotspots()

                    set_file_loaded_into_edit_depsgraph(False)
                    break
    except Exception as e:
        print(f"Anvil Level Design: Error in depsgraph handler: {e}", flush=True)
    finally:
        finish_performance_mode_report(performance_report)
