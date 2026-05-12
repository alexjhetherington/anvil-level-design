"""Cross-object edit-mode undo support for texture paint operations.

Blender's edit-mode undo stack only restores the active edit mesh.  Alt-click
texture paint can also write to another object's mesh, so we store before/after
snapshots for those target writes and key them to a tiny marker on the active
edit BMesh.  When Blender undo/redo crosses that marker, lifecycle handlers
restore the external target meshes to the matching snapshot.
"""

import bmesh
import bpy

from ..core.logging import debug_log


_FRONTIER_LAYER = "_anvil_cross_object_undo_frontier"

_next_transaction_id = 0
_transactions = {}
_last_frontiers = {}
_active_transactions_by_source = {}
_undone_transactions_by_source = {}
_undo_pre_state = {}
_redo_pre_state = {}


def _source_key(obj):
    return obj.data.name


def _target_key(obj):
    return obj.name


def _material_slots_snapshot(me):
    slots = []
    for mat in me.materials:
        mat_name = ""
        if mat is not None:
            try:
                mat_name = mat.name
            except ReferenceError:
                mat = None
                mat_name = ""
        slots.append({
            'name': mat_name,
            'material': mat,
        })
    return slots


def _snapshot_object(obj, bm):
    bm.faces.ensure_lookup_table()

    faces = []
    for face in bm.faces:
        faces.append({
            'material_index': face.material_index,
            'loop_count': len(face.loops),
        })

    uv_layers = []
    for layer_index in range(len(bm.loops.layers.uv)):
        uv_layer = bm.loops.layers.uv[layer_index]
        uv_layers.append({
            'name': uv_layer.name,
            'faces': [
                [(loop[uv_layer].uv.x, loop[uv_layer].uv.y)
                 for loop in face.loops]
                for face in bm.faces
            ],
        })

    return {
        'object_name': obj.name,
        'object': obj,
        'mesh_name': obj.data.name,
        'material_slots': _material_slots_snapshot(obj.data),
        'faces': faces,
        'uv_layers': uv_layers,
    }


def _resolve_object(snapshot):
    obj = snapshot.get('object')
    if obj is not None:
        try:
            if obj.name == snapshot['object_name'] and obj.data is not None:
                return obj
        except ReferenceError:
            pass
    return bpy.data.objects.get(snapshot['object_name'])


def _resolve_material(slot):
    mat = slot.get('material')
    if mat is not None:
        try:
            mat.name
            return mat
        except ReferenceError:
            pass

    mat_name = slot.get('name', "")
    if mat_name:
        return bpy.data.materials.get(mat_name)
    return None


def _restore_material_slots(me, slots):
    me.materials.clear()
    for slot in slots:
        mat = _resolve_material(slot)
        if mat is not None:
            me.materials.append(mat)


def _restore_snapshot(snapshot):
    obj = _resolve_object(snapshot)
    if obj is None or obj.type != 'MESH' or obj.data is None:
        debug_log("[CrossObjectUndo] Restore skipped: target object missing")
        return

    me = obj.data

    _restore_material_slots(me, snapshot['material_slots'])

    if me.is_editmode:
        bm = bmesh.from_edit_mesh(me)
        should_free = False
    else:
        bm = bmesh.new()
        bm.from_mesh(me)
        should_free = True

    try:
        bm.faces.ensure_lookup_table()
        snapshot_faces = snapshot['faces']
        if len(snapshot_faces) != len(bm.faces):
            debug_log("[CrossObjectUndo] Restore skipped: target topology changed")
            return

        for face, face_snapshot in zip(bm.faces, snapshot_faces):
            face.material_index = face_snapshot['material_index']

        for layer_snapshot in snapshot['uv_layers']:
            uv_layer = bm.loops.layers.uv.get(layer_snapshot['name'])
            if uv_layer is None:
                uv_layer = bm.loops.layers.uv.new(layer_snapshot['name'])
                bm.faces.ensure_lookup_table()

            for face, uvs in zip(bm.faces, layer_snapshot['faces']):
                if len(uvs) != len(face.loops):
                    continue
                for loop, uv in zip(face.loops, uvs):
                    loop[uv_layer].uv.x = uv[0]
                    loop[uv_layer].uv.y = uv[1]

        if me.is_editmode:
            bmesh.update_edit_mesh(me)
        else:
            bm.to_mesh(me)
            me.update()
    finally:
        if should_free:
            bm.free()


def begin_transaction(source_obj):
    global _next_transaction_id
    _next_transaction_id += 1
    transaction_id = _next_transaction_id
    source_key = _source_key(source_obj)
    _transactions[transaction_id] = {
        'source_key': source_key,
        'targets': {},
    }
    return transaction_id


def record_target_before(transaction_id, obj, bm):
    transaction = _transactions.get(transaction_id)
    if transaction is None:
        return

    key = _target_key(obj)
    if key in transaction['targets']:
        return

    transaction['targets'][key] = {
        'before': _snapshot_object(obj, bm),
        'after': None,
    }


def record_target_after(transaction_id, obj, bm):
    transaction = _transactions.get(transaction_id)
    if transaction is None:
        return

    key = _target_key(obj)
    entry = transaction['targets'].get(key)
    if entry is None:
        return

    entry['after'] = _snapshot_object(obj, bm)


def transaction_has_targets(transaction_id):
    transaction = _transactions.get(transaction_id)
    return transaction is not None and bool(transaction['targets'])


def discard_transaction(transaction_id):
    if transaction_id in _transactions:
        del _transactions[transaction_id]


def write_frontier_marker(source_obj, transaction_id):
    me = source_obj.data
    if not me.is_editmode:
        return

    bm = bmesh.from_edit_mesh(me)
    if len(bm.verts) == 0:
        return

    layer = bm.verts.layers.int.get(_FRONTIER_LAYER)
    if layer is None:
        layer = bm.verts.layers.int.new(_FRONTIER_LAYER)
        bm.verts.ensure_lookup_table()

    for vert in bm.verts:
        vert[layer] = transaction_id

    bmesh.update_edit_mesh(me)
    source_key = _source_key(source_obj)
    active_transactions = _active_transactions_by_source.setdefault(source_key, [])
    if transaction_id not in active_transactions:
        active_transactions.append(transaction_id)
    _undone_transactions_by_source[source_key] = []
    _last_frontiers[source_key] = transaction_id


def _read_frontier_from_context():
    context = bpy.context
    if context.mode != 'EDIT_MESH':
        return None, None

    obj = context.active_object
    if obj is None or obj.type != 'MESH' or obj.data is None:
        return None, None

    me = obj.data
    if not me.is_editmode:
        return None, None

    try:
        bm = bmesh.from_edit_mesh(me)
    except (ReferenceError, RuntimeError):
        return None, None

    layer = bm.verts.layers.int.get(_FRONTIER_LAYER)
    if layer is None or len(bm.verts) == 0:
        return _source_key(obj), 0

    frontier = 0
    for vert in bm.verts:
        marker_value = vert[layer]
        if marker_value > frontier:
            frontier = marker_value
    return _source_key(obj), frontier


def _restore_transaction(transaction_id, snapshot_name, source_key):
    transaction = _transactions.get(transaction_id)
    if transaction is None or transaction['source_key'] != source_key:
        return

    for entry in transaction['targets'].values():
        snapshot = entry.get(snapshot_name)
        if snapshot is not None:
            _restore_snapshot(snapshot)


def handle_undo_pre():
    source_key, current_frontier = _read_frontier_from_context()
    if source_key is None:
        return

    active_transactions = _active_transactions_by_source.get(source_key, [])
    active_transaction = None
    if active_transactions:
        active_transaction = active_transactions[-1]

    _undo_pre_state[source_key] = {
        'frontier': current_frontier,
        'active_transaction': active_transaction,
    }


def handle_undo_post():
    source_key, current_frontier = _read_frontier_from_context()
    if source_key is None:
        return

    previous_frontier = _last_frontiers.get(source_key)
    if previous_frontier is None:
        _last_frontiers[source_key] = current_frontier
        return

    pre_state = _undo_pre_state.pop(source_key, None)
    active_transactions = _active_transactions_by_source.setdefault(source_key, [])
    undone_transactions = _undone_transactions_by_source.setdefault(source_key, [])

    active_transaction = active_transactions[-1] if active_transactions else None
    pre_active_transaction = None
    pre_frontier = previous_frontier
    if pre_state is not None:
        pre_active_transaction = pre_state['active_transaction']
        pre_frontier = pre_state['frontier']

    transaction_to_restore = None
    if active_transaction is not None and current_frontier < active_transaction:
        transaction_to_restore = active_transaction
    elif (pre_active_transaction is not None
          and pre_frontier >= pre_active_transaction
          and current_frontier < pre_active_transaction):
        transaction_to_restore = pre_active_transaction

    if transaction_to_restore is not None:
        _restore_transaction(transaction_to_restore, 'before', source_key)
        if active_transactions and active_transactions[-1] == transaction_to_restore:
            active_transactions.pop()
        undone_transactions.append(transaction_to_restore)
        if active_transactions:
            _last_frontiers[source_key] = active_transactions[-1]
        else:
            _last_frontiers[source_key] = current_frontier
        return

    _last_frontiers[source_key] = current_frontier


def handle_redo_pre():
    source_key, current_frontier = _read_frontier_from_context()
    if source_key is None:
        return

    undone_transactions = _undone_transactions_by_source.get(source_key, [])
    redo_transaction = None
    if undone_transactions:
        redo_transaction = undone_transactions[-1]

    _redo_pre_state[source_key] = {
        'frontier': current_frontier,
        'redo_transaction': redo_transaction,
    }


def handle_redo_post():
    source_key, current_frontier = _read_frontier_from_context()
    if source_key is None:
        return

    previous_frontier = _last_frontiers.get(source_key)
    if previous_frontier is None:
        _last_frontiers[source_key] = current_frontier
        return

    pre_state = _redo_pre_state.pop(source_key, None)
    active_transactions = _active_transactions_by_source.setdefault(source_key, [])
    undone_transactions = _undone_transactions_by_source.setdefault(source_key, [])

    redo_transaction = None
    if pre_state is not None:
        redo_transaction = pre_state['redo_transaction']
    if redo_transaction is None and undone_transactions:
        redo_transaction = undone_transactions[-1]

    if redo_transaction is not None and current_frontier >= redo_transaction:
        _restore_transaction(redo_transaction, 'after', source_key)
        if undone_transactions and undone_transactions[-1] == redo_transaction:
            undone_transactions.pop()
        if redo_transaction not in active_transactions:
            active_transactions.append(redo_transaction)
        _last_frontiers[source_key] = redo_transaction
        return

    _last_frontiers[source_key] = current_frontier


def reset():
    global _next_transaction_id
    _next_transaction_id = 0
    _transactions.clear()
    _last_frontiers.clear()
    _active_transactions_by_source.clear()
    _undone_transactions_by_source.clear()
    _undo_pre_state.clear()
    _redo_pre_state.clear()
