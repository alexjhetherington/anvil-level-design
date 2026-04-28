"""Helpers for interpreting Blender modal operator state."""


PASSIVE_MODAL_OPERATOR_IDS = frozenset({
    'LEVELDESIGN_OT_prefab_browser_interaction',
    'LEVELDESIGN_OT_texture_browser_interaction',
})


def modal_operator_ids(window):
    if window is None:
        return set()
    try:
        return set(op.bl_idname for op in window.modal_operators)
    except ReferenceError:
        return set()


def blocking_modal_operator_ids(window):
    return modal_operator_ids(window) - PASSIVE_MODAL_OPERATOR_IDS
