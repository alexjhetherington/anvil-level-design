"""Configurable modal hotkeys shared by face and grid snapping."""

from bpy.types import Operator

from ..core.workspace_check import is_level_design_workspace


SNAPPING_HOTKEYS = (
    ('TOP', "leveldesign.snapping_texture_edge_top", 'W'),
    ('LEFT', "leveldesign.snapping_texture_edge_left", 'A'),
    ('BOTTOM', "leveldesign.snapping_texture_edge_bottom", 'S'),
    ('RIGHT', "leveldesign.snapping_texture_edge_right", 'D'),
    ('VERTICAL_FIT', "leveldesign.snapping_toggle_vertical_fit", 'Q'),
    ('HORIZONTAL_FIT', "leveldesign.snapping_toggle_horizontal_fit", 'E'),
    ('RESET_SCALE', "leveldesign.snapping_reset_scale", 'R'),
)

SNAPPING_TEXTURE_EDGES = {'TOP', 'LEFT', 'BOTTOM', 'RIGHT'}

_SNAPPING_ACTION_IDS = {
    operator_id: action
    for action, operator_id, _default_key in SNAPPING_HOTKEYS
}


def _event_matches_keymap_item(event, keymap_item):
    if not getattr(keymap_item, "active", True):
        return False
    if keymap_item.value != 'ANY' and event.value != keymap_item.value:
        return False
    if event.type != keymap_item.type:
        return False
    if getattr(keymap_item, "any", False):
        return True

    for attr_name in ('ctrl', 'shift', 'alt', 'oskey'):
        if getattr(event, attr_name, False) != getattr(keymap_item, attr_name, False):
            return False

    key_modifier = getattr(keymap_item, "key_modifier", 'NONE')
    if key_modifier != 'NONE':
        return getattr(event, "key_modifier", 'NONE') == key_modifier
    return True


def _effective_snapping_keymap_items(window_manager):
    user_keyconfig = window_manager.keyconfigs.user
    addon_keyconfig = window_manager.keyconfigs.addon
    user_keymap = user_keyconfig.keymaps.get("Mesh") if user_keyconfig else None
    addon_keymap = addon_keyconfig.keymaps.get("Mesh") if addon_keyconfig else None

    effective_items = []
    for _action, operator_id, _default_key in SNAPPING_HOTKEYS:
        user_items = []
        if user_keymap:
            user_items = [
                keymap_item
                for keymap_item in user_keymap.keymap_items
                if keymap_item.idname == operator_id
            ]
        if user_items:
            effective_items.extend(user_items)
            continue

        if addon_keymap:
            effective_items.extend(
                keymap_item
                for keymap_item in addon_keymap.keymap_items
                if keymap_item.idname == operator_id
            )

    return effective_items


def snapping_action_for_event(window_manager, event):
    for keymap_item in _effective_snapping_keymap_items(window_manager):
        if _event_matches_keymap_item(event, keymap_item):
            return _SNAPPING_ACTION_IDS[keymap_item.idname]
    return None


def _key_type_label(key_type):
    labels = {
        'LEFTMOUSE': "LMB",
        'MIDDLEMOUSE': "MMB",
        'RIGHTMOUSE': "RMB",
        'WHEELUPMOUSE': "Wheel Up",
        'WHEELDOWNMOUSE': "Wheel Down",
    }
    if key_type in labels:
        return labels[key_type]
    if len(key_type) == 1:
        return key_type
    return key_type.replace('_', ' ').title()


def _keymap_item_label(keymap_item):
    if not getattr(keymap_item, "active", True) or keymap_item.type == 'NONE':
        return "Unbound"

    parts = []
    key_modifier = getattr(keymap_item, "key_modifier", 'NONE')
    if key_modifier != 'NONE':
        parts.append(_key_type_label(key_modifier))
    if getattr(keymap_item, "ctrl", False):
        parts.append("Ctrl")
    if getattr(keymap_item, "shift", False):
        parts.append("Shift")
    if getattr(keymap_item, "alt", False):
        parts.append("Alt")
    if getattr(keymap_item, "oskey", False):
        parts.append("OS")
    parts.append(_key_type_label(keymap_item.type))
    return "+".join(parts)


def snapping_shortcut_labels(window_manager):
    labels = {
        action: "Unbound"
        for action, _operator_id, _default_key in SNAPPING_HOTKEYS
    }
    for keymap_item in _effective_snapping_keymap_items(window_manager):
        action = _SNAPPING_ACTION_IDS[keymap_item.idname]
        labels[action] = _keymap_item_label(keymap_item)
    return labels


class _SnappingHotkeyOperator(Operator):
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        return {'PASS_THROUGH'}


class LEVELDESIGN_OT_snapping_texture_edge_top(_SnappingHotkeyOperator):
    """Select the top texture edge while face or grid snapping"""
    bl_idname = "leveldesign.snapping_texture_edge_top"
    bl_label = "Snapping Texture Edge Top"


class LEVELDESIGN_OT_snapping_texture_edge_left(_SnappingHotkeyOperator):
    """Select the left texture edge while face or grid snapping"""
    bl_idname = "leveldesign.snapping_texture_edge_left"
    bl_label = "Snapping Texture Edge Left"


class LEVELDESIGN_OT_snapping_texture_edge_bottom(_SnappingHotkeyOperator):
    """Select the bottom texture edge while face or grid snapping"""
    bl_idname = "leveldesign.snapping_texture_edge_bottom"
    bl_label = "Snapping Texture Edge Bottom"


class LEVELDESIGN_OT_snapping_texture_edge_right(_SnappingHotkeyOperator):
    """Select the right texture edge while face or grid snapping"""
    bl_idname = "leveldesign.snapping_texture_edge_right"
    bl_label = "Snapping Texture Edge Right"


class LEVELDESIGN_OT_snapping_toggle_vertical_fit(_SnappingHotkeyOperator):
    """Toggle vertical texture fitting while face or grid snapping"""
    bl_idname = "leveldesign.snapping_toggle_vertical_fit"
    bl_label = "Snapping Toggle Vertical Fit"


class LEVELDESIGN_OT_snapping_toggle_horizontal_fit(_SnappingHotkeyOperator):
    """Toggle horizontal texture fitting while face or grid snapping"""
    bl_idname = "leveldesign.snapping_toggle_horizontal_fit"
    bl_label = "Snapping Toggle Horizontal Fit"


class LEVELDESIGN_OT_snapping_reset_scale(_SnappingHotkeyOperator):
    """Reset texture scale while face or grid snapping"""
    bl_idname = "leveldesign.snapping_reset_scale"
    bl_label = "Snapping Reset Scale"


classes = (
    LEVELDESIGN_OT_snapping_texture_edge_top,
    LEVELDESIGN_OT_snapping_texture_edge_left,
    LEVELDESIGN_OT_snapping_texture_edge_bottom,
    LEVELDESIGN_OT_snapping_texture_edge_right,
    LEVELDESIGN_OT_snapping_toggle_vertical_fit,
    LEVELDESIGN_OT_snapping_toggle_horizontal_fit,
    LEVELDESIGN_OT_snapping_reset_scale,
)
