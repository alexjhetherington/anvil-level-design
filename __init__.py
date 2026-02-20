bl_info = {
    "name": "Anvil Level Design",
    "author": "Alex Hetherington",
    "version": (1, 2, 5),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Level Design",
    "description": "TrenchBroom-style UV tools, texture application, and grid controls for level design",
    "category": "3D View",
}

import bpy
from bpy.props import FloatProperty

from . import properties
from . import handlers
from . import operators
from . import panels
from . import workspace
from . import hotspot_mapping


# Default movement keys
DEFAULT_KEY_FORWARD = 'W'
DEFAULT_KEY_BACKWARD = 'S'
DEFAULT_KEY_LEFT = 'A'
DEFAULT_KEY_RIGHT = 'D'
DEFAULT_KEY_UP = 'E'
DEFAULT_KEY_DOWN = 'Q'

# Movement key bindings (keymap, keymap_item) tuples
freelook_movement_keymaps = []

# Direction names for display
MOVEMENT_DIRECTIONS = [
    ('forward', "Forward", DEFAULT_KEY_FORWARD),
    ('backward', "Backward", DEFAULT_KEY_BACKWARD),
    ('left', "Left", DEFAULT_KEY_LEFT),
    ('right', "Right", DEFAULT_KEY_RIGHT),
    ('up', "Up", DEFAULT_KEY_UP),
    ('down', "Down", DEFAULT_KEY_DOWN),
]


def get_movement_key(direction):
    """Get the configured key for a movement direction, or None if disabled."""
    wm = bpy.context.window_manager
    kc_user = wm.keyconfigs.user
    if kc_user:
        km = kc_user.keymaps.get("3D View")
        if km:
            for kmi in km.keymap_items:
                if (kmi.idname == "leveldesign.freelook_movement_key" and
                    hasattr(kmi.properties, 'direction') and
                    kmi.properties.direction == direction):
                    # Return None if the keymap item is disabled
                    if not kmi.active:
                        return None
                    return kmi.type
    # Return default if not found
    for d, name, default in MOVEMENT_DIRECTIONS:
        if d == direction:
            return default
    return None


def get_movement_keys_map():
    """Get a dict mapping key types to movement directions (only active bindings)."""
    keys = {}
    for direction, name, default in MOVEMENT_DIRECTIONS:
        key_type = get_movement_key(direction)
        if key_type is not None:
            keys[key_type] = direction
    return keys


class LEVELDESIGN_OT_freelook_movement_key(bpy.types.Operator):
    """Placeholder operator for freelook movement key bindings (handled by modal)"""
    bl_idname = "leveldesign.freelook_movement_key"
    bl_label = "Freelook Movement Key"
    bl_options = {'INTERNAL'}

    direction: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        from .utils import is_level_design_workspace
        return is_level_design_workspace()

    def execute(self, context):
        # This operator is never actually executed - it's just for keymap UI
        return {'PASS_THROUGH'}


class LEVELDESIGN_OT_restore_default_keybindings(bpy.types.Operator):
    """Restore all addon keybindings to their default values"""
    bl_idname = "leveldesign.restore_default_keybindings"
    bl_label = "Restore Default Keybindings"
    bl_options = {'REGISTER'}

    def execute(self, context):
        wm = context.window_manager
        kc_addon = wm.keyconfigs.addon
        kc_user = wm.keyconfigs.user
        if kc_addon and kc_user:
            # Find addon keymaps that contain our items, then restore the user versions
            for km_addon in kc_addon.keymaps:
                has_addon_items = any(
                    kmi.idname.startswith("leveldesign.")
                    for kmi in km_addon.keymap_items
                )
                if has_addon_items:
                    # Restore the user keymap (which references addon keymap as default)
                    km_user = kc_user.keymaps.get(km_addon.name)
                    if km_user:
                        km_user.restore_to_default()

        self.report({'INFO'}, "Keybindings restored to defaults")
        return {'FINISHED'}


class LevelDesignPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    mouse_sensitivity: FloatProperty(
        name="Mouse Sensitivity",
        description="Mouse look sensitivity for freelook camera",
        default=0.006,
        min=0.001,
        max=0.05,
        precision=4,
    )

    move_speed: FloatProperty(
        name="Move Speed",
        description="Movement speed for freelook camera (adjust with scroll wheel)",
        default=0.1,
        min=0.001,
        max=10.0,
        precision=3,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "mouse_sensitivity")
        layout.prop(self, "move_speed")

        # Workspaces section
        layout.separator()
        layout.label(text="Workspaces")
        row = layout.row()
        row.operator("leveldesign.create_level_design_workspace")
        row.enabled = not workspace.level_design_workspace_exists()
        row = layout.row()
        row.operator("leveldesign.create_hotspot_mapping_workspace")
        row.enabled = not workspace.hotspot_mapping_workspace_exists()

        # Keybindings section
        layout.separator()
        row = layout.row()
        row.label(text="Keybindings")
        row.operator("leveldesign.restore_default_keybindings", text="Restore Defaults")
        layout.label(text="Context Menu is a default Blender item but is here because by default this addon remaps it", icon='INFO')

        wm = context.window_manager
        kc_addon = wm.keyconfigs.addon
        kc_user = wm.keyconfigs.user
        if kc_addon:
            # Collect all addon keymap items with context
            # We iterate addon keymaps to find our items, then look up the user's version
            keymap_entries = []
            for km_addon in kc_addon.keymaps:
                # Find the corresponding user keymap
                km_user = kc_user.keymaps.get(km_addon.name) if kc_user else None

                for kmi_addon in km_addon.keymap_items:
                    if kmi_addon.idname.startswith("leveldesign."):
                        base_name = kmi_addon.name if kmi_addon.name else kmi_addon.idname

                        # Find the matching user keymap item
                        kmi_user = None
                        if km_user:
                            for kmi in km_user.keymap_items:
                                if kmi.idname == kmi_addon.idname:
                                    # For operators with properties, match on properties too
                                    if kmi_addon.idname == "leveldesign.ortho_view":
                                        if (hasattr(kmi.properties, "view_type") and
                                            hasattr(kmi_addon.properties, "view_type") and
                                            kmi.properties.view_type == kmi_addon.properties.view_type):
                                            kmi_user = kmi
                                            break
                                    elif kmi_addon.idname == "leveldesign.freelook_movement_key":
                                        if (hasattr(kmi.properties, "direction") and
                                            hasattr(kmi_addon.properties, "direction") and
                                            kmi.properties.direction == kmi_addon.properties.direction):
                                            kmi_user = kmi
                                            break
                                    elif kmi_addon.idname in ("leveldesign.backface_select",
                                                              "leveldesign.backface_object_select"):
                                        # Match on extend + loop properties
                                        props_match = True
                                        for prop_name in ("extend", "loop"):
                                            if (hasattr(kmi_addon.properties, prop_name) and
                                                hasattr(kmi.properties, prop_name)):
                                                if getattr(kmi_addon.properties, prop_name) != getattr(kmi.properties, prop_name):
                                                    props_match = False
                                                    break
                                        if props_match:
                                            kmi_user = kmi
                                            break
                                    else:
                                        kmi_user = kmi
                                        break

                        # Use user keymap item if found, otherwise fall back to addon
                        kmi_display = kmi_user if kmi_user else kmi_addon

                        # Check for property-based differentiation
                        if kmi_addon.idname == "leveldesign.ortho_view" and hasattr(kmi_addon.properties, "view_type"):
                            display_name = f"{base_name} - {kmi_addon.properties.view_type.title()}"
                        elif kmi_addon.idname == "leveldesign.freelook_movement_key" and hasattr(kmi_addon.properties, "direction"):
                            display_name = f"Freelook {kmi_addon.properties.direction.title()}"
                        elif kmi_addon.idname == "leveldesign.backface_select":
                            extend = getattr(kmi_addon.properties, "extend", False)
                            loop = getattr(kmi_addon.properties, "loop", False)
                            if loop and extend:
                                suffix = "Extend Loop"
                            elif loop:
                                suffix = "Loop"
                            elif extend:
                                suffix = "Extend"
                            else:
                                suffix = "Click"
                            display_name = f"Select ({suffix})"
                        elif kmi_addon.idname == "leveldesign.backface_object_select":
                            extend = getattr(kmi_addon.properties, "extend", False)
                            display_name = f"Object Select ({'Extend' if extend else 'Click'})"
                        else:
                            # Add keymap context in brackets for mode-based differentiation
                            display_name = f"{base_name} ({km_addon.name})"

                        keymap_entries.append((display_name, km_addon, kmi_display))

            # Sort alphabetically by display name
            keymap_entries.sort(key=lambda x: x[0].lower())

            # Draw sorted entries
            for display_name, km, kmi in keymap_entries:
                col = layout.column()
                row = col.row(align=True)
                row.label(text=display_name)
                row.prop(kmi, "map_type", text="")
                row.prop(kmi, "type", text="", full_event=True)
                row.prop(kmi, "active", text="", emboss=False)


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_freelook_movement_key)
    bpy.utils.register_class(LEVELDESIGN_OT_restore_default_keybindings)
    bpy.utils.register_class(LevelDesignPreferences)
    properties.register()
    handlers.register()
    operators.register()
    panels.register()
    workspace.register()
    hotspot_mapping.register()

    # Register movement key keymaps
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name="3D View", space_type='VIEW_3D')
        for direction, display_name, default_key in MOVEMENT_DIRECTIONS:
            kmi = km.keymap_items.new(
                "leveldesign.freelook_movement_key",
                default_key, 'PRESS'
            )
            kmi.properties.direction = direction
            freelook_movement_keymaps.append((km, kmi))


def unregister():
    # Remove movement key keymaps
    for km, kmi in freelook_movement_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    freelook_movement_keymaps.clear()

    hotspot_mapping.unregister()
    workspace.unregister()
    panels.unregister()
    operators.unregister()
    handlers.unregister()
    properties.unregister()
    bpy.utils.unregister_class(LevelDesignPreferences)
    bpy.utils.unregister_class(LEVELDESIGN_OT_restore_default_keybindings)
    bpy.utils.unregister_class(LEVELDESIGN_OT_freelook_movement_key)


if __name__ == "__main__":
    register()
