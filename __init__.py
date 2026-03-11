bl_info = {
    "name": "Anvil Level Design",
    "author": "Alex Hetherington",
    "version": (1, 3, 3),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Level Design",
    "description": "TrenchBroom-style UV tools, texture application, and grid controls for level design",
    "category": "3D View",
}

import bpy

from . import properties
from . import handlers
from . import operators
from . import panels
from . import workspace
from . import hotspot_mapping


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

    def draw(self, context):
        layout = self.layout

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

        # Category definitions: order matters for display
        CATEGORY_ORDER = ["Navigation", "Selection", "UV", "Tools", "Other"]
        CATEGORY_MAP = {
            "leveldesign.walk_navigation_hold": "Navigation",
            "leveldesign.ortho_view": "Navigation",
            "leveldesign.ortho_pan": "Navigation",
            "leveldesign.context_menu": "Navigation",
            "leveldesign.backface_select": "Selection",
            "leveldesign.backface_object_select": "Selection",
            "leveldesign.backface_paint_select": "Selection",
            "leveldesign.select_linked": "Selection",
            "leveldesign.select_invalid_uvs": "Selection",
            "leveldesign.face_uv_mode": "UV",
            "leveldesign.face_aligned_project": "UV",
            "leveldesign.align_uv": "UV",
            "leveldesign.fit_to_face": "UV",
            "leveldesign.force_apply_texture": "UV",
            "leveldesign.apply_image_to_face": "UV",
            "leveldesign.pick_image_from_face": "UV",
            "leveldesign.grid_scale_up": "Tools",
            "leveldesign.grid_scale_down": "Tools",
            "leveldesign.line_mode_activate": "Tools",
            "leveldesign.box_builder": "Tools",
            "leveldesign.cube_cut": "Tools",
            "leveldesign.context_weld": "Tools",
        }

        wm = context.window_manager
        kc_addon = wm.keyconfigs.addon
        kc_user = wm.keyconfigs.user
        if kc_addon:
            # Collect all addon keymap items with context
            # We iterate addon keymaps to find our items, then look up the user's version
            categorized_entries = {cat: [] for cat in CATEGORY_ORDER}
            walk_nav_primary_kmi = None
            walk_nav_all_user_kmis = []
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
                                    elif kmi_addon.idname == "leveldesign.select_linked":
                                        if (hasattr(kmi.properties, "normal_mode") and
                                            hasattr(kmi_addon.properties, "normal_mode") and
                                            kmi.properties.normal_mode == kmi_addon.properties.normal_mode):
                                            kmi_user = kmi
                                            break
                                    elif kmi_addon.idname == "leveldesign.align_uv":
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

                        # Deduplicate walk navigation — show one combined entry for all modes
                        if kmi_addon.idname == "leveldesign.walk_navigation_hold":
                            walk_nav_all_user_kmis.append(kmi_display)
                            if walk_nav_primary_kmi is None:
                                walk_nav_primary_kmi = kmi_display
                                category = CATEGORY_MAP.get(kmi_addon.idname, "Other")
                                categorized_entries[category].append((base_name, km_addon, kmi_display))
                            continue

                        # Check for property-based differentiation
                        if kmi_addon.idname == "leveldesign.ortho_view" and hasattr(kmi_addon.properties, "view_type"):
                            display_name = f"{base_name} - {kmi_addon.properties.view_type.title()}"
                        elif kmi_addon.idname == "leveldesign.select_linked":
                            mode = getattr(kmi_addon.properties, "normal_mode", "NONE")
                            if mode == 'EXPAND':
                                suffix = "Expand"
                            elif mode == 'SHRINK':
                                suffix = "Shrink"
                            else:
                                suffix = "Linked"
                            display_name = f"Select Linked ({suffix})"
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
                        elif kmi_addon.idname == "leveldesign.face_aligned_project":
                            display_name = "Face Aligned Project"
                        elif kmi_addon.idname == "leveldesign.align_uv":
                            direction = getattr(kmi_addon.properties, "direction", "CENTER")
                            display_name = f"UV Align ({direction.title()})"
                        elif kmi_addon.idname == "leveldesign.fit_to_face":
                            display_name = "Fit to Face"
                        elif kmi_addon.idname == "leveldesign.select_invalid_uvs":
                            display_name = "Select Invalid UVs"
                        else:
                            # Add keymap context in brackets for mode-based differentiation
                            display_name = f"{base_name} ({km_addon.name})"

                        category = CATEGORY_MAP.get(kmi_addon.idname, "Other")
                        categorized_entries[category].append((display_name, km_addon, kmi_display))

            # Draw categorized entries
            for category in CATEGORY_ORDER:
                entries = categorized_entries[category]
                if not entries:
                    continue
                entries.sort(key=lambda x: x[0].lower())

                box = layout.box()
                box.label(text=category)
                for display_name, km, kmi in entries:
                    row = box.row(align=True)
                    row.label(text=display_name)
                    row.prop(kmi, "map_type", text="")
                    row.prop(kmi, "type", text="", full_event=True)
                    row.prop(kmi, "active", text="", emboss=False)

            # Sync walk navigation keybinds across all modes
            if walk_nav_primary_kmi:
                for kmi in walk_nav_all_user_kmis:
                    if kmi is walk_nav_primary_kmi:
                        continue
                    if kmi.map_type != walk_nav_primary_kmi.map_type:
                        kmi.map_type = walk_nav_primary_kmi.map_type
                    for attr in ('type', 'value', 'ctrl', 'shift', 'alt', 'oskey', 'active'):
                        if getattr(kmi, attr) != getattr(walk_nav_primary_kmi, attr):
                            setattr(kmi, attr, getattr(walk_nav_primary_kmi, attr))
                from .operators import walk_navigation
                walk_navigation.sync_confirm_key(walk_nav_primary_kmi.type)


def register():
    print("Anvil Level Design: Debug logging is DISABLED (toggle in Anvil Settings > Debug)", flush=True)
    bpy.utils.register_class(LEVELDESIGN_OT_restore_default_keybindings)
    bpy.utils.register_class(LevelDesignPreferences)
    properties.register()
    handlers.register()
    operators.register()
    panels.register()
    workspace.register()
    hotspot_mapping.register()


def unregister():
    hotspot_mapping.unregister()
    workspace.unregister()
    panels.unregister()
    operators.unregister()
    handlers.unregister()
    properties.unregister()
    bpy.utils.unregister_class(LevelDesignPreferences)
    bpy.utils.unregister_class(LEVELDESIGN_OT_restore_default_keybindings)


if __name__ == "__main__":
    register()
