bl_info = {
    "name": "Anvil Level Design",
    "author": "Alex Hetherington",
    "version": (1, 6, 4),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > Level Design",
    "description": "TrenchBroom-style UV tools, texture application, and grid controls for level design",
    "category": "3D View",
}

import bpy
import blf

from . import properties
from . import handlers
from . import operators
from . import panels
from . import workspace
from . import hotspot_mapping
from .core.workspace_check import is_level_design_workspace
from .operators.gltf_export_extension import glTF2ExportUserExtension

_MINIMUM_BLENDER_VERSION = bl_info["blender"]
_VERSION_OK = bpy.app.version >= _MINIMUM_BLENDER_VERSION

_version_warning_handle = None


def _draw_version_warning():
    if not is_level_design_workspace():
        return

    region = bpy.context.region
    if not region:
        return

    font_id = 0
    min_ver = ".".join(str(v) for v in _MINIMUM_BLENDER_VERSION)
    cur_ver = ".".join(str(v) for v in bpy.app.version[:3])

    center_x = region.width // 2
    center_y = region.height // 2

    # Title
    blf.size(font_id, 32)
    blf.color(font_id, 1.0, 0.2, 0.2, 1.0)
    title = "UNSUPPORTED BLENDER VERSION"
    tw, th = blf.dimensions(font_id, title)
    blf.position(font_id, center_x - tw / 2, center_y + 20, 0)
    blf.draw(font_id, title)

    # Details
    blf.size(font_id, 20)
    blf.color(font_id, 1.0, 0.8, 0.4, 1.0)
    detail = f"Anvil Level Design requires Blender {min_ver} or newer (current: {cur_ver})"
    dw, dh = blf.dimensions(font_id, detail)
    blf.position(font_id, center_x - dw / 2, center_y - 20, 0)
    blf.draw(font_id, detail)


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


class LEVELDESIGN_OT_pref_double_pixels(bpy.types.Operator):
    bl_idname = "leveldesign.pref_double_pixels"
    bl_label = "x2"
    bl_description = "Double default pixels per meter"

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        prefs.pref_pixels_per_meter = min(int(prefs.pref_pixels_per_meter * 2), 4096)
        return {'FINISHED'}


class LEVELDESIGN_OT_pref_halve_pixels(bpy.types.Operator):
    bl_idname = "leveldesign.pref_halve_pixels"
    bl_label = "/2"
    bl_description = "Halve default pixels per meter"

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        prefs.pref_pixels_per_meter = max(int(prefs.pref_pixels_per_meter / 2), 1)
        return {'FINISHED'}


class LEVELDESIGN_OT_set_pref_interpolation(bpy.types.Operator):
    """Set the default interpolation mode in preferences"""
    bl_idname = "leveldesign.set_pref_interpolation"
    bl_label = "Set Pref Interpolation"

    interpolation: bpy.props.StringProperty()

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        prefs.pref_default_interpolation = self.interpolation
        return {'FINISHED'}


class LevelDesignPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    # === New File Defaults ===
    pref_pixels_per_meter: bpy.props.IntProperty(
        name="Pixels per Meter",
        description="Default pixels per meter for new files. Per-file setting is in Anvil (Settings) > Texture Settings",
        default=128,
        min=1,
        max=4096,
    )

    pref_default_interpolation: bpy.props.EnumProperty(
        name="Default Interpolation",
        description="Default interpolation mode for new files. Per-file setting is in Anvil (Settings) > Default Material Settings",
        items=[
            ('Closest', "Closest", "No interpolation (pixelated)"),
            ('Linear', "Linear", "Linear interpolation (smooth)"),
        ],
        default='Linear',
    )

    pref_default_texture_as_alpha: bpy.props.BoolProperty(
        name="Texture as Alpha",
        description="Default texture-as-alpha for new files. Per-file setting is in Anvil (Settings) > Default Material Settings",
        default=False,
    )

    pref_default_vertex_colors: bpy.props.BoolProperty(
        name="Vertex Colors",
        description="Default vertex colors for new files. Per-file setting is in Anvil (Settings) > Default Material Settings",
        default=False,
    )

    pref_default_roughness: bpy.props.FloatProperty(
        name="Roughness",
        description="Default roughness for new files. Per-file setting is in Anvil (Settings) > Default Material Settings",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    pref_default_metallic: bpy.props.FloatProperty(
        name="Metallic",
        description="Default metallic for new files. Per-file setting is in Anvil (Settings) > Default Material Settings",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    pref_default_emission_strength: bpy.props.FloatProperty(
        name="Emission Strength",
        description="Default emission strength for new files. Per-file setting is in Anvil (Settings) > Default Material Settings",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    pref_default_emission_color: bpy.props.FloatVectorProperty(
        name="Emission Color",
        description="Default emission color for new files. Per-file setting is in Anvil (Settings) > Default Material Settings",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
    )

    pref_default_specular: bpy.props.FloatProperty(
        name="Specular",
        description="Default specular (IOR Level) for new files. Per-file setting is in Anvil (Settings) > Default Material Settings",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    show_pref_experimental: bpy.props.BoolProperty(
        name="Experimental Settings",
        default=False,
    )

    # === Viewport Defaults (applied on load, override startup.blend) ===
    pref_default_unit_system: bpy.props.EnumProperty(
        name="Unit System",
        description="Scene unit system applied to new (unsaved) files on load",
        items=[
            ('NONE', "None", "No units"),
            ('METRIC', "Metric", "Metric units"),
            ('IMPERIAL', "Imperial", "Imperial units"),
        ],
        default='NONE',
    )

    pref_default_grid_subdivisions: bpy.props.IntProperty(
        name="Grid Subdivisions",
        description="Grid subdivisions applied to every 3D viewport on load",
        default=1,
        min=1,
        max=1024,
    )

    pref_default_show_extra_edge_length: bpy.props.BoolProperty(
        name="Show Edge Length",
        description="Show selected edge lengths in every 3D viewport on load (Measurement overlay)",
        default=True,
    )

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

        # New File Defaults section
        layout.separator()
        layout.label(text="New File Defaults")
        box = layout.box()
        box.label(text="Applied to new (unsaved) files. Per-file overrides in Anvil (Settings) sidebar.", icon='INFO')
        box.label(text="Changes here will not affect the current file.")

        box.separator()
        box.label(text="Texture Settings:")
        row = box.row(align=True)
        sub = row.row(align=True)
        sub.scale_x = 0.4
        sub.operator("leveldesign.pref_halve_pixels", text="/2")
        row.prop(self, "pref_pixels_per_meter")
        sub = row.row(align=True)
        sub.scale_x = 0.4
        sub.operator("leveldesign.pref_double_pixels", text="x2")

        box.separator()
        box.label(text="Viewport Defaults:")
        box.prop(self, "pref_default_grid_subdivisions")
        box.prop(self, "pref_default_show_extra_edge_length")

        box.separator()
        box.label(text="Scene Settings:")
        box.prop(self, "pref_default_unit_system")

        box.separator()
        box.label(text="Default Material Settings:")

        row = box.row(align=True)
        row.operator(
            "leveldesign.set_pref_interpolation",
            text="Closest",
            depress=(self.pref_default_interpolation == 'Closest'),
        ).interpolation = 'Closest'
        row.operator(
            "leveldesign.set_pref_interpolation",
            text="Linear",
            depress=(self.pref_default_interpolation == 'Linear'),
        ).interpolation = 'Linear'

        row = box.row()
        row.prop(
            self, "pref_default_texture_as_alpha",
            text="Texture as Alpha",
            icon='CHECKBOX_HLT' if self.pref_default_texture_as_alpha else 'CHECKBOX_DEHLT',
        )
        row = box.row()
        row.prop(
            self, "pref_default_vertex_colors",
            text="Vertex Colors",
            icon='CHECKBOX_HLT' if self.pref_default_vertex_colors else 'CHECKBOX_DEHLT',
        )
        box.prop(self, "pref_default_roughness")
        box.prop(self, "pref_default_metallic")

        # Experimental settings (collapsible)
        box.separator()
        row = box.row()
        row.prop(
            self, "show_pref_experimental",
            icon='DISCLOSURE_TRI_DOWN' if self.show_pref_experimental else 'DISCLOSURE_TRI_RIGHT',
            emboss=False,
        )
        if self.show_pref_experimental:
            sub = box.box()
            col = sub.column(align=True)
            col.scale_y = 0.7
            col.label(text="These settings may change in future")
            col.label(text="versions of Anvil as they are not")
            col.label(text="widely supported on game engine import.")
            sub.separator()
            sub.prop(self, "pref_default_emission_strength")
            row = sub.row(align=True)
            row.label(text="Emission Color")
            row.prop(self, "pref_default_emission_color", text="")
            sub.prop(self, "pref_default_specular")

        # Keybindings section
        layout.separator()
        row = layout.row()
        row.label(text="Keybindings")
        row.operator("leveldesign.restore_default_keybindings", text="Restore Defaults")

        # Category definitions: order matters for display
        REMAPPED_CATEGORY = "Remapped Defaults"
        CATEGORY_ORDER = [REMAPPED_CATEGORY, "Navigation", "Selection", "UV", "Tools", "Other"]
        CATEGORY_MAP = {
            "leveldesign.walk_navigation_hold": "Navigation",
            "leveldesign.ortho_view": "Navigation",
            "leveldesign.ortho_pan": "Navigation",
            "leveldesign.context_menu": REMAPPED_CATEGORY,
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
            "leveldesign.stretch_apply_image_to_face": "UV",
            "leveldesign.stretch_pick_image_from_face": "UV",
            "leveldesign.apply_uv_transform_to_face": "UV",
            "leveldesign.pick_uv_transform_from_face": "UV",
            "leveldesign.grid_scale_up": "Tools",
            "leveldesign.grid_scale_down": "Tools",
            "leveldesign.toggle_snap_mode": "Tools",
            "leveldesign.line_mode_activate": "Tools",
            "leveldesign.box_builder": "Tools",
            "leveldesign.cube_cut": "Tools",
            "leveldesign.knife_cut": "Tools",
            "leveldesign.context_weld": "Tools",
            "leveldesign.toggle_grid_overlay": "Tools",
            "leveldesign.uv_transform_modal": "UV",
            "leveldesign.snapping_mode_dispatch": "UV",
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
            context_menu_primary_kmi = None
            context_menu_all_user_kmis = []
            for km_addon in kc_addon.keymaps:
                # Find the corresponding user keymap
                km_user = kc_user.keymaps.get(km_addon.name) if kc_user else None

                for kmi_addon in km_addon.keymap_items:
                    # Check if this is our vertex paint menu remap
                    is_vp_menu = (kmi_addon.idname == "wm.call_panel"
                                  and hasattr(kmi_addon.properties, "name")
                                  and kmi_addon.properties.name == 'VIEW3D_PT_paint_vertex_context_menu')
                    if kmi_addon.idname.startswith("leveldesign.") or is_vp_menu:
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
                                            kmi.properties.normal_mode == kmi_addon.properties.normal_mode and
                                            hasattr(kmi.properties, "extend") and
                                            hasattr(kmi_addon.properties, "extend") and
                                            kmi.properties.extend == kmi_addon.properties.extend):
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

                        # Deduplicate context menu — show one combined entry for all modes
                        if kmi_addon.idname == "leveldesign.context_menu":
                            context_menu_all_user_kmis.append(kmi_display)
                            if context_menu_primary_kmi is None:
                                context_menu_primary_kmi = kmi_display
                                category = CATEGORY_MAP.get(kmi_addon.idname, "Other")
                                categorized_entries[category].append(("Context Menu (Right Click)", km_addon, kmi_display))
                            continue

                        # Check for property-based differentiation
                        if kmi_addon.idname == "leveldesign.ortho_view" and hasattr(kmi_addon.properties, "view_type"):
                            display_name = f"{base_name} - {kmi_addon.properties.view_type.title()}"
                        elif kmi_addon.idname == "leveldesign.select_linked":
                            mode = getattr(kmi_addon.properties, "normal_mode", "NONE")
                            extend = getattr(kmi_addon.properties, "extend", False)
                            if mode == 'EXPAND':
                                suffix = "Expand"
                            elif mode == 'SHRINK':
                                suffix = "Shrink"
                            elif extend:
                                suffix = "Extend"
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
                        elif kmi_addon.idname == "leveldesign.snapping_mode_dispatch":
                            display_name = "Face / Grid Snapping Mode"
                        elif is_vp_menu:
                            display_name = "Vertex Paint Color Menu"
                        else:
                            # Add keymap context in brackets for mode-based differentiation
                            display_name = f"{base_name} ({km_addon.name})"

                        if is_vp_menu:
                            category = REMAPPED_CATEGORY
                        else:
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
                if category == REMAPPED_CATEGORY:
                    box.label(text="Blender default hotkeys that have been remapped by this addon", icon='INFO')
                if category == "Navigation":
                    box.label(text="Walk navigation controls can be found in the Blender keymap menu at: 3D View -> View3D Walk Modal", icon='INFO')
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

            # Sync context menu keybinds across all modes
            if context_menu_primary_kmi:
                for kmi in context_menu_all_user_kmis:
                    if kmi is context_menu_primary_kmi:
                        continue
                    if kmi.map_type != context_menu_primary_kmi.map_type:
                        kmi.map_type = context_menu_primary_kmi.map_type
                    for attr in ('type', 'value', 'ctrl', 'shift', 'alt', 'oskey', 'active'):
                        if getattr(kmi, attr) != getattr(context_menu_primary_kmi, attr):
                            setattr(kmi, attr, getattr(context_menu_primary_kmi, attr))


def register():
    global _version_warning_handle

    print("Anvil Level Design: Debug logging is DISABLED (toggle in Anvil Settings > Debug)", flush=True)

    if not _VERSION_OK:
        min_ver = ".".join(str(v) for v in _MINIMUM_BLENDER_VERSION)
        cur_ver = ".".join(str(v) for v in bpy.app.version[:3])
        print(f"Anvil Level Design: WARNING - Blender {cur_ver} is not supported. "
              f"Minimum required version is {min_ver}.", flush=True)
        _version_warning_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_version_warning, (), 'WINDOW', 'POST_PIXEL'
        )

    bpy.utils.register_class(LEVELDESIGN_OT_restore_default_keybindings)
    bpy.utils.register_class(LEVELDESIGN_OT_pref_double_pixels)
    bpy.utils.register_class(LEVELDESIGN_OT_pref_halve_pixels)
    bpy.utils.register_class(LEVELDESIGN_OT_set_pref_interpolation)
    bpy.utils.register_class(LevelDesignPreferences)

    # Make face orientation front face transparent so only back faces are highlighted
    theme_3d = bpy.context.preferences.themes[0].view_3d
    theme_3d.face_front = (theme_3d.face_front[0], theme_3d.face_front[1], theme_3d.face_front[2], 0.0)

    properties.register()
    handlers.register()
    operators.register()
    panels.register()
    workspace.register()
    hotspot_mapping.register()


def unregister():
    global _version_warning_handle

    if _version_warning_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_version_warning_handle, 'WINDOW')
        _version_warning_handle = None

    hotspot_mapping.unregister()
    workspace.unregister()
    panels.unregister()
    operators.unregister()
    handlers.unregister()
    properties.unregister()
    bpy.utils.unregister_class(LevelDesignPreferences)
    bpy.utils.unregister_class(LEVELDESIGN_OT_set_pref_interpolation)
    bpy.utils.unregister_class(LEVELDESIGN_OT_pref_halve_pixels)
    bpy.utils.unregister_class(LEVELDESIGN_OT_pref_double_pixels)
    bpy.utils.unregister_class(LEVELDESIGN_OT_restore_default_keybindings)


if __name__ == "__main__":
    register()
