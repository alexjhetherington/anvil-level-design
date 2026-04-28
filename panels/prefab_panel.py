"""Anvil (Prefabs) sidebar panels."""

import bpy
from bpy.types import Panel, UIList

from ..core.workspace_check import is_level_design_workspace


class LEVELDESIGN_UL_prefab_libraries(UIList):
    """Library list shown in the Libraries panel."""
    bl_idname = "LEVELDESIGN_UL_prefab_libraries"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        filepath = item.filepath or ""
        label = bpy.path.basename(filepath) if filepath else "<empty>"
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text=label, icon='FILE_BLEND')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=label, icon='FILE_BLEND')


class LEVELDESIGN_PT_prefab_mode_panel(Panel):
    """Prefab scene mode (Scene vs Library)"""
    bl_label = "Mode"
    bl_idname = "LEVELDESIGN_PT_prefab_mode_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil Prefabs'
    bl_order = 0

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        row = layout.row(align=True)
        row.operator(
            "leveldesign.set_prefab_mode",
            text="Scene",
            depress=(scene.anvil_prefab_mode == 'SCENE'),
        ).mode = 'SCENE'
        row.operator(
            "leveldesign.set_prefab_mode",
            text="Library",
            depress=(scene.anvil_prefab_mode == 'LIBRARY'),
        ).mode = 'LIBRARY'

        if scene.anvil_prefab_mode == 'LIBRARY':
            sub = layout.column(align=True)
            sub.scale_y = 0.8


class LEVELDESIGN_PT_prefab_controls_panel(Panel):
    """Prefab Controls"""
    bl_label = "Controls"
    bl_idname = "LEVELDESIGN_PT_prefab_controls_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil Prefabs'
    bl_order = 10

    @classmethod
    def poll(cls, context):
        return (is_level_design_workspace()
                and context.scene.anvil_prefab_mode == 'SCENE')

    def draw(self, context):
        layout = self.layout
        props = context.scene.level_design_props

        layout.use_property_decorate = False

        top = layout.row(align=True)
        top.operator(
            "leveldesign.prefab_refresh_libraries",
            text="Refresh Libraries",
            icon='FILE_REFRESH',
        )

        row = layout.row(align=True)
        row.prop(props, "prefab_inherit_normal", text="Inherit Normal", toggle=True)

        row = layout.row(align=True)
        row.prop(props, "prefab_random_scale_enabled", text="Random Size", toggle=True)
        reset_size = row.operator(
            "leveldesign.prefab_reset_random_transform",
            text="",
            icon='FILE_REFRESH',
        )
        reset_size.target = 'SIZE'

        if props.prefab_random_scale_enabled:
            scale_col = layout.column(align=True)
            row = scale_col.row(align=True)
            row.label(text="Min")
            row.prop(props, "prefab_random_scale_min", text="")
            row.prop(
                props,
                "prefab_random_scale_min_linked",
                text="",
                icon='LINKED' if props.prefab_random_scale_min_linked else 'UNLINKED',
            )
            row = scale_col.row(align=True)
            row.label(text="Max")
            row.prop(props, "prefab_random_scale_max", text="")
            row.prop(
                props,
                "prefab_random_scale_max_linked",
                text="",
                icon='LINKED' if props.prefab_random_scale_max_linked else 'UNLINKED',
            )

        row = layout.row(align=True)
        row.prop(
            props,
            "prefab_random_rotation_enabled",
            text="Random Rotation",
            toggle=True,
        )
        reset_rotation = row.operator(
            "leveldesign.prefab_reset_random_transform",
            text="",
            icon='FILE_REFRESH',
        )
        reset_rotation.target = 'ROTATION'

        if props.prefab_random_rotation_enabled:
            rotation_col = layout.column(align=True)
            row = rotation_col.row(align=True)
            row.label(text="Min")
            row.prop(props, "prefab_random_rotation_min", text="")
            row = rotation_col.row(align=True)
            row.label(text="Max")
            row.prop(props, "prefab_random_rotation_max", text="")


class LEVELDESIGN_PT_prefab_libraries_panel(Panel):
    """Prefab Libraries (.blend files)"""
    bl_label = "Libraries"
    bl_idname = "LEVELDESIGN_PT_prefab_libraries_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil Prefabs'
    bl_order = 20

    @classmethod
    def poll(cls, context):
        return (is_level_design_workspace()
                and context.scene.anvil_prefab_mode == 'SCENE')

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        row = layout.row()
        row.template_list(
            "LEVELDESIGN_UL_prefab_libraries",
            "",
            scene,
            "anvil_prefab_libraries",
            scene,
            "anvil_prefab_active_library_index",
            rows=4,
        )

        col = row.column(align=True)
        col.operator("leveldesign.prefab_add_library", text="", icon='ADD')
        col.operator("leveldesign.prefab_remove_library", text="", icon='REMOVE')

        idx = scene.anvil_prefab_active_library_index
        if 0 <= idx < len(scene.anvil_prefab_libraries):
            entry = scene.anvil_prefab_libraries[idx]
            sub = layout.column(align=True)
            sub.scale_y = 0.8
            sub.label(text=entry.filepath, icon='FILE_FOLDER')


class LEVELDESIGN_PT_prefab_prefabs_panel(Panel):
    """Prefab asset listing per library"""
    bl_label = "Prefabs"
    bl_idname = "LEVELDESIGN_PT_prefab_prefabs_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil Prefabs'
    bl_order = 30

    @classmethod
    def poll(cls, context):
        return (is_level_design_workspace()
                and context.scene.anvil_prefab_mode == 'SCENE')

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.operator(
            "leveldesign.prefab_browser",
            text="Browse Prefabs",
            icon='ASSET_MANAGER',
        )

        if not scene.anvil_prefab_libraries:
            layout.label(text="No libraries added", icon='INFO')
            return


class LEVELDESIGN_PT_prefab_library_prefabs_panel(Panel):
    """Prefab asset management for library scenes"""
    bl_label = "Prefabs"
    bl_idname = "LEVELDESIGN_PT_prefab_library_prefabs_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil Prefabs'
    bl_order = 10

    @classmethod
    def poll(cls, context):
        return (is_level_design_workspace()
                and context.scene.anvil_prefab_mode == 'LIBRARY')

    def draw(self, context):
        from ..prefabs.assets import iter_scene_prefab_assets

        layout = self.layout
        scene = context.scene

        info = layout.column(align=True)
        info.scale_y = 0.8
        info.label(text="Anvil prefabs are Blender Object Assets.", icon='INFO')
        info.label(text="Collection Assets are not supported.")
        info.label(text="Previews will be generated on save.")

        col = layout.column(align=True)
        col.operator(
            "leveldesign.prefab_make_free_objects_assets",
            text="Make All Free Objects Assets",
            icon='OBJECT_DATA',
        )

        layout.operator(
            "leveldesign.prefab_generate_previews",
            text="Generate Previews",
            icon='RENDER_STILL',
        )
        
        layout.separator()

        assets = iter_scene_prefab_assets(scene)
        if not assets:
            layout.label(text="No prefab assets", icon='INFO')
            return

        list_box = layout.box()
        for asset_type, asset_name in assets:
            row = list_box.row(align=True)
            select_op = row.operator(
                "leveldesign.prefab_select_asset",
                text=asset_name,
                icon='OBJECT_DATA',
            )
            select_op.asset_type = asset_type
            select_op.asset_name = asset_name
            clear_op = row.operator(
                "leveldesign.prefab_clear_asset",
                text="",
                icon='REMOVE',
            )
            clear_op.asset_type = asset_type
            clear_op.asset_name = asset_name


classes = (
    LEVELDESIGN_UL_prefab_libraries,
    LEVELDESIGN_PT_prefab_mode_panel,
    LEVELDESIGN_PT_prefab_library_prefabs_panel,
    LEVELDESIGN_PT_prefab_controls_panel,
    LEVELDESIGN_PT_prefab_libraries_panel,
    LEVELDESIGN_PT_prefab_prefabs_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
