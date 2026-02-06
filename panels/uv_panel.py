import bpy
from bpy.types import Panel

from ..utils import (
    get_selected_face_count,
    get_texture_node_from_material,
    get_viewport_grid_settings,
    find_material_with_image,
    get_principled_bsdf_from_material,
    is_texture_alpha_connected,
)
from ..handlers import get_active_image


class LEVELDESIGN_PT_grid_panel(Panel):
    """Grid Settings Panel"""

    bl_label = "Grid"
    bl_idname = "LEVELDESIGN_PT_grid_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    def draw(self, context):
        layout = self.layout

        overlay = get_viewport_grid_settings(context)
        if overlay:
            box = layout.box()
            box.label(
                text=f"Grid Size: {overlay.grid_scale}  [ / ]", icon='GRID'
            )


class LEVELDESIGN_PT_uv_lock_panel(Panel):
    """UV Lock Settings"""

    bl_label = "UV Lock"
    bl_idname = "LEVELDESIGN_PT_uv_lock_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    def draw(self, context):
        layout = self.layout
        obj = context.object
        in_edit_mode = context.mode == 'EDIT_MESH'

        row = layout.row()

        if obj and obj.type == 'MESH':
            row.enabled = in_edit_mode
            row.prop(
                obj,
                "anvil_uv_lock",
                text="UV Lock",
                toggle=True,
                icon='LOCKED' if obj.anvil_uv_lock else 'UNLOCKED',
            )

            if obj.anvil_uv_lock:
                layout.label(text="Texture locked to geometry", icon='INFO')
            else:
                layout.label(text="Texture at world scale", icon='INFO')

            if not in_edit_mode:
                layout.label(text="(Edit Mode required)", icon='INFO')
        else:
            row.enabled = False
            row.label(text="UV Lock")
            layout.label(text="Select a mesh object", icon='INFO')


class LEVELDESIGN_PT_uv_settings_panel(Panel):
    """UV Settings (Scale, Rotation, Offset)"""

    bl_label = "UV Settings"
    bl_idname = "LEVELDESIGN_PT_uv_settings_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    def draw(self, context):
        layout = self.layout
        props = context.scene.level_design_props
        has_selection = get_selected_face_count(context) > 0

        col = layout.column(align=True)
        col.enabled = has_selection

        # Scale row with link toggle - always show the same UI
        row = col.row(align=True)
        row.prop(props, "texture_scale_u")
        row.prop(props, "texture_scale_v")
        row.prop(
            props,
            "texture_scale_linked",
            text="",
            icon='LINKED' if props.texture_scale_linked else 'UNLINKED',
        )

        col.prop(props, "texture_rotation")

        # Offset row
        row = col.row(align=True)
        row.prop(props, "texture_offset_x")
        row.prop(props, "texture_offset_y")


class LEVELDESIGN_PT_uv_shortcuts_panel(Panel):
    """UV Shortcuts (Projection and Alignment)"""

    bl_label = "UV Shortcuts"
    bl_idname = "LEVELDESIGN_PT_uv_shortcuts_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    def draw(self, context):
        layout = self.layout
        props = context.scene.level_design_props

        # Projection with scale
        obj = context.object
        has_mesh = obj is not None and obj.type == 'MESH'
        row = layout.row(align=True)
        row.enabled = has_mesh
        row.operator(
            "leveldesign.face_aligned_project",
            text="Face-Aligned Project",
            icon='MOD_UVPROJECT',
        )
        row.prop(props, "projection_scale", text="")

        # Alignment
        row = layout.row(align=True)
        row.operator(
            "leveldesign.align_uv", text="Center", icon='ALIGN_CENTER'
        ).direction = 'CENTER'
        row.operator(
            "leveldesign.fit_to_face",
            text="Fit to Face",
            icon='FULLSCREEN_ENTER',
        )


class LEVELDESIGN_PT_hotspotting_panel(Panel):
    """Hotspotting Controls"""

    bl_label = "Hotspotting"
    bl_idname = "LEVELDESIGN_PT_hotspotting_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    def draw(self, context):
        layout = self.layout
        props = context.scene.level_design_props
        obj = context.object

        # Auto hotspot checkbox
        layout.prop(props, "auto_hotspot")

        # Hotspot seam mode dropdown
        layout.prop(props, "hotspot_seam_mode", text="")

        # Allow combined faces checkbox (per-object)
        row = layout.row()
        if obj and obj.type == 'MESH':
            row.prop(obj, "anvil_allow_combined_faces")
        else:
            row.enabled = False
            row.label(text="Allow Combined Faces")

        # Hotspot size weight (per-object)
        row = layout.row()
        if obj and obj.type == 'MESH':
            row.prop(obj, "anvil_hotspot_size_weight")
        else:
            row.enabled = False
            row.label(text="Size Weight")

        # Apply Hotspot button (works in both edit and object mode)
        row = layout.row()
        row.operator(
            "leveldesign.apply_hotspot",
            text="Apply Hotspot",
            icon='UV_DATA',
        )


class LEVELDESIGN_PT_texture_preview_panel(Panel):
    """Texture Preview"""

    bl_label = "Texture Preview"
    bl_idname = "LEVELDESIGN_PT_texture_preview_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    def draw(self, context):
        layout = self.layout

        image = get_active_image()

        if image:
            layout.label(text=image.name)
            if image.preview:
                icon_id = image.preview.icon_id
                if icon_id:
                    layout.template_icon(icon_value=icon_id, scale=8.0)
                else:
                    image.preview_ensure()
                    layout.label(text="(Loading preview...)")
            else:
                image.preview_ensure()
                layout.label(text="(Loading preview...)")

            # Material settings
            mat = find_material_with_image(image)
            tex = get_texture_node_from_material(mat)
            bsdf = get_principled_bsdf_from_material(mat) if mat else None

            layout.separator()

            row = layout.row(align=True)
            if tex:
                row.operator(
                    "leveldesign.set_interpolation_closest",
                    text="Closest",
                    depress=(tex.interpolation == 'Closest'),
                )
                row.operator(
                    "leveldesign.set_interpolation_linear",
                    text="Linear",
                    depress=(tex.interpolation == 'Linear'),
                )
            else:
                row.enabled = False
                row.label(text="Closest / Linear")

            layout.separator()

            # Texture alpha checkbox
            row = layout.row()
            if mat:
                alpha_connected = is_texture_alpha_connected(mat)
                row.operator(
                    "leveldesign.toggle_texture_alpha",
                    text="Texture as Alpha",
                    icon=(
                        'CHECKBOX_HLT' if alpha_connected else 'CHECKBOX_DEHLT'
                    ),
                    depress=alpha_connected,
                )
            else:
                row.enabled = False
                row.label(text="Texture as Alpha: No material")

            # Roughness slider
            row = layout.row()
            if bsdf:
                row.prop(
                    bsdf.inputs["Roughness"], "default_value", text="Roughness"
                )
            else:
                row.enabled = False
                row.label(text="Roughness: No material")

            # Fix alpha bleed button
            layout.separator()
            layout.operator(
                "leveldesign.fix_alpha_bleed", icon='IMAGE_RGB_ALPHA'
            )
        else:
            layout.label(text="No texture selected")


class LEVELDESIGN_PT_texture_settings_panel(Panel):
    """Texture Settings (Pixels per Meter)"""

    bl_label = "Texture Settings"
    bl_idname = "LEVELDESIGN_PT_texture_settings_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    def draw(self, context):
        layout = self.layout
        props = context.scene.level_design_props

        row = layout.row(align=True)

        # halve button
        sub = row.row(align=True)
        sub.scale_x = 0.4
        sub.operator("leveldesign.halve_pixels", text="/2")

        # main property
        row.prop(props, "pixels_per_meter")

        # double button
        sub = row.row(align=True)
        sub.scale_x = 0.4
        sub.operator("leveldesign.double_pixels", text="x2")

        layout.separator()

        in_object_mode = context.mode == 'OBJECT'
        row = layout.row()
        row.enabled = in_object_mode
        row.operator("leveldesign.cleanup_unused_materials", icon='BRUSH_DATA')
        if not in_object_mode:
            layout.label(text="(Requires Object Mode)", icon='INFO')


class LEVELDESIGN_PT_export_panel(Panel):
    """Export Panel"""

    bl_label = "Export"
    bl_idname = "LEVELDESIGN_PT_export_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    def draw(self, context):
        layout = self.layout
        props = context.scene.level_design_props

        has_last_export = props.last_export_filepath != ""

        col = layout.column()
        col.enabled = has_last_export
        col.operator("leveldesign.export_glb_quick", icon='EXPORT')

        if has_last_export:
            import os

            filename = os.path.basename(props.last_export_filepath)
            layout.label(text=f"File: {filename}", icon='FILE')
            layout.label(text=f"Scale: {props.last_export_scale}")
        else:
            layout.label(text="No previous export", icon='INFO')
            layout.label(text="Use File > Export > GLB Scaled first")


classes = (
    LEVELDESIGN_PT_grid_panel,
    LEVELDESIGN_PT_uv_lock_panel,
    LEVELDESIGN_PT_uv_settings_panel,
    LEVELDESIGN_PT_uv_shortcuts_panel,
    LEVELDESIGN_PT_hotspotting_panel,
    LEVELDESIGN_PT_texture_preview_panel,
    LEVELDESIGN_PT_texture_settings_panel,
    LEVELDESIGN_PT_export_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
