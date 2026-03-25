import bpy
from bpy.types import Panel, Operator

from ..utils import (
    get_selected_face_count,
    get_texture_node_from_material,
    find_material_with_image,
    get_principled_bsdf_from_material,
    is_texture_alpha_connected,
    is_vertex_colors_enabled,
    is_level_design_workspace,
)
from ..operators.grid_tools import get_unit_label
from ..handlers import (
    get_active_image,
    get_previous_image,
    get_multi_face_mode,
    is_multi_face_unset_scale,
    is_multi_face_unset_rotation,
    is_multi_face_unset_offset,
    get_selected_faces_share_image,
    get_all_selected_hotspot,
)


class LEVELDESIGN_PT_status_panel(Panel):
    """Status Panel"""

    bl_label = "Status"
    bl_idname = "LEVELDESIGN_PT_status_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def draw(self, context):
        layout = self.layout

        props = context.scene.level_design_props
        anvil_scale = props.anvil_grid_scale
        if anvil_scale == 0.0:
            anvil_scale = 1.0
        unit_settings = context.scene.unit_settings
        label = get_unit_label(unit_settings.system, unit_settings.length_unit)

        box = layout.box()
        if label:
            box.label(
                text=f"Grid Size: {anvil_scale}  ({label})  [ / ]", icon='GRID'
            )
        else:
            box.label(
                text=f"Grid Size: {anvil_scale}  [ / ]", icon='GRID'
            )

        from ..operators.weld import get_weld_display_name
        weld_mode = props.weld_mode
        weld_name = get_weld_display_name(weld_mode)
        box = layout.box()
        if weld_mode != 'NONE':
            box.label(text=f"Next Weld: {weld_name}  [ W ]", icon='AUTOMERGE_ON')
        else:
            box.label(text="Next Weld: None", icon='AUTOMERGE_ON')


class LEVELDESIGN_OT_set_active_render_uv(Operator):
    """Set the active render UV map"""
    bl_idname = "leveldesign.set_active_render_uv"
    bl_label = "Set Active Render UV"
    bl_options = {'INTERNAL'}

    uv_name: bpy.props.StringProperty()

    def execute(self, context):
        obj = context.object
        if not obj or obj.type != 'MESH':
            return {'CANCELLED'}

        me = obj.data
        for uv_map in me.uv_layers:
            uv_map.active_render = (uv_map.name == self.uv_name)

        # Sync settings now that we're in an operator context (writing allowed)
        from ..utils import sync_uv_map_settings
        sync_uv_map_settings(obj)

        return {'FINISHED'}


class LEVELDESIGN_OT_toggle_uv_lock(Operator):
    """Toggle UV map lock (sticker mode)"""
    bl_idname = "leveldesign.toggle_uv_lock"
    bl_label = "Toggle UV Lock"
    bl_options = {'INTERNAL'}

    uv_name: bpy.props.StringProperty()

    def execute(self, context):
        obj = context.object
        if not obj or obj.type != 'MESH':
            return {'CANCELLED'}

        from ..utils import sync_uv_map_settings
        sync_uv_map_settings(obj)

        for setting in obj.anvil_uv_map_settings:
            if setting.name == self.uv_name:
                setting.locked = not setting.locked
                return {'FINISHED'}

        return {'CANCELLED'}


class LEVELDESIGN_PT_uv_lock_panel(Panel):
    """UV Maps - per-layer lock/unlock"""

    bl_label = "UV Maps"
    bl_idname = "LEVELDESIGN_PT_uv_lock_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def draw(self, context):
        layout = self.layout
        obj = context.object
        in_edit_mode = context.mode == 'EDIT_MESH'
        has_mesh = obj and obj.type == 'MESH'

        if not in_edit_mode:
            layout.label(text="(Edit Mode required)", icon='INFO')

        if not has_mesh:
            row = layout.row()
            row.enabled = False
            row.label(text="No mesh object")
            return

        me = obj.data
        if not me.uv_layers:
            row = layout.row()
            row.enabled = False
            row.label(text="No UV maps")
            return

        if len(me.uv_layers) > 1:
            layout.label(text="Multi-UV maps is experimental!", icon='ERROR')

        settings = obj.anvil_uv_map_settings
        # Build a lookup of existing settings (read-only in draw context)
        settings_by_name = {s.name: s for s in settings}

        for uv_map in me.uv_layers:
            row = layout.row(align=True)
            row.enabled = in_edit_mode

            setting = settings_by_name.get(uv_map.name)

            # Render active icon (click to set)
            icon_sub = row.row(align=True)
            icon_sub.scale_x = 1.2
            op = icon_sub.operator(
                "leveldesign.set_active_render_uv",
                text="",
                icon='RESTRICT_RENDER_OFF' if uv_map.active_render else 'RESTRICT_RENDER_ON',
                depress=uv_map.active_render,
            )
            op.uv_name = uv_map.name

            row.separator(factor=0.5)

            # UV map name
            row.label(text=uv_map.name)

            # Lock toggle
            is_locked = setting.locked if setting is not None else False
            op = row.operator(
                "leveldesign.toggle_uv_lock",
                text="Locked" if is_locked else "Unlocked",
                icon='LOCKED' if is_locked else 'UNLOCKED',
                depress=is_locked,
            )
            op.uv_name = uv_map.name


class LEVELDESIGN_PT_uv_settings_panel(Panel):
    """UV Settings (Scale, Rotation, Offset)"""

    bl_label = "UV Settings"
    bl_idname = "LEVELDESIGN_PT_uv_settings_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def draw(self, context):
        layout = self.layout
        props = context.scene.level_design_props
        in_face_mode = (context.mode == 'EDIT_MESH' and
                        context.tool_settings.mesh_select_mode[2])
        has_selection = in_face_mode and get_selected_face_count(context) > 0
        multi_face = has_selection and get_multi_face_mode()
        all_hotspot = has_selection and get_all_selected_hotspot()

        # Warn if non-uniform object scale paired with non-uniform UV scale
        obj = context.object
        if obj and obj.type == 'MESH':
            s = obj.scale
            obj_non_uniform = (abs(s.x - s.y) > 1e-4
                               or abs(s.x - s.z) > 1e-4)
            uv_non_uniform = abs(props.texture_scale_u
                                 - props.texture_scale_v) > 1e-4
            if obj_non_uniform and uv_non_uniform:
                col_warn = layout.column(align=True)
                col_warn.label(
                    text="Non-Uniform object scale.",
                    icon='ERROR',
                )
                col_warn.label(
                    text="Apply Object Scale (Ctrl+A).",
                )

        if all_hotspot:
            layout.label(text="Hotspot texture", icon='INFO')

        col = layout.column(align=True)

        col.enabled = has_selection and not all_hotspot

        # Scale row with link toggle
        scale_row = col.row(align=True)
        scale_row.alert = multi_face and is_multi_face_unset_scale()
        scale_row.prop(props, "texture_scale_u")
        scale_row.prop(props, "texture_scale_v")
        scale_row.prop(
            props,
            "texture_scale_linked",
            text="",
            icon='LINKED' if props.texture_scale_linked else 'UNLINKED',
        )

        # Rotation
        rot_row = col.row(align=True)
        rot_row.alert = multi_face and is_multi_face_unset_rotation()
        rot_row.prop(props, "texture_rotation")

        # Offset row
        off_row = col.row(align=True)
        off_row.alert = multi_face and is_multi_face_unset_offset()
        off_row.prop(props, "texture_offset_x")
        off_row.prop(props, "texture_offset_y")


class LEVELDESIGN_PT_uv_shortcuts_panel(Panel):
    """UV Shortcuts (Projection and Alignment)"""

    bl_label = "UV Shortcuts"
    bl_idname = "LEVELDESIGN_PT_uv_shortcuts_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

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

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

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

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def draw(self, context):
        import bmesh

        layout = self.layout

        in_face_mode = (context.mode == 'EDIT_MESH' and
                        context.tool_settings.mesh_select_mode[2])
        image = get_active_image() if in_face_mode else None
        multi_face = in_face_mode and get_multi_face_mode()

        # In multi-face mode, check if all selected faces share the same image
        show_multi_texture_placeholder = False
        if multi_face and image is not None:
            obj = context.object
            if obj and obj.type == 'MESH' and context.mode == 'EDIT_MESH':
                bm = bmesh.from_edit_mesh(obj.data)
                bm.faces.ensure_lookup_table()
                shared, shared_image = get_selected_faces_share_image(obj, bm, obj.data)
                if not shared:
                    show_multi_texture_placeholder = True
                    image = None  # Don't show preview for mixed textures

        if show_multi_texture_placeholder:
            layout.label(text="Multiple textures")
            box = layout.box()
            box.scale_y = 8.0
            box.label(text="")
        elif image:
            layout.label(text=image.name)
            if image.preview:
                icon_id = image.preview.icon_id
                if icon_id:
                    layout.template_icon(icon_value=icon_id, scale=8.0)
                else:
                    image.preview_ensure()
                    box = layout.box()
                    box.scale_y = 8.0
                    box.label(text="")
            else:
                image.preview_ensure()
                box = layout.box()
                box.scale_y = 8.0
                box.label(text="")

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

            # Vertex colors checkbox
            row = layout.row()
            if mat:
                vc_enabled = is_vertex_colors_enabled(mat)
                row.operator(
                    "leveldesign.toggle_vertex_colors",
                    text="Vertex Colors",
                    icon=(
                        'CHECKBOX_HLT' if vc_enabled else 'CHECKBOX_DEHLT'
                    ),
                    depress=vc_enabled,
                )
            else:
                row.enabled = False
                row.label(text="Vertex Colors: No material")

            # Roughness slider
            row = layout.row()
            if bsdf:
                row.prop(
                    bsdf.inputs["Roughness"], "default_value", text="Roughness"
                )
            else:
                row.enabled = False
                row.label(text="Roughness: No material")

            # Metallic slider
            row = layout.row()
            if bsdf:
                row.prop(
                    bsdf.inputs["Metallic"], "default_value", text="Metallic"
                )
            else:
                row.enabled = False
                row.label(text="Metallic: No material")

            # Experimental settings (collapsible)
            layout.separator()
            props = context.scene.level_design_props
            row = layout.row()
            row.prop(
                props, "show_experimental_settings",
                icon='DISCLOSURE_TRI_DOWN' if props.show_experimental_settings else 'DISCLOSURE_TRI_RIGHT',
                emboss=False,
            )
            if props.show_experimental_settings:
                box = layout.box()
                col = box.column(align=True)
                col.scale_y = 0.7
                col.label(text="These settings may change in future")
                col.label(text="versions of Anvil as they are not")
                col.label(text="widely supported on game engine import.")
                box.separator()

                if bsdf:
                    box.prop(bsdf.inputs["Emission Strength"], "default_value", text="Emission Strength")
                    row = box.row(align=True)
                    row.label(text="Emission Color")
                    row.prop(bsdf.inputs["Emission Color"], "default_value", text="")
                    box.prop(bsdf.inputs["Specular IOR Level"], "default_value", text="Specular")
                else:
                    col = box.column()
                    col.enabled = False
                    col.label(text="Emission Strength: No material")
                    col.label(text="Emission Color: No material")
                    col.label(text="Specular: No material")

            # Fix alpha bleed button
            layout.separator()
            layout.operator(
                "leveldesign.fix_alpha_bleed", icon='IMAGE_RGB_ALPHA'
            )
        else:
            prev_image = get_previous_image()
            if prev_image:
                layout.enabled = False
                layout.label(text=prev_image.name)
                if prev_image.preview:
                    icon_id = prev_image.preview.icon_id
                    if icon_id:
                        layout.template_icon(icon_value=icon_id, scale=8.0)
                    else:
                        prev_image.preview_ensure()
                        box = layout.box()
                        box.scale_y = 8.0
                        box.label(text="")
                else:
                    prev_image.preview_ensure()
                    box = layout.box()
                    box.scale_y = 8.0
                    box.label(text="")
            else:
                layout.label(text="No texture selected")
                box = layout.box()
                box.scale_y = 8.0
                box.label(text="")


class LEVELDESIGN_PT_texture_settings_panel(Panel):
    """Texture Settings (Pixels per Meter)"""

    bl_label = "Texture Settings"
    bl_idname = "LEVELDESIGN_PT_texture_settings_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil (Settings)'

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

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


class LEVELDESIGN_PT_default_material_settings_panel(Panel):
    """Default Material Settings"""

    bl_label = "Default Material Settings"
    bl_idname = "LEVELDESIGN_PT_default_material_settings_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil (Settings)'

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def draw(self, context):
        layout = self.layout
        props = context.scene.level_design_props

        # Interpolation toggle
        row = layout.row(align=True)
        row.operator(
            "leveldesign.set_default_interpolation",
            text="Closest",
            depress=(props.default_interpolation == 'Closest'),
        ).interpolation = 'Closest'
        row.operator(
            "leveldesign.set_default_interpolation",
            text="Linear",
            depress=(props.default_interpolation == 'Linear'),
        ).interpolation = 'Linear'

        # Texture as alpha
        row = layout.row()
        row.prop(
            props, "default_texture_as_alpha",
            text="Texture as Alpha",
            icon='CHECKBOX_HLT' if props.default_texture_as_alpha else 'CHECKBOX_DEHLT',
        )

        # Vertex colors
        row = layout.row()
        row.prop(
            props, "default_vertex_colors",
            text="Vertex Colors",
            icon='CHECKBOX_HLT' if props.default_vertex_colors else 'CHECKBOX_DEHLT',
        )

        # Roughness
        layout.prop(props, "default_roughness")

        # Metallic
        layout.prop(props, "default_metallic")

        # Experimental settings (collapsible)
        layout.separator()
        row = layout.row()
        row.prop(
            props, "show_default_experimental_settings",
            icon='DISCLOSURE_TRI_DOWN' if props.show_default_experimental_settings else 'DISCLOSURE_TRI_RIGHT',
            emboss=False,
        )
        if props.show_default_experimental_settings:
            box = layout.box()
            col = box.column(align=True)
            col.scale_y = 0.7
            col.label(text="These settings may change in future")
            col.label(text="versions of Anvil as they are not")
            col.label(text="widely supported on game engine import.")
            box.separator()

            box.prop(props, "default_emission_strength")
            row = box.row(align=True)
            row.label(text="Emission Color")
            row.prop(props, "default_emission_color", text="")
            box.prop(props, "default_specular")


class LEVELDESIGN_PT_export_panel(Panel):
    """Export Panel"""

    bl_label = "Export"
    bl_idname = "LEVELDESIGN_PT_export_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil (Export)'

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

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


class LEVELDESIGN_OT_toggle_debug_logging(Operator):
    """Toggle debug logging to the console"""

    bl_idname = "leveldesign.toggle_debug_logging"
    bl_label = "Toggle Debug Logging"

    def execute(self, context):
        props = context.scene.level_design_props
        props.debug_logging = not props.debug_logging
        state = "enabled" if props.debug_logging else "disabled"
        print(f"Anvil Level Design: Debug logging {state}", flush=True)
        return {'FINISHED'}


class LEVELDESIGN_PT_debug_panel(Panel):
    """Debug options"""

    bl_label = "Debug"
    bl_idname = "LEVELDESIGN_PT_debug_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anvil (Settings)'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def draw(self, context):
        layout = self.layout
        layout.operator(
            "leveldesign.toggle_debug_logging",
            text="Debug Logging",
            depress=context.scene.level_design_props.debug_logging,
            icon='CONSOLE',
        )

        from ..operators.overlap_check import is_overlap_check_active, get_overlap_count
        active = is_overlap_check_active()
        count = get_overlap_count()
        if active and count > 0:
            text = f"Overlapping Faces ({count})"
        else:
            text = "Overlapping Faces"
        layout.operator(
            "leveldesign.toggle_overlap_check",
            text=text,
            depress=active,
            icon='ERROR',
        )


classes = (
    LEVELDESIGN_PT_status_panel,
    LEVELDESIGN_OT_set_active_render_uv,
    LEVELDESIGN_OT_toggle_uv_lock,
    LEVELDESIGN_PT_uv_lock_panel,
    LEVELDESIGN_PT_uv_settings_panel,
    LEVELDESIGN_PT_uv_shortcuts_panel,
    LEVELDESIGN_PT_hotspotting_panel,
    LEVELDESIGN_PT_texture_preview_panel,
    LEVELDESIGN_PT_texture_settings_panel,
    LEVELDESIGN_PT_default_material_settings_panel,
    LEVELDESIGN_PT_export_panel,
    LEVELDESIGN_OT_toggle_debug_logging,
    LEVELDESIGN_PT_debug_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
