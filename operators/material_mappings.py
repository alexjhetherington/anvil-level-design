"""Fix Material Mappings analysis UI and lifecycle prompt."""

import bpy
from bpy.props import BoolProperty, CollectionProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Menu, Operator, PropertyGroup, UIList

from ..core.material_mappings import (
    analyze_material_mappings,
    apply_material_mapping_choices,
    eligible_materials,
    mapping_choice_conflicts,
)
from ..core.material_shader import shader_images
from ..core.workspace_check import is_level_design_workspace


_addon_keymaps = []
_MATERIAL_COLUMN_FACTOR = 0.34
_CURRENT_MAPPING_COLUMN_FACTOR = 0.46
_ARROW_COLUMN_FACTOR = 0.1


class AnvilMaterialMappingRow(PropertyGroup):
    material: PointerProperty(type=bpy.types.Material)
    initial_image: PointerProperty(type=bpy.types.Image)
    selected_image: PointerProperty(type=bpy.types.Image)
    was_mapped: BoolProperty(default=False)


def _mapping_rows(window_manager):
    return window_manager.anvil_material_mapping_rows


def _active_mapping_row(window_manager):
    rows = _mapping_rows(window_manager)
    index = window_manager.anvil_material_mapping_active_index
    if index < 0 or index >= len(rows):
        return None
    return rows[index]


def _mapping_row_matches_filters(row, search_text, unmanaged_after_only):
    if unmanaged_after_only and row.selected_image is not None:
        return False
    normalized_search = search_text.strip().casefold()
    if not normalized_search:
        return True
    current_mapping = (
        row.initial_image.name
        if row.was_mapped and row.initial_image is not None
        else "Unmanaged"
    )
    after_mapping = (
        row.selected_image.name
        if row.selected_image is not None
        else "Unmanaged"
    )
    searchable_values = (
        row.material.name if row.material is not None else "",
        current_mapping,
        after_mapping,
    )
    return any(
        normalized_search in value.casefold()
        for value in searchable_values
    )


def _visible_mapping_row_indices(window_manager):
    search_text = window_manager.anvil_material_mapping_search
    unmanaged_after_only = (
        window_manager.anvil_material_mapping_unmanaged_after_only
    )
    return [
        index
        for index, row in enumerate(_mapping_rows(window_manager))
        if _mapping_row_matches_filters(
            row,
            search_text,
            unmanaged_after_only,
        )
    ]


def _ensure_active_mapping_row_visible(window_manager):
    visible_indices = _visible_mapping_row_indices(window_manager)
    if not visible_indices:
        window_manager.anvil_material_mapping_active_index = 0
        return
    active_index = window_manager.anvil_material_mapping_active_index
    if active_index not in visible_indices:
        window_manager.anvil_material_mapping_active_index = visible_indices[0]


def _update_material_mapping_filters(window_manager, _context):
    _ensure_active_mapping_row_visible(window_manager)


def _populate_mapping_rows(window_manager):
    rows = _mapping_rows(window_manager)
    rows.clear()
    analyses = analyze_material_mappings(eligible_materials())
    for analysis in analyses:
        row = rows.add()
        row.material = analysis.material
        row.selected_image = analysis.suggested_image
        row.was_mapped = analysis.is_already_mapped
        if analysis.is_already_mapped:
            row.initial_image = analysis.suggested_image
    window_manager.anvil_material_mapping_active_index = 0
    _ensure_active_mapping_row_visible(window_manager)


def _mapping_conflict_material_pointers(window_manager):
    choices = [
        (row.material, row.selected_image)
        for row in _mapping_rows(window_manager)
        if row.material is not None
    ]
    conflicts = mapping_choice_conflicts(choices)
    return {
        material.as_pointer()
        for materials in conflicts.values()
        for material in materials
    }


class LEVELDESIGN_UL_material_mappings(UIList):
    def draw_filter(self, context, layout):
        layout.prop(
            context.window_manager,
            "anvil_material_mapping_unmanaged_after_only",
            text="After fixing: Unmanaged only",
            icon='FILTER',
            toggle=True,
        )

    def filter_items(self, context, data, property_name):
        search_text = context.window_manager.anvil_material_mapping_search
        unmanaged_after_only = (
            context.window_manager.anvil_material_mapping_unmanaged_after_only
        )
        flags = [
            self.bitflag_filter_item
            if _mapping_row_matches_filters(
                row,
                search_text,
                unmanaged_after_only,
            )
            else 0
            for row in getattr(data, property_name)
        ]
        return flags, []

    def draw_item(
            self, context, layout, data, item, icon, active_data,
            active_property, index):
        conflict_pointers = _mapping_conflict_material_pointers(context.window_manager)
        material = item.material
        if material is None:
            layout.label(text="Missing material", icon='ERROR')
            return
        is_conflict = material.as_pointer() in conflict_pointers
        initial_image = item.initial_image if item.was_mapped else None
        initial_icon = 'CHECKMARK' if initial_image is not None else 'DOT'
        selected_icon = (
            'ERROR'
            if is_conflict
            else 'CHECKMARK' if item.selected_image is not None else 'DOT'
        )
        row = layout.row(align=True)
        row.alert = is_conflict
        columns = row.split(factor=_MATERIAL_COLUMN_FACTOR)
        columns.label(text=material.name, icon='MATERIAL')
        transition = columns.split(
            factor=_CURRENT_MAPPING_COLUMN_FACTOR,
            align=True,
        )
        transition.label(
            text=initial_image.name if initial_image is not None else "Unmanaged",
            icon=initial_icon,
        )
        result = transition.split(factor=_ARROW_COLUMN_FACTOR, align=True)
        result.label(text="", icon='TRIA_RIGHT')
        result.label(
            text=(
                item.selected_image.name
                if item.selected_image is not None
                else "Unmanaged"
            ),
            icon=selected_icon,
        )


class LEVELDESIGN_OT_select_material_mapping_candidate(Operator):
    """Use this shader image as the material's proposed primary image"""

    bl_idname = "leveldesign.select_material_mapping_candidate"
    bl_label = "Select Material Mapping Candidate"
    bl_options = {'INTERNAL'}

    image_name: StringProperty()

    def execute(self, context):
        row = _active_mapping_row(context.window_manager)
        image = bpy.data.images.get(self.image_name)
        if row is None or image is None:
            return {'CANCELLED'}
        row.selected_image = image
        _ensure_active_mapping_row_visible(context.window_manager)
        return {'FINISHED'}


class LEVELDESIGN_MT_material_mapping_candidates(Menu):
    bl_idname = "LEVELDESIGN_MT_material_mapping_candidates"
    bl_label = "Images in Shader"

    def draw(self, context):
        layout = self.layout
        row = _active_mapping_row(context.window_manager)
        if row is None or row.material is None:
            layout.label(text="No active material")
            return
        images = shader_images(row.material)
        if not images:
            layout.label(text="Shader contains no images", icon='INFO')
            return
        for image in images:
            icon_id = 0
            try:
                if image.preview is None or not image.preview.icon_id:
                    image.preview_ensure()
                if image.preview is not None:
                    icon_id = image.preview.icon_id
            except (ReferenceError, RuntimeError):
                pass
            if icon_id:
                operator = layout.operator(
                    "leveldesign.select_material_mapping_candidate",
                    text=image.name,
                    icon_value=icon_id,
                )
            else:
                operator = layout.operator(
                    "leveldesign.select_material_mapping_candidate",
                    text=image.name,
                    icon='IMAGE_DATA',
                )
            operator.image_name = image.name


class LEVELDESIGN_OT_clear_material_mapping_choice(Operator):
    """Leave this material unmanaged by Anvil"""

    bl_idname = "leveldesign.clear_material_mapping_choice"
    bl_label = "Leave Material Unmanaged"
    bl_description = "Do not map an image to this material"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        row = _active_mapping_row(context.window_manager)
        if row is None:
            return {'CANCELLED'}
        row.selected_image = None
        _ensure_active_mapping_row_visible(context.window_manager)
        return {'FINISHED'}


def finish_browser_mapping_selection(material_name, image):
    window_manager = bpy.context.window_manager
    rows = _mapping_rows(window_manager)
    for row in rows:
        if row.material is not None and row.material.name == material_name:
            if image is not None:
                row.selected_image = image
            break
    _ensure_active_mapping_row_visible(window_manager)
    for window in bpy.context.window_manager.windows:
        try:
            for area in window.screen.areas:
                area.tag_redraw()
        except ReferenceError:
            continue


class LEVELDESIGN_OT_browse_material_mapping_image(Operator):
    """Choose any primary image using the Texture Browser"""

    bl_idname = "leveldesign.browse_material_mapping_image"
    bl_label = "Choose in Texture Browser"
    bl_description = "Temporarily open the Texture Browser to choose any primary image"
    bl_options = {'INTERNAL'}

    material_name: StringProperty()

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        from ..texture_browser import browser

        material = bpy.data.materials.get(self.material_name)
        if material is None:
            self.report({'ERROR'}, "Material was not found")
            return {'CANCELLED'}
        browser.begin_material_mapping_image_selection(material.name)
        result = bpy.ops.leveldesign.texture_browser()
        if result == {'CANCELLED'}:
            browser.cancel_material_mapping_image_selection()
        return result


class LEVELDESIGN_OT_fix_material_mappings(Operator):
    """Review and repair Anvil's one-to-one image/material mappings"""

    bl_idname = "leveldesign.fix_material_mappings"
    bl_label = "Fix Material Mappings"
    bl_description = "Review which primary image maps to each Anvil material"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        for scene in bpy.data.scenes:
            scene.anvil_material_mapping_prompt_handled = True
        _populate_mapping_rows(context.window_manager)
        return context.window_manager.invoke_props_dialog(self, width=850)

    def draw(self, context):
        layout = self.layout
        window_manager = context.window_manager
        rows = _mapping_rows(window_manager)
        conflicts = _mapping_conflict_material_pointers(window_manager)
        initially_mapped_count = sum(1 for row in rows if row.was_mapped)
        mapped_count = sum(1 for row in rows if row.selected_image is not None)

        summary = layout.row()
        summary.scale_y = 0.6
        summary.label(
            text=(
                f"{len(rows)} local materials. {initially_mapped_count} -> "
                f"{mapped_count} will be mapped. You can open this window "
                "at any time with Shift-4."
            )
        )
        if conflicts:
            conflict_label = layout.row()
            conflict_label.alert = True
            conflict_label.label(text=f"{len(conflicts)} conflicts", icon='ERROR')

        layout.prop(
            window_manager,
            "anvil_material_mapping_search",
            text="",
            icon='VIEWZOOM',
        )

        header = layout.row(align=True)
        header_columns = header.split(factor=_MATERIAL_COLUMN_FACTOR)
        header_columns.label(text="Material")
        header_transition = header_columns.split(
            factor=_CURRENT_MAPPING_COLUMN_FACTOR,
            align=True,
        )
        header_transition.label(text="Current mapping")
        header_result = header_transition.split(
            factor=_ARROW_COLUMN_FACTOR,
            align=True,
        )
        header_result.label(text="")
        header_result.label(text="After fixing")

        layout.template_list(
            "LEVELDESIGN_UL_material_mappings",
            "",
            window_manager,
            "anvil_material_mapping_rows",
            window_manager,
            "anvil_material_mapping_active_index",
            rows=9,
            sort_lock=True,
        )

        active = _active_mapping_row(window_manager)
        if active is not None and not _mapping_row_matches_filters(
                active,
                window_manager.anvil_material_mapping_search,
                window_manager.anvil_material_mapping_unmanaged_after_only,
        ):
            active = None
        if active is not None and active.material is not None:
            detail = layout.box()
            initial_image = active.initial_image if active.was_mapped else None
            detail.label(
                text=(
                    f"{active.material.name}, "
                    + (
                        f"currently mapped to {initial_image.name}"
                        if initial_image is not None
                        else "currently unmanaged"
                    )
                ),
                icon='MATERIAL',
            )
            controls = detail.row(align=True)
            candidate_text = (
                active.selected_image.name
                if active.selected_image is not None
                else "Choose image from shader"
            )
            controls.menu(
                "LEVELDESIGN_MT_material_mapping_candidates",
                text=candidate_text,
                icon='IMAGE_DATA',
            )
            browse = controls.operator(
                "leveldesign.browse_material_mapping_image",
                text="Browse All",
                icon='VIEWZOOM',
            )
            browse.material_name = active.material.name
            controls.operator(
                "leveldesign.clear_material_mapping_choice",
                text="Unmanaged",
                icon='X',
            )
            if active.material.as_pointer() in conflicts:
                conflict = detail.row()
                conflict.alert = True
                conflict.label(
                    text="This image is also selected for another material. Choose only one.",
                    icon='ERROR',
                )

        layout.separator()
        layout.prop(
            window_manager,
            "anvil_fix_material_mappings_rename",
            text=(
                "Rename mapped materials using the current naming pattern: "
                f"{context.scene.level_design_props.default_material_name_pattern}"
            ),
        )

    def execute(self, context):
        window_manager = context.window_manager
        choices = [
            (row.material, row.selected_image)
            for row in _mapping_rows(window_manager)
            if row.material is not None
        ]
        pattern = context.scene.level_design_props.default_material_name_pattern
        try:
            mapped_count, renamed_count = apply_material_mapping_choices(
                choices,
                window_manager.anvil_fix_material_mappings_rename,
                pattern,
            )
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report(
            {'INFO'},
            f"Mapped {mapped_count} material(s); renamed {renamed_count}",
        )
        return {'FINISHED'}

def _material_mapping_prompt_timer():
    if not bpy.data.filepath:
        return None
    scenes = list(bpy.data.scenes)
    if not scenes or all(scene.anvil_material_mapping_prompt_handled for scene in scenes):
        return None
    if not eligible_materials():
        for scene in scenes:
            scene.anvil_material_mapping_prompt_handled = True
        return None
    if not is_level_design_workspace():
        return 1.0
    windows = list(bpy.context.window_manager.windows)
    window = bpy.context.window or (windows[0] if windows else None)
    if window is None:
        return 1.0
    try:
        with bpy.context.temp_override(window=window):
            bpy.ops.leveldesign.fix_material_mappings('INVOKE_DEFAULT')
    except RuntimeError:
        return 1.0
    return None


def schedule_material_mapping_prompt():
    if not bpy.app.timers.is_registered(_material_mapping_prompt_timer):
        bpy.app.timers.register(_material_mapping_prompt_timer, first_interval=0.5)


classes = (
    AnvilMaterialMappingRow,
    LEVELDESIGN_UL_material_mappings,
    LEVELDESIGN_OT_select_material_mapping_candidate,
    LEVELDESIGN_MT_material_mapping_candidates,
    LEVELDESIGN_OT_clear_material_mapping_choice,
    LEVELDESIGN_OT_browse_material_mapping_image,
    LEVELDESIGN_OT_fix_material_mappings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.anvil_material_mapping_rows = CollectionProperty(
        type=AnvilMaterialMappingRow,
    )
    bpy.types.WindowManager.anvil_material_mapping_active_index = IntProperty(default=0)
    bpy.types.WindowManager.anvil_material_mapping_search = StringProperty(
        name="Search Material Mappings",
        description="Filter by material, current mapping, or mapping after fixing",
        update=_update_material_mapping_filters,
    )
    bpy.types.WindowManager.anvil_material_mapping_unmanaged_after_only = BoolProperty(
        name="Unmanaged After Fixing Only",
        description="Show only materials that will remain unmanaged after fixing",
        default=False,
        update=_update_material_mapping_filters,
    )
    bpy.types.WindowManager.anvil_fix_material_mappings_rename = BoolProperty(default=False)

    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig is not None:
        for keymap_name, space_type in (
                ("Object Mode", 'EMPTY'),
                ("Mesh", 'EMPTY')):
            keymap = keyconfig.keymaps.new(name=keymap_name, space_type=space_type)
            keymap_item = keymap.keymap_items.new(
                "leveldesign.fix_material_mappings",
                'FOUR',
                'PRESS',
                shift=True,
                head=True,
            )
            _addon_keymaps.append((keymap, keymap_item))

    schedule_material_mapping_prompt()


def unregister():
    if bpy.app.timers.is_registered(_material_mapping_prompt_timer):
        bpy.app.timers.unregister(_material_mapping_prompt_timer)
    for keymap, keymap_item in _addon_keymaps:
        try:
            keymap.keymap_items.remove(keymap_item)
        except ReferenceError:
            pass
    _addon_keymaps.clear()

    del bpy.types.WindowManager.anvil_fix_material_mappings_rename
    del bpy.types.WindowManager.anvil_material_mapping_unmanaged_after_only
    del bpy.types.WindowManager.anvil_material_mapping_search
    del bpy.types.WindowManager.anvil_material_mapping_active_index
    del bpy.types.WindowManager.anvil_material_mapping_rows
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
