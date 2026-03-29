"""
Hotspot Mapping - Operators

Operators for managing hotspots: assign hotspottable, add/delete/select hotspots.
"""

import os

import bpy
from bpy.props import StringProperty, BoolProperty

from . import json_storage
from ..utils import debug_log


class HOTSPOT_OT_assign_hotspottable(bpy.types.Operator):
    """Mark the current image as hotspottable"""
    bl_idname = "hotspot.assign_hotspottable"
    bl_label = "Assign Hotspottable"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        if space and space.type == 'IMAGE_EDITOR':
            return space.image is not None
        return False

    def execute(self, context):
        space = context.space_data
        image = space.image

        if not image:
            self.report({'WARNING'}, "No image selected")
            return {'CANCELLED'}

        # Get image dimensions
        width = image.size[0] if image.size[0] > 0 else 1
        height = image.size[1] if image.size[1] > 0 else 1

        # Add to hotspots.json
        if json_storage.add_texture_as_hotspottable(image.name, width, height):
            self.report({'INFO'}, f"Added '{image.name}' as hotspottable")
            # Update active texture in scene props
            context.scene.hotspot_mapping_props.active_texture = image.name
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to save hotspots.json")
            return {'CANCELLED'}


class HOTSPOT_OT_remove_hotspottable(bpy.types.Operator):
    """Remove the current image from hotspottable list"""
    bl_idname = "hotspot.remove_hotspottable"
    bl_label = "Remove Hotspottable"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        if space and space.type == 'IMAGE_EDITOR' and space.image:
            return json_storage.is_texture_hotspottable(space.image.name)
        return False

    def execute(self, context):
        space = context.space_data
        image = space.image

        if json_storage.remove_texture_as_hotspottable(image.name):
            self.report({'INFO'}, f"Removed '{image.name}' from hotspottable")
            # Clear active texture
            context.scene.hotspot_mapping_props.active_texture = ""
            context.scene.hotspot_mapping_props.active_hotspot_id = ""
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to update hotspots.json")
            return {'CANCELLED'}


class HOTSPOT_OT_add_hotspot(bpy.types.Operator):
    """Add a new hotspot to the current texture"""
    bl_idname = "hotspot.add_hotspot"
    bl_label = "Add Hotspot"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        if space and space.type == 'IMAGE_EDITOR' and space.image:
            return json_storage.is_texture_hotspottable(space.image.name)
        return False

    def execute(self, context):
        space = context.space_data
        image = space.image
        texture_name = image.name

        # Get image dimensions
        img_width = image.size[0] if image.size[0] > 0 else 256
        img_height = image.size[1] if image.size[1] > 0 else 256

        # Default hotspot: 128x128 centered in image
        hotspot_size = 128
        x = (img_width - hotspot_size) // 2
        y = (img_height - hotspot_size) // 2

        # Clamp to valid range
        x = max(0, min(x, img_width - hotspot_size))
        y = max(0, min(y, img_height - hotspot_size))

        new_id = json_storage.add_hotspot(
            texture_name, x, y, hotspot_size, hotspot_size
        )

        if new_id:
            self.report({'INFO'}, f"Added hotspot: {new_id}")
            # Select the new hotspot
            context.scene.hotspot_mapping_props.active_hotspot_id = new_id
            # Force redraw
            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.tag_redraw()
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to add hotspot")
            return {'CANCELLED'}


class HOTSPOT_OT_delete_hotspot(bpy.types.Operator):
    """Delete a hotspot"""
    bl_idname = "hotspot.delete_hotspot"
    bl_label = "Delete Hotspot"
    bl_options = {'REGISTER', 'UNDO'}

    hotspot_id: StringProperty(
        name="Hotspot ID",
        description="ID of the hotspot to delete",
    )

    @classmethod
    def poll(cls, context):
        space = context.space_data
        if space and space.type == 'IMAGE_EDITOR' and space.image:
            return json_storage.is_texture_hotspottable(space.image.name)
        return False

    def execute(self, context):
        space = context.space_data
        image = space.image
        texture_name = image.name

        if not self.hotspot_id:
            self.report({'WARNING'}, "No hotspot specified")
            return {'CANCELLED'}

        if json_storage.delete_hotspot(texture_name, self.hotspot_id):
            self.report({'INFO'}, f"Deleted hotspot: {self.hotspot_id}")
            # Clear selection if we deleted the active hotspot
            props = context.scene.hotspot_mapping_props
            if props.active_hotspot_id == self.hotspot_id:
                props.active_hotspot_id = ""
            # Force redraw
            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.tag_redraw()
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to delete hotspot")
            return {'CANCELLED'}


class HOTSPOT_OT_select_hotspot(bpy.types.Operator):
    """Select a hotspot for editing"""
    bl_idname = "hotspot.select_hotspot"
    bl_label = "Select Hotspot"
    bl_options = {'REGISTER', 'UNDO'}

    hotspot_id: StringProperty(
        name="Hotspot ID",
        description="ID of the hotspot to select",
    )

    def execute(self, context):
        props = context.scene.hotspot_mapping_props
        props.active_hotspot_id = self.hotspot_id
        debug_log(f"[Hotspots] Selected: {self.hotspot_id}")
        # Force redraw
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()
        return {'FINISHED'}


class HOTSPOT_OT_browse_file_path(bpy.types.Operator):
    """Browse for a hotspot JSON file to link"""
    bl_idname = "hotspot.browse_file_path"
    bl_label = "Set Hotspot File"

    filepath: StringProperty(
        name="File Path",
        description="Path to the hotspots JSON file",
        subtype='FILE_PATH',
    )

    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'},
    )

    confirm_pending: BoolProperty(
        default=False,
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    resolved_path: StringProperty(
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        # Second call (after confirmation dialog) - apply directly
        if self.confirm_pending:
            self.confirm_pending = False
            _apply_file_path(context, self.resolved_path)
            return {'FINISHED'}

        filepath = self.filepath
        if not filepath:
            return {'CANCELLED'}

        # Ensure .json extension
        if not filepath.lower().endswith('.json'):
            filepath += '.json'

        absolute_path = os.path.normpath(os.path.abspath(filepath))

        # Check if we need a confirmation dialog
        file_exists = os.path.exists(absolute_path)
        has_hotspots = json_storage.scene_has_hotspots()

        if has_hotspots and file_exists:
            self.confirm_pending = True
            self.resolved_path = absolute_path
            return context.window_manager.invoke_props_dialog(self)

        _apply_file_path(context, absolute_path)
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.label(text="Loading this file will replace your current hotspot data.", icon='ERROR')


class HOTSPOT_OT_clear_file_path(bpy.types.Operator):
    """Unlink the external hotspot file (data stays in .blend only)"""
    bl_idname = "hotspot.clear_file_path"
    bl_label = "Unlink Hotspot File"

    @classmethod
    def poll(cls, context):
        scene = context.scene
        if scene and hasattr(scene, 'hotspot_mapping_props'):
            return bool(scene.hotspot_mapping_props.hotspots_file_path)
        return False

    def execute(self, context):
        context.scene.hotspot_mapping_props.hotspots_file_path = ""
        self.report({'INFO'}, "Hotspot file unlinked. Data remains in .blend file.")
        return {'FINISHED'}


def _apply_file_path(context, absolute_path):
    """Set the hotspot file path and load/create the file.

    Args:
        context: Blender context.
        absolute_path: Absolute path to the hotspot JSON file.
    """
    props = context.scene.hotspot_mapping_props

    # Store as relative if blend is saved, absolute otherwise
    stored_path = json_storage.make_path_relative(absolute_path)
    props.hotspots_file_path = stored_path

    # Load from file (creates if not found)
    json_storage.load_from_file(absolute_path)

    # Force UI redraw
    for area in context.screen.areas:
        area.tag_redraw()


classes = (
    HOTSPOT_OT_assign_hotspottable,
    HOTSPOT_OT_remove_hotspottable,
    HOTSPOT_OT_add_hotspot,
    HOTSPOT_OT_delete_hotspot,
    HOTSPOT_OT_select_hotspot,
    HOTSPOT_OT_browse_file_path,
    HOTSPOT_OT_clear_file_path,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
