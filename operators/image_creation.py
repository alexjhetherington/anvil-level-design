"""Operator UI for creating simple image files."""

import os

import bpy
from bpy.props import (
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import Operator

from ..core.image_creation import (
    CreatedImageLoadError,
    ImageFileCreationError,
    ImageTextRenderError,
    create_image,
)
from ..core.workspace_check import (
    is_hotspot_mapping_workspace,
    is_level_design_workspace,
)
from ..texture_browser.previews import invalidate_texture_browser_preview_cache


_TEXTURE_BROWSER_POPUP_KEY = "anvil_texture_browser_popup"


def _display_path(path):
    return os.path.abspath(bpy.path.abspath(path)) if path else ""


def _png_filename(filename):
    filename = os.path.basename(filename.strip())
    if filename and not filename.lower().endswith(".png"):
        filename += ".png"
    return filename


def _image_filepath(directory, filename):
    filename = _png_filename(filename)
    return os.path.join(_display_path(directory), filename) if filename else ""


class LEVELDESIGN_OT_texture_browser_create_image(Operator):
    """Create a solid-colour PNG with optional centred text"""

    bl_idname = "leveldesign.texture_browser_create_image"
    bl_label = "Create Texture"
    bl_description = "Create a PNG with a chosen background and optional centred text"
    bl_options = {'REGISTER'}

    directory: StringProperty(
        name="Directory",
        subtype='DIR_PATH',
        options={'HIDDEN'},
    )
    filename: StringProperty(name="Filename", default="texture.png")
    width: IntProperty(name="Width", default=256, min=1, max=8192)
    height: IntProperty(name="Height", default=256, min=1, max=8192)
    color: FloatVectorProperty(
        name="Background Colour",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.5, 0.5, 0.5, 1.0),
    )
    text: StringProperty(
        name="Text",
        description="Optional text centred and automatically sized on the image",
        default="",
        options={'TEXTEDIT_UPDATE'},
    )
    text_color: FloatVectorProperty(
        name="Text Colour",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
    )

    @classmethod
    def poll(cls, context):
        screen = getattr(getattr(context, "window", None), "screen", None)
        try:
            is_texture_browser_popup = bool(
                screen is not None
                and screen.get(_TEXTURE_BROWSER_POPUP_KEY, False)
            )
        except ReferenceError:
            is_texture_browser_popup = False
        context_is_allowed = (
            is_level_design_workspace()
            or is_hotspot_mapping_workspace()
            or is_texture_browser_popup
        )
        collection_index = getattr(
            context.window_manager,
            "anvil_texture_browser_collection_index",
            -1,
        )
        return context_is_allowed and collection_index == -1

    def invoke(self, context, event):
        if not self.directory:
            self.directory = _display_path(getattr(
                context.window_manager,
                "anvil_texture_browser_folder_path",
                "",
            ))
        filepath = _image_filepath(self.directory, self.filename)
        confirm_text = (
            "Overwrite"
            if filepath and os.path.exists(filepath)
            else "Create Texture"
        )
        return context.window_manager.invoke_props_dialog(
            self,
            width=420,
            confirm_text=confirm_text,
        )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "filename")
        dimensions = layout.row(align=True)
        dimensions.prop(self, "width")
        dimensions.prop(self, "height")
        layout.prop(self, "color")
        layout.prop(self, "text")
        text_color_row = layout.row()
        text_color_row.enabled = bool(self.text.strip())
        text_color_row.prop(self, "text_color")
        filepath = _image_filepath(self.directory, self.filename)
        if filepath and os.path.exists(filepath):
            warning = layout.row()
            warning.alert = True
            warning.label(
                text="A file with this name will be overwritten",
                icon='ERROR',
            )

    def execute(self, context):
        directory = _display_path(self.directory)
        if not directory or not os.path.isdir(directory):
            self.report({'ERROR'}, "Current texture folder does not exist")
            return {'CANCELLED'}

        filename = _png_filename(self.filename)
        if not filename:
            self.report({'ERROR'}, "Enter a filename")
            return {'CANCELLED'}
        filepath = os.path.join(directory, filename)
        text = self.text.strip()

        try:
            create_image(
                filepath,
                self.width,
                self.height,
                tuple(self.color),
                text,
                tuple(self.text_color),
            )
        except ImageFileCreationError as exc:
            self.report({'ERROR'}, f"Could not create image: {exc}")
            return {'CANCELLED'}
        except ImageTextRenderError as exc:
            self.report({'ERROR'}, f"Could not add text to image: {exc}")
            return {'CANCELLED'}
        except CreatedImageLoadError as exc:
            self.report({'ERROR'}, f"Created file but could not load image: {exc}")
            return {'CANCELLED'}

        invalidate_texture_browser_preview_cache()
        if context.area is not None:
            context.area.tag_redraw()
        self.report({'INFO'}, f"Created {filename}")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_texture_browser_create_image)


def unregister():
    bpy.utils.unregister_class(LEVELDESIGN_OT_texture_browser_create_image)
