"""Persistent user settings for the texture browser."""

import bpy
from bpy.props import CollectionProperty, IntProperty, StringProperty


class AnvilTextureBrowserFavoriteFolder(bpy.types.PropertyGroup):
    """A user-level favorite folder for the texture browser."""

    path: StringProperty(
        name="Folder",
        description="Absolute folder path",
        subtype='DIR_PATH',
    )


def texture_browser_suffix_update(self, context):
    suffix = self.suffix.strip().lower()
    if suffix != self.suffix:
        self.suffix = suffix


class AnvilTextureBrowserSuffixFilter(bpy.types.PropertyGroup):
    """A suffix include/exclude rule for the texture browser."""

    suffix: StringProperty(
        name="Suffix",
        description="File suffix, such as .png or _normal.png",
        update=texture_browser_suffix_update,
    )


class AnvilTextureBrowserCollectionItem(bpy.types.PropertyGroup):
    """A file stored in a user-level texture browser collection."""

    filepath: StringProperty(
        name="File",
        description="Absolute file path",
        subtype='FILE_PATH',
    )


class AnvilTextureBrowserCollection(bpy.types.PropertyGroup):
    """A named user-level collection of texture browser files."""

    name: StringProperty(
        name="Name",
        description="Collection name",
    )
    files: CollectionProperty(type=AnvilTextureBrowserCollectionItem)
    active_file_index: IntProperty(
        name="Active File Index",
        default=0,
    )


classes = (
    AnvilTextureBrowserFavoriteFolder,
    AnvilTextureBrowserSuffixFilter,
    AnvilTextureBrowserCollectionItem,
    AnvilTextureBrowserCollection,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
