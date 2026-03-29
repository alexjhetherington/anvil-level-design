"""
Hotspot Mapping - Properties

Scene-level properties for tracking hotspot editing state.
Stores hotspot data in Blender properties for undo support.
"""

import bpy
from bpy.props import (StringProperty, PointerProperty, BoolProperty,
                       IntProperty)


# Snap size options (powers of 2)
SNAP_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_SNAP_SIZE = 32


class HotspotMappingProperties(bpy.types.PropertyGroup):
    """Properties for hotspot mapping state"""

    active_texture: StringProperty(
        name="Active Texture",
        description="Name of the texture currently being edited",
    )

    active_line_idx: IntProperty(
        name="Active Line",
        description="Index of the currently selected bisecting line (-1 = none)",
        default=-1,
    )

    snap_enabled: BoolProperty(
        name="Snap to Pixels",
        description="Snap hotspot positions and sizes to pixel grid",
        default=True,
    )

    snap_size: IntProperty(
        name="Snap Size",
        description="Pixel grid size for snapping",
        default=DEFAULT_SNAP_SIZE,
        min=1,
        max=512,
    )

    hotspots_file_path: StringProperty(
        name="Hotspots File",
        description="Path to external hotspots JSON file. Blank means data lives only in the .blend file",
        default="",
        subtype='FILE_PATH',
    )

    # Store hotspot data as JSON string for undo support
    # This is the source of truth; JSON file is synced on save/load
    hotspots_json: StringProperty(
        name="Hotspots Data",
        description="JSON string containing all hotspot data (for undo support)",
        default="",
    )


def register():
    bpy.utils.register_class(HotspotMappingProperties)
    bpy.types.Scene.hotspot_mapping_props = PointerProperty(
        type=HotspotMappingProperties
    )


def unregister():
    del bpy.types.Scene.hotspot_mapping_props
    bpy.utils.unregister_class(HotspotMappingProperties)
