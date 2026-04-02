"""
Hotspot Mapping - Panels

N-panel UI for Image Editor to manage hotspots.
"""

import bpy
from bpy.types import Panel

from . import json_storage
from ..core.workspace_check import is_hotspot_mapping_workspace


class HOTSPOT_PT_main_panel(Panel):
    """Hotspot Mapping Panel"""

    bl_label = "Hotspot Mapping"
    bl_idname = "HOTSPOT_PT_main_panel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Anvil'

    @classmethod
    def poll(cls, context):
        return is_hotspot_mapping_workspace()

    def draw(self, context):
        layout = self.layout
        space = context.space_data
        props = context.scene.hotspot_mapping_props

        # Hotspot file path selector
        box = layout.box()
        box.label(text="Hotspot File", icon='FILE')
        row = box.row(align=True)
        if props.hotspots_file_path:
            row.label(text=props.hotspots_file_path, icon='LINKED')
            row.operator("hotspot.clear_file_path", text="", icon='X')
        else:
            row.label(text="None (data in .blend only)")
        box.operator("hotspot.browse_file_path", text="Browse...",
                     icon='FILEBROWSER')

        layout.separator()

        # Check if image is open
        if not space.image:
            layout.label(text="Open an image to define hotspots",
                         icon='INFO')
            return

        image = space.image
        texture_name = image.name

        # Check if this texture is hotspottable
        if not json_storage.is_texture_hotspottable(texture_name):
            layout.operator(
                "hotspot.assign_hotspottable",
                text="Assign Hotspottable",
                icon='ADD'
            )
            return

        # Texture is hotspottable - show management UI

        # Header with remove option
        row = layout.row()
        row.label(text=f"Texture: {texture_name}", icon='TEXTURE')
        row.operator(
            "hotspot.remove_hotspottable",
            text="",
            icon='X'
        )

        layout.separator()

        # Show cells
        cells = json_storage.get_cells_with_orientations(texture_name)

        if cells:
            box = layout.box()
            box.label(text=f"Hotspots ({len(cells)})", icon='MESH_GRID')
            for i, (cx, cy, cw, ch, orientation, key) in enumerate(cells):
                row = box.row(align=True)

                # Label
                sub = row.row()
                sub.scale_x = 0.5
                sub.label(text=f"#{i + 1}")

                # Orientation type button
                orientation_symbols = {
                    'Any': '●',
                    'Upwards': '↑',
                    'Floor': '⌊',
                    'Ceiling': '⌈',
                }
                symbol = orientation_symbols.get(orientation, '*')
                sub = row.row()
                sub.scale_x = 1.0
                op = sub.operator(
                    "hotspot.cycle_orientation",
                    text=f"{symbol} {orientation}",
                )
                op.cell_key = key

                # Show dimensions
                sub = row.row()
                sub.scale_x = 0.8
                sub.label(text=f"{cw}x{ch}")

        layout.separator()

        # Instructions
        box = layout.box()
        box.label(text="Usage", icon='QUESTION')
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text="Click to add full bisecting line")
        col.label(text="Ctrl+Click for anchored partial line")
        col.label(text="Drag existing lines to move")
        col.label(text="X / Del to remove selected line")

        # Snapping settings
        layout.separator()
        box = layout.box()
        box.label(text="Snapping", icon='SNAP_ON')

        row = box.row(align=True)
        row.prop(props, "snap_enabled", text="")
        sub = row.row(align=True)
        sub.enabled = props.snap_enabled
        sub.prop(props, "snap_size", text="Grid")
        sub.operator("hotspot.snap_size_down", text="", icon='REMOVE')
        sub.operator("hotspot.snap_size_up", text="", icon='ADD')


classes = (
    HOTSPOT_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
