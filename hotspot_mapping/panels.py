"""
Hotspot Mapping - Panels

N-panel UI for Image Editor to manage hotspots.
"""

import bpy
from bpy.types import Panel

from . import json_storage
from ..utils import is_hotspot_mapping_workspace


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

        # Check if blend file is saved
        if not bpy.data.filepath:
            layout.label(text="Save .blend file first", icon='ERROR')
            return

        # Check if image is open
        if not space.image:
            layout.label(text="Open an image to define hotspots", icon='INFO')
            return

        image = space.image
        texture_name = image.name

        # Check if this texture is hotspottable
        if not json_storage.is_texture_hotspottable(texture_name):
            # Show assign button
            layout.operator(
                "hotspot.assign_hotspottable",
                text="Assign Hotspottable",
                icon='ADD'
            )
            return

        # Texture is hotspottable - show hotspot management UI

        # Header with remove option
        row = layout.row()
        row.label(text=f"Texture: {texture_name}", icon='TEXTURE')
        row.operator(
            "hotspot.remove_hotspottable",
            text="",
            icon='X'
        )

        layout.separator()

        # Get hotspots for this texture
        hotspots = json_storage.get_texture_hotspots(texture_name)

        # Hotspot list
        if hotspots:
            box = layout.box()
            for hotspot in hotspots:
                hotspot_id = hotspot.get("id", "")
                is_active = (props.active_hotspot_id == hotspot_id)
                orientation = hotspot.get("orientation_type", "Any")

                row = box.row(align=True)

                # Selection button (whole row acts as selector)
                op = row.operator(
                    "hotspot.select_hotspot",
                    text=hotspot_id,
                    depress=is_active,
                    icon='RADIOBUT_ON' if is_active else 'RADIOBUT_OFF'
                )
                op.hotspot_id = hotspot_id

                # Orientation type button (cycles on click)
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
                op.hotspot_id = hotspot_id

                # Show dimensions
                sub = row.row()
                sub.scale_x = 0.8
                sub.label(text=f"{hotspot.get('width', 0)}x{hotspot.get('height', 0)}")

                # Delete button
                op = row.operator(
                    "hotspot.delete_hotspot",
                    text="",
                    icon='TRASH'
                )
                op.hotspot_id = hotspot_id
        else:
            layout.label(text="No hotspots defined", icon='INFO')

        layout.separator()

        # Add hotspot button
        layout.operator(
            "hotspot.add_hotspot",
            text="Add Hotspot",
            icon='ADD'
        )

        # Show active hotspot details
        if props.active_hotspot_id:
            hotspot = json_storage.get_hotspot_by_id(
                texture_name, props.active_hotspot_id
            )
            if hotspot:
                layout.separator()
                box = layout.box()
                box.label(text="Selected Hotspot", icon='RESTRICT_SELECT_OFF')
                col = box.column(align=True)
                col.label(text=f"Position: ({hotspot.get('x', 0)}, {hotspot.get('y', 0)})")
                col.label(text=f"Size: {hotspot.get('width', 0)} x {hotspot.get('height', 0)}")
                col.label(text=f"Orientation: {hotspot.get('orientation_type', 'Any')}")

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
