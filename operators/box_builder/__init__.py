"""
Box Builder

A modal tool for creating box meshes using the 3-state draw workflow.
Designed for level design workflows.
"""

import bpy
from . import operator
from ..modal_draw import preview


# Keymap items to track for cleanup
_addon_keymaps = []


def register():
    """Register the box builder operator and keymap."""
    operator.register()

    # Register keymap
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Mesh', space_type='EMPTY')

        # B key to activate box builder in edit mode
        kmi = km.keymap_items.new(
            operator.MESH_OT_box_builder.bl_idname,
            type='B',
            value='PRESS',
            ctrl=False,
            shift=False,
            alt=False
        )

        _addon_keymaps.append((km, kmi))

        # B key in object mode
        km_obj = kc.keymaps.new(name='Object Mode', space_type='EMPTY')
        kmi_obj = km_obj.keymap_items.new(
            operator.MESH_OT_box_builder.bl_idname,
            type='B',
            value='PRESS',
            ctrl=False,
            shift=False,
            alt=False
        )
        _addon_keymaps.append((km_obj, kmi_obj))


def unregister():
    """Unregister the box builder operator and keymap."""
    # Clean up keymap
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    # Clean up any active preview
    preview.cleanup_preview()

    operator.unregister()
