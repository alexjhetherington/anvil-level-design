"""
Box Create Tool

A modal tool for creating box meshes using the 3-state draw workflow.
Designed for level design workflows.
"""

import bpy
from . import operator
from ..modal_draw import preview


# Keymap items to track for cleanup
_addon_keymaps = []


def register():
    """Register the box create operator and keymap."""
    operator.register()

    # Register keymap
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Mesh', space_type='EMPTY')

        # B key to activate box create in edit mode
        kmi = km.keymap_items.new(
            operator.MESH_OT_box_create.bl_idname,
            type='B',
            value='PRESS',
            ctrl=False,
            shift=False,
            alt=False
        )

        _addon_keymaps.append((km, kmi))


def unregister():
    """Unregister the box create operator and keymap."""
    # Clean up keymap
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    # Clean up any active preview
    preview.cleanup_preview()

    operator.unregister()
