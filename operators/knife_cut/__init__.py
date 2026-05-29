"""
Knife Cut Tool

A modal tool for grid-snapped knife cuts on mesh geometry.
Designed for level design workflows.

This module is architecturally separate from the rest of the addon
to maintain clean boundaries and easy modification.
"""

import bpy
from . import operator


# Keymap items to track for cleanup
_addon_keymaps = []


def register():
    """Register the knife cut operator and keymap."""
    operator.register()

    # Register keymap
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Mesh', space_type='EMPTY')

        # K key to activate knife cut in edit mode
        kmi = km.keymap_items.new(
            operator.MESH_OT_knife_cut.bl_idname,
            type='K',
            value='PRESS',
            ctrl=False,
            shift=False,
            alt=False
        )

        _addon_keymaps.append((km, kmi))


def unregister():
    """Unregister the knife cut operator and keymap."""
    # Clean up keymap
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    operator.unregister()
