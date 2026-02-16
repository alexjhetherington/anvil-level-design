"""
Modal Draw - Shared infrastructure for modal 3-state draw operators.

Provides the base operator, preview drawing, snapping, and utility functions
used by cube_cut, box_builder, and future modal draw tools.
"""

import bpy

from ...utils import is_level_design_workspace


class LEVELDESIGN_OT_line_mode_activate(bpy.types.Operator):
    """Placeholder operator for line mode modifier key binding (never actually executes)"""
    bl_idname = "leveldesign.line_mode_activate"
    bl_label = "Line Mode (Draw Rotated)"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        return {'PASS_THROUGH'}


# Keymap items to track for cleanup
_addon_keymaps = []


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_line_mode_activate)

    # Register line mode keymap
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Mesh', space_type='EMPTY')
        kmi = km.keymap_items.new(
            "leveldesign.line_mode_activate",
            'LEFT_SHIFT', 'PRESS'
        )
        _addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    _addon_keymaps.clear()

    bpy.utils.unregister_class(LEVELDESIGN_OT_line_mode_activate)
