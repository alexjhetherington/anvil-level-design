import bpy
from bpy.types import Operator

from ..core.workspace_check import is_level_design_workspace


class LEVELDESIGN_OT_toggle_edge_length_display(Operator):
    """Toggle selected edge length display in the current 3D View."""
    bl_idname = "leveldesign.toggle_edge_length_display"
    bl_label = "Toggle Edge Length Display"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        if not is_level_design_workspace():
            return False
        space = context.space_data
        return space is not None and space.type == 'VIEW_3D'

    def execute(self, context):
        space = context.space_data
        if space is None or space.type != 'VIEW_3D':
            return {'CANCELLED'}

        overlay = space.overlay
        overlay.show_extra_edge_length = not overlay.show_extra_edge_length
        if context.area is not None:
            context.area.tag_redraw()
        return {'FINISHED'}


classes = (
    LEVELDESIGN_OT_toggle_edge_length_display,
)

addon_keymaps = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    km = kc.keymaps.new(name='3D View', space_type='VIEW_3D')
    kmi = km.keymap_items.new(
        "leveldesign.toggle_edge_length_display",
        'NONE',
        'PRESS',
        head=True,
    )
    addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
