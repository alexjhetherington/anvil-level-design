import bpy
from bpy.types import Operator

from ..utils import is_level_design_workspace

# Precalculated grid scale steps (doubling / halving)
GRID_SCALES = [
    0.0078125,
    0.015625, 0.03125, 0.0625,
    0.125, 0.25, 0.5,
    1, 2, 4,
    8, 16, 32,
    64, 128, 256,
    512, 1024
]


def find_closest_scale(value):
    return min(range(len(GRID_SCALES)), key=lambda i: abs(GRID_SCALES[i] - value))


def get_all_3d_view_overlays():
    """Get overlay settings for all 3D views across all screens."""
    overlays = []
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        overlays.append(space.overlay)
    return overlays


def set_all_grid_scales(scale):
    """Set grid scale on all 3D views."""
    for overlay in get_all_3d_view_overlays():
        overlay.grid_scale = scale


class LEVELDESIGN_OT_grid_scale_up(Operator):
    """Increase grid scale to next step"""
    bl_idname = "leveldesign.grid_scale_up"
    bl_label = "Grid Scale Up"

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        overlays = get_all_3d_view_overlays()
        if not overlays:
            return {'CANCELLED'}

        # Use current viewport's scale as reference
        current_scale = overlays[0].grid_scale
        idx = find_closest_scale(current_scale)
        idx = min(idx + 1, len(GRID_SCALES) - 1)
        set_all_grid_scales(GRID_SCALES[idx])
        return {'FINISHED'}


class LEVELDESIGN_OT_grid_scale_down(Operator):
    """Decrease grid scale to previous step"""
    bl_idname = "leveldesign.grid_scale_down"
    bl_label = "Grid Scale Down"

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        overlays = get_all_3d_view_overlays()
        if not overlays:
            return {'CANCELLED'}

        # Use current viewport's scale as reference
        current_scale = overlays[0].grid_scale
        idx = find_closest_scale(current_scale)
        idx = max(idx - 1, 0)
        set_all_grid_scales(GRID_SCALES[idx])
        return {'FINISHED'}


classes = (
    LEVELDESIGN_OT_grid_scale_up,
    LEVELDESIGN_OT_grid_scale_down,
)

addon_keymaps = []

KEYMAPS_TO_REGISTER = [
    ("Object Mode", 'EMPTY'),
    ("Mesh", 'EMPTY'),
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Hotkeys
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    for km_name, space_type in KEYMAPS_TO_REGISTER:
        km = kc.keymaps.new(name=km_name, space_type=space_type)

        kmi = km.keymap_items.new(
            "leveldesign.grid_scale_up", 'RIGHT_BRACKET', 'PRESS',
            head=True
        )
        addon_keymaps.append((km, kmi))

        kmi = km.keymap_items.new(
            "leveldesign.grid_scale_down", 'LEFT_BRACKET', 'PRESS',
            head=True
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
