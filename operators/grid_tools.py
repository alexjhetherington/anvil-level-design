import bpy
from bpy.types import Operator

from ..core.workspace_check import is_level_design_workspace

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

# Maps (unit_system, length_unit) to the factor for blender_grid_scale = anvil_scale * factor
_UNIT_FACTORS = {
    ('NONE', ''): 1.0,
    ('METRIC', 'ADAPTIVE'): 1.0,
    ('METRIC', 'KILOMETERS'): 1000.0,
    ('METRIC', 'METERS'): 1.0,
    ('METRIC', 'CENTIMETERS'): 0.01,
    ('METRIC', 'MILLIMETERS'): 0.001,
    ('METRIC', 'MICROMETERS'): 0.000001,
    ('IMPERIAL', 'ADAPTIVE'): 1.0,
    ('IMPERIAL', 'MILES'): 5280.0,
    ('IMPERIAL', 'FEET'): 1.0,
    ('IMPERIAL', 'INCHES'): 1.0 / 12.0,
    ('IMPERIAL', 'THOU'): 1.0 / 12000.0,
}

# Maps (unit_system, length_unit) to a human-readable label
_UNIT_LABELS = {
    ('METRIC', 'KILOMETERS'): 'kilometers',
    ('METRIC', 'METERS'): 'meters',
    ('METRIC', 'CENTIMETERS'): 'centimeters',
    ('METRIC', 'MILLIMETERS'): 'millimeters',
    ('METRIC', 'MICROMETERS'): 'micrometers',
    ('IMPERIAL', 'MILES'): 'miles',
    ('IMPERIAL', 'FEET'): 'feet',
    ('IMPERIAL', 'INCHES'): 'inches',
    ('IMPERIAL', 'THOU'): 'thou',
}


def get_unit_factor(unit_system, length_unit):
    """Return the conversion factor for the given unit settings."""
    return _UNIT_FACTORS.get((unit_system, length_unit), 1.0)


# Maps (unit_system, length_unit) to the actual unit size in meters.
# Used by the modal tools (cube cut, box builder) to compute snap grid size
# independently of overlay.grid_scale, so snap positions match Blender's
# unit-aware floor grid.
_SNAP_UNIT_METERS = {
    ('METRIC', 'KILOMETERS'): 1000.0,
    ('METRIC', 'METERS'): 1.0,
    ('METRIC', 'CENTIMETERS'): 0.01,
    ('METRIC', 'MILLIMETERS'): 0.001,
    ('METRIC', 'MICROMETERS'): 0.000001,
    ('IMPERIAL', 'MILES'): 1609.344,
    ('IMPERIAL', 'FEET'): 0.3048,
    ('IMPERIAL', 'INCHES'): 0.0254,
    ('IMPERIAL', 'THOU'): 0.0000254,
}


def get_snap_unit_size(unit_system, length_unit, scale_length):
    """Return the snap grid unit size (in Blender units) for the given settings.

    This converts the display unit size from meters to Blender units using
    scale_length, so snap points land on the same positions Blender draws
    its floor grid lines.
    """
    unit_meters = _SNAP_UNIT_METERS.get((unit_system, length_unit))
    if unit_meters is None:
        # ADAPTIVE or NONE — no unit-specific adjustment
        return 1.0
    if scale_length > 0:
        return unit_meters / scale_length
    return unit_meters


def get_unit_label(unit_system, length_unit):
    """Return a lowercase display label for the given unit settings, or empty string."""
    return _UNIT_LABELS.get((unit_system, length_unit), '')


def apply_anvil_grid_scale(anvil_scale, unit_system, length_unit):
    """Compute blender grid scale from anvil scale and unit settings, then apply."""
    factor = get_unit_factor(unit_system, length_unit)
    set_all_grid_scales(anvil_scale * factor)


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
        props = context.scene.level_design_props
        current_scale = props.anvil_grid_scale
        if current_scale == 0.0:
            current_scale = 1.0
        idx = find_closest_scale(current_scale)
        idx = min(idx + 1, len(GRID_SCALES) - 1)
        new_scale = GRID_SCALES[idx]
        props.anvil_grid_scale = new_scale
        unit_settings = context.scene.unit_settings
        apply_anvil_grid_scale(new_scale, unit_settings.system, unit_settings.length_unit)
        return {'FINISHED'}


class LEVELDESIGN_OT_toggle_snap_mode(Operator):
    """Toggle between incremental and grid snapping"""
    bl_idname = "leveldesign.toggle_snap_mode"
    bl_label = "Toggle Snap Mode"

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        tool_settings = context.scene.tool_settings
        snap = tool_settings.snap_elements
        has_increment = 'INCREMENT' in snap
        has_grid = 'GRID' in snap
        if has_increment and not has_grid:
            tool_settings.snap_elements = {'GRID'}
        elif has_grid and not has_increment:
            tool_settings.snap_elements = {'INCREMENT'}
        else:
            # Both or neither — default to incremental
            tool_settings.snap_elements = {'INCREMENT'}
        return {'FINISHED'}


class LEVELDESIGN_OT_grid_scale_down(Operator):
    """Decrease grid scale to previous step"""
    bl_idname = "leveldesign.grid_scale_down"
    bl_label = "Grid Scale Down"

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def execute(self, context):
        props = context.scene.level_design_props
        current_scale = props.anvil_grid_scale
        if current_scale == 0.0:
            current_scale = 1.0
        idx = find_closest_scale(current_scale)
        idx = max(idx - 1, 0)
        new_scale = GRID_SCALES[idx]
        props.anvil_grid_scale = new_scale
        unit_settings = context.scene.unit_settings
        apply_anvil_grid_scale(new_scale, unit_settings.system, unit_settings.length_unit)
        return {'FINISHED'}


def get_snap_mode_icon(tool_settings):
    """Return the appropriate icon for the current snap mode."""
    snap = tool_settings.snap_elements
    has_increment = 'INCREMENT' in snap
    has_grid = 'GRID' in snap
    if has_increment and not has_grid:
        return 'SNAP_INCREMENT'
    elif has_grid and not has_increment:
        return 'SNAP_GRID'
    else:
        return 'GRID'


classes = (
    LEVELDESIGN_OT_grid_scale_up,
    LEVELDESIGN_OT_grid_scale_down,
    LEVELDESIGN_OT_toggle_snap_mode,
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

        kmi = km.keymap_items.new(
            "leveldesign.toggle_snap_mode", 'G', 'PRESS',
            ctrl=True,
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
