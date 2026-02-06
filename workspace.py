import os

import bpy

from .utils import LEVEL_DESIGN_WORKSPACE_NAME, HOTSPOT_MAPPING_WORKSPACE_NAME


def _get_workspaces_blend_path():
    addon_dir = os.path.dirname(__file__)
    return os.path.join(addon_dir, "workspaces.blend")


def level_design_workspace_exists():
    """Check if the Level Design workspace already exists"""
    return LEVEL_DESIGN_WORKSPACE_NAME in bpy.data.workspaces


def hotspot_mapping_workspace_exists():
    """Check if the Hotspot Mapping workspace already exists."""
    return HOTSPOT_MAPPING_WORKSPACE_NAME in bpy.data.workspaces


def create_level_design_workspace():
    """Create Level Design workspace by appending from workspaces.blend.

    Used by the manual preferences button. Does not control tab ordering.
    """
    if level_design_workspace_exists():
        return False

    filepath = _get_workspaces_blend_path()
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.workspaces = [LEVEL_DESIGN_WORKSPACE_NAME]

    return level_design_workspace_exists()


def create_hotspot_mapping_workspace():
    """Create Hotspot Mapping workspace by appending from workspaces.blend.

    Used by the manual preferences button. Does not control tab ordering.
    """
    if hotspot_mapping_workspace_exists():
        return False

    filepath = _get_workspaces_blend_path()
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.workspaces = [HOTSPOT_MAPPING_WORKSPACE_NAME]

    return hotspot_mapping_workspace_exists()


class LEVELDESIGN_OT_create_level_design_workspace(bpy.types.Operator):
    """Create the Level Design workspace from the bundled template"""
    bl_idname = "leveldesign.create_level_design_workspace"
    bl_label = "Create Level Design Workspace"

    def execute(self, context):
        if create_level_design_workspace():
            self.report({'INFO'}, "Created Level Design workspace")
        else:
            self.report({'WARNING'}, "Level Design workspace already exists")
        return {'FINISHED'}


class LEVELDESIGN_OT_create_hotspot_mapping_workspace(bpy.types.Operator):
    """Create the Hotspot Mapping workspace from the bundled template"""
    bl_idname = "leveldesign.create_hotspot_mapping_workspace"
    bl_label = "Create Hotspot Mapping Workspace"

    def execute(self, context):
        if create_hotspot_mapping_workspace():
            self.report({'INFO'}, "Created Hotspot Mapping workspace")
        else:
            self.report({'WARNING'}, "Hotspot Mapping workspace already exists")
        return {'FINISHED'}


def _setup_addon_workspaces():
    """Create addon workspaces and configure scene settings on first enable."""
    create_hotspot_mapping_workspace()
    create_level_design_workspace()
    bpy.ops.workspace.reorder_to_front()

    # Configure scene settings
    scene = bpy.context.scene
    scene.unit_settings.system = 'NONE'
    scene.tool_settings.use_snap = True
    scene.tool_settings.snap_elements = {'INCREMENT'}

    # Set grid subdivisions to 1 on all 3D viewports
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.overlay.grid_subdivisions = 1

    #bpy.context.window.workspace = bpy.data.workspaces[LEVEL_DESIGN_WORKSPACE_NAME]


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_create_level_design_workspace)
    bpy.utils.register_class(LEVELDESIGN_OT_create_hotspot_mapping_workspace)

    # Create workspaces and configure settings on addon enable
    bpy.app.timers.register(_setup_addon_workspaces, first_interval=0.2)


def unregister():
    bpy.utils.unregister_class(LEVELDESIGN_OT_create_hotspot_mapping_workspace)
    bpy.utils.unregister_class(LEVELDESIGN_OT_create_level_design_workspace)
