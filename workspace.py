import os

import bpy

from .core.workspace_check import LEVEL_DESIGN_WORKSPACE_NAME, HOTSPOT_MAPPING_WORKSPACE_NAME


# Workspace and scene setup flow
# ================================
#
# bpy.data is restricted during addon registration, so register() defers setup via a timer.
# The _specialized_template_active flag prevents the timer from re-creating workspaces
# after the splash watcher has removed them.
#
# On Blender launch:
#   register() schedules setup timer
#   on_load_post fires, filepath is empty -> resets flag, runs setup, subscribes splash watcher
#   Timer fires -> setup (no-op, workspaces already exist)
#
# On File -> New or selecting General from splash screen:
#   on_load_post fires, filepath is empty -> resets flag, runs setup, subscribes fresh splash watcher
#
# On selecting a specialised splash template (Sculpting, 2D Animation, etc.):
#   Blender calls reset_all which re-runs register() -> schedules setup timer
#   on_load_post fires, filepath is empty -> resets flag, runs setup, subscribes splash watcher
#   Splash watcher fires -> detects specialised workspace -> sets flag, removes addon workspaces
#   Timer fires -> flag is set, skips setup
#
# On opening a saved .blend file:
#   on_load_post fires, filepath is non-empty -> resets flag, skips setup
#
# On enabling the addon mid-session (first install):
#   register() schedules setup timer -> timer fires -> runs setup


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


def setup_addon_workspaces():
    """Create addon workspaces and configure scene settings for new files."""
    if _specialized_template_active:
        return
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


def remove_addon_workspaces():
    """Remove addon workspaces if they exist."""
    to_remove = []
    for name in (LEVEL_DESIGN_WORKSPACE_NAME, HOTSPOT_MAPPING_WORKSPACE_NAME):
        ws = bpy.data.workspaces.get(name)
        if ws:
            to_remove.append(ws)
    if to_remove:
        bpy.data.batch_remove(to_remove)


# Workspaces that indicate the user chose a specialised splash screen template
_SPECIALIZED_WORKSPACES = {
    "2D Animation", "Sculpting", "Storyboarding",
    "Motion Tracking", "Video Editing",
}

_splash_watcher_owner = object()
_specialized_template_active = False


def reset_specialized_template_flag():
    """Reset the flag so setup_addon_workspaces runs on the next load."""
    global _specialized_template_active
    _specialized_template_active = False


def _on_workspace_changed():
    """One-shot callback: if the first workspace switch in an unsaved file
    is to a specialised template, remove the addon workspaces."""
    global _specialized_template_active
    bpy.msgbus.clear_by_owner(_splash_watcher_owner)

    if bpy.data.filepath:
        return

    workspace_name = bpy.context.window.workspace.name
    if workspace_name in _SPECIALIZED_WORKSPACES:
        _specialized_template_active = True
        remove_addon_workspaces()


def subscribe_splash_watcher():
    """Subscribe to workspace changes to detect specialised splash selection."""
    bpy.msgbus.clear_by_owner(_splash_watcher_owner)

    window = bpy.context.window
    bpy.msgbus.subscribe_rna(
        key=window.path_resolve("workspace", False),
        owner=_splash_watcher_owner,
        args=(),
        notify=_on_workspace_changed,
    )


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_create_level_design_workspace)
    bpy.utils.register_class(LEVELDESIGN_OT_create_hotspot_mapping_workspace)

    # bpy.data is restricted during addon_utils.enable (both startup and mid-session).
    # on_load_post handles setup for startup and file loads. For mid-session first install,
    # on_load_post won't fire, so we use a timer to defer until bpy.data is available.
    bpy.app.timers.register(setup_addon_workspaces, first_interval=0.0)


def unregister():
    bpy.msgbus.clear_by_owner(_splash_watcher_owner)
    bpy.utils.unregister_class(LEVELDESIGN_OT_create_hotspot_mapping_workspace)
    bpy.utils.unregister_class(LEVELDESIGN_OT_create_level_design_workspace)
