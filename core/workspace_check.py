import bpy


LEVEL_DESIGN_WORKSPACE_NAME = "Level Design"
HOTSPOT_MAPPING_WORKSPACE_NAME = "Hotspot Mapping"


def is_level_design_workspace():
    """Check if the current workspace is the Level Design workspace."""
    try:
        return bpy.context.workspace.name == LEVEL_DESIGN_WORKSPACE_NAME
    except AttributeError:
        return False


def is_hotspot_mapping_workspace():
    """Check if the current workspace is the Hotspot Mapping workspace."""
    try:
        return bpy.context.workspace.name == HOTSPOT_MAPPING_WORKSPACE_NAME
    except AttributeError:
        return False
