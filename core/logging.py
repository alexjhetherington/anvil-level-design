import bpy


DEBUG_KEEP_HOTSPOT_SEAMS = False


def debug_log(msg):
    """Print debug message if logging is enabled via the Debug panel."""
    try:
        enabled = bpy.context.scene.level_design_props.debug_logging
    except AttributeError:
        enabled = False
    if enabled:
        print(msg, flush=True)
