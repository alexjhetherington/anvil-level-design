"""
Hotspot Mapping Module

Allows users to define rectangular hotspot regions on textures.
Data is stored in scene properties (for undo). Optionally synced to an
external JSON file chosen by the user.
"""

import bpy
from bpy.app.handlers import persistent

from . import properties
from . import operators
from . import panels
from . import gizmo
from . import json_storage


# Stores the resolved absolute path before a save, so save_post can
# re-relativize after a Save As changes the .blend location.
_pre_save_resolved_path = None


@persistent
def _on_load_post(dummy):
    """Sync hotspot data from JSON file after loading .blend file."""
    json_storage.sync_from_file()


@persistent
def _on_save_pre(dummy):
    """Resolve the hotspot file path and sync data to disk before saving."""
    global _pre_save_resolved_path
    _pre_save_resolved_path = json_storage.resolve_filepath()

    if _pre_save_resolved_path is not None:
        json_storage.save_to_file(_pre_save_resolved_path)


@persistent
def _on_save_post(dummy):
    """Re-relativize the hotspot file path after saving.

    Handles Save As: the .blend path may have changed, so the relative
    path needs to be recomputed against the new location.
    """
    global _pre_save_resolved_path

    scene = bpy.context.scene
    if not scene or not hasattr(scene, 'hotspot_mapping_props'):
        _pre_save_resolved_path = None
        return

    props = scene.hotspot_mapping_props
    if not props.hotspots_file_path:
        _pre_save_resolved_path = None
        return

    # Use the absolute path we resolved before saving
    absolute_path = _pre_save_resolved_path
    _pre_save_resolved_path = None

    if absolute_path is None:
        return

    # Re-relativize against the (potentially new) .blend location
    new_relative = json_storage.make_path_relative(absolute_path)
    if new_relative != props.hotspots_file_path:
        props.hotspots_file_path = new_relative


def register():
    properties.register()
    operators.register()
    panels.register()
    gizmo.register()
    # Register handlers for file sync
    bpy.app.handlers.load_post.append(_on_load_post)
    bpy.app.handlers.save_pre.append(_on_save_pre)
    bpy.app.handlers.save_post.append(_on_save_post)


def unregister():
    # Unregister handlers
    if _on_save_post in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.remove(_on_save_post)
    if _on_save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_on_save_pre)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)

    gizmo.unregister()
    panels.unregister()
    operators.unregister()
    properties.unregister()
