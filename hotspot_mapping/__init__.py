"""
Hotspot Mapping Module

Allows users to define rectangular hotspot regions on textures.
Data is stored in scene properties (for undo) and synced to hotspots.json.
"""

import bpy
from bpy.app.handlers import persistent

from . import properties
from . import operators
from . import panels
from . import gizmo
from . import json_storage


@persistent
def _on_load_post(dummy):
    """Sync hotspot data from JSON file after loading .blend file."""
    json_storage.sync_from_file()


@persistent
def _on_save_pre(dummy):
    """Sync hotspot data to JSON file before saving .blend file."""
    json_storage.sync_to_file()


def register():
    properties.register()
    operators.register()
    panels.register()
    gizmo.register()
    # Register handlers for file sync
    bpy.app.handlers.load_post.append(_on_load_post)
    bpy.app.handlers.save_pre.append(_on_save_pre)


def unregister():
    # Unregister handlers
    if _on_save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_on_save_pre)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)

    gizmo.unregister()
    panels.unregister()
    operators.unregister()
    properties.unregister()
