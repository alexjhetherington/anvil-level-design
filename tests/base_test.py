import os
import unittest
import bpy

from ..utils import LEVEL_DESIGN_WORKSPACE_NAME
from ..workspace import create_level_design_workspace

# Set by run_tests.py when --save flag is used
save_outputs = False
output_dir = ""

_initialized = False


def _get_window():
    """Get the active window, working around timer callbacks lacking window context."""
    return bpy.context.window or bpy.context.window_manager.windows[0]


def _redraw():
    """Force Blender to redraw the UI so test results are visible."""
    window = _get_window()
    with bpy.context.temp_override(window=window):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)


def _purge_all():
    """Remove all user data from the blend file."""
    # Switch to object mode if possible, so deletions don't fail
    window = _get_window()
    with bpy.context.temp_override(window=window):
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass

    # Deselect and unlink all objects from every collection
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # Purge orphan data blocks (meshes, materials, images, node groups, etc.)
    categories = [
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.node_groups,
        bpy.data.textures,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.curves,
        bpy.data.armatures,
    ]
    for category in categories:
        for block in list(category):
            category.remove(block, do_unlink=True)

    # Remove extra collections (keep the scene's master collection)
    scene = bpy.context.scene
    for child in list(scene.collection.children):
        scene.collection.children.unlink(child)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)

    # Remove extra scenes
    for s in list(bpy.data.scenes):
        if s != scene:
            bpy.data.scenes.remove(s)


def activate_level_design_workspace():
    """Ensure Level Design workspace exists and queue a switch to it.

    Must be called from a timer callback. The workspace switch is deferred
    until Blender processes the next event loop cycle, so tests must run
    in a LATER timer.
    """
    create_level_design_workspace()
    window = _get_window()
    window.workspace = bpy.data.workspaces[LEVEL_DESIGN_WORKSPACE_NAME]


def _init_scene():
    """One-time initialization: purge everything, set properties.

    Called after the workspace switch has been processed by Blender's
    event loop (i.e. we are already in Level Design workspace).
    """
    global _initialized
    _purge_all()
    bpy.context.scene.level_design_props.pixels_per_meter = 1024
    _initialized = True


def _clean_scene():
    """Between-test cleanup: purge data, stay in Level Design workspace."""
    _purge_all()
    bpy.context.scene.level_design_props.pixels_per_meter = 1024


class AnvilTestCase(unittest.TestCase):
    """Base test case for Anvil Level Design addon tests.

    Provides setup/teardown that ensures a clean Blender scene between tests.
    """

    @classmethod
    def setUpClass(cls):
        global _initialized
        if not _initialized:
            _init_scene()

    def setUp(self):
        pass

    def tearDown(self):
        _redraw()

        if save_outputs:
            test_name = self.id()  # e.g. anvil_level_design.tests.test_smoke.SmokeTest.test_passes
            filepath = os.path.join(output_dir, f"{test_name}.blend")
            try:
                window = _get_window()
                with bpy.context.temp_override(window=window):
                    bpy.ops.wm.save_as_mainfile(filepath=filepath)
            except Exception as e:
                print(f"Level Design Tools: WARNING - Could not save {filepath}: {e}")

        _clean_scene()
