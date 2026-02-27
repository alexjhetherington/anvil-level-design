import os
import unittest
import bpy

from ..utils import LEVEL_DESIGN_WORKSPACE_NAME
from ..workspace import create_level_design_workspace

# Set by run_tests.py when --save flag is used
save_outputs = False
output_dir = ""


def _get_window():
    """Get the active window, working around timer callbacks lacking window context."""
    return bpy.context.window or bpy.context.window_manager.windows[0]


def _redraw():
    """Force Blender to redraw the UI so test results are visible."""
    window = _get_window()
    with bpy.context.temp_override(window=window):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)


def _reset_scene():
    """Reset to empty scene with Level Design workspace active."""
    window = _get_window()
    with bpy.context.temp_override(window=window):
        bpy.ops.wm.read_homefile(use_empty=True)
    create_level_design_workspace()
    window = _get_window()
    window.workspace = bpy.data.workspaces[LEVEL_DESIGN_WORKSPACE_NAME]
    bpy.context.scene.level_design_props.pixels_per_meter = 1024


class AnvilTestCase(unittest.TestCase):
    """Base test case for Anvil Level Design addon tests.

    Provides setup/teardown that ensures a clean Blender scene between tests.
    """

    @classmethod
    def setUpClass(cls):
        _reset_scene()

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

        _reset_scene()
