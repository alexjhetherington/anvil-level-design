import os
import unittest
import bpy

# Set by run_tests.py when --save flag is used
save_outputs = False
output_dir = ""


class AnvilTestCase(unittest.TestCase):
    """Base test case for Anvil Level Design addon tests.

    Provides setup/teardown that ensures a clean Blender scene between tests.
    """

    @classmethod
    def setUpClass(cls):
        bpy.ops.wm.read_homefile(use_empty=True)

    def setUp(self):
        pass

    def tearDown(self):
        if save_outputs:
            test_name = self.id()  # e.g. anvil_level_design.tests.test_smoke.SmokeTest.test_passes
            filepath = os.path.join(output_dir, f"{test_name}.blend")
            bpy.ops.wm.save_as_mainfile(filepath=filepath)

        bpy.ops.wm.read_homefile(use_empty=True)
