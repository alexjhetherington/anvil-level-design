"""Test runner script invoked by Blender in GUI mode.

Usage:
    blender --python tests/run_tests.py
    blender --python tests/run_tests.py -- test_smoke
    blender --python tests/run_tests.py -- --save
    blender --python tests/run_tests.py -- --save test_smoke.SmokeTest.test_passes
"""

import sys
import os
import shutil
import unittest

# Add the parent of anvil_level_design to sys.path so imports resolve
addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_dir = os.path.dirname(addon_dir)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

# Ensure the addon is enabled (it may already be loaded from the installed copy)
import bpy
import addon_utils

addon_loaded = False
for mod in addon_utils.modules():
    if mod.__name__ == "anvil_level_design" or mod.__name__ == "anvil-level-design":
        if addon_utils.check(mod.__name__)[1]:
            addon_loaded = True
            break

if not addon_loaded:
    bpy.ops.preferences.addon_enable(module="anvil_level_design")


def get_user_args():
    """Return arguments after the '--' separator, if any."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def main():
    user_args = get_user_args()

    # Check for --save flag
    save_mode = "--save" in user_args
    if save_mode:
        user_args.remove("--save")

        from anvil_level_design.tests import base_test
        base_test.save_outputs = True
        base_test.output_dir = os.path.join(addon_dir, "test_outputs")

        # Clean and recreate output directory
        if os.path.exists(base_test.output_dir):
            try:
                shutil.rmtree(base_test.output_dir)
            except PermissionError:
                print("Level Design Tools: ERROR - Cannot clean test_outputs/. "
                      "Close any .blend files open from that folder and retry.")
                os._exit(1)
        os.makedirs(base_test.output_dir)

    loader = unittest.TestLoader()
    tests_dir = os.path.dirname(os.path.abspath(__file__))

    if user_args:
        # Load specific test modules/classes/methods passed after --
        suite = unittest.TestSuite()
        for name in user_args:
            # Prefix with package name so unittest can resolve it
            qualified = f"anvil_level_design.tests.{name}"
            suite.addTests(loader.loadTestsFromName(qualified))
    else:
        # Discover all test_*.py files in the tests directory
        suite = loader.discover(tests_dir, pattern="test_*.py", top_level_dir=repo_dir)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # os._exit works from timer context to terminate Blender
    os._exit(0 if result.wasSuccessful() else 1)


# Defer test execution so Blender's GUI and depsgraph are fully initialized
bpy.app.timers.register(main, first_interval=1.0)
