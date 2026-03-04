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

    # Ensure the view layer's active collection points to the master collection
    # (removing child collections can leave bpy.context.collection as None)
    view_layer = _get_window().view_layer
    view_layer.active_layer_collection = view_layer.layer_collection

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
    Includes event simulation helpers for testing modal operators.
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

    # ------------------------------------------------------------------
    # Event simulation helpers (for generator-based async tests)
    # ------------------------------------------------------------------

    _DIGIT_MAP = {
        '0': 'ZERO', '1': 'ONE', '2': 'TWO', '3': 'THREE', '4': 'FOUR',
        '5': 'FIVE', '6': 'SIX', '7': 'SEVEN', '8': 'EIGHT', '9': 'NINE',
    }

    def _get_3d_viewport_center(self):
        """Return (x, y) pixel coordinates at the center of the 3D viewport."""
        window = _get_window()
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                x = area.x + area.width // 2
                y = area.y + area.height // 2
                return x, y
        raise RuntimeError("No 3D viewport found in current workspace")

    # Map event types to unicode characters for event_simulate
    _UNICODE_MAP = {
        'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4',
        'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9',
        'MINUS': '-', 'PERIOD': '.', 'RET': '\r',
    }

    def _simulate_key_tap(self, event_type):
        """Generator: simulate a key press + release with yields between."""
        window = _get_window()
        mx, my = self._get_3d_viewport_center()
        unicode_ch = self._UNICODE_MAP.get(event_type, '')
        if unicode_ch:
            window.event_simulate(
                type=event_type, value='PRESS', x=mx, y=my, unicode=unicode_ch
            )
        else:
            window.event_simulate(
                type=event_type, value='PRESS', x=mx, y=my
            )
        yield 0.05
        window.event_simulate(
            type=event_type, value='RELEASE', x=mx, y=my
        )
        yield 0.05

    def _simulate_number(self, value):
        """Generator: type a numeric value (handles negative via MINUS key)."""
        text = str(value)
        if text.startswith('-'):
            yield from self._simulate_key_tap('MINUS')
            text = text[1:]
        for ch in text:
            if ch == '.':
                yield from self._simulate_key_tap('PERIOD')
            elif ch in self._DIGIT_MAP:
                yield from self._simulate_key_tap(self._DIGIT_MAP[ch])

    def simulate_extrude(self, axis, value):
        """Generator: simulate a full extrude via event_simulate.

        Sends E (extrude) -> axis key -> numeric value -> Enter, yielding
        between each step so Blender's event loop processes the events.
        Then yields extra settle time for the depsgraph handler to fire.

        Args:
            axis: 'X', 'Y', or 'Z'
            value: numeric distance (can be negative)
        """
        mx, my = self._get_3d_viewport_center()
        window = _get_window()

        # E to start extrude
        yield from self._simulate_key_tap('E')
        yield 0.3

        # MOUSEMOVE to initialize the modal's tracking state
        window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=mx, y=my)
        yield 0.1

        # Axis constraint
        yield from self._simulate_key_tap(axis.upper())
        yield 0.1

        # Type the numeric value
        yield from self._simulate_number(value)
        yield 0.1

        # Confirm with Enter
        yield from self._simulate_key_tap('RET')

        # Settle time for depsgraph handler to detect and process the extrude
        yield 0.5
