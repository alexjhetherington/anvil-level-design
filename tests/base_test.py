import os
import unittest
import bpy
from mathutils import Quaternion, Vector

from ..core.workspace_check import LEVEL_DESIGN_WORKSPACE_NAME
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
    from ..handlers import (
        active_image, auto_hotspot, face_cache, mode_tracking,
        uv_world_scale,
        face_data_cache, on_depsgraph_update,
    )
    from ..core.materials import reset_duplicate_material_consolidation
    from ..handlers.cross_object_undo import reset as reset_cross_object_undo
    from ..handlers.lifecycle import set_undo_in_progress

    # Clear handler state that holds references to Blender data blocks
    # before removing the data, preventing dangling pointer access.
    active_image.reset()
    face_data_cache.clear()
    face_cache.reset()
    auto_hotspot.reset()
    mode_tracking.reset_mode_tracking()
    reset_duplicate_material_consolidation()
    uv_world_scale._tracked_modal_operators = set()
    from ..handlers import depsgraph as _depsgraph_mod
    _depsgraph_mod._last_cleaned_spin_fingerprint = None
    set_undo_in_progress(False)
    reset_cross_object_undo()

    # Reset weld module state (clears transient flags like _weld_op_running).
    # Mesh-stored weld data is cleaned up when objects are deleted below.
    from ..operators import weld as _weld_mod
    _weld_mod._weld_op_running = False
    _weld_mod._weld_just_stored = False
    _weld_mod.clear_repeat_prefab_override()
    props = bpy.context.scene.level_design_props
    props.weld_mode = 'NONE'
    props.weld_depth = 0.0
    props.weld_prefab_library_index = -1
    props.weld_prefab_object_name = ""
    props.weld_prefab_asset_type = "OBJECT"
    props.weld_prefab_rotation = 0.0
    props.prefab_inherit_normal = True
    props.prefab_random_scale_enabled = False
    props.prefab_random_scale_min = (1.0, 1.0, 1.0)
    props.prefab_random_scale_min_linked = True
    props.prefab_random_scale_max = (1.0, 1.0, 1.0)
    props.prefab_random_scale_max_linked = True
    props.prefab_random_rotation_enabled = False
    props.prefab_random_rotation_min = (0.0, 0.0, 0.0)
    props.prefab_random_rotation_max = (0.0, 0.0, 0.0)
    props.gltf_anvil_enabled = True
    props.gltf_anvil_scale = 1.0
    props.gltf_anvil_apply_modifiers = True
    props.gltf_anvil_separate_loose = True
    props.gltf_anvil_always_combine_materials = True
    props.gltf_anvil_debug = False
    prefab_libraries = getattr(bpy.context.scene, "anvil_prefab_libraries", None)
    if prefab_libraries is not None:
        prefab_libraries.clear()

    # Temporarily remove the depsgraph handler during cleanup to prevent it
    # from firing while objects/meshes are being deleted (which can cause
    # access violations from accessing freed data).
    handler_was_registered = on_depsgraph_update in bpy.app.handlers.depsgraph_update_post
    if handler_was_registered:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)

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

    # Force depsgraph to settle after object removal before removing data
    # blocks. Without this, deferred depsgraph evaluation can access freed
    # object/mesh data and cause access violations.
    bpy.context.view_layer.update()

    # Flush any background icon preview jobs that hold pointers to material/
    # image data blocks. If we delete those blocks while a preview job is
    # still running, its endjob callback writes to freed memory
    # (icon_preview_endjob crash). The DRAW_WIN_SWAP processes pending
    # wm_jobs, letting them finish before we remove the data they reference.
    with bpy.context.temp_override(window=window):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

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

    # Force another depsgraph settle after all removals
    bpy.context.view_layer.update()

    # Reset state that tests may have changed: viewport orientation,
    # mesh select mode, etc. Without this, tests that change these leak
    # state into subsequent tests (e.g. face select mode from texture_apply
    # breaks edge extrude in uv_extend).
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bpy.context.tool_settings.use_mesh_automerge = False

    for area in window.screen.areas:
        if area.type == 'VIEW_3D':
            rv3d = area.spaces.active.region_3d
            rv3d.view_rotation = Quaternion((0.8186, 0.4341, -0.1653, -0.3279))
            rv3d.view_location = Vector((0.0, 0.0, 0.0))
            rv3d.view_distance = 5.0
            rv3d.view_perspective = 'PERSP'

    # Force a synchronous redraw BEFORE re-registering the depsgraph handler.
    # This makes the outliner (and other editors) rebuild their internal caches
    # from the now-empty scene while the depsgraph handler is still detached,
    # preventing it from firing during the redraw and accessing freed data.
    with bpy.context.temp_override(window=window):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    # Re-register the depsgraph handler
    if handler_was_registered:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)


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
        if not save_outputs:
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

    def _get_3d_view_context(self):
        """Return a complete override context for 3D View modal operators."""
        window = _get_window()
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = None
            for area_region in area.regions:
                if area_region.type == 'WINDOW':
                    region = area_region
                    break
            if region is None:
                continue
            space = area.spaces.active
            return {
                "window": window,
                "screen": window.screen,
                "area": area,
                "region": region,
                "space_data": space,
                "region_data": space.region_3d,
            }
        raise RuntimeError("No 3D viewport found in current workspace")

    def refresh_face_cache(self):
        """Refresh cached face UVs after tests edit BMesh data directly."""
        from ..handlers.face_cache import cache_face_data
        cache_face_data(bpy.context)

    # Map event types to unicode characters for event_simulate
    _UNICODE_MAP = {
        'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4',
        'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9',
        'MINUS': '-', 'PERIOD': '.', 'RET': '\r',
        'B': 'b', 'E': 'e', 'X': 'x', 'Y': 'y', 'Z': 'z',
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
        yield
        window.event_simulate(
            type=event_type, value='RELEASE', x=mx, y=my
        )
        yield

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

    def simulate_extrude(self, value, axis=None):
        """Generator: simulate a full extrude via event_simulate.

        Sends E (extrude) -> [axis key] -> numeric value -> Enter, yielding
        between each step so Blender's event loop processes the events.
        Then yields extra settle time for the depsgraph handler to fire.

        Args:
            value: numeric distance (can be negative)
            axis: 'X', 'Y', or 'Z', or None to extrude along the face normal
        """
        mx, my = self._get_3d_viewport_center()
        window = _get_window()

        # Face-normal extrudes still use the E key path. Axis-constrained
        # edge extrudes invoke the same modal operator directly because raw E
        # key dispatch can be swallowed by Blender's timer-driven test focus.
        if axis is None:
            yield from self._simulate_key_tap('E')
        else:
            with bpy.context.temp_override(**self._get_3d_view_context()):
                bpy.ops.mesh.extrude_region_move('INVOKE_DEFAULT')
            yield
        yield
        yield

        # MOUSEMOVE to initialize the modal's tracking state
        window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=mx + 1, y=my + 1)
        yield
        yield

        # Axis constraint (skip if None — extrude along normal)
        if axis is not None:
            yield from self._simulate_key_tap(axis.upper())
            yield

        # Type the numeric value
        yield from self._simulate_number(value)

        # Confirm with Enter
        yield from self._simulate_key_tap('RET')

        # Settle time for depsgraph handler to detect and process the extrude
        yield
