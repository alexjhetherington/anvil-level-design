import bmesh
import bpy
from mathutils import Vector

from .base_test import AnvilTestCase
from .helpers import create_textured_cube, _get_context_override


class WeldUndoStackTest(AnvilTestCase):
    """Test that the weld state follows the full undo chain correctly.

    Simulates the real workflow:
      cube cut → corridor → something else → undo × 3
    and verifies weld_mode at every step.
    """

    def _undo_ctx(self):
        """Build a full context override for ed.undo (needs window + screen)."""
        window = bpy.context.window or bpy.context.window_manager.windows[0]
        screen = window.screen
        area = next(a for a in screen.areas if a.type == 'VIEW_3D')
        region = next(r for r in area.regions if r.type == 'WINDOW')
        return {"window": window, "screen": screen, "area": area, "region": region}

    def test_weld_undo_chain(self):
        """Walk through: cut → corridor → other op → undo × 3.

        Expected weld_mode at each step:
        1. After simulated cube cut (set_weld): CORRIDOR
        2. After corridor (W):                  NONE
        3. After 'something else':              NONE
        4. Ctrl+Z (undo something else):        NONE  (still post-corridor)
        5. Ctrl+Z (undo corridor):              CORRIDOR
        6. Ctrl+Z (undo to before cut):         NONE
        """
        obj = create_textured_cube("weld_stack", 1.0, 1.0)
        obj_name = obj.name

        ctx = _get_context_override()
        undo_ctx = self._undo_ctx()

        # Enter edit mode
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        # --- Baseline undo step (before weld setup) ---
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="Before weld setup")

        # Delete the top face to create a boundary edge loop
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = False
        for f in bm.faces:
            if f.normal.z > 0.9:
                f.select = True
                break
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.delete(type='FACE')

        yield 0.5

        # Select the top boundary edges
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'EDGE'}
        for e in bm.edges:
            e.select = all(v.co.z > 0.9 for v in e.verts)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

        # Simulate cube cut setting the weld state
        from ..operators.weld import set_weld_from_edge_selection
        set_weld_from_edge_selection(bpy.context, 0.5, (0, 0, -1), -0.5,
                                         Vector((0, 0, 0)), Vector((1, 0, 1)),
                                         Vector((1, 0, 0)), Vector((0, 0, 1)))

        yield 0.5

        # --- Step 1: After simulated cube cut → CORRIDOR ---
        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'CORRIDOR',
                         "Step 1: weld should be CORRIDOR after cube cut")

        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="After cube cut")

        # --- Step 2: Execute corridor → NONE ---
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        yield 0.5

        self.assertEqual(props.weld_mode, 'NONE',
                         "Step 2: weld should be NONE after corridor")

        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="After corridor")

        # --- Step 3: Do something else (select all) → still NONE ---
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.select_all(action='SELECT')

        yield 0.5

        self.assertEqual(props.weld_mode, 'NONE',
                         "Step 3: weld should be NONE after other op")

        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="After something else")

        # --- Step 4: Ctrl+Z (undo 'something else') → NONE ---
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo()

        yield 0.5

        obj = bpy.data.objects[obj_name]
        bpy.context.view_layer.objects.active = obj
        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'NONE',
                         "Step 4: weld should be NONE (still post-corridor)")

        # --- Step 5: Ctrl+Z (undo corridor) → CORRIDOR ---
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo()

        yield 0.5

        obj = bpy.data.objects[obj_name]
        bpy.context.view_layer.objects.active = obj
        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'CORRIDOR',
                         "Step 5: weld should be CORRIDOR after undoing corridor")
        self.assertAlmostEqual(props.weld_depth, 0.5, places=3,
                               msg="Step 5: weld depth should be 0.5")

        # --- Step 6: Ctrl+Z (undo to before cut) → NONE ---
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo()

        yield 0.5

        obj = bpy.data.objects[obj_name]
        bpy.context.view_layer.objects.active = obj
        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'NONE',
                         "Step 6: weld should be NONE before cube cut")


