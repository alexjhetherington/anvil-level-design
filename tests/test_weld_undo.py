import bmesh
import bpy
from mathutils import Vector

from ..operators.box_builder.geometry import execute_box_builder
from ..operators.weld import set_weld_from_edge_selection, set_weld_from_box_builder
from .base_test import AnvilTestCase
from .helpers import create_textured_cube, create_vertical_plane, _get_context_override


def _undo_ctx():
    """Build a full context override for ed.undo (needs window + screen)."""
    window = bpy.context.window or bpy.context.window_manager.windows[0]
    screen = window.screen
    area = next(a for a in screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return {"window": window, "screen": screen, "area": area, "region": region}


class CorridorWeldUndoTest(AnvilTestCase):
    """Test corridor weld undo: weld → undo → verify mode → re-weld → verify geometry."""

    def test_corridor_weld_undo_and_reweld(self):
        """Corridor: weld → undo → verify CORRIDOR → re-weld → verify geometry.

        Uses operator-based geometry (delete face) so the undo system properly
        tracks subsequent BMesh layer changes.
        """
        obj = create_textured_cube("corridor_undo", 1.0, 1.0)
        obj_name = obj.name
        ctx = _get_context_override()
        uctx = _undo_ctx()

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo_push(message="Baseline")

        # Delete the top face via operator to create a boundary edge loop
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = f.normal.z > 0.9
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

        # Set weld state (stored in BMesh layers)
        set_weld_from_edge_selection(bpy.context, 0.5, (0, 0, -1), -0.5)

        yield 0.5

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'CORRIDOR',
                         "Should be CORRIDOR after setup")

        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo_push(message="After weld setup")

        # Execute corridor weld
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        yield 0.5

        self.assertEqual(props.weld_mode, 'NONE',
                         "Should be NONE after corridor weld")

        # Push after operator so undo goes to the pre-operator push
        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo_push(message="After corridor")

        # --- Undo corridor ---
        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo()

        yield 0.5

        obj = bpy.data.objects[obj_name]
        bpy.context.view_layer.objects.active = obj
        props = bpy.context.scene.level_design_props

        self.assertEqual(props.weld_mode, 'CORRIDOR',
                         "Should be CORRIDOR after undoing corridor")

        # --- Re-execute corridor weld ---
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        yield 0.5

        self.assertEqual(props.weld_mode, 'NONE',
                         "Should be NONE after re-corridor")


class BridgeWeldUndoTest(AnvilTestCase):
    """Test bridge weld undo: weld → undo → verify mode → re-weld → verify geometry."""

    def test_bridge_weld_undo_and_reweld(self):
        """Bridge: weld → undo → verify BRIDGE → re-weld → verify 6 faces."""
        # Create two planes and join them via operators
        plane_a = create_vertical_plane("bridge_a")
        plane_b = create_vertical_plane("bridge_b")

        import math
        ctx = _get_context_override()
        uctx = _undo_ctx()

        # Rotate plane_a 180° so normals face away
        plane_a.select_set(True)
        plane_b.select_set(False)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        plane_a.rotation_euler.z = math.pi
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(rotation=True)

        # Move plane_b 1 unit away
        plane_b.location.y = 1.0
        bpy.context.view_layer.objects.active = plane_b
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(location=True)

        # Join into one object
        plane_a.select_set(True)
        plane_b.select_set(True)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.join()

        obj = plane_a
        obj_name = obj.name

        # Enter edit mode
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        # Select all edges via operator
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.select_all(action='SELECT')

        yield 0.5

        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'EDGE'}
        for e in bm.edges:
            e.select = True
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

        set_weld_from_edge_selection(bpy.context, 1.0, (0, 1, 0), 1.0)

        yield 0.5

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'BRIDGE',
                         "Should be BRIDGE with 2 edge groups")

        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo_push(message="After bridge setup")

        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        yield 0.5

        bm = bmesh.from_edit_mesh(obj.data)
        self.assertEqual(len(bm.faces), 6,
                         f"Should have 6 faces after bridge, got {len(bm.faces)}")
        self.assertEqual(props.weld_mode, 'NONE',
                         "Should be NONE after bridge weld")

        # Push after operator
        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo_push(message="After bridge")

        # --- Undo bridge ---
        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo()

        yield 0.5

        obj = bpy.data.objects[obj_name]
        bpy.context.view_layer.objects.active = obj
        props = bpy.context.scene.level_design_props

        self.assertEqual(props.weld_mode, 'BRIDGE',
                         "Should be BRIDGE after undoing bridge")

        bm = bmesh.from_edit_mesh(obj.data)
        self.assertEqual(len(bm.faces), 2,
                         f"Should have 2 faces after undo, got {len(bm.faces)}")

        # --- Re-execute bridge weld ---
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        yield 0.5

        bm = bmesh.from_edit_mesh(obj.data)
        self.assertEqual(len(bm.faces), 6,
                         f"Should have 6 faces after re-bridge, got {len(bm.faces)}")


class InvertWeldUndoTest(AnvilTestCase):
    """Test invert weld undo: weld → undo → verify mode → re-weld → verify normals."""

    def test_invert_weld_undo_and_reweld(self):
        """Invert: weld → undo → verify INVERT → re-weld → verify normals flipped."""
        mesh = bpy.data.meshes.new("invert_undo")
        obj = bpy.data.objects.new("invert_undo", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        obj_name = obj.name

        ctx = _get_context_override()
        uctx = _undo_ctx()

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo_push(message="Before box")

        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        result = execute_box_builder(
            Vector((0, 0, 0)), Vector((1, 0, 1)), 1.0,
            Vector((1, 0, 0)), Vector((0, 0, 1)), Vector((0, 1, 0)),
            obj, ppm, Vector((0, -1, 0)),
        )
        self.assertTrue(result[0], result[1])

        face_verts = result[2] if len(result) > 2 else []
        set_weld_from_box_builder(bpy.context, face_verts)

        yield 0.5

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'INVERT',
                         "Should be INVERT after box build")

        # Record normals before invert
        bm = bmesh.from_edit_mesh(obj.data)
        bm.normal_update()
        bm.faces.ensure_lookup_table()
        normals_before = {f.index: tuple(round(v, 4) for v in f.normal) for f in bm.faces}

        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo_push(message="After weld setup")

        # Execute invert weld
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)
        self.assertEqual(props.weld_mode, 'NONE',
                         "Should be NONE after invert")

        # Verify normals are flipped
        bm = bmesh.from_edit_mesh(obj.data)
        bm.normal_update()
        bm.faces.ensure_lookup_table()
        for f in bm.faces:
            before = normals_before.get(f.index)
            if before is not None:
                after = tuple(round(v, 4) for v in f.normal)
                for i in range(3):
                    self.assertAlmostEqual(after[i], -before[i], places=2,
                                           msg=f"Face {f.index} normal not flipped")

        # Push after operator so undo goes to the pre-operator push
        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo_push(message="After invert")

        # --- Undo invert ---
        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo()

        yield 0.5

        obj = bpy.data.objects[obj_name]
        bpy.context.view_layer.objects.active = obj
        props = bpy.context.scene.level_design_props

        self.assertEqual(props.weld_mode, 'INVERT',
                         "Should be INVERT after undoing invert")

        # Verify normals are back to original
        bm = bmesh.from_edit_mesh(obj.data)
        bm.normal_update()
        bm.faces.ensure_lookup_table()
        for f in bm.faces:
            before = normals_before.get(f.index)
            if before is not None:
                restored = tuple(round(v, 4) for v in f.normal)
                for i in range(3):
                    self.assertAlmostEqual(restored[i], before[i], places=2,
                                           msg=f"Face {f.index} normal not restored after undo")

        # --- Re-execute invert weld ---
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        # Verify normals are flipped again
        bm = bmesh.from_edit_mesh(obj.data)
        bm.normal_update()
        bm.faces.ensure_lookup_table()
        for f in bm.faces:
            before = normals_before.get(f.index)
            if before is not None:
                after = tuple(round(v, 4) for v in f.normal)
                for i in range(3):
                    self.assertAlmostEqual(after[i], -before[i], places=2,
                                           msg=f"Face {f.index} normal not flipped on re-weld")
