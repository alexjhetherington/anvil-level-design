import bmesh
import bpy
from mathutils import Vector

from .base_test import AnvilTestCase
from .helpers import _get_context_override

from ..operators.box_builder.geometry import execute_box_builder, execute_box_builder_object_mode
from ..operators.weld import set_weld_from_box_builder, set_weld_from_box_builder_object_mode


class BoxBuilderWeldTest(AnvilTestCase):
    """Test that box builder sets the correct weld mode based on surrounding geometry."""

    def _undo_ctx(self):
        """Build a full context override for ed.undo (needs window + screen)."""
        window = bpy.context.window or bpy.context.window_manager.windows[0]
        screen = window.screen
        area = next(a for a in screen.areas if a.type == 'VIEW_3D')
        region = next(r for r in area.regions if r.type == 'WINDOW')
        return {"window": window, "screen": screen, "area": area, "region": region}

    def _create_empty_mesh(self, name):
        """Create an empty mesh object in edit mode."""
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        return obj

    def _create_mesh_with_plane(self, name, verts):
        """Create a mesh object with a single quad face and enter edit mode.

        Args:
            name: Object name
            verts: List of 4 Vector-like vertex positions (winding determines normal)
        """
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bm_verts = [bm.verts.new(v) for v in verts]
        bm.faces.new(bm_verts)
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        return obj

    # ------------------------------------------------------------------
    # Object mode tests (box builder creates a new standalone object)
    # ------------------------------------------------------------------

    def test_standalone_box_object_mode_invert(self):
        """Object-mode box is always standalone → weld should be INVERT.

        This is the production path: user is in object mode, builds a box,
        the operator creates a new object and sets weld_mode = 'INVERT'.
        """
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        success, msg = execute_box_builder_object_mode(
            Vector((0, 0, 0)), Vector((1, 0, 1)), 1.0,
            Vector((1, 0, 0)), Vector((0, 0, 1)), Vector((0, 1, 0)),
            ppm, Vector((0, -1, 0)),
        )
        self.assertTrue(success, msg)

        # Production path: operator stores INVERT on object and sets scene prop
        set_weld_from_box_builder_object_mode(bpy.context.active_object)
        bpy.context.scene.level_design_props.weld_mode = 'INVERT'

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'INVERT',
                         "Object-mode standalone box should be INVERT")

        # Verify weld execution works from object mode
        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result,
                       "Weld INVERT should succeed from object mode")
        self.assertEqual(props.weld_mode, 'NONE',
                         "Weld mode should be NONE after executing invert")

    # ------------------------------------------------------------------
    # Edit mode tests (box builder adds geometry to existing mesh)
    # ------------------------------------------------------------------

    def test_standalone_box_edit_mode_invert(self):
        """Edit-mode box in empty mesh → weld should be INVERT."""
        obj = self._create_empty_mesh("standalone_box")
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        result = execute_box_builder(
            Vector((0, 0, 0)), Vector((1, 0, 1)), 1.0,
            Vector((1, 0, 0)), Vector((0, 0, 1)), Vector((0, 1, 0)),
            obj, ppm, Vector((0, -1, 0)),
        )
        self.assertTrue(result[0], result[1])

        # Production path: operator calls set_weld_from_box_builder after
        # execute_box_builder in edit mode
        face_verts = result[2] if len(result) > 2 else []
        set_weld_from_box_builder(bpy.context, face_verts)

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'INVERT',
                         "Edit-mode standalone box should set weld to INVERT")

    def test_box_on_plane_normal_side_none(self):
        """Box built on the side a plane's normal points to should be NONE.

        Plane at Y=0 with normal +Y. Box extends from Y=0 to Y=1.
        The box face at Y=0 has outward normal -Y, anti-parallel to the
        plane's +Y normal. They are coplanar and overlap -> NONE.
        """
        # Plane at Y=0, normal +Y (winding: CCW when viewed from +Y)
        obj = self._create_mesh_with_plane("plane_normal_side", [
            (0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0),
        ])
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        result = execute_box_builder(
            Vector((0, 0, 0)), Vector((1, 0, 1)), 1.0,
            Vector((1, 0, 0)), Vector((0, 0, 1)), Vector((0, 1, 0)),
            obj, ppm, Vector((0, -1, 0)),
        )
        self.assertTrue(result[0], result[1])

        face_verts = result[2] if len(result) > 2 else []
        set_weld_from_box_builder(bpy.context, face_verts)

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'NONE',
                         "Box on plane normal side should set weld to NONE")

    def test_box_on_plane_opposite_side_invert(self):
        """Box built on the opposite side of a plane should be INVERT.

        Plane at Y=0 with normal +Y. Box extends from Y=0 to Y=-1.
        The box face at Y=0 has outward normal +Y, parallel to the plane's
        +Y normal. Not anti-parallel, so no blocking overlap -> INVERT.
        """
        # Same plane at Y=0, normal +Y
        obj = self._create_mesh_with_plane("plane_opposite_side", [
            (0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0),
        ])
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        # Negative depth: box extends in -Y direction
        result = execute_box_builder(
            Vector((0, 0, 0)), Vector((1, 0, 1)), -1.0,
            Vector((1, 0, 0)), Vector((0, 0, 1)), Vector((0, 1, 0)),
            obj, ppm, Vector((0, -1, 0)),
        )
        self.assertTrue(result[0], result[1])

        face_verts = result[2] if len(result) > 2 else []
        set_weld_from_box_builder(bpy.context, face_verts)

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'INVERT',
                         "Box on plane opposite side should set weld to INVERT")

    def test_box_plane_intersects_invert(self):
        """Plane that intersects the box (not coplanar with any face) should be INVERT.

        Plane at Y=0.5, normal +Y. Box spans Y=0 to Y=1.
        No box face lies at Y=0.5, so no coplanar match -> INVERT.
        """
        # Plane at Y=0.5, normal +Y
        obj = self._create_mesh_with_plane("plane_intersects", [
            (0, 0.5, 0), (0, 0.5, 1), (1, 0.5, 1), (1, 0.5, 0),
        ])
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        result = execute_box_builder(
            Vector((0, 0, 0)), Vector((1, 0, 1)), 1.0,
            Vector((1, 0, 0)), Vector((0, 0, 1)), Vector((0, 1, 0)),
            obj, ppm, Vector((0, -1, 0)),
        )
        self.assertTrue(result[0], result[1])

        face_verts = result[2] if len(result) > 2 else []
        set_weld_from_box_builder(bpy.context, face_verts)

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'INVERT',
                         "Box with intersecting plane should set weld to INVERT")

    # ------------------------------------------------------------------
    # Undo tests
    # ------------------------------------------------------------------

    def test_edit_mode_invert_survives_undo(self):
        """Edit-mode: build box → deselect → undo should restore INVERT.

        Verifies the weld stack correctly restores INVERT after undo
        re-selects the box edges.
        """
        obj = self._create_empty_mesh("undo_edit")
        obj_name = obj.name
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        ctx = _get_context_override()
        undo_ctx = self._undo_ctx()

        # Baseline undo step
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="Before box")

        # Build box
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
                         "Step 1: should be INVERT after box build")

        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="After box")

        # Deselect all → weld should clear to NONE
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.select_all(action='DESELECT')

        yield 0.5

        self.assertEqual(props.weld_mode, 'NONE',
                         "Step 2: should be NONE after deselect")

        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="After deselect")

        # Undo deselect → should restore INVERT
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo()

        yield 0.5

        obj = bpy.data.objects[obj_name]
        bpy.context.view_layer.objects.active = obj
        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'INVERT',
                         "Step 3: should be INVERT after undo")

    def test_edit_mode_undo_weld_restores_invert(self):
        """Edit-mode: build box → weld invert → undo should restore INVERT.

        Verifies that undoing a weld invert re-derives INVERT from the
        consumed entries list.
        """
        obj = self._create_empty_mesh("undo_weld")
        obj_name = obj.name
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        ctx = _get_context_override()
        undo_ctx = self._undo_ctx()

        # Baseline undo step
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="Before box")

        # Build box
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
                         "Step 1: should be INVERT after box build")

        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="After box")

        # Execute weld invert
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result,
                       "Weld INVERT should succeed")
        self.assertEqual(props.weld_mode, 'NONE',
                         "Step 2: should be NONE after weld invert")

        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="After weld")

        # Undo weld → should restore INVERT
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo()

        yield 0.5

        obj = bpy.data.objects[obj_name]
        bpy.context.view_layer.objects.active = obj
        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'INVERT',
                         "Step 3: should be INVERT after undoing weld")
