import bmesh
import bpy
from mathutils import Vector

from ..operators.cube_cut.geometry import execute_cube_cut
from ..operators.weld import set_weld_from_edge_selection
from .base_test import AnvilTestCase
from .helpers import create_textured_cube, _get_context_override


# Cube cut parameters: cut from bottom (z=0) to middle (z=0.5) of a plain cube
_FIRST_VERTEX = Vector((0.25, -0.5, 0.0))
_SECOND_VERTEX = Vector((0.75, -0.5, 0.5))
_DEPTH = 2.0
_LOCAL_X = Vector((1, 0, 0))
_LOCAL_Y = Vector((0, 0, 1))
_LOCAL_Z = Vector((0, 1, 0))


def _setup_plain_cube(name):
    """Create a plain textured cube in edit mode with all faces selected.

    Returns (obj, ctx).
    """
    obj = create_textured_cube(name, 1.0, 1.0, face_aligned=True)
    ctx = _get_context_override()

    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.select_mode = {'FACE'}
    for f in bm.faces:
        f.select = True
    bmesh.update_edit_mesh(obj.data)

    return obj, ctx


def _execute_cube_cut_and_set_weld(test_case, ctx):
    """Execute the cube cut and set up weld state. Returns success bool."""
    with bpy.context.temp_override(**ctx):
        success, msg = execute_cube_cut(
            bpy.context,
            _FIRST_VERTEX, _SECOND_VERTEX, _DEPTH,
            _LOCAL_X, _LOCAL_Y, _LOCAL_Z,
        )
    test_case.assertTrue(success, msg)

    extrude_dir = -_LOCAL_Z  # (0, -1, 0)
    back_point = _FIRST_VERTEX + _LOCAL_Z * _DEPTH
    back_plane_offset = back_point.dot(extrude_dir.normalized())

    set_weld_from_edge_selection(
        bpy.context, abs(_DEPTH), extrude_dir, back_plane_offset,
        _FIRST_VERTEX, _SECOND_VERTEX, _LOCAL_X, _LOCAL_Y,
    )

    return success


def _vert_key(v):
    """Return a hashable key from a vertex position, rounded to 4dp."""
    return (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))


class FoldedPlaneWeldCubeTest(AnvilTestCase):
    """Test folded plane weld after cube cut on a plain cube (bottom-to-middle cut)."""

    def test_folded_plane_weld_on_plain_cube_bottom_to_middle_cut(self):
        """Cube cut a plain cube from z=0 to z=0.5 (bottom to middle),
        then execute folded plane weld.

        The cut removes a slot from the bottom of the cube between x=0.25
        and x=0.75, creating a non-coplanar boundary edge loop that wraps
        around the front, bottom, and back faces.
        """
        obj, ctx = _setup_plain_cube("weld_folded_cube")

        _execute_cube_cut_and_set_weld(self, ctx)

        yield 0.1

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'FOLDED_PLANE',
                         "Should be FOLDED_PLANE after cube cut on plain cube "
                         "with bottom-to-middle cut")

        # Execute folded plane weld
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        self.assertEqual(props.weld_mode, 'NONE',
                         "Weld mode should be NONE after folded plane weld")

        # Let depsgraph handler fire to apply UVs
        yield 0.5

        # --- Geometry assertions ---
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        bm.normal_update()

        self.assertEqual(len(bm.verts), 16)
        self.assertEqual(len(bm.faces), 15)

        # Verify all vertex positions
        expected_verts = sorted([
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),
            (0.25, 0.0, 0.0), (0.25, 0.0, 0.5),
            (0.25, 1.0, 0.0), (0.25, 1.0, 0.5),
            (0.75, 0.0, 0.0), (0.75, 0.0, 0.5),
            (0.75, 1.0, 0.0), (0.75, 1.0, 0.5),
            (1.0, 0.0, 0.0), (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 1.0, 1.0),
        ])
        actual_verts = sorted([_vert_key(v) for v in bm.verts])
        self.assertEqual(actual_verts, expected_verts)

        # All faces should be valid (no degenerate faces)
        for face in bm.faces:
            self.assertGreaterEqual(len(face.verts), 3,
                                    "All faces should have at least 3 verts")
            area = face.calc_area()
            self.assertGreater(area, 1e-6,
                               f"Face should have non-zero area, got {area}")

        # Check that UVs exist on all faces
        uv_layer = bm.loops.layers.uv.active
        self.assertIsNotNone(uv_layer, "Should have an active UV layer")

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()
