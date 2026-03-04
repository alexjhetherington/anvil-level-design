import bmesh
import bpy
from mathutils import Vector

from ..utils import derive_transform_from_uvs
from ..operators.cube_cut.geometry import execute_cube_cut
from .base_test import AnvilTestCase
from .helpers import create_textured_cube, _get_context_override


def _face_key(face):
    """Return a hashable key from the face's normal and centroid."""
    c = face.calc_center_median()
    n = face.normal
    return (round(n.x), round(n.y), round(n.z),
            round(c.x, 2), round(c.y, 2), round(c.z, 2))


class CubeCutTest(AnvilTestCase):
    """Test cube cut geometry and UV preservation."""

    def test_through_hole(self):
        """Cut a hole through a textured cube and verify UVs on remaining faces."""
        obj = create_textured_cube("cc_cube", 1.0, 1.0, face_aligned=True)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        # Select all faces so cube cut processes them
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(obj.data)

        # Cut a hole through the cube along the Y axis.
        # The cube spans (0,0,0) to (1,1,1).
        # The cut rectangle is at x=[0.25,0.75], z=[0.25,0.75],
        # extending from y=-0.5 to y=1.5 (fully through the cube).
        with bpy.context.temp_override(**ctx):
            success, msg = execute_cube_cut(
                bpy.context,
                Vector((0.25, -0.5, 0.25)),
                Vector((0.75, -0.5, 0.75)),
                2.0,
                Vector((1, 0, 0)),
                Vector((0, 0, 1)),
                Vector((0, 1, 0)),
            )

        self.assertTrue(success, msg)

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        self.assertEqual(len(bm.faces), 12,
                         "Should have 4 uncut + 4 front frame + 4 back frame faces")

        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        # Expected transforms per face, keyed by (normal, centroid).
        # The source cube uses face-aligned projection, so uncut faces
        # inherit those rotations. Cut frame pieces preserve the original
        # face's UV mapping via planar re-projection.
        # Key: (nx, ny, nz, cx, cy, cz)
        expected = {
            # Uncut faces
            (-1, 0, 0, 0.0, 0.5, 0.5):   (90.0, 0.0, 0.0),    # left
            (1, 0, 0, 1.0, 0.5, 0.5):     (90.0, 0.0, 0.0),    # right
            (0, 0, -1, 0.5, 0.5, 0.0):    (0.0, 0.0, 0.0),     # bottom
            (0, 0, 1, 0.5, 0.5, 1.0):     (180.0, 0.0, 0.0),   # top
            # Front frame (-Y normal at y=0)
            (0, -1, 0, 0.88, 0.0, 0.5):   (-90.0, 0.75, 0.75),
            (0, -1, 0, 0.5, 0.0, 0.88):   (0.0, 0.25, 0.75),
            (0, -1, 0, 0.5, 0.0, 0.12):   (180.0, 0.75, 0.25),
            (0, -1, 0, 0.12, 0.0, 0.5):   (90.0, 0.25, 0.25),
            # Back frame (+Y normal at y=1)
            (0, 1, 0, 0.5, 1.0, 0.88):    (180.0, 0.75, 0.75),
            (0, 1, 0, 0.12, 1.0, 0.5):    (-90.0, 0.25, 0.75),
            (0, 1, 0, 0.5, 1.0, 0.12):    (0.0, 0.25, 0.25),
            (0, 1, 0, 0.88, 1.0, 0.5):    (90.0, 0.75, 0.25),
        }

        for face in bm.faces:
            key = _face_key(face)
            self.assertIn(key, expected, f"Unexpected face key {key}")
            rot, off_x, off_y = expected[key]
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertAlmostEqual(
                t['scale_u'], 1.0, places=2,
                msg=f"Face {key} scale_u={t['scale_u']}")
            self.assertAlmostEqual(
                t['scale_v'], 1.0, places=2,
                msg=f"Face {key} scale_v={t['scale_v']}")
            self.assertAlmostEqual(
                t['rotation'], rot, places=2,
                msg=f"Face {key} rotation={t['rotation']}")
            self.assertAlmostEqual(
                t['offset_x'], off_x, places=2,
                msg=f"Face {key} offset_x={t['offset_x']}")
            self.assertAlmostEqual(
                t['offset_y'], off_y, places=2,
                msg=f"Face {key} offset_y={t['offset_y']}")
