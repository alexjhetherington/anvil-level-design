import bmesh
import bpy
from mathutils import Vector

from ..handlers import set_active_image
from ..operators.box_builder.geometry import execute_box_builder
from ..operators.cube_cut.geometry import execute_cube_cut
from ..operators.weld import set_weld_from_box_builder, set_weld_from_edge_selection
from ..core.uv_projection import derive_transform_from_uvs
from .base_test import AnvilTestCase
from .helpers import _get_context_override, TEXTURE_PATH


# NOTE: Two interesting observations from this test's UV snapshot:
# 1. Two floor faces have non-axis-aligned rotations that are valid:
#    c=(0.17,0.67,0.0) rot=-75.96°  and  c=(0.83,0.67,0.0) rot=-104.04°


def _r(v):
    """Round to 2dp, normalizing -0.0 to 0.0."""
    r = round(v, 2)
    return 0.0 if r == 0.0 else r


def _face_key(face):
    """Hashable key from face normal + center, rounded to 2dp."""
    c = face.calc_center_median()
    n = face.normal
    return (_r(n.x), _r(n.y), _r(n.z), _r(c.x), _r(c.y), _r(c.z))


# Expected UV transforms keyed by (nx, ny, nz, cx, cy, cz).
# Values are (rotation, scale_u, scale_v).
_EXPECTED = {
    (0.0, 1.0, 0.0, 0.5, 0.0, 0.88):       (0.00, 1.0, 1.0),
    (0.0, -1.0, 0.0, 0.5, 1.0, 0.62):       (180.00, 1.0, 1.0),
    (1.0, 0.0, 0.0, 0.0, 0.5, 0.5):         (90.00, 1.0, 1.0),
    (-1.0, 0.0, 0.0, 1.0, 0.5, 0.5):        (0.00, 1.0, 1.0),
    (0.0, 1.0, 0.0, 0.5, -0.5, 0.38):       (90.00, 1.0, 1.0),
    (0.0, 0.0, -1.0, 0.5, 0.5, 1.0):        (0.00, 1.0, 1.0),
    (0.0, 1.0, 0.0, 0.12, 0.0, 0.44):       (-90.00, 1.0, 1.0),
    (0.0, 1.0, 0.0, 0.88, 0.0, 0.44):       (90.00, 1.0, 1.0),
    (-1.0, 0.0, 0.0, 0.75, -0.25, 0.38):    (-90.00, 1.0, 1.0),
    (0.0, 0.0, 1.0, 0.08, 0.33, 0.0):       (-90.00, 1.0, 1.0),
    (0.0, 0.0, 1.0, 0.92, 0.33, 0.0):       (0.00, 1.0, 1.0),
    (0.0, -1.0, 0.0, 0.5, 1.25, 0.12):      (-90.00, 1.0, 1.0),
    (0.0, 0.0, -1.0, 0.5, -0.25, 0.75):     (180.00, 1.0, 1.0),
    (0.0, 0.0, 1.0, 0.5, -0.25, 0.0):       (180.00, 1.0, 1.0),
    (1.0, 0.0, 0.0, 0.25, -0.25, 0.38):     (90.00, 1.0, 1.0),
    (0.0, -1.0, 0.0, 0.88, 1.0, 0.31):      (90.00, 1.0, 1.0),
    (0.0, -1.0, 0.0, 0.12, 1.0, 0.31):      (-90.00, 1.0, 1.0),
    (0.0, 0.0, 1.0, 0.17, 0.67, 0.0):       (-75.96, 1.0, 1.0),
    (-1.0, 0.0, 0.0, 0.75, 1.12, 0.12):     (-90.00, 1.0, 1.0),
    (0.0, 0.0, 1.0, 0.83, 0.67, 0.0):       (75.96, 1.0, 1.0),
    (0.0, 0.0, 1.0, 0.5, 0.5, 0.0):         (90.00, 1.0, 1.0),
    (1.0, 0.0, 0.0, 0.25, 1.12, 0.12):      (90.00, 1.0, 1.0),
    (0.0, 0.0, -1.0, 0.5, 1.12, 0.25):      (0.00, 1.0, 1.0),
    (0.0, 0.0, 1.0, 0.5, 1.12, 0.0):        (0.00, 1.0, 1.0),
}


class BoxCutDualCorridorTest(AnvilTestCase):
    """Build box → invert → two opposing corridors, assert all UVs."""

    def test_opposing_corridors_uvs(self):
        """Build two opposing corridors and verify every face's UV transform."""
        image = bpy.data.images.load(TEXTURE_PATH, check_existing=True)
        set_active_image(image)

        mesh = bpy.data.meshes.new("dual_corridor")
        obj = bpy.data.objects.new("dual_corridor", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        # Box builder: 1x1x1 cube
        box_result = execute_box_builder(
            Vector((0, 0, 0)), Vector((1, 0, 1)), 1.0,
            Vector((1, 0, 0)), Vector((0, 0, 1)), Vector((0, 1, 0)),
            obj, ppm, Vector((0, -1, 0)),
        )
        self.assertTrue(box_result[0], box_result[1])

        # Weld invert
        face_verts = box_result[2] if len(box_result) > 2 else []
        set_weld_from_box_builder(bpy.context, face_verts)
        with bpy.context.temp_override(**ctx):
            bpy.ops.leveldesign.context_weld()

        # First cut: y=0 face outward (-Y)
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(obj.data)

        with bpy.context.temp_override(**ctx):
            success, msg = execute_cube_cut(
                bpy.context,
                Vector((0.25, 0.25, 0.0)),
                Vector((0.75, 0.25, 0.75)),
                0.75,
                Vector((1, 0, 0)),
                Vector((0, 0, 1)),
                Vector((0, -1, 0)),
            )
        self.assertTrue(success, msg)

        # First corridor weld
        set_weld_from_edge_selection(
            bpy.context, 0.75, (0, 1, 0), -0.5,
            Vector((0.25, 0.25, 0.0)), Vector((0.75, 0.25, 0.75)),
            Vector((1, 0, 0)), Vector((0, 0, 1)),
            0,
        )
        with bpy.context.temp_override(**ctx):
            bpy.ops.leveldesign.context_weld()

        yield 0.5

        bm = bmesh.from_edit_mesh(obj.data)
        self.assertEqual(len(bm.faces), 15, "After first corridor should have 15 faces")

        # Second cut: y=1 wall outward (+Y)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = False
        bmesh.update_edit_mesh(obj.data)

        with bpy.context.temp_override(**ctx):
            success, msg = execute_cube_cut(
                bpy.context,
                Vector((0.25, 1.0, 0.0)),
                Vector((0.75, 1.0, 0.25)),
                0.25,
                Vector((1, 0, 0)),
                Vector((0, 0, 1)),
                Vector((0, 1, 0)),
            )
        self.assertTrue(success, msg)

        # Second corridor weld
        set_weld_from_edge_selection(
            bpy.context, 0.25, (-0.0, 1.0, -0.0), 1.25,
            Vector((0.25, 1.0, 0.0)), Vector((0.75, 1.0, 0.25)),
            Vector((1, 0, 0)), Vector((0, 0, 1)),
            0,
        )
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        yield 0.5

        # Assert geometry
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        self.assertEqual(len(bm.faces), 24,
                         f"Should have 24 faces, got {len(bm.faces)}")

        # Assert every face's UV transform matches expected snapshot
        uv_layer = bm.loops.layers.uv[0]

        for face in bm.faces:
            key = _face_key(face)
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            c = face.calc_center_median()
            n = face.normal
            label = (f"n=({n.x:.2f},{n.y:.2f},{n.z:.2f}) "
                     f"c=({c.x:.2f},{c.y:.2f},{c.z:.2f})")

            self.assertIn(key, _EXPECTED,
                          f"Unexpected face {label}")
            self.assertIsNotNone(t, f"Face {label} has no UV transform")

            expected_rot, expected_su, expected_sv = _EXPECTED[key]
            self.assertAlmostEqual(
                t['rotation'], expected_rot, places=1,
                msg=f"Face {label} rotation={t['rotation']:.2f}, expected {expected_rot:.2f}")
            self.assertAlmostEqual(
                t['scale_u'], expected_su, places=2,
                msg=f"Face {label} scale_u={t['scale_u']:.4f}, expected {expected_su}")
            self.assertAlmostEqual(
                t['scale_v'], expected_sv, places=2,
                msg=f"Face {label} scale_v={t['scale_v']:.4f}, expected {expected_sv}")

        # Mark faces for visual identification of non-determinism
        _marks = {
            # Corridor 1 floor
            (0.0, 0.0, 1.0, 0.5, -0.25, 0.0): 0.1,
            # Corridor 1 left wall
            (1.0, 0.0, 0.0, 0.25, -0.25, 0.38): 5.0,
            # Central floor
            (0.0, 0.0, 1.0, 0.5, 0.5, 0.0): 10.0,
        }
        for face in bm.faces:
            key = _face_key(face)
            if key in _marks:
                _scale = _marks[key]
                for loop in face.loops:
                    loop[uv_layer].uv *= _scale

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()
