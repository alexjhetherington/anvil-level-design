from unittest.mock import patch

import bmesh
import bpy
from mathutils import Vector

from ..utils import derive_transform_from_uvs, get_image_from_material
from .. import handlers as handlers_module
from ..handlers import set_active_image, apply_texture_from_file_browser
from ..operators.box_builder.geometry import execute_box_builder
from .base_test import AnvilTestCase
from .helpers import create_vertical_plane, _get_context_override, TEXTURE_PATH


def _face_key(face):
    """Return a hashable key from the face's normal and centroid."""
    c = face.calc_center_median()
    n = face.normal
    return (round(n.x), round(n.y), round(n.z),
            round(c.x, 2), round(c.y, 2), round(c.z, 2))


class BoxBuilderTest(AnvilTestCase):
    """Test box builder geometry creation and UV mapping."""

    def test_previous_texture(self):
        """Box builder uses the previously selected texture when no face is active."""
        # Load the test image and seed it as the active image
        image = bpy.data.images.load(TEXTURE_PATH, check_existing=True)
        set_active_image(image)

        # Create an empty mesh object and enter edit mode
        mesh = bpy.data.meshes.new("bb_prev_tex")
        obj = bpy.data.objects.new("bb_prev_tex", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        # Build a 1x1x1 box: rectangle in XZ plane, depth along +Y
        success, msg = execute_box_builder(
            Vector((0, 0, 0)),
            Vector((1, 0, 1)),
            1.0,
            Vector((1, 0, 0)),
            Vector((0, 0, 1)),
            Vector((0, 1, 0)),
            obj, ppm, False,
        )

        self.assertTrue(success, msg)

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        self.assertEqual(len(bm.faces), 6, "Box should have 6 faces")

        # Face-aligned projection maps world coordinates onto the best
        # projection plane per face.  Rotation varies by orientation.
        # Key: (nx, ny, nz, cx, cy, cz)
        expected = {
            (0, -1, 0, 0.5, 0.0, 0.5):  (90.0, 0.0, 0.0),   # front
            (0, 1, 0, 0.5, 1.0, 0.5):   (0.0, 0.0, 0.0),     # back
            (-1, 0, 0, 0.0, 0.5, 0.5):  (0.0, 0.0, 0.0),     # left
            (1, 0, 0, 1.0, 0.5, 0.5):   (90.0, 0.0, 0.0),    # right
            (0, 0, -1, 0.5, 0.5, 0.0):  (0.0, 0.0, 0.0),     # bottom
            (0, 0, 1, 0.5, 0.5, 1.0):   (90.0, 0.0, 0.0),    # top
        }

        for face in bm.faces:
            key = _face_key(face)
            self.assertIn(key, expected, f"Unexpected face key {key}")
            rot, off_x, off_y = expected[key]
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertAlmostEqual(
                t['scale_u'], 1.0, places=3,
                msg=f"Face {key} scale_u={t['scale_u']}")
            self.assertAlmostEqual(
                t['scale_v'], 1.0, places=3,
                msg=f"Face {key} scale_v={t['scale_v']}")
            self.assertAlmostEqual(
                t['rotation'], rot, places=3,
                msg=f"Face {key} rotation={t['rotation']}")
            self.assertAlmostEqual(
                t['offset_x'], off_x, places=3,
                msg=f"Face {key} offset_x={t['offset_x']}")
            self.assertAlmostEqual(
                t['offset_y'], off_y, places=3,
                msg=f"Face {key} offset_y={t['offset_y']}")

    def test_file_browser_texture_apply(self):
        """Applying a texture via file browser route to a blank box."""
        # Create an empty mesh object and enter edit mode
        mesh = bpy.data.meshes.new("bb_fb_apply")
        obj = bpy.data.objects.new("bb_fb_apply", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        # Build a 1x1x1 box with NO active image — box builder skips
        # material/UV assignment when there is no image available
        success, msg = execute_box_builder(
            Vector((0, 0, 0)),
            Vector((1, 0, 1)),
            1.0,
            Vector((1, 0, 0)),
            Vector((0, 0, 1)),
            Vector((0, 1, 0)),
            obj, ppm, False,
        )
        self.assertTrue(success, msg)

        # Verify box was created with no material
        self.assertEqual(len(obj.data.materials), 0,
                         "Box should have no materials before texture apply")

        # Switch to object mode — file browser apply works on all faces
        # of the selected object when in object mode
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

        # Apply texture via the file browser route, mocking the file
        # browser selection to return our test texture path
        with patch.object(handlers_module, 'get_selected_image_path',
                          return_value=TEXTURE_PATH):
            apply_texture_from_file_browser()

        # Verify material was applied
        self.assertEqual(len(obj.data.materials), 1,
                         "Box should have 1 material after texture apply")
        mat = obj.data.materials[0]
        image = get_image_from_material(mat)
        self.assertIsNotNone(image, "Material should have an image texture")

        # Re-enter edit mode to inspect UVs
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        self.assertEqual(len(bm.faces), 6, "Box should have 6 faces")

        # Blank faces get per-face local projection with clean defaults
        # from _apply_regular_uv_projection: scale=1.0, rotation=0.0, offset=0.0
        for face in bm.faces:
            key = _face_key(face)
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertAlmostEqual(
                t['scale_u'], 1.0, places=3,
                msg=f"Face {key} scale_u={t['scale_u']}")
            self.assertAlmostEqual(
                t['scale_v'], 1.0, places=3,
                msg=f"Face {key} scale_v={t['scale_v']}")
            self.assertAlmostEqual(
                t['rotation'], 0.0, places=3,
                msg=f"Face {key} rotation={t['rotation']}")
            self.assertAlmostEqual(
                t['offset_x'], 0.0, places=3,
                msg=f"Face {key} offset_x={t['offset_x']}")
            self.assertAlmostEqual(
                t['offset_y'], 0.0, places=3,
                msg=f"Face {key} offset_y={t['offset_y']}")

    def test_active_face_texture(self):
        """Box builder copies texture from the actively selected face."""
        # Create a vertical plane (textured with dev_orange_wall.png)
        obj = create_vertical_plane("bb_active_face")

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        # Select and activate the plane face so box builder copies from it
        bm.select_mode = {'FACE'}
        source_face = bm.faces[0]
        source_face.select = True
        bm.faces.active = source_face
        bmesh.update_edit_mesh(obj.data)

        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        # Build a 1x1x1 box offset along X to avoid overlapping the plane
        success, msg = execute_box_builder(
            Vector((2, 0, 0)),
            Vector((3, 0, 1)),
            1.0,
            Vector((1, 0, 0)),
            Vector((0, 0, 1)),
            Vector((0, 1, 0)),
            obj, ppm, False,
        )

        self.assertTrue(success, msg)

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        # 1 original plane face + 6 box faces
        self.assertEqual(len(bm.faces), 7,
                         "Should have 1 plane + 6 box faces")

        # Expected transforms per face, keyed by (normal, centroid).
        # Rotation varies by projection angle from the source face; offsets
        # are always zero because source and targets share aligned origins.
        # Key: (nx, ny, nz, cx, cy, cz)
        expected = {
            (0, -1, 0, 0.5, 0.0, 0.5):  (0.0, 0.0, 0.0),    # source plane
            (0, -1, 0, 2.5, 0.0, 0.5):  (90.0, 0.0, 0.0),   # box front
            (0, 1, 0, 2.5, 1.0, 0.5):   (180.0, 0.0, 0.0),   # box back
            (-1, 0, 0, 2.0, 0.5, 0.5):  (180.0, 0.0, 0.0),   # box left
            (1, 0, 0, 3.0, 0.5, 0.5):   (90.0, 0.0, 0.0),    # box right
            (0, 0, -1, 2.5, 0.5, 0.0):  (0.0, 0.0, 0.0),     # box bottom
            (0, 0, 1, 2.5, 0.5, 1.0):   (90.0, 0.0, 0.0),    # box top
        }

        for face in bm.faces:
            key = _face_key(face)
            self.assertIn(key, expected, f"Unexpected face key {key}")
            rot, off_x, off_y = expected[key]
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertAlmostEqual(
                t['scale_u'], 1.0, places=3,
                msg=f"Face {key} scale_u={t['scale_u']}")
            self.assertAlmostEqual(
                t['scale_v'], 1.0, places=3,
                msg=f"Face {key} scale_v={t['scale_v']}")
            self.assertAlmostEqual(
                t['rotation'], rot, places=3,
                msg=f"Face {key} rotation={t['rotation']}")
            self.assertAlmostEqual(
                t['offset_x'], off_x, places=3,
                msg=f"Face {key} offset_x={t['offset_x']}")
            self.assertAlmostEqual(
                t['offset_y'], off_y, places=3,
                msg=f"Face {key} offset_y={t['offset_y']}")
