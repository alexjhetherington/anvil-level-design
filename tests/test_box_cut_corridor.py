import bmesh
import bpy
from mathutils import Vector

from ..handlers import set_active_image
from ..operators.box_builder.geometry import execute_box_builder
from ..operators.cube_cut.geometry import execute_cube_cut
from ..operators.weld import set_weld_from_box_builder, set_weld_from_edge_selection
from ..utils import derive_transform_from_uvs
from .base_test import AnvilTestCase
from .helpers import _get_context_override, TEXTURE_PATH


def _vert_key(v):
    """Return a hashable key from a vertex position, rounded to 4dp."""
    return (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))


class BoxCutCorridorTest(AnvilTestCase):
    """Integration test: box build → weld invert → cube cut outward → weld corridor."""

    def test_box_invert_cut_corridor_weld(self):
        """Build a cube, invert normals, cube cut wall outward, corridor weld.

        1. Box builder creates a 1×1×1 cube at (0,0,0)-(1,1,1).
        2. Weld invert flips normals inward (room interior).
        3. Cube cut punches a horizontal hole in the y=0 wall going
           outward (-Y). Hole at x=[0.25,0.75], z=[0.0,0.75] — a
           wide, low opening like a corridor a character walks through.
        4. Corridor weld fills the hole and extrudes to y=-0.5.

        Final geometry: 15 faces, 16 verts.
        """
        # -- Step 1: Box builder creates a textured cube --
        image = bpy.data.images.load(TEXTURE_PATH, check_existing=True)
        set_active_image(image)

        mesh = bpy.data.meshes.new("box_cut_corridor")
        obj = bpy.data.objects.new("box_cut_corridor", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        box_result = execute_box_builder(
            Vector((0, 0, 0)), Vector((1, 0, 1)), 1.0,
            Vector((1, 0, 0)), Vector((0, 0, 1)), Vector((0, 1, 0)),
            obj, ppm, Vector((0, -1, 0)),
        )
        self.assertTrue(box_result[0], box_result[1])

        bm = bmesh.from_edit_mesh(obj.data)
        self.assertEqual(len(bm.verts), 8, "Box should have 8 verts")
        self.assertEqual(len(bm.faces), 6, "Box should have 6 faces")
        self.assertEqual(len(obj.data.materials), 1,
                         "Box should have a material from the active image")

        # -- Step 2: Weld invert --
        face_verts = box_result[2] if len(box_result) > 2 else []
        set_weld_from_box_builder(bpy.context, face_verts)

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'INVERT',
                         "Standalone box should set weld to INVERT")

        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)
        self.assertEqual(props.weld_mode, 'NONE',
                         "Weld mode should be NONE after invert")

        # Verify normals flipped inward: each face normal should point
        # toward the cube center (0.5, 0.5, 0.5).
        bm = bmesh.from_edit_mesh(obj.data)
        bm.normal_update()
        center = Vector((0.5, 0.5, 0.5))
        for face in bm.faces:
            to_center = center - face.calc_center_median()
            self.assertGreater(face.normal.dot(to_center), 0,
                               f"Face at {face.calc_center_median()} should point inward")

        # -- Step 3: Cube cut from y=0 face outward --
        # Select all faces for cube cut
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(obj.data)

        # Cut through the y=0 wall going outward (-Y).
        # Rectangle on the wall: x=[0.25,0.75], z=[0.0,0.75]
        # Cuboid extends from y=0.25 out to y=-0.5
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

        yield 0.1

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        # Edge-aligned cut: y=0 wall gets 3 frame pieces (no bottom strip),
        # z=0 floor gets split where the cut rectangle touches it.
        self.assertEqual(len(bm.faces), 10,
                         f"After cube cut should have 10 faces, got {len(bm.faces)}")
        self.assertEqual(len(bm.verts), 12,
                         f"After cube cut should have 12 verts, got {len(bm.verts)}")

        # Verify 4 boundary edges selected
        selected_edges = [e for e in bm.edges if e.select]
        self.assertEqual(len(selected_edges), 4,
                         f"Should have 4 boundary edges selected, got {len(selected_edges)}")

        # -- Step 4: Weld corridor --
        # Replicate what the cube cut operator does:
        # extrude_dir = -local_z = (0, 1, 0)
        # back_point = first_vertex + local_z * depth
        #            = (0.25, 0.25, 0.0) + (0, -0.75, 0) = (0.25, -0.5, 0.0)
        # back_plane_offset = back_point · extrude_dir = -0.5
        set_weld_from_edge_selection(
            bpy.context, 0.75, (0, 1, 0), -0.5,
        )

        self.assertEqual(props.weld_mode, 'CORRIDOR',
                         "Should be CORRIDOR after cube cut on single face")

        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)
        self.assertEqual(props.weld_mode, 'NONE',
                         "Weld mode should be NONE after corridor")

        # Let depsgraph handler fire
        yield 0.5

        # -- Geometry assertions --
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        self.assertEqual(len(bm.verts), 16,
                         f"After corridor should have 16 verts, got {len(bm.verts)}")
        self.assertEqual(len(bm.faces), 15,
                         f"After corridor should have 15 faces, got {len(bm.faces)}")

        # Verify corridor cap verts exist at y=-0.5
        cap_verts = [v for v in bm.verts
                     if abs(v.co.y - (-0.5)) < 0.001]
        self.assertEqual(len(cap_verts), 4,
                         f"Should have 4 cap verts at y=-0.5, got {len(cap_verts)}")

        expected_cap_positions = sorted([
            (0.25, -0.5, 0.0), (0.75, -0.5, 0.0),
            (0.75, -0.5, 0.75), (0.25, -0.5, 0.75),
        ])
        actual_cap_positions = sorted([_vert_key(v) for v in cap_verts])
        self.assertEqual(actual_cap_positions, expected_cap_positions,
                         "Cap verts should be at the correct positions")

        # -- UV assertions: all faces should have scale=1 and clean rotation --
        uv_layer = bm.loops.layers.uv[0]
        clean_rotations = {0.0, 90.0, -90.0, 180.0, -180.0, 270.0, -270.0, 360.0}

        for face in bm.faces:
            c = face.calc_center_median()
            n = face.normal
            label = (f"n=({n.x:.1f},{n.y:.1f},{n.z:.1f}) "
                     f"c=({c.x:.2f},{c.y:.2f},{c.z:.2f})")
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertIsNotNone(t, f"Face {label} has no UV transform")
            self.assertAlmostEqual(
                t['scale_u'], 1.0, places=2,
                msg=f"Face {label} scale_u={t['scale_u']:.4f}")
            self.assertAlmostEqual(
                t['scale_v'], 1.0, places=2,
                msg=f"Face {label} scale_v={t['scale_v']:.4f}")
            # Rotation should be a clean multiple of 90
            rot = t['rotation']
            closest = min(clean_rotations, key=lambda r: abs(r - rot))
            self.assertAlmostEqual(
                rot, closest, places=1,
                msg=f"Face {label} rotation={rot:.2f} (expected clean 90° multiple)")

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()
