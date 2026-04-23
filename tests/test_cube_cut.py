import bmesh
import bpy
from mathutils import Vector

from ..core.uv_projection import derive_transform_from_uvs
from ..operators.cube_cut.geometry import execute_cube_cut
from .base_test import AnvilTestCase
from .helpers import (
    create_textured_cube,
    add_uv_layer_face_aligned,
    _get_context_override,
    _apply_material_face_aligned,
)


def _face_key(face):
    """Return a hashable key from the face's normal and centroid."""
    c = face.calc_center_median()
    n = face.normal
    return (round(n.x), round(n.y), round(n.z),
            round(c.x, 2), round(c.y, 2), round(c.z, 2))


class CubeCutTest(AnvilTestCase):
    """Test cube cut geometry and UV preservation."""

    def test_cube_cut_through_hole(self):
        """Cut a hole through a textured cube and verify UVs on remaining faces."""
        obj = create_textured_cube("cc_cube", 1.0, 1.0, face_aligned=True)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        add_uv_layer_face_aligned(obj, "UVMap.001", 0.5)

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
        uv_layer = bm.loops.layers.uv[0]
        uv_layer2 = bm.loops.layers.uv[1]
        bm.faces.ensure_lookup_table()

        self.assertEqual(len(bm.faces), 12,
                         "Should have 4 uncut + 4 front frame + 4 back frame faces")

        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        # Expected transforms per face, keyed by (normal, centroid).
        # The source cube uses face-aligned projection, so uncut faces
        # inherit those rotations. Cut frame pieces preserve the original
        # face's UV mapping via planar re-projection.
        # Key: (nx, ny, nz, cx, cy, cz)
        # Values: (rotation, L1_offset_x, L1_offset_y, L2_offset_x, L2_offset_y)
        expected = {
            # Uncut faces
            (-1, 0, 0, 0.0, 0.5, 0.5):   (90.0, 0.0, 0.0, 0.0, 0.0),
            (1, 0, 0, 1.0, 0.5, 0.5):     (90.0, 0.0, 0.0, 0.0, 0.0),
            (0, 0, -1, 0.5, 0.5, 0.0):    (0.0, 0.0, 0.0, 0.0, 0.0),
            (0, 0, 1, 0.5, 0.5, 1.0):     (180.0, 0.0, 0.0, 0.0, 0.0),
            # Front frame (-Y normal at y=0)
            (0, -1, 0, 0.88, 0.0, 0.5):   (-90.0, 0.75, 0.75, 0.5, 0.5),
            (0, -1, 0, 0.5, 0.0, 0.88):   (0.0, 0.25, 0.75, 0.5, 0.5),
            (0, -1, 0, 0.5, 0.0, 0.12):   (180.0, 0.75, 0.25, 0.5, 0.5),
            (0, -1, 0, 0.12, 0.0, 0.5):   (90.0, 0.25, 0.25, 0.5, 0.5),
            # Back frame (+Y normal at y=1) — face_aligned_project flips U
            # on +Y faces, so offsets on U and some rotations flip relative
            # to the mirrored counterparts on the front frame.
            (0, 1, 0, 0.5, 1.0, 0.88):    (0.0, 0.25, 0.75, 0.5, 0.5),
            (0, 1, 0, 0.12, 1.0, 0.5):    (-90.0, 0.75, 0.75, 0.5, 0.5),
            (0, 1, 0, 0.5, 1.0, 0.12):    (180.0, 0.75, 0.25, 0.5, 0.5),
            (0, 1, 0, 0.88, 1.0, 0.5):    (90.0, 0.25, 0.25, 0.5, 0.5),
        }

        for face in bm.faces:
            key = _face_key(face)
            self.assertIn(key, expected, f"Unexpected face key {key}")
            rot, off_x, off_y, off2_x, off2_y = expected[key]

            # Layer 1: scale=1.0
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertAlmostEqual(
                t['scale_u'], 1.0, places=2,
                msg=f"Face {key} L1 scale_u={t['scale_u']}")
            self.assertAlmostEqual(
                t['scale_v'], 1.0, places=2,
                msg=f"Face {key} L1 scale_v={t['scale_v']}")
            self.assertAlmostEqual(
                t['rotation'], rot, places=2,
                msg=f"Face {key} L1 rotation={t['rotation']}")
            self.assertAlmostEqual(
                t['offset_x'], off_x, places=2,
                msg=f"Face {key} L1 offset_x={t['offset_x']}")
            self.assertAlmostEqual(
                t['offset_y'], off_y, places=2,
                msg=f"Face {key} L1 offset_y={t['offset_y']}")

            # Layer 2: scale=0.5, same rotation
            t2 = derive_transform_from_uvs(face, uv_layer2, ppm, obj.data)
            self.assertAlmostEqual(
                t2['scale_u'], 0.5, places=2,
                msg=f"Face {key} L2 scale_u={t2['scale_u']}")
            self.assertAlmostEqual(
                t2['scale_v'], 0.5, places=2,
                msg=f"Face {key} L2 scale_v={t2['scale_v']}")
            self.assertAlmostEqual(
                t2['rotation'], rot, places=2,
                msg=f"Face {key} L2 rotation={t2['rotation']}")
            self.assertAlmostEqual(
                t2['offset_x'], off2_x, places=2,
                msg=f"Face {key} L2 offset_x={t2['offset_x']}")
            self.assertAlmostEqual(
                t2['offset_y'], off2_y, places=2,
                msg=f"Face {key} L2 offset_y={t2['offset_y']}")

        # Verify edge select mode after cube cut
        self.assertEqual(
            list(bpy.context.tool_settings.mesh_select_mode),
            [False, True, False],
            "Cube cut should finish in edge select mode"
        )

        # Verify only cut boundary edges are selected
        selected_edges = [e for e in bm.edges if e.select]
        # Through-hole cut creates 2 rectangular openings (front and back),
        # each with 4 boundary edges = 8 total
        self.assertEqual(len(selected_edges), 8,
                         f"Should have 8 boundary edges selected (4 per opening), got {len(selected_edges)}")

        # All selected edges should have exactly 1 linked face (they're on the open boundary)
        for e in selected_edges:
            self.assertEqual(len(e.link_faces), 1,
                             f"Boundary edge should have exactly 1 linked face, got {len(e.link_faces)}")

        # No faces should be selected
        selected_faces = [f for f in bm.faces if f.select]
        self.assertEqual(len(selected_faces), 0, "No faces should be selected after cube cut")

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()

    def test_cube_cut_through_hole_then_bridge(self):
        """Cut a through-hole then bridge the two openings, verify bridged face UVs."""
        obj = create_textured_cube("cc_bridge", 1.0, 1.0, face_aligned=True)

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

        # Cube cut leaves 8 boundary edges selected (4 per opening).
        # Bridge edge loops connects the two openings.
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops()

        # Let depsgraph handler fire to apply UVs to the bridged faces
        yield 0.5

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv[0]
        bm.faces.ensure_lookup_table()

        # 12 original faces from cube cut + 4 bridged tunnel faces = 16
        self.assertEqual(len(bm.faces), 16,
                         f"Should have 16 faces after bridge, got {len(bm.faces)}")

        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        # The 4 bridged faces form a tunnel through the cube along Y.
        # bridge_edge_loops can produce either winding depending on internal
        # edge ordering, so each face has two acceptable UV projections.
        # Key: (nx, ny, nz, cx, cy, cz)
        # Values: list of (scale_u, scale_v, rotation, offset_x, offset_y)
        # Bridged faces get UVs from _project_new_faces (handlers/auto_hotspot.py),
        # which copies the transform from a "best neighbor" adjacent face. Two
        # sources of variation:
        #  (a) bridge_edge_loops can pick either winding, changing loop order.
        #  (b) _get_best_neighbor_face ranks neighbors and breaks ties by
        #      bmesh edge/face iteration order — not deterministic across
        #      suites, since prior tests' operations perturb internal order.
        # Before face_aligned_project was un-mirrored, (b) collapsed into two
        # equivalent forms so the test only needed two alternatives. Now the
        # V-flipped and non-flipped sources produce distinct outputs, giving
        # up to four valid forms per face.
        bridged_expected = {
            (1, 0, 0, 0.25, 0.5, 0.5):   [
                (1.0, 1.0, 90.0, 0.25, 0.25),
                (1.0, 1.0, -90.0, 0.25, 0.75),
                (1.0, 1.0, 90.0, 0.75, 0.25),
                (1.0, 1.0, -90.0, 0.75, 0.75),
            ],
            (-1, 0, 0, 0.75, 0.5, 0.5):  [
                (1.0, 1.0, -90.0, 0.75, 0.75),
                (1.0, 1.0, 90.0, 0.75, 0.25),
                (1.0, 1.0, -90.0, 0.25, 0.75),
                (1.0, 1.0, 90.0, 0.25, 0.25),
            ],
            (0, 0, 1, 0.5, 0.5, 0.25):   [
                (1.0, 1.0, 180.0, 0.75, 0.25),
                (1.0, 1.0, 0.0, 0.25, 0.25),
                (1.0, 1.0, 180.0, 0.75, 0.75),
                (1.0, 1.0, 0.0, 0.25, 0.75),
            ],
            (0, 0, -1, 0.5, 0.5, 0.75):  [
                (1.0, 1.0, 0.0, 0.25, 0.75),
                (1.0, 1.0, 180.0, 0.75, 0.75),
                (1.0, 1.0, 0.0, 0.25, 0.25),
                (1.0, 1.0, 180.0, 0.75, 0.25),
            ],
        }

        found_bridged = 0
        for face in bm.faces:
            key = _face_key(face)
            if key in bridged_expected:
                found_bridged += 1
                t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
                alternatives = bridged_expected[key]
                matched = False
                for exp_su, exp_sv, exp_rot, exp_ox, exp_oy in alternatives:
                    if (abs(t['scale_u'] - exp_su) < 0.01 and
                            abs(t['scale_v'] - exp_sv) < 0.01 and
                            abs(t['rotation'] - exp_rot) < 0.01 and
                            abs(t['offset_x'] - exp_ox) < 0.01 and
                            abs(t['offset_y'] - exp_oy) < 0.01):
                        matched = True
                        break
                self.assertTrue(
                    matched,
                    f"Bridged face {key}: got su={t['scale_u']:.2f} "
                    f"sv={t['scale_v']:.2f} rot={t['rotation']:.2f} "
                    f"ox={t['offset_x']:.2f} oy={t['offset_y']:.2f}, "
                    f"expected one of {alternatives}")

        self.assertEqual(found_bridged, 4,
                         f"Should find 4 bridged faces, found {found_bridged}")

    def test_cube_cut_edge_aligned_hole(self):
        """Cut a hole where the top of the cut aligns with the top of the cube."""
        obj = create_textured_cube("cc_cube_edge", 1.0, 1.0, face_aligned=True)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        add_uv_layer_face_aligned(obj, "UVMap.001", 0.5)

        # Select all faces so cube cut processes them
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(obj.data)

        # Cut a hole through the cube along the Y axis.
        # The cube spans (0,0,0) to (1,1,1).
        # The cut rectangle is at x=[0.25,0.75], z=[0.5,1.0],
        # extending from y=-0.5 to y=1.5 (fully through the cube).
        # The top of the cut aligns with the top of the cube.
        with bpy.context.temp_override(**ctx):
            success, msg = execute_cube_cut(
                bpy.context,
                Vector((0.25, -0.5, 0.5)),
                Vector((0.75, -0.5, 1.0)),
                2.0,
                Vector((1, 0, 0)),
                Vector((0, 0, 1)),
                Vector((0, 1, 0)),
            )

        self.assertTrue(success, msg)

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        self.assertEqual(len(bm.faces), 11,
                         "Edge-aligned cut should produce 11 faces (no degenerate strip)")

        # Verify edge select mode after cube cut
        self.assertEqual(
            list(bpy.context.tool_settings.mesh_select_mode),
            [False, True, False],
            "Cube cut should finish in edge select mode"
        )

        # No faces should be selected
        selected_faces = [f for f in bm.faces if f.select]
        self.assertEqual(len(selected_faces), 0, "No faces should be selected after cube cut")

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

    def test_cube_cut_in_concave_ngon_face_produces_expected_geometry(self):
        """Cutting a rectangle out of a concave U-shaped n-gon produces
        the expected set of vertices and edges.

        The host face is a 14x8 rectangle at X=0 with a 4x6 rectangular
        notch cut out of its bottom. The cube cut takes a 2x1 slice from
        the solid band above the notch. The bridge from the bottom-right
        cut corner must reach (10, 6) (the top-right of the notch) rather
        than (14, 0), which is the closest exterior vert by angle but is
        occluded by the notch.
        """
        # U-shape at X=0: outer rectangle Y=[0,14], Z=[0,8] with a notch
        # at Y=[6,10], Z=[0,6]. Winding gives face normal = (-1, 0, 0).
        mesh = bpy.data.meshes.new("u_face")
        bm_new = bmesh.new()
        loop = [
            (0, 6, 6), (0, 6, 0), (0, 0, 0), (0, 0, 8),
            (0, 14, 8), (0, 14, 0), (0, 10, 0), (0, 10, 6),
        ]
        verts = [bm_new.verts.new(p) for p in loop]
        bm_new.faces.new(verts)
        bm_new.to_mesh(mesh)
        bm_new.free()

        obj = bpy.data.objects.new("u_face", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        _apply_material_face_aligned(obj, 5.0)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(mesh)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(mesh)

        # Cut a 2x1 rectangle at Y=[5,7], Z=[6.5,7.5] (solid band above
        # the notch). These parameters match the user's reproduction.
        with bpy.context.temp_override(**ctx):
            success, msg = execute_cube_cut(
                bpy.context,
                Vector((0.0, 7.0, 7.5)),
                Vector((0.0, 5.0, 6.5)),
                0.0,
                Vector((0.0, -1.0, 0.0)),
                Vector((0.0, 0.0, -1.0)),
                Vector((-1.0, 0.0, 0.0)),
            )

        self.assertTrue(success, msg)

        bm = bmesh.from_edit_mesh(mesh)

        def r(v):
            return (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))

        verts_actual = sorted(r(v) for v in bm.verts if v.is_valid)
        edges_actual = sorted(
            tuple(sorted((r(e.verts[0]), r(e.verts[1]))))
            for e in bm.edges if e.is_valid
        )

        bmesh.update_edit_mesh(mesh)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

        expected_verts = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 8.0),
            (0.0, 5.0, 6.5),
            (0.0, 5.0, 7.5),
            (0.0, 6.0, 0.0),
            (0.0, 6.0, 6.0),
            (0.0, 7.0, 6.5),
            (0.0, 7.0, 7.5),
            (0.0, 10.0, 0.0),
            (0.0, 10.0, 6.0),
            (0.0, 14.0, 0.0),
            (0.0, 14.0, 8.0),
        ]
        expected_edges = [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 8.0)),
            ((0.0, 0.0, 0.0), (0.0, 5.0, 6.5)),
            ((0.0, 0.0, 0.0), (0.0, 6.0, 0.0)),
            ((0.0, 0.0, 8.0), (0.0, 5.0, 7.5)),
            ((0.0, 0.0, 8.0), (0.0, 14.0, 8.0)),
            ((0.0, 5.0, 6.5), (0.0, 5.0, 7.5)),
            ((0.0, 5.0, 6.5), (0.0, 6.0, 6.0)),
            ((0.0, 5.0, 6.5), (0.0, 7.0, 6.5)),
            ((0.0, 5.0, 7.5), (0.0, 7.0, 7.5)),
            ((0.0, 6.0, 0.0), (0.0, 6.0, 6.0)),
            ((0.0, 6.0, 6.0), (0.0, 10.0, 6.0)),
            ((0.0, 7.0, 6.5), (0.0, 7.0, 7.5)),
            ((0.0, 7.0, 6.5), (0.0, 10.0, 6.0)),
            ((0.0, 7.0, 7.5), (0.0, 14.0, 8.0)),
            ((0.0, 10.0, 0.0), (0.0, 10.0, 6.0)),
            ((0.0, 10.0, 0.0), (0.0, 14.0, 0.0)),
            ((0.0, 10.0, 6.0), (0.0, 14.0, 8.0)),
            ((0.0, 14.0, 0.0), (0.0, 14.0, 8.0)),
        ]

        self.assertEqual(verts_actual, expected_verts)
        self.assertEqual(edges_actual, expected_edges)
        obj.data.update()
