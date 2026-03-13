import bmesh
import bpy
from mathutils import Vector

from ..utils import derive_transform_from_uvs
from ..operators.cube_cut.geometry import execute_cube_cut
from ..operators.weld import set_weld_from_edge_selection
from .base_test import AnvilTestCase
from .helpers import create_vertical_plane, _get_context_override, _apply_material


def _create_sloped_plane(name):
    """Create a 1x1 plane tilted 45 deg in the YZ plane.

    Vertices: (0,0,0), (1,0,0), (1,1,1), (0,1,1)
    Normal approx: (0, -0.707, 0.707)

    Textured with dev_orange_wall.png.
    Returns the new object in object mode.
    """
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    v0 = bm.verts.new((0, 0, 0))
    v1 = bm.verts.new((1, 0, 0))
    v2 = bm.verts.new((1, 1, 1))
    v3 = bm.verts.new((0, 1, 1))
    bm.faces.new((v0, v1, v2, v3))

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    _apply_material(obj, 1.0, 1.0)

    return obj


def _face_key(face):
    """Return a hashable key from the face's normal and centroid."""
    c = face.calc_center_median()
    n = face.normal
    return (round(n.x), round(n.y), round(n.z),
            round(c.x, 2), round(c.y, 2), round(c.z, 2))


def _vert_key(v):
    """Return a hashable key from a vertex position, rounded to 4dp."""
    return (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))


def _face_verts_sorted(face):
    """Return sorted list of vertex position tuples for a face."""
    return sorted([_vert_key(v) for v in face.verts])


class CorridorWeldVerticalTest(AnvilTestCase):
    """Test corridor weld after cube cut on a vertical plane."""

    def test_corridor_on_vertical_plane(self):
        """Cube cut a hole in a textured vertical plane, then corridor weld.

        The vertical plane faces -Y. A horizontal cube cut creates a rectangular
        hole. Corridor weld fills the hole and extrudes along the filled face's
        normal (+Y). Since the cube cut direction is also +Y, the corridor
        geometry is correct.

        Geometry after corridor:
        - 4 frame faces (from cube cut) on the original plane at y=0
        - 1 inner face at y=0 closing the hole (normal +Y, faces into corridor)
        - 1 back face at y=0.5 (normal -Y, end of corridor)
        - 4 side faces connecting y=0 to y=0.5
        """
        obj = create_vertical_plane("corridor_vert")

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        # Select all faces for cube cut
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(obj.data)

        # Cube cut: hole at x=[0.25,0.75], z=[0.25,0.75], passing through y=0
        # Cuboid spans y=[-0.25, 0.25], fully enclosing the plane at y=0
        with bpy.context.temp_override(**ctx):
            success, msg = execute_cube_cut(
                bpy.context,
                Vector((0.25, -0.25, 0.25)),
                Vector((0.75, -0.25, 0.75)),
                0.5,
                Vector((1, 0, 0)),
                Vector((0, 0, 1)),
                Vector((0, 1, 0)),
            )
        self.assertTrue(success, msg)

        yield 0.1

        # Set up weld state (simulating what the cube cut operator does)
        set_weld_from_edge_selection(bpy.context, 0.5)

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'CORRIDOR',
                         "Should be CORRIDOR mode after cube cut on single plane")

        # Execute corridor weld
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        # Let depsgraph handler fire to apply UVs
        yield 0.5

        # --- Geometry assertions ---
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        self.assertEqual(len(bm.verts), 12)
        self.assertEqual(len(bm.faces), 10)

        # Verify all vertex positions
        expected_verts = sorted([
            # Original plane corners
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0), (0.0, 0.0, 1.0),
            # Hole corners (from cube cut) at y=0
            (0.25, 0.0, 0.25), (0.75, 0.0, 0.25),
            (0.75, 0.0, 0.75), (0.25, 0.0, 0.75),
            # Extruded corridor verts at y=0.5 (face normal +Y * depth 0.5)
            (0.25, 0.5, 0.25), (0.75, 0.5, 0.25),
            (0.75, 0.5, 0.75), (0.25, 0.5, 0.75),
        ])
        actual_verts = sorted([_vert_key(v) for v in bm.verts])
        self.assertEqual(actual_verts, expected_verts)

        # Verify each face's vertices
        expected_face_verts = {
            # Frame faces (original plane, normal -Y)
            (0, -1, 0, 0.5, 0.0, 0.88): sorted([
                (0.0, 0.0, 1.0), (0.25, 0.0, 0.75),
                (0.75, 0.0, 0.75), (1.0, 0.0, 1.0)]),
            (0, -1, 0, 0.88, 0.0, 0.5): sorted([
                (0.75, 0.0, 0.25), (0.75, 0.0, 0.75),
                (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)]),
            (0, -1, 0, 0.5, 0.0, 0.12): sorted([
                (0.0, 0.0, 0.0), (0.25, 0.0, 0.25),
                (0.75, 0.0, 0.25), (1.0, 0.0, 0.0)]),
            (0, -1, 0, 0.12, 0.0, 0.5): sorted([
                (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                (0.25, 0.0, 0.25), (0.25, 0.0, 0.75)]),
            # Inner face (closing the hole at y=0, normal +Y)
            (0, 1, 0, 0.5, 0.0, 0.5): sorted([
                (0.25, 0.0, 0.25), (0.25, 0.0, 0.75),
                (0.75, 0.0, 0.25), (0.75, 0.0, 0.75)]),
            # Back face (end of corridor at y=0.5, normal -Y)
            (0, -1, 0, 0.5, 0.5, 0.5): sorted([
                (0.25, 0.5, 0.25), (0.25, 0.5, 0.75),
                (0.75, 0.5, 0.25), (0.75, 0.5, 0.75)]),
            # Side faces (connecting y=0 to y=0.5)
            (-1, 0, 0, 0.75, 0.25, 0.5): sorted([
                (0.75, 0.0, 0.25), (0.75, 0.0, 0.75),
                (0.75, 0.5, 0.25), (0.75, 0.5, 0.75)]),
            (0, 0, 1, 0.5, 0.25, 0.25): sorted([
                (0.25, 0.0, 0.25), (0.25, 0.5, 0.25),
                (0.75, 0.0, 0.25), (0.75, 0.5, 0.25)]),
            (0, 0, -1, 0.5, 0.25, 0.75): sorted([
                (0.25, 0.0, 0.75), (0.25, 0.5, 0.75),
                (0.75, 0.0, 0.75), (0.75, 0.5, 0.75)]),
            (1, 0, 0, 0.25, 0.25, 0.5): sorted([
                (0.25, 0.0, 0.25), (0.25, 0.0, 0.75),
                (0.25, 0.5, 0.25), (0.25, 0.5, 0.75)]),
        }

        for face in bm.faces:
            key = _face_key(face)
            self.assertIn(key, expected_face_verts,
                          f"Unexpected face key {key}")
            self.assertEqual(_face_verts_sorted(face), expected_face_verts[key],
                             f"Face {key} has wrong vertices")

        # --- UV assertions ---
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        uv_layer = bm.loops.layers.uv[0]

        # Expected UV transforms per face.
        # Key: face_key → (rotation, offset_x, offset_y)
        # All faces have scale_u=1.0, scale_v=1.0.
        expected_uvs = {
            # Frame faces
            (0, -1, 0, 0.5, 0.0, 0.88):  (0.0, 0.25, 0.75),
            (0, -1, 0, 0.88, 0.0, 0.5):  (-90.0, 0.75, 0.75),
            (0, -1, 0, 0.5, 0.0, 0.12):  (180.0, 0.75, 0.25),
            (0, -1, 0, 0.12, 0.0, 0.5):  (90.0, 0.25, 0.25),
            # Inner face
            (0, 1, 0, 0.5, 0.0, 0.5):    (180.0, 0.25, 0.75),
            # Back face
            (0, -1, 0, 0.5, 0.5, 0.5):   (0.0, 0.0, 0.0),
            # Side faces
            (-1, 0, 0, 0.75, 0.25, 0.5): (90.0, 0.75, 0.25),
            (0, 0, 1, 0.5, 0.25, 0.25):  (0.0, 0.25, 0.25),
            (0, 0, -1, 0.5, 0.25, 0.75): (180.0, 0.75, 0.75),
            (1, 0, 0, 0.25, 0.25, 0.5):  (-90.0, 0.25, 0.75),
        }

        for face in bm.faces:
            key = _face_key(face)
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertIsNotNone(t, f"Face {key} has no UV transform")

            self.assertAlmostEqual(t['scale_u'], 1.0, places=2,
                                   msg=f"Face {key} scale_u={t['scale_u']}")
            self.assertAlmostEqual(t['scale_v'], 1.0, places=2,
                                   msg=f"Face {key} scale_v={t['scale_v']}")

            rot, ox, oy = expected_uvs[key]
            self.assertAlmostEqual(t['rotation'], rot, places=2,
                                   msg=f"Face {key} rotation={t['rotation']}")
            self.assertAlmostEqual(t['offset_x'], ox, places=2,
                                   msg=f"Face {key} offset_x={t['offset_x']}")
            self.assertAlmostEqual(t['offset_y'], oy, places=2,
                                   msg=f"Face {key} offset_y={t['offset_y']}")

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()


class CorridorWeldSlopedTest(AnvilTestCase):
    """Test corridor weld after cube cut on a sloped plane.

    Demonstrates the current behavior where the corridor extrudes along
    the face normal (perpendicular to the slope) rather than along the
    cube cut direction (horizontal). This causes:
    - The corridor to extend diagonally instead of horizontally
    - The back face to be sloped (parallel to the front face) instead of
      vertical (perpendicular to the cube cut direction)
    """

    def test_corridor_on_sloped_plane(self):
        """Cube cut a hole in a textured 45 deg sloped plane, then corridor weld.

        The sloped plane goes from (0,0,0)-(1,0,0) at the bottom to
        (0,1,1)-(1,1,1) at the top (45 deg tilt in YZ plane).

        A horizontal cube cut (local_z along +Y) creates a hole where
        the cuboid intersects the slope. Corridor weld fills the hole
        and extrudes along the face normal.

        Current behavior (bug): extrusion follows face normal (0, 0.707, -0.707),
        producing a corridor that goes diagonally. The back face is parallel to
        the sloped front face.

        Expected behavior: extrusion should follow the cube cut direction (+Y),
        producing a horizontal corridor. The back face should be vertical.
        """
        obj = _create_sloped_plane("corridor_slope")

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        # Select all faces for cube cut
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(obj.data)

        # Cube cut: horizontal cut through the sloped plane
        # The sloped plane satisfies z=y on the surface.
        # Cuboid: x=[0.25,0.75], z=[0.25,0.75], y=[0.0, 1.0]
        with bpy.context.temp_override(**ctx):
            success, msg = execute_cube_cut(
                bpy.context,
                Vector((0.25, 0.0, 0.25)),
                Vector((0.75, 0.0, 0.75)),
                1.0,
                Vector((1, 0, 0)),
                Vector((0, 0, 1)),
                Vector((0, 1, 0)),
            )
        self.assertTrue(success, msg)

        yield 0.1

        # Set up weld state
        set_weld_from_edge_selection(bpy.context, 1.0)

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'CORRIDOR',
                         "Should be CORRIDOR mode after cube cut on sloped plane")

        # Execute corridor weld
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        # Let depsgraph handler fire to apply UVs
        yield 0.5

        # --- Geometry assertions ---
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        self.assertEqual(len(bm.verts), 12)
        self.assertEqual(len(bm.faces), 10)

        # d = 1/sqrt(2) ≈ 0.7071: the face normal component magnitude.
        # Extrusion displacement = face_normal * depth
        #   = (0, 0.7071, -0.7071) * 1.0
        # So extruded verts are offset by (0, +0.7071, -0.7071) from hole corners.
        #
        # BUG: these extruded positions are wrong. Correct behavior would
        # extrude along +Y (cube cut direction), giving back-face verts at
        # constant z values (not sloped).
        expected_verts = sorted([
            # Original plane corners
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
            # Hole corners on slope (where cuboid intersects plane z=y)
            (0.25, 0.25, 0.25), (0.75, 0.25, 0.25),
            (0.75, 0.75, 0.75), (0.25, 0.75, 0.75),
            # Extruded verts (current buggy behavior: along face normal)
            # (0.25, 0.25+0.7071, 0.25-0.7071):
            (0.25, 0.9571, -0.4571),
            # (0.75, 0.25+0.7071, 0.25-0.7071):
            (0.75, 0.9571, -0.4571),
            # (0.75, 0.75+0.7071, 0.75-0.7071):
            (0.75, 1.4571, 0.0429),
            # (0.25, 0.75+0.7071, 0.75-0.7071):
            (0.25, 1.4571, 0.0429),
        ])
        actual_verts = sorted([_vert_key(v) for v in bm.verts])
        self.assertEqual(actual_verts, expected_verts)

        # Verify each face's vertices
        expected_face_verts = {
            # Frame faces (original sloped plane, normal ≈ (0, -0.71, 0.71))
            (0, -1, 1, 0.5, 0.88, 0.88): sorted([
                (0.0, 1.0, 1.0), (0.25, 0.75, 0.75),
                (0.75, 0.75, 0.75), (1.0, 1.0, 1.0)]),
            (0, -1, 1, 0.88, 0.5, 0.5): sorted([
                (0.75, 0.25, 0.25), (0.75, 0.75, 0.75),
                (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)]),
            (0, -1, 1, 0.5, 0.12, 0.12): sorted([
                (0.0, 0.0, 0.0), (0.25, 0.25, 0.25),
                (0.75, 0.25, 0.25), (1.0, 0.0, 0.0)]),
            (0, -1, 1, 0.12, 0.5, 0.5): sorted([
                (0.0, 0.0, 0.0), (0.0, 1.0, 1.0),
                (0.25, 0.25, 0.25), (0.25, 0.75, 0.75)]),
            # Inner face (closing hole on slope, normal ≈ (0, 0.71, -0.71))
            (0, 1, -1, 0.5, 0.5, 0.5): sorted([
                (0.25, 0.25, 0.25), (0.25, 0.75, 0.75),
                (0.75, 0.25, 0.25), (0.75, 0.75, 0.75)]),
            # Back face (end of corridor, normal ≈ (0, -0.71, 0.71))
            # BUG: this face is sloped (parallel to the front face).
            # Correct behavior: back face should be vertical with all
            # verts at a constant y, and z matching the hole z values.
            (0, -1, 1, 0.5, 1.21, -0.21): sorted([
                (0.25, 0.9571, -0.4571), (0.25, 1.4571, 0.0429),
                (0.75, 0.9571, -0.4571), (0.75, 1.4571, 0.0429)]),
            # Side faces
            (-1, 0, 0, 0.75, 0.85, 0.15): sorted([
                (0.75, 0.25, 0.25), (0.75, 0.75, 0.75),
                (0.75, 0.9571, -0.4571), (0.75, 1.4571, 0.0429)]),
            (0, 1, 1, 0.5, 0.6, -0.1): sorted([
                (0.25, 0.25, 0.25), (0.25, 0.9571, -0.4571),
                (0.75, 0.25, 0.25), (0.75, 0.9571, -0.4571)]),
            (0, -1, -1, 0.5, 1.1, 0.4): sorted([
                (0.25, 0.75, 0.75), (0.25, 1.4571, 0.0429),
                (0.75, 0.75, 0.75), (0.75, 1.4571, 0.0429)]),
            (1, 0, 0, 0.25, 0.85, 0.15): sorted([
                (0.25, 0.25, 0.25), (0.25, 0.75, 0.75),
                (0.25, 0.9571, -0.4571), (0.25, 1.4571, 0.0429)]),
        }

        for face in bm.faces:
            key = _face_key(face)
            self.assertIn(key, expected_face_verts,
                          f"Unexpected face key {key}")
            self.assertEqual(_face_verts_sorted(face), expected_face_verts[key],
                             f"Face {key} has wrong vertices")

        # --- UV assertions ---
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        uv_layer = bm.loops.layers.uv[0]

        # Expected UV transforms per face.
        # Key: face_key → (rotation, offset_x, offset_y)
        # All faces have scale_u=1.0, scale_v=1.0.
        expected_uvs = {
            # Frame faces
            (0, -1, 1, 0.5, 0.88, 0.88):  (0.0, 0.25, 0.06),
            (0, -1, 1, 0.88, 0.5, 0.5):   (-90.0, 0.75, 0.06),
            (0, -1, 1, 0.5, 0.12, 0.12):  (180.0, 0.75, 0.35),
            (0, -1, 1, 0.12, 0.5, 0.5):   (90.0, 0.25, 0.35),
            # Inner face
            (0, 1, -1, 0.5, 0.5, 0.5):    (180.0, 0.25, 0.06),
            # Back face
            (0, -1, 1, 0.5, 1.21, -0.21): (0.0, 0.0, 0.0),
            # Side faces
            (-1, 0, 0, 0.75, 0.85, 0.15): (90.0, 0.75, 0.35),
            (0, 1, 1, 0.5, 0.6, -0.1):    (0.0, 0.25, 0.35),
            (0, -1, -1, 0.5, 1.1, 0.4):   (180.0, 0.75, 0.06),
            (1, 0, 0, 0.25, 0.85, 0.15):  (-90.0, 0.25, 0.06),
        }

        for face in bm.faces:
            key = _face_key(face)
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertIsNotNone(t, f"Face {key} has no UV transform")

            self.assertAlmostEqual(t['scale_u'], 1.0, places=2,
                                   msg=f"Face {key} scale_u={t['scale_u']}")
            self.assertAlmostEqual(t['scale_v'], 1.0, places=2,
                                   msg=f"Face {key} scale_v={t['scale_v']}")

            rot, ox, oy = expected_uvs[key]
            self.assertAlmostEqual(t['rotation'], rot, places=2,
                                   msg=f"Face {key} rotation={t['rotation']}")
            self.assertAlmostEqual(t['offset_x'], ox, places=2,
                                   msg=f"Face {key} offset_x={t['offset_x']}")
            self.assertAlmostEqual(t['offset_y'], oy, places=2,
                                   msg=f"Face {key} offset_y={t['offset_y']}")

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()
