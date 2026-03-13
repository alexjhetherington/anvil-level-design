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

        The vertical plane faces -Y (winding order gives -Y normal).
        A horizontal cube cut creates a rectangular hole. Corridor weld
        fills the hole and extrudes inward (+Y, negated face normal).

        Geometry after corridor (9 faces, 12 verts):
        - 4 frame faces (from cube cut) on the original plane at y=0
        - 1 back face at y=0.5 (end of corridor)
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
        set_weld_from_edge_selection(bpy.context, 0.5, (0, 1, 0))

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
        self.assertEqual(len(bm.faces), 9)

        # Verify all vertex positions
        expected_verts = sorted([
            # Original plane corners
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0), (0.0, 0.0, 1.0),
            # Hole corners (from cube cut) at y=0
            (0.25, 0.0, 0.25), (0.75, 0.0, 0.25),
            (0.75, 0.0, 0.75), (0.25, 0.0, 0.75),
            # Extruded corridor verts at y=+0.5 (negated -Y normal * depth 0.5)
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
            # Back face (end of corridor at y=0.5)
            (0, -1, 0, 0.5, 0.5, 0.5): sorted([
                (0.25, 0.5, 0.25), (0.25, 0.5, 0.75),
                (0.75, 0.5, 0.25), (0.75, 0.5, 0.75)]),
            # Side faces (connecting y=0 to y=0.5)
            (0, 0, 1, 0.5, 0.25, 0.25): sorted([
                (0.25, 0.0, 0.25), (0.25, 0.5, 0.25),
                (0.75, 0.0, 0.25), (0.75, 0.5, 0.25)]),
            (0, 0, -1, 0.5, 0.25, 0.75): sorted([
                (0.25, 0.0, 0.75), (0.25, 0.5, 0.75),
                (0.75, 0.0, 0.75), (0.75, 0.5, 0.75)]),
            (1, 0, 0, 0.25, 0.25, 0.5): sorted([
                (0.25, 0.0, 0.25), (0.25, 0.0, 0.75),
                (0.25, 0.5, 0.25), (0.25, 0.5, 0.75)]),
            (-1, 0, 0, 0.75, 0.25, 0.5): sorted([
                (0.75, 0.0, 0.25), (0.75, 0.0, 0.75),
                (0.75, 0.5, 0.25), (0.75, 0.5, 0.75)]),
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

        expected_uvs = {
            # Frame faces
            (0, -1, 0, 0.5, 0.0, 0.88):  (0.0, 0.25, 0.75),
            (0, -1, 0, 0.88, 0.0, 0.5):  (-90.0, 0.75, 0.75),
            (0, -1, 0, 0.5, 0.0, 0.12):  (180.0, 0.75, 0.25),
            (0, -1, 0, 0.12, 0.0, 0.5):  (90.0, 0.25, 0.25),
            # Back face (UV projected from adjacent frame face)
            (0, -1, 0, 0.5, 0.5, 0.5):   (-90.0, 0.25, 0.75),
            # Side faces
            (0, 0, 1, 0.5, 0.25, 0.25):  (0.0, 0.25, 0.25),
            (0, 0, -1, 0.5, 0.25, 0.75): (180.0, 0.75, 0.75),
            (1, 0, 0, 0.25, 0.25, 0.5):  (-90.0, 0.25, 0.75),
            (-1, 0, 0, 0.75, 0.25, 0.5): (90.0, 0.75, 0.25),
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

    The corridor extrudes along the cube cut direction (local_z), not
    the face normal.  For a sloped plane this produces a horizontal
    corridor even though the face normal is diagonal.
    """

    def test_corridor_on_sloped_plane(self):
        """Cube cut a hole in a textured 45 deg sloped plane, then corridor weld.

        The sloped plane goes from (0,0,0)-(1,0,0) at the bottom to
        (0,1,1)-(1,1,1) at the top (45 deg tilt in YZ plane).

        A horizontal cube cut (local_z along +Y) creates a hole where
        the cuboid intersects the slope. Corridor weld fills the hole
        and extrudes along the cube cut direction (+Y), producing a
        horizontal corridor.  The back face is vertical.
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
        set_weld_from_edge_selection(bpy.context, 1.0, (0, 1, 0))

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
        self.assertEqual(len(bm.faces), 9)

        # Extrusion goes along +Y (cube cut direction) * depth 1.0
        # Hole corners offset by (0, +1, 0):
        #   (0.25, 0.25, 0.25) → (0.25, 1.25, 0.25)
        #   (0.25, 0.75, 0.75) → (0.25, 1.75, 0.75)
        expected_verts = sorted([
            # Original plane corners
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
            # Hole corners on slope (where cuboid intersects plane z=y)
            (0.25, 0.25, 0.25), (0.75, 0.25, 0.25),
            (0.75, 0.75, 0.75), (0.25, 0.75, 0.75),
            # Extruded verts (along +Y, horizontal corridor)
            (0.25, 1.25, 0.25),
            (0.75, 1.25, 0.25),
            (0.75, 1.75, 0.75),
            (0.25, 1.75, 0.75),
        ])
        actual_verts = sorted([_vert_key(v) for v in bm.verts])
        self.assertEqual(actual_verts, expected_verts)

        # Verify each face's vertices
        expected_face_verts = {
            # Frame faces (original sloped plane, normal ~(0, -0.71, 0.71))
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
            # Back face (end of corridor at y+1.0, same normal as sloped front)
            (0, -1, 1, 0.5, 1.5, 0.5): sorted([
                (0.25, 1.25, 0.25), (0.25, 1.75, 0.75),
                (0.75, 1.25, 0.25), (0.75, 1.75, 0.75)]),
            # Side faces (connecting hole to extruded verts along +Y)
            # Bottom (z=0.25 plane, normal +Z facing into corridor)
            (0, 0, 1, 0.5, 0.75, 0.25): sorted([
                (0.25, 0.25, 0.25), (0.25, 1.25, 0.25),
                (0.75, 0.25, 0.25), (0.75, 1.25, 0.25)]),
            # Top (z=0.75 plane, normal -Z facing into corridor)
            (0, 0, -1, 0.5, 1.25, 0.75): sorted([
                (0.25, 0.75, 0.75), (0.25, 1.75, 0.75),
                (0.75, 0.75, 0.75), (0.75, 1.75, 0.75)]),
            # Left (x=0.25 plane)
            (1, 0, 0, 0.25, 1.0, 0.5): sorted([
                (0.25, 0.25, 0.25), (0.25, 0.75, 0.75),
                (0.25, 1.25, 0.25), (0.25, 1.75, 0.75)]),
            # Right (x=0.75 plane)
            (-1, 0, 0, 0.75, 1.0, 0.5): sorted([
                (0.75, 0.25, 0.25), (0.75, 0.75, 0.75),
                (0.75, 1.25, 0.25), (0.75, 1.75, 0.75)]),
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

        # Verify all faces have valid UV transforms at scale 1
        for face in bm.faces:
            key = _face_key(face)
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertIsNotNone(t, f"Face {key} has no UV transform")
            self.assertAlmostEqual(t['scale_u'], 1.0, places=2,
                                   msg=f"Face {key} scale_u={t['scale_u']}")
            self.assertAlmostEqual(t['scale_v'], 1.0, places=2,
                                   msg=f"Face {key} scale_v={t['scale_v']}")

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()
