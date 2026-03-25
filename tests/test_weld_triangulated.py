import bmesh
import bpy
from mathutils import Vector

from ..utils import derive_transform_from_uvs
from ..operators.cube_cut.geometry import execute_cube_cut
from ..operators.weld import set_weld_from_edge_selection
from .base_test import AnvilTestCase
from .helpers import create_textured_cube, _get_context_override


# Cube cut parameters used by both test classes
_FIRST_VERTEX = Vector((0.25, -0.5, 0.5))
_SECOND_VERTEX = Vector((0.75, -0.5, 3.5))
_DEPTH = 2.0
_LOCAL_X = Vector((1, 0, 0))
_LOCAL_Y = Vector((0, 0, 1))
_LOCAL_Z = Vector((0, 1, 0))


def _setup_subdivided_triangulated_cube(test_case, name):
    """Create a subdivided cube with triangulated top and raised centre.

    Returns (obj, ctx) with the object in edit mode, all faces selected.
    """
    obj = create_textured_cube(name, 1.0, 1.0, face_aligned=True)
    ctx = _get_context_override()

    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')

    # Subdivide the entire cube
    bm = bmesh.from_edit_mesh(obj.data)
    for f in bm.faces:
        f.select = True
    bmesh.update_edit_mesh(obj.data)

    with bpy.context.temp_override(**ctx):
        bpy.ops.mesh.subdivide()

    # Triangulate the top 4 faces with diagonals pointing to the centre
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    centre_vert = None
    for v in bm.verts:
        if (abs(v.co.x - 0.5) < 1e-4 and
                abs(v.co.y - 0.5) < 1e-4 and
                abs(v.co.z - 1.0) < 1e-4):
            centre_vert = v
            break
    test_case.assertIsNotNone(centre_vert,
                              "Should find centre vertex at (0.5, 0.5, 1.0)")

    top_faces = [f for f in bm.faces if f.normal.z > 0.9]
    test_case.assertEqual(len(top_faces), 4,
                          "Should have 4 top faces after subdivide")

    for face in top_faces:
        verts = list(face.verts)
        centre_idx = verts.index(centre_vert)
        opposite = verts[(centre_idx + 2) % 4]
        bmesh.utils.face_split(face, centre_vert, opposite)

    bmesh.update_edit_mesh(obj.data)

    # Raise the centre vertex 1 unit
    bm.select_mode = {'VERT'}
    for v in bm.verts:
        v.select = False
    centre_vert.select = True
    bmesh.update_edit_mesh(obj.data)

    with bpy.context.temp_override(**ctx):
        bpy.ops.transform.translate(value=(0, 0, 1.0))

    # Verify raise
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    raised = None
    for v in bm.verts:
        if (abs(v.co.x - 0.5) < 1e-4 and
                abs(v.co.y - 0.5) < 1e-4 and
                abs(v.co.z - 2.0) < 1e-4):
            raised = v
            break
    test_case.assertIsNotNone(raised,
                              "Centre vertex should be at (0.5, 0.5, 2.0)")

    # Select all faces for cube cut
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

    # Replicate what the cube cut operator does
    extrude_dir = -_LOCAL_Z  # (0, -1, 0)
    back_point = _FIRST_VERTEX + _LOCAL_Z * _DEPTH
    back_plane_offset = back_point.dot(extrude_dir.normalized())

    set_weld_from_edge_selection(
        bpy.context, abs(_DEPTH), extrude_dir, back_plane_offset,
        _FIRST_VERTEX, _SECOND_VERTEX, _LOCAL_X, _LOCAL_Y,
        0,
    )

    return success


def _vert_key(v):
    """Return a hashable key from a vertex position, rounded to 4dp."""
    return (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))


def _face_key(face):
    """Return a hashable key from the face's normal and centroid."""
    c = face.calc_center_median()
    n = face.normal
    return (round(n.x), round(n.y), round(n.z),
            round(c.x, 2), round(c.y, 2), round(c.z, 2))


def _face_verts_sorted(face):
    """Return sorted list of vertex position tuples for a face."""
    return sorted([_vert_key(v) for v in face.verts])


class FoldedPlaneWeldTest(AnvilTestCase):
    """Test folded plane weld after cube cut on a subdivided triangulated cube."""

    def test_folded_plane_weld_on_subdivided_triangulated_cube_with_raised_centre(self):
        """Subdivide a cube, triangulate the top 4 faces, raise the centre vertex,
        cube cut, then execute folded plane weld.

        The cube cut creates a non-coplanar boundary edge loop that wraps
        around the triangulated top. The folded plane weld fills side faces
        of the cuboid with new geometry: 2 side faces (left/right at x=0.25
        and x=0.75) triangulated from 7-vertex polygons, and a bottom face
        (z=0.5) split into quads from a 6-vertex hexagon.
        """
        obj, ctx = _setup_subdivided_triangulated_cube(self, "weld_folded")

        _execute_cube_cut_and_set_weld(self, ctx)

        yield 0.1

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'FOLDED_PLANE',
                         "Should be FOLDED_PLANE after cube cut on non-planar surface")

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

        self.assertEqual(len(bm.verts), 37)
        self.assertEqual(len(bm.faces), 39)

        # Verify all vertex positions
        expected_verts = sorted([
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0),
            (0.0, 0.5, 0.0), (0.0, 0.5, 0.5), (0.0, 0.5, 1.0),
            (0.0, 1.0, 0.0), (0.0, 1.0, 0.5), (0.0, 1.0, 1.0),
            (0.25, 0.0, 0.5), (0.25, 0.0, 1.0),
            (0.25, 0.25, 1.5), (0.25, 0.5, 1.5), (0.25, 0.75, 1.5),
            (0.25, 1.0, 0.5), (0.25, 1.0, 1.0),
            (0.5, 0.0, 0.0), (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 1.0, 0.0), (0.5, 1.0, 0.5),
            (0.75, 0.0, 0.5), (0.75, 0.0, 1.0),
            (0.75, 0.25, 1.5), (0.75, 0.5, 1.5), (0.75, 0.75, 1.5),
            (0.75, 1.0, 0.5), (0.75, 1.0, 1.0),
            (1.0, 0.0, 0.0), (1.0, 0.0, 0.5), (1.0, 0.0, 1.0),
            (1.0, 0.5, 0.0), (1.0, 0.5, 0.5), (1.0, 0.5, 1.0),
            (1.0, 1.0, 0.0), (1.0, 1.0, 0.5), (1.0, 1.0, 1.0),
        ])
        actual_verts = sorted([_vert_key(v) for v in bm.verts])
        self.assertEqual(actual_verts, expected_verts)

        # Verify each face by its key (normal + centroid) and vertices
        expected_face_verts = {
            # Original cube faces (bottom z=0)
            (0, 0, -1, 0.25, 0.25, 0.0): sorted([(0.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0)]),
            (0, 0, -1, 0.25, 0.75, 0.0): sorted([(0.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (0.5, 1.0, 0.0)]),
            (0, 0, -1, 0.75, 0.25, 0.0): sorted([(0.5, 0.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0.0, 0.0), (1.0, 0.5, 0.0)]),
            (0, 0, -1, 0.75, 0.75, 0.0): sorted([(0.5, 0.5, 0.0), (0.5, 1.0, 0.0), (1.0, 0.5, 0.0), (1.0, 1.0, 0.0)]),
            # Left face (x=0)
            (-1, 0, 0, 0.0, 0.25, 0.25): sorted([(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.5, 0.0), (0.0, 0.5, 0.5)]),
            (-1, 0, 0, 0.0, 0.25, 0.75): sorted([(0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.0, 0.5, 0.5), (0.0, 0.5, 1.0)]),
            (-1, 0, 0, 0.0, 0.75, 0.25): sorted([(0.0, 0.5, 0.0), (0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.0, 1.0, 0.5)]),
            (-1, 0, 0, 0.0, 0.75, 0.75): sorted([(0.0, 0.5, 0.5), (0.0, 0.5, 1.0), (0.0, 1.0, 0.5), (0.0, 1.0, 1.0)]),
            # Right face (x=1)
            (1, 0, 0, 1.0, 0.25, 0.25): sorted([(1.0, 0.0, 0.0), (1.0, 0.0, 0.5), (1.0, 0.5, 0.0), (1.0, 0.5, 0.5)]),
            (1, 0, 0, 1.0, 0.25, 0.75): sorted([(1.0, 0.0, 0.5), (1.0, 0.0, 1.0), (1.0, 0.5, 0.5), (1.0, 0.5, 1.0)]),
            (1, 0, 0, 1.0, 0.75, 0.25): sorted([(1.0, 0.5, 0.0), (1.0, 0.5, 0.5), (1.0, 1.0, 0.0), (1.0, 1.0, 0.5)]),
            (1, 0, 0, 1.0, 0.75, 0.75): sorted([(1.0, 0.5, 0.5), (1.0, 0.5, 1.0), (1.0, 1.0, 0.5), (1.0, 1.0, 1.0)]),
            # Front face (y=0) - remaining faces after cube cut
            (0, -1, 0, 0.19, 0.0, 0.25): sorted([(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.25, 0.0, 0.5), (0.5, 0.0, 0.0)]),
            (0, -1, 0, 0.42, 0.0, 0.33): sorted([(0.25, 0.0, 0.5), (0.5, 0.0, 0.0), (0.5, 0.0, 0.5)]),
            (0, -1, 0, 0.12, 0.0, 0.75): sorted([(0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.25, 0.0, 0.5), (0.25, 0.0, 1.0)]),
            (0, -1, 0, 0.17, 0.08, 1.17): sorted([(0.0, 0.0, 1.0), (0.25, 0.0, 1.0), (0.25, 0.25, 1.5)]),
            (0, -1, 0, 0.69, 0.0, 0.25): sorted([(0.5, 0.0, 0.0), (0.5, 0.0, 0.5), (0.75, 0.0, 0.5), (1.0, 0.0, 0.0)]),
            (0, -1, 0, 0.92, 0.0, 0.33): sorted([(0.75, 0.0, 0.5), (1.0, 0.0, 0.0), (1.0, 0.0, 0.5)]),
            (0, -1, 0, 0.88, 0.0, 0.75): sorted([(0.75, 0.0, 0.5), (0.75, 0.0, 1.0), (1.0, 0.0, 0.5), (1.0, 0.0, 1.0)]),
            (0, -1, 0, 0.83, 0.08, 1.17): sorted([(0.75, 0.0, 1.0), (0.75, 0.25, 1.5), (1.0, 0.0, 1.0)]),
            # Back face (y=1)
            (0, 1, 0, 0.19, 1.0, 0.25): sorted([(0.0, 1.0, 0.0), (0.0, 1.0, 0.5), (0.25, 1.0, 0.5), (0.5, 1.0, 0.0)]),
            (0, 1, 0, 0.12, 1.0, 0.75): sorted([(0.0, 1.0, 0.5), (0.0, 1.0, 1.0), (0.25, 1.0, 0.5), (0.25, 1.0, 1.0)]),
            (0, 1, 0, 0.17, 0.92, 1.17): sorted([(0.0, 1.0, 1.0), (0.25, 0.75, 1.5), (0.25, 1.0, 1.0)]),
            (0, 1, 0, 0.5, 1.0, 0.38): sorted([(0.25, 1.0, 0.5), (0.5, 1.0, 0.0), (0.5, 1.0, 0.5), (0.75, 1.0, 0.5)]),
            (0, 1, 0, 0.81, 1.0, 0.25): sorted([(0.5, 1.0, 0.0), (0.75, 1.0, 0.5), (1.0, 1.0, 0.0), (1.0, 1.0, 0.5)]),
            (0, 1, 0, 0.88, 1.0, 0.75): sorted([(0.75, 1.0, 0.5), (0.75, 1.0, 1.0), (1.0, 1.0, 0.5), (1.0, 1.0, 1.0)]),
            (0, 1, 0, 0.83, 0.92, 1.17): sorted([(0.75, 0.75, 1.5), (0.75, 1.0, 1.0), (1.0, 1.0, 1.0)]),
            # Triangulated top remnants (diagonal faces between original cube sides and cut)
            (-1, 0, 0, 0.12, 0.31, 1.25): sorted([(0.0, 0.0, 1.0), (0.0, 0.5, 1.0), (0.25, 0.25, 1.5), (0.25, 0.5, 1.5)]),
            (-1, 0, 0, 0.12, 0.69, 1.25): sorted([(0.0, 0.5, 1.0), (0.0, 1.0, 1.0), (0.25, 0.5, 1.5), (0.25, 0.75, 1.5)]),
            (1, 0, 0, 0.88, 0.31, 1.25): sorted([(0.75, 0.25, 1.5), (0.75, 0.5, 1.5), (1.0, 0.0, 1.0), (1.0, 0.5, 1.0)]),
            (1, 0, 0, 0.88, 0.69, 1.25): sorted([(0.75, 0.5, 1.5), (0.75, 0.75, 1.5), (1.0, 0.5, 1.0), (1.0, 1.0, 1.0)]),
        }

        # Check non-weld faces exactly
        for face in bm.faces:
            key = _face_key(face)
            if key in expected_face_verts:
                self.assertEqual(_face_verts_sorted(face), expected_face_verts[key],
                                 f"Face {key} has wrong vertices")

        # Check weld faces by vertex coverage (triangulation is non-deterministic)
        # Collinear intermediate verts (*.5, 1.5) are removed from the weld
        # polygon to avoid degenerate triangles during triangulation.
        left_weld_verts = {(0.25, 0.0, 0.5), (0.25, 0.0, 1.0), (0.25, 0.25, 1.5),
                           (0.25, 0.75, 1.5),
                           (0.25, 1.0, 0.5), (0.25, 1.0, 1.0)}
        right_weld_verts = {(0.75, 0.0, 0.5), (0.75, 0.0, 1.0), (0.75, 0.25, 1.5),
                            (0.75, 0.75, 1.5),
                            (0.75, 1.0, 0.5), (0.75, 1.0, 1.0)}
        bottom_weld_verts = {(0.25, 0.0, 0.5), (0.5, 0.0, 0.5), (0.75, 0.0, 0.5),
                             (0.75, 1.0, 0.5), (0.5, 1.0, 0.5), (0.25, 1.0, 0.5)}

        # Collect vertices from weld faces (faces at x=0.25 with +x normal,
        # x=0.75 with -x normal, z=0.5 with +z normal)
        actual_left = set()
        actual_right = set()
        actual_bottom = set()
        for face in bm.faces:
            key = _face_key(face)
            if key in expected_face_verts:
                continue  # Skip non-weld faces
            fv = set(_vert_key(v) for v in face.verts)
            n = face.normal
            if abs(n.x - 1.0) < 0.1 and all(abs(v[0] - 0.25) < 0.01 for v in fv):
                actual_left |= fv
            elif abs(n.x + 1.0) < 0.1 and all(abs(v[0] - 0.75) < 0.01 for v in fv):
                actual_right |= fv
            elif abs(n.z - 1.0) < 0.1 and all(abs(v[2] - 0.5) < 0.01 for v in fv):
                actual_bottom |= fv

        self.assertEqual(actual_left, left_weld_verts,
                         "Left weld faces should cover all 6 non-collinear boundary vertices")
        self.assertEqual(actual_right, right_weld_verts,
                         "Right weld faces should cover all 6 non-collinear boundary vertices")
        self.assertEqual(actual_bottom, bottom_weld_verts,
                         "Bottom weld faces should cover all 6 boundary vertices")

        # --- UV assertions ---
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        uv_layer = bm.loops.layers.uv.active

        expected_uvs = {
            # Original cube faces (bottom z=0)
            (0, 0, -1, 0.25, 0.25, 0.0): (0.0, 0.0, 0.5),
            (0, 0, -1, 0.25, 0.75, 0.0): (0.0, 0.0, 0.0),
            (0, 0, -1, 0.75, 0.25, 0.0): (0.0, 0.5, 0.5),
            (0, 0, -1, 0.75, 0.75, 0.0): (0.0, 0.5, 0.0),
            # Left face (x=0)
            (-1, 0, 0, 0.0, 0.25, 0.25): (90.0, 0.0, 0.0),
            (-1, 0, 0, 0.0, 0.25, 0.75): (90.0, 0.0, 0.5),
            (-1, 0, 0, 0.0, 0.75, 0.25): (90.0, 0.5, 0.0),
            (-1, 0, 0, 0.0, 0.75, 0.75): (90.0, 0.5, 0.5),
            # Right face (x=1)
            (1, 0, 0, 1.0, 0.25, 0.25): (90.0, 0.5, 0.0),
            (1, 0, 0, 1.0, 0.25, 0.75): (90.0, 0.5, 0.5),
            (1, 0, 0, 1.0, 0.75, 0.25): (90.0, 0.0, 0.0),
            (1, 0, 0, 1.0, 0.75, 0.75): (90.0, 0.0, 0.5),
            # Front face (y=0)
            (0, -1, 0, 0.19, 0.0, 0.25): (180.0, 0.25, 0.5),
            (0, -1, 0, 0.42, 0.0, 0.33): (90.0, 0.5, 0.0),
            (0, -1, 0, 0.12, 0.0, 0.75): (-90.0, 0.0, 0.0),
            (0, -1, 0, 0.17, 0.08, 1.17): (0.0, 0.0, 0.0),
            (0, -1, 0, 0.69, 0.0, 0.25): (180.0, 0.75, 0.5),
            (0, -1, 0, 0.92, 0.0, 0.33): (90.0, 0.0, 0.0),
            (0, -1, 0, 0.88, 0.0, 0.75): (-90.0, 0.75, 0.0),
            (0, -1, 0, 0.83, 0.08, 1.17): (-90.0, 0.75, 0.84),
            # Back face (y=1)
            (0, 1, 0, 0.19, 1.0, 0.25): (90.0, 0.0, 0.0),
            (0, 1, 0, 0.12, 1.0, 0.75): (-90.0, 0.75, 0.0),
            (0, 1, 0, 0.17, 0.92, 1.17): (90.0, 0.25, 0.44),
            (0, 1, 0, 0.5, 1.0, 0.38): (180.0, 0.75, 0.5),
            (0, 1, 0, 0.81, 1.0, 0.25): (180.0, 0.25, 0.5),
            (0, 1, 0, 0.88, 1.0, 0.75): (-90.0, 0.0, 0.0),
            (0, 1, 0, 0.83, 0.92, 1.17): (180.0, 0.0, 0.0),
            # Triangulated top remnants
            (-1, 0, 0, 0.12, 0.31, 1.25): (90.0, 0.84, 0.25),
            (-1, 0, 0, 0.12, 0.69, 1.25): (90.0, 0.56, 0.5),
            (1, 0, 0, 0.88, 0.31, 1.25): (-90.0, 0.44, 0.5),
            (1, 0, 0, 0.88, 0.69, 1.25): (-90.0, 0.44, 0.75),
        }

        non_unit_scale_faces = set()
        if uv_layer:
            for face in bm.faces:
                key = _face_key(face)
                t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
                if t is None or key not in expected_uvs:
                    continue
                rot, ox, oy = expected_uvs[key]
                if key not in non_unit_scale_faces:
                    self.assertAlmostEqual(t['scale_u'], 1.0, places=1,
                                           msg=f"Face {key} scale_u={t['scale_u']}")
                    self.assertAlmostEqual(t['scale_v'], 1.0, places=1,
                                           msg=f"Face {key} scale_v={t['scale_v']}")
                self.assertAlmostEqual(t['rotation'], rot, places=1,
                                       msg=f"Face {key} rotation={t['rotation']}")
                self.assertAlmostEqual(t['offset_x'], ox, places=1,
                                       msg=f"Face {key} offset_x={t['offset_x']}")
                self.assertAlmostEqual(t['offset_y'], oy, places=1,
                                       msg=f"Face {key} offset_y={t['offset_y']}")

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()


def _undo_ctx():
    """Build a full context override for ed.undo (needs window + screen)."""
    window = bpy.context.window or bpy.context.window_manager.windows[0]
    screen = window.screen
    area = next(a for a in screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return {"window": window, "screen": screen, "area": area, "region": region}


class FoldedPlaneWeldUndoTest(AnvilTestCase):
    """Test folded plane weld undo: weld → undo → verify mode → re-weld."""

    def test_folded_plane_weld_undo_and_reweld(self):
        """Folded plane: weld → undo → verify FOLDED_PLANE → re-weld → verify geometry."""
        obj, ctx = _setup_subdivided_triangulated_cube(self, "weld_folded_undo")
        obj_name = obj.name
        uctx = _undo_ctx()

        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo_push(message="Before cube cut")

        _execute_cube_cut_and_set_weld(self, ctx)

        yield 0.1

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'FOLDED_PLANE',
                         "Should be FOLDED_PLANE after setup")

        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo_push(message="After weld setup")

        # Count faces before weld
        bm = bmesh.from_edit_mesh(obj.data)
        faces_before_weld = len(bm.faces)

        # Execute folded plane weld
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        yield 0.5

        self.assertEqual(props.weld_mode, 'NONE',
                         "Should be NONE after folded plane weld")

        bm = bmesh.from_edit_mesh(obj.data)
        faces_after_weld = len(bm.faces)
        self.assertGreater(faces_after_weld, faces_before_weld,
                           "Weld should have created new faces")

        # Push after operator so undo goes to the pre-operator push
        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo_push(message="After folded plane")

        # --- Undo folded plane ---
        with bpy.context.temp_override(**uctx):
            bpy.ops.ed.undo()

        yield 0.5

        obj = bpy.data.objects[obj_name]
        bpy.context.view_layer.objects.active = obj
        props = bpy.context.scene.level_design_props

        self.assertEqual(props.weld_mode, 'FOLDED_PLANE',
                         "Should be FOLDED_PLANE after undoing weld")

        bm = bmesh.from_edit_mesh(obj.data)
        self.assertEqual(len(bm.faces), faces_before_weld,
                         "Should have original face count after undo")

        # --- Re-execute folded plane weld ---
        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)

        yield 0.5

        self.assertEqual(props.weld_mode, 'NONE',
                         "Should be NONE after re-weld")

        bm = bmesh.from_edit_mesh(obj.data)
        self.assertEqual(len(bm.faces), faces_after_weld,
                         "Should have same face count after re-weld")
