import bmesh
import bpy
from mathutils import Vector

from ..operators.cube_cut.geometry import execute_cube_cut
from .base_test import AnvilTestCase
from .helpers import create_textured_cube, _get_context_override


class WeldTriangulatedCubeCutTest(AnvilTestCase):
    """Test cube cut on a subdivided cube with triangulated top and raised centre vertex."""

    def test_cube_cut_on_subdivided_triangulated_cube_with_raised_centre(self):
        """Subdivide a cube, triangulate the top 4 faces, raise the centre vertex,
        then cube cut 0.5 units either side of centre from centre to 3 units up."""
        obj = create_textured_cube("weld_tri", 1.0, 1.0, face_aligned=True)

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

        # Triangulate only the top 4 faces with diagonals pointing to the centre.
        # Use face_split to control which diagonal is used: always split through
        # the centre vertex (0.5, 0.5, 1.0) to the opposite corner.
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
        self.assertIsNotNone(centre_vert, "Should find centre vertex at (0.5, 0.5, 1.0)")

        top_faces = [f for f in bm.faces if f.normal.z > 0.9]
        self.assertEqual(len(top_faces), 4, "Should have 4 top faces after subdivide")

        for face in top_faces:
            verts = list(face.verts)
            centre_idx = verts.index(centre_vert)
            opposite = verts[(centre_idx + 2) % 4]
            bmesh.utils.face_split(face, centre_vert, opposite)

        bmesh.update_edit_mesh(obj.data)

        # Select the centre vertex and raise it 1 unit
        bm.select_mode = {'VERT'}
        for v in bm.verts:
            v.select = False
        centre_vert.select = True
        bmesh.update_edit_mesh(obj.data)

        with bpy.context.temp_override(**ctx):
            bpy.ops.transform.translate(value=(0, 0, 1.0))

        # Verify the vertex was raised
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        raised = None
        for v in bm.verts:
            if (abs(v.co.x - 0.5) < 1e-4 and
                    abs(v.co.y - 0.5) < 1e-4 and
                    abs(v.co.z - 2.0) < 1e-4):
                raised = v
                break
        self.assertIsNotNone(raised, "Centre vertex should be at (0.5, 0.5, 2.0) after raise")

        # Select all faces for cube cut
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(obj.data)

        # Cube cut: 0.5 units either side of cube centre (0.5, 0.5, 0.5) in X,
        # from centre height (z=0.5) to 3 units up (z=3.5),
        # cutting through in Y
        with bpy.context.temp_override(**ctx):
            success, msg = execute_cube_cut(
                bpy.context,
                Vector((0.25, -0.5, 0.5)),
                Vector((0.75, -0.5, 3.5)),
                2.0,
                Vector((1, 0, 0)),
                Vector((0, 0, 1)),
                Vector((0, 1, 0)),
            )

        self.assertTrue(success, msg)

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        self.assertGreater(len(bm.faces), 0, "Should have faces remaining after cube cut")

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()
