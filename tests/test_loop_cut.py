import bmesh
import bpy
from mathutils import Vector

from ..properties import apply_uv_to_face
from ..utils import derive_transform_from_uvs
from .base_test import AnvilTestCase
from .helpers import create_vertical_plane, _get_context_override


def _create_two_face_plane(name):
    """Create a vertical plane and extrude its right edge to make two faces.

    Face 0: original plane at x=[0,1], z=[0,1]
    Face 1: extruded plane at x=[1,2], z=[0,1]

    Returns the object in object mode.
    """
    obj = create_vertical_plane(name)

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)

    # Select the right edge (x=1) and extrude along +X
    bm.select_mode = {'EDGE'}
    for v in bm.verts:
        v.select_set(False)
    for e in bm.edges:
        e.select_set(False)
    for f in bm.faces:
        f.select_set(False)

    bm.edges.ensure_lookup_table()
    for edge in bm.edges:
        if all(abs(v.co.x - 1.0) < 1e-5 for v in edge.verts):
            edge.select_set(True)
            break
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)

    with bpy.context.temp_override(**ctx):
        bpy.ops.mesh.extrude_region_move(
            TRANSFORM_OT_translate={"value": Vector((1, 0, 0))}
        )

    bmesh.update_edit_mesh(obj.data)
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')

    return obj


def _apply_per_face_rotation(obj, rotation_by_center_x):
    """Apply different rotations to faces based on their center X coordinate.

    Args:
        obj: Object (will be put into edit mode and back)
        rotation_by_center_x: dict mapping approximate center X to rotation degrees
    """
    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()
    me = obj.data
    ppm = bpy.context.scene.level_design_props.pixels_per_meter
    mat = me.materials[0] if me.materials else None

    for face in bm.faces:
        cx = face.calc_center_median().x
        for target_x, rotation in rotation_by_center_x.items():
            if abs(cx - target_x) < 0.3:
                apply_uv_to_face(face, uv_layer, 1.0, 1.0, rotation, 0.0, 0.0,
                                 mat, ppm, me)
                break

    bmesh.update_edit_mesh(me)
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')


def _find_loop_cut_edges(bm):
    """Find the vertical edges for a horizontal loop cut.

    A horizontal loop cut on this 2-face strip splits each face into top/bottom
    halves by cutting through the vertical edges (those running along Z).

    Returns list of BMEdge.
    """
    loop_edges = []
    for edge in bm.edges:
        v0, v1 = edge.verts
        # Vertical edge (same X, same Y) spanning Z
        if (abs(v0.co.x - v1.co.x) < 1e-5 and
                abs(v0.co.y - v1.co.y) < 1e-5 and
                abs(v0.co.z - v1.co.z) > 0.5):
            loop_edges.append(edge)
    return loop_edges


class LoopCutTest(AnvilTestCase):
    """Test loop cut UV preservation across faces with different rotations."""

    def test_loop_cut_preserves_rotation(self):
        """Loop cut two faces with different rotations preserves each face's rotation.

        Creates two coplanar faces: left at rotation=0, right at rotation=45.
        Re-enters edit mode then loop cuts across both.
        Expected: 4 faces -- two with rotation~0, two with rotation~45.
        """
        obj = _create_two_face_plane("loopcut_rot")
        yield 0.5

        # Apply rotation=0 to left face (center x~0.5), rotation=45 to right (center x~1.5)
        _apply_per_face_rotation(obj, {0.5: 0.0, 1.5: 45.0})
        yield 0.5

        # Verify initial state: 2 faces with correct rotations
        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv[0]
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        self.assertEqual(len(bm.faces), 2, "Should start with 2 faces")

        for face in bm.faces:
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertIsNotNone(t, f"Face {face.index} transform should be derivable")
            cx = face.calc_center_median().x
            if cx < 1.0:
                self.assertAlmostEqual(t['rotation'], 0.0, places=1,
                                       msg=f"Left face rotation should be 0, got {t['rotation']}")
            else:
                self.assertAlmostEqual(t['rotation'], 45.0, places=1,
                                       msg=f"Right face rotation should be 45, got {t['rotation']}")

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

        # Re-enter edit mode (this is what triggers the bug -- fresh edit session)
        yield 0.5
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        yield 0.5  # Let depsgraph handler rebuild cache

        # Select the horizontal loop edges and subdivide them (equivalent to a loop cut)
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()

        loop_edges = _find_loop_cut_edges(bm)
        self.assertGreaterEqual(len(loop_edges), 2,
                                f"Should find at least 2 horizontal spanning edges, got {len(loop_edges)}")

        # Select loop edges
        bm.select_mode = {'EDGE'}
        for v in bm.verts:
            v.select_set(False)
        for e in bm.edges:
            e.select_set(False)
        for f in bm.faces:
            f.select_set(False)

        for edge in loop_edges:
            edge.select_set(True)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

        # Subdivide selected edges (this is what loop cut does internally)
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.subdivide(number_cuts=1)
        yield 0.5  # Let depsgraph handler process topology change

        # Verify: should now have 4 faces
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv[0]
        self.assertEqual(len(bm.faces), 4, f"Should have 4 faces after loop cut, got {len(bm.faces)}")

        # Check each face's rotation matches its parent face
        for face in bm.faces:
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertIsNotNone(t, f"Face {face.index} transform should be derivable")
            cx = face.calc_center_median().x
            if cx < 1.0:
                expected_rot = 0.0
            else:
                expected_rot = 45.0
            self.assertAlmostEqual(
                t['rotation'], expected_rot, places=1,
                msg=f"Face at x={cx:.2f} rotation should be {expected_rot}, got {t['rotation']}"
            )
            self.assertAlmostEqual(t['scale_u'], 1.0, places=1,
                                   msg=f"Face at x={cx:.2f} scale_u should be 1.0, got {t['scale_u']}")
            self.assertAlmostEqual(t['scale_v'], 1.0, places=1,
                                   msg=f"Face at x={cx:.2f} scale_v should be 1.0, got {t['scale_v']}")

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
