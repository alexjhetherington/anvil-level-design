import bmesh
import bpy
from mathutils import Vector

from ..operators.cube_cut.geometry import execute_cube_cut
from ..operators.weld import set_weld_from_edge_selection, snapshot_coplanar_sides
from .base_test import AnvilTestCase
from .helpers import create_textured_cube, _get_context_override


# Cube cut parameters: cut from bottom (z=0) to middle (z=0.5) of a plain cube
_FIRST_VERTEX = Vector((0.25, -0.5, 0.0))
_SECOND_VERTEX = Vector((0.75, -0.5, 0.5))
_DEPTH = 2.0
_LOCAL_X = Vector((1, 0, 0))
_LOCAL_Y = Vector((0, 0, 1))
_LOCAL_Z = Vector((0, 1, 0))


def _setup_plain_cube(name):
    """Create a plain textured cube in edit mode with all faces selected.

    Returns (obj, ctx).
    """
    obj = create_textured_cube(name, 1.0, 1.0, face_aligned=True)
    ctx = _get_context_override()

    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.select_mode = {'FACE'}
    for f in bm.faces:
        f.select = True
    bmesh.update_edit_mesh(obj.data)

    return obj, ctx


def _compute_cuboid_params_local(obj, v1, v2, lx, ly):
    """Compute cuboid params in object local space (mirrors operator logic)."""
    w2l = obj.matrix_world.inverted()
    w2l_rot = w2l.to_3x3()
    origin = w2l @ v1
    local_lx = (w2l_rot @ Vector(lx)).normalized()
    local_ly = (w2l_rot @ Vector(ly)).normalized()
    diff_world = Vector(v2) - Vector(v1)
    cdx = abs(diff_world.dot(Vector(lx))) * (w2l_rot @ Vector(lx)).length
    cdy = abs(diff_world.dot(Vector(ly))) * (w2l_rot @ Vector(ly)).length
    local_diff = (w2l @ Vector(v2)) - origin
    if local_diff.dot(local_lx) < 0:
        local_lx = -local_lx
    if local_diff.dot(local_ly) < 0:
        local_ly = -local_ly
    return (origin, local_lx, local_ly, cdx, cdy)


def _snapshot_and_cut_and_set_weld(test_case, ctx, obj, v1, v2, depth, lx, ly, lz):
    """Snapshot coplanar faces, execute cube cut, and set weld state."""
    # Snapshot before cut
    bm = bmesh.from_edit_mesh(obj.data)
    cuboid_local = _compute_cuboid_params_local(obj, v1, v2, lx, ly)
    coplanar_blocked = snapshot_coplanar_sides(bm, cuboid_local)

    with bpy.context.temp_override(**ctx):
        success, msg = execute_cube_cut(
            bpy.context, v1, v2, depth, lx, ly, lz,
        )
    test_case.assertTrue(success, msg)

    extrude_dir = -lz
    back_point = v1 + lz * depth
    back_plane_offset = back_point.dot(extrude_dir.normalized())

    set_weld_from_edge_selection(
        bpy.context, abs(depth), extrude_dir, back_plane_offset,
        v1, v2, lx, ly, coplanar_blocked,
    )

    return success


def _vert_key(v):
    """Return a hashable key from a vertex position, rounded to 4dp."""
    return (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))


class FoldedPlaneWeldCubeTest(AnvilTestCase):
    """Test folded plane weld after cube cut on a plain cube (bottom-to-middle cut)."""

    def test_folded_plane_weld_on_plain_cube_bottom_to_middle_cut(self):
        """Cube cut a plain cube from z=0 to z=0.5 (bottom to middle),
        then execute folded plane weld.

        The cut removes a slot from the bottom of the cube between x=0.25
        and x=0.75, creating a non-coplanar boundary edge loop that wraps
        around the front, bottom, and back faces.
        """
        obj, ctx = _setup_plain_cube("weld_folded_cube")

        _snapshot_and_cut_and_set_weld(
            self, ctx, obj,
            _FIRST_VERTEX, _SECOND_VERTEX, _DEPTH,
            _LOCAL_X, _LOCAL_Y, _LOCAL_Z,
        )

        yield 0.1

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'FOLDED_PLANE',
                         "Should be FOLDED_PLANE after cube cut on plain cube "
                         "with bottom-to-middle cut")

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

        self.assertEqual(len(bm.verts), 16)
        self.assertEqual(len(bm.faces), 14)

        # Verify all vertex positions
        expected_verts = sorted([
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),
            (0.25, 0.0, 0.0), (0.25, 0.0, 0.5),
            (0.25, 1.0, 0.0), (0.25, 1.0, 0.5),
            (0.75, 0.0, 0.0), (0.75, 0.0, 0.5),
            (0.75, 1.0, 0.0), (0.75, 1.0, 0.5),
            (1.0, 0.0, 0.0), (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 1.0, 1.0),
        ])
        actual_verts = sorted([_vert_key(v) for v in bm.verts])
        self.assertEqual(actual_verts, expected_verts)

        # The bottom of the cube (z=0) should NOT have a weld face because
        # it is coplanar with the existing bottom faces that were cut.
        for face in bm.faces:
            if abs(face.normal.z) > 0.9:
                for v in face.verts:
                    # No weld face should span the cut slot at z=0
                    if abs(v.co.z) < 1e-4:
                        self.assertTrue(
                            v.co.x <= 0.25 + 1e-4 or v.co.x >= 0.75 - 1e-4,
                            f"No bottom weld face should exist in the cut slot, "
                            f"but found face with vert at {tuple(v.co)}")

        # All faces should be valid (no degenerate faces)
        for face in bm.faces:
            self.assertGreaterEqual(len(face.verts), 3,
                                    "All faces should have at least 3 verts")
            area = face.calc_area()
            self.assertGreater(area, 1e-6,
                               f"Face should have non-zero area, got {area}")

        # Check that UVs exist on all faces
        uv_layer = bm.loops.layers.uv.active
        self.assertIsNotNone(uv_layer, "Should have an active UV layer")

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()


class FoldedPlaneWeldCubeDoubleTest(AnvilTestCase):
    """Test two successive folded plane welds on a plain cube."""

    def test_folded_plane_weld_on_plain_cube_bottom_then_top_corner_cut(self):
        """Cube cut a plain cube from z=0 to z=0.5 (bottom to middle),
        weld it, then cube cut the top-right corner (x=0.75..1, z=0.75..1)
        and weld again.
        """
        # --- First cut: bottom to middle (same as single-cut test) ---
        obj = create_textured_cube("weld_folded_double", 1.0, 1.0,
                                   face_aligned=True)
        ctx = _get_context_override()

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(obj.data)

        first_v1 = Vector((0.25, -0.5, 0.0))
        first_v2 = Vector((0.75, -0.5, 0.5))
        first_depth = 2.0
        lx = Vector((1, 0, 0))
        ly = Vector((0, 0, 1))
        lz = Vector((0, 1, 0))

        _snapshot_and_cut_and_set_weld(
            self, ctx, obj,
            first_v1, first_v2, first_depth, lx, ly, lz,
        )

        yield 0.1

        props = bpy.context.scene.level_design_props
        self.assertEqual(props.weld_mode, 'FOLDED_PLANE')

        with bpy.context.temp_override(**ctx):
            result = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result)
        self.assertEqual(props.weld_mode, 'NONE')

        yield 0.5

        # --- Second cut: top-right corner (x=0.75..1, z=0.75..1) ---
        # Re-select all faces for the second cut
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(obj.data)

        second_v1 = Vector((0.75, -0.5, 0.75))
        second_v2 = Vector((1.0, -0.5, 1.0))
        second_depth = 2.0

        _snapshot_and_cut_and_set_weld(
            self, ctx, obj,
            second_v1, second_v2, second_depth, lx, ly, lz,
        )

        yield 0.1

        # Diagnostic: inspect boundary edges after second cut
        bm = bmesh.from_edit_mesh(obj.data)
        sel_edges = [e for e in bm.edges if e.select]
        print(f"Level Design Tools: Second cut selected {len(sel_edges)} edges")
        for e in sel_edges:
            v0, v1 = e.verts
            print(f"  edge: ({v0.co.x:.2f},{v0.co.y:.2f},{v0.co.z:.2f})"
                  f" -> ({v1.co.x:.2f},{v1.co.y:.2f},{v1.co.z:.2f})"
                  f" link_faces={len(e.link_faces)}")
        print(f"Level Design Tools: weld_mode = {props.weld_mode}")

        self.assertIn(props.weld_mode, ('BRIDGE', 'FOLDED_PLANE'),
                      "Second cut should produce a weld mode")

        with bpy.context.temp_override(**ctx):
            result2 = bpy.ops.leveldesign.context_weld()
        self.assertIn('FINISHED', result2)
        self.assertEqual(props.weld_mode, 'NONE')

        yield 0.5

        # --- Geometry assertions ---
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        bm.normal_update()

        self.assertEqual(len(bm.verts), 22)
        self.assertEqual(len(bm.faces), 21)

        expected_verts = sorted([
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),
            (0.25, 0.0, 0.0), (0.25, 0.0, 0.5),
            (0.25, 1.0, 0.0), (0.25, 1.0, 0.5),
            (0.75, 0.0, 0.0), (0.75, 0.0, 0.5),
            (0.75, 0.0, 0.75), (0.75, 0.0, 1.0),
            (0.75, 1.0, 0.0), (0.75, 1.0, 0.5),
            (0.75, 1.0, 0.75), (0.75, 1.0, 1.0),
            (0.875, 0.0, 0.75), (0.875, 1.0, 0.75),
            (1.0, 0.0, 0.0), (1.0, 0.0, 0.75),
            (1.0, 1.0, 0.0), (1.0, 1.0, 0.75),
        ])
        actual_verts = sorted([_vert_key(v) for v in bm.verts])
        self.assertEqual(actual_verts, expected_verts)

        # All faces should be valid
        for face in bm.faces:
            self.assertGreaterEqual(len(face.verts), 3,
                                    "All faces should have at least 3 verts")
            area = face.calc_area()
            self.assertGreater(area, 1e-6,
                               f"Face should have non-zero area, got {area}")

        uv_layer = bm.loops.layers.uv.active
        self.assertIsNotNone(uv_layer, "Should have an active UV layer")

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()
