import bmesh
import bpy
from mathutils import Vector

from ..core.uv_projection import apply_uv_to_face
from ..core.uv_projection import compute_uv_projection_from_face
from ..core.uv_projection import derive_transform_from_uvs
from .base_test import AnvilTestCase
from .helpers import (
    create_textured_cube, create_vertical_plane, edit_mesh_cache_is_current,
    wait_for_condition, _get_context_override,
)


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
        obj: Object (will be left in edit mode)
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
                abs(v0.co.z - v1.co.z) > 0.3):
            loop_edges.append(edge)
    return loop_edges


def _select_only_edges(bm, edges):
    """Select only the supplied edges in edge-select mode."""
    bm.select_mode = {'EDGE'}
    for vertex in bm.verts:
        vertex.select_set(False)
    for edge in bm.edges:
        edge.select_set(False)
    for face in bm.faces:
        face.select_set(False)
    for edge in edges:
        edge.select_set(True)
    bm.select_flush_mode()


def _execute_centered_loop_cut(test_case, bm, edge, mesh):
    """Run Blender's loop-cut operator on one edge without sliding it."""
    bm.edges.index_update()
    edge_index = edge.index
    bmesh.update_edit_mesh(mesh)
    with bpy.context.temp_override(**test_case._get_3d_view_context()):
        result = bpy.ops.mesh.loopcut(
            number_cuts=1,
            smoothness=0.0,
            object_index=0,
            edge_index=edge_index,
            mesh_select_mode_init=(False, True, False),
        )
    test_case.assertEqual(result, {'FINISHED'})


def _vertex_position_key(vertex):
    return tuple(round(value, 6) for value in vertex.co)


def _face_uvs_by_vertex(face, uv_layer):
    return {
        _vertex_position_key(loop.vert): tuple(loop[uv_layer].uv)
        for loop in face.loops
    }


def _projected_uvs_by_vertex(face, projection):
    u_axis, v_axis, origin_uv, origin_pos, _source_normal = projection
    return {
        _vertex_position_key(vertex): (
            origin_uv.x + (vertex.co - origin_pos).dot(u_axis),
            origin_uv.y + (vertex.co - origin_pos).dot(v_axis),
        )
        for vertex in face.verts
    }


class LoopCutTest(AnvilTestCase):
    """Test loop cut UV preservation across faces with different rotations."""

    def test_dissolve_middle_edge_after_plane_loop_cut_preserves_face_uvs(self):
        """Dissolving a plane loop cut preserves the pre-dissolve UV mapping."""
        obj = create_vertical_plane("loop_cut_dissolve_plane")
        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        for vertex in bm.verts:
            vertex.co.y = vertex.co.z
            vertex.co.z = 0.0
        bm.normal_update()
        bmesh.update_edit_mesh(obj.data)
        self.refresh_face_cache()

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv[0]
        source_projection = compute_uv_projection_from_face(
            bm.faces[0],
            uv_layer,
        )
        self.assertIsNotNone(source_projection)
        horizontal_edges = [
            edge for edge in bm.edges
            if abs(edge.verts[0].co.y - edge.verts[1].co.y) < 1e-6
            and abs(edge.verts[0].co.z - edge.verts[1].co.z) < 1e-6
            and abs(edge.verts[0].co.x - edge.verts[1].co.x) > 0.9
        ]
        self.assertEqual(len(horizontal_edges), 2)
        _execute_centered_loop_cut(
            self,
            bm,
            horizontal_edges[0],
            obj.data,
        )
        yield from wait_for_condition(
            lambda: (
                edit_mesh_cache_is_current()
                and len(bmesh.from_edit_mesh(obj.data).faces) == 2
            ),
            "Plane loop cut and UV projection did not finish",
        )

        bm = bmesh.from_edit_mesh(obj.data)
        middle_edges = [
            edge for edge in bm.edges
            if all(abs(vertex.co.x - 0.5) < 1e-6 for vertex in edge.verts)
            and all(abs(vertex.co.z) < 1e-6 for vertex in edge.verts)
            and abs(edge.verts[0].co.y - edge.verts[1].co.y) > 0.9
        ]
        self.assertEqual(len(middle_edges), 1)
        _select_only_edges(bm, middle_edges)
        bmesh.update_edit_mesh(obj.data)

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.dissolve_edges(
                use_verts=True,
                use_face_split=False,
            )
        yield from wait_for_condition(
            lambda: (
                edit_mesh_cache_is_current()
                and len(bmesh.from_edit_mesh(obj.data).faces) == 1
            ),
            "Plane edge dissolve and UV projection did not finish",
        )

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv[0]
        self.assertEqual(len(bm.faces), 1)
        actual_uvs = _face_uvs_by_vertex(bm.faces[0], uv_layer)
        expected_uvs = _projected_uvs_by_vertex(
            bm.faces[0],
            source_projection,
        )
        mismatched_vertices = [
            position for position, expected_uv in expected_uvs.items()
            if any(
                abs(actual_uvs[position][axis] - expected_uv[axis]) > 1e-6
                for axis in range(2)
            )
        ]

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()

        self.assertEqual(set(actual_uvs), set(expected_uvs))
        self.assertEqual(
            mismatched_vertices,
            [],
            "Dissolving the plane loop-cut edge changed the face UV mapping",
        )

    def test_dissolve_top_middle_edge_after_cube_loop_cut_preserves_top_face_uvs(self):
        """Dissolving one top loop edge preserves the top surface UV mapping."""
        obj = create_textured_cube(
            "loop_cut_dissolve_cube",
            1.0,
            1.0,
            use_box_project=True,
        )
        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        self.refresh_face_cache()

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv[0]
        initial_top_faces = [face for face in bm.faces if face.normal.z > 0.9]
        self.assertEqual(len(initial_top_faces), 1)
        x_axis_edges = [
            edge for edge in bm.edges
            if abs(edge.verts[0].co.y - edge.verts[1].co.y) < 1e-6
            and abs(edge.verts[0].co.z - edge.verts[1].co.z) < 1e-6
            and abs(edge.verts[0].co.x - edge.verts[1].co.x) > 0.9
        ]
        self.assertEqual(len(x_axis_edges), 4)
        _execute_centered_loop_cut(
            self,
            bm,
            x_axis_edges[0],
            obj.data,
        )
        yield from wait_for_condition(
            lambda: (
                edit_mesh_cache_is_current()
                and len(bmesh.from_edit_mesh(obj.data).faces) == 10
            ),
            "Cube loop cut and UV projection did not finish",
        )

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv[0]
        top_faces = [face for face in bm.faces if face.normal.z > 0.9]
        self.assertEqual(len(top_faces), 2)
        source_projections = [
            compute_uv_projection_from_face(face, uv_layer)
            for face in top_faces
        ]
        self.assertTrue(all(source_projections))
        top_middle_edges = [
            edge for edge in bm.edges
            if all(abs(vertex.co.x - 0.5) < 1e-6 for vertex in edge.verts)
            and all(abs(vertex.co.z - 1.0) < 1e-6 for vertex in edge.verts)
            and abs(edge.verts[0].co.y - edge.verts[1].co.y) > 0.9
        ]
        self.assertEqual(len(top_middle_edges), 1)
        _select_only_edges(bm, top_middle_edges)
        bmesh.update_edit_mesh(obj.data)

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.dissolve_edges(
                use_verts=True,
                use_face_split=False,
            )
        yield from wait_for_condition(
            lambda: (
                edit_mesh_cache_is_current()
                and len(bmesh.from_edit_mesh(obj.data).faces) == 9
            ),
            "Cube top-edge dissolve and UV projection did not finish",
        )

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv[0]
        top_faces = [face for face in bm.faces if face.normal.z > 0.9]
        self.assertEqual(len(top_faces), 1)
        actual_uvs = _face_uvs_by_vertex(top_faces[0], uv_layer)
        expected_uv_candidates = [
            _projected_uvs_by_vertex(top_faces[0], source_projection)
            for source_projection in source_projections
        ]
        matching_source_count = sum(
            all(
                abs(actual_uvs[position][axis] - expected_uvs[position][axis])
                <= 1e-6
                for position in expected_uvs
                for axis in range(2)
            )
            for expected_uvs in expected_uv_candidates
        )

        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update()

        self.assertEqual(set(actual_uvs), set(expected_uv_candidates[0]))
        self.assertGreaterEqual(
            matching_source_count,
            1,
            "Dissolved top face did not use either old top face's UV mapping",
        )

    def test_loop_cut_preserves_rotation(self):
        """Loop cut two faces with different rotations preserves each face's rotation.

        Creates two coplanar faces: left at rotation=0, right at rotation=45.
        Expected: 4 faces -- two with rotation~0, two with rotation~45.
        """
        obj = _create_two_face_plane("loopcut_rot")
        yield from wait_for_condition(
            lambda: len(obj.data.polygons) == 2,
            "Two-face plane setup did not finish",
        )

        # Deform the right face: move the 2 outside verts (x=2) inward 0.25 on Z
        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        for v in bm.verts:
            if abs(v.co.x - 2.0) < 1e-5:
                if v.co.z < 0.5:
                    v.co.z += 0.25
                else:
                    v.co.z -= 0.25
        bmesh.update_edit_mesh(obj.data)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
        yield from wait_for_condition(
            lambda: (
                len(obj.data.polygons) == 2
                and sum(abs(vertex.co.x - 2.0) < 1e-5
                        for vertex in obj.data.vertices) == 2
            ),
            "Plane deformation did not finish",
        )

        # Apply rotation=0 to left face (center x~0.5), rotation=45 to right
        # (center x~1.5) after the setup deformation is already in place.
        _apply_per_face_rotation(obj, {0.5: 0.0, 1.5: 45.0})
        self.refresh_face_cache()

        # Verify initial state: 2 faces with correct rotations
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

        self.refresh_face_cache()

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
        yield from wait_for_condition(
            lambda: (
                edit_mesh_cache_is_current()
                and len(bmesh.from_edit_mesh(obj.data).faces) >= 4
            ),
            "Loop cut topology and UV projection did not finish",
        )

        # Verify: should now have 4 faces
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv[0]
        self.assertGreaterEqual(len(bm.faces), 4, f"Should have at least 4 faces after loop cut, got {len(bm.faces)}")

        # Check each face's rotation, scale, and offset after loop cut.
        # Expected offsets with set_uv_from_source_params approach:
        #   Left faces (rot=0):  bottom (0, 0),       top (0, 0.5)
        #   Right faces (rot=45): bottom (0.3536, 0.3536), top (0, 0)
        expected = {
            (0.5, 0.25): {'rot': 0.0, 'off_x': 0.0, 'off_y': 0.0},
            (0.5, 0.75): {'rot': 0.0, 'off_x': 0.0, 'off_y': 0.5},
            (1.5, 0.31): {'rot': 45.0, 'off_x': 0.3536, 'off_y': 0.3536},
            (1.5, 0.69): {'rot': 45.0, 'off_x': 0.0, 'off_y': 0.0},
        }
        for face in bm.faces:
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            self.assertIsNotNone(t, f"Face {face.index} transform should be derivable")
            cx = round(face.calc_center_median().x, 2)
            cz = round(face.calc_center_median().z, 2)
            key = (cx, cz)
            self.assertIn(key, expected, f"Unexpected face center ({cx}, {cz})")
            exp = expected[key]
            label = f"Face at x={cx},z={cz}"
            self.assertAlmostEqual(
                t['rotation'], exp['rot'], places=1,
                msg=f"{label} rotation should be {exp['rot']}, got {t['rotation']}"
            )
            self.assertAlmostEqual(t['scale_u'], 1.0, places=1,
                                   msg=f"{label} scale_u should be 1.0, got {t['scale_u']}")
            self.assertAlmostEqual(t['scale_v'], 1.0, places=1,
                                   msg=f"{label} scale_v should be 1.0, got {t['scale_v']}")
            self.assertAlmostEqual(t['offset_x'], exp['off_x'], places=2,
                                   msg=f"{label} offset_x should be {exp['off_x']}, got {t['offset_x']}")
            self.assertAlmostEqual(t['offset_y'], exp['off_y'], places=2,
                                   msg=f"{label} offset_y should be {exp['off_y']}, got {t['offset_y']}")

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
