import bmesh
import bpy
import math
from mathutils import Vector

from ..operators.cube_cut.geometry import execute_cube_cut
from ..core.uv_projection import derive_transform_from_uvs
from .base_test import AnvilTestCase
from .helpers import create_vertical_plane, _get_context_override


def _select_edges_by_filter(bm, me, edge_filter):
    """Select all edges that satisfy edge_filter(edge)."""
    bm.select_mode = {'EDGE'}
    for v in bm.verts:
        v.select_set(False)
    for e in bm.edges:
        e.select_set(False)
    for f in bm.faces:
        f.select_set(False)

    bm.edges.ensure_lookup_table()
    count = 0
    for edge in bm.edges:
        if edge_filter(edge):
            edge.select_set(True)
            count += 1
    bm.select_flush_mode()
    bmesh.update_edit_mesh(me)
    return count


class BridgeBevelCorridorTest(AnvilTestCase):
    """Create two planes with holes, bridge them, then bevel the top corridor edges."""

    def test_bridge_and_bevel_corridor_3_segments(self):
        # 1. Create two vertical planes facing each other, 2 units apart
        #    plane_a at y=0 facing -Y, plane_b at y=2 facing +Y
        plane_a = create_vertical_plane("corridor_a")
        plane_b = create_vertical_plane("corridor_b")

        ctx = _get_context_override()

        # plane_a already faces +Y (outward), no rotation needed.

        # Rotate plane_b 180 degrees so it faces -Y (outward)
        plane_b.select_set(True)
        plane_a.select_set(False)
        bpy.context.view_layer.objects.active = plane_b
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        plane_b.rotation_euler.z = math.pi
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(rotation=True)

        # Move plane_b to Y=2
        plane_b.location.y = 2.0
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(location=True)

        # 2. Cut a hole in each plane using execute_cube_cut
        for plane in [plane_a, plane_b]:
            plane_a.select_set(plane is plane_a)
            plane_b.select_set(plane is plane_b)
            bpy.context.view_layer.objects.active = plane
            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='EDIT')

            bm = bmesh.from_edit_mesh(plane.data)
            bm.select_mode = {'FACE'}
            for f in bm.faces:
                f.select = True
            bmesh.update_edit_mesh(plane.data)

            # Determine Y position of this plane
            bm.verts.ensure_lookup_table()
            plane_y = bm.verts[0].co.y

            # Cut hole at x=[0.25,0.75], z=[0.25,0.75]
            # Cuboid passes through the plane along Y
            with bpy.context.temp_override(**ctx):
                success, msg = execute_cube_cut(
                    bpy.context,
                    Vector((0.25, plane_y - 0.25, 0.25)),
                    Vector((0.75, plane_y - 0.25, 0.75)),
                    0.5,
                    Vector((1, 0, 0)),
                    Vector((0, 0, 1)),
                    Vector((0, 1, 0)),
                )
            assert success, f"Cube cut failed on {plane.name}: {msg}"

            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='OBJECT')

        yield 0.5

        # 3. Join both planes into one object
        plane_a.select_set(True)
        plane_b.select_set(True)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.join()
        obj = plane_a

        # 4. Enter edit mode, select boundary edges (hole edges), and bridge
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)

        # Select only the hole boundary edges (not outer plane edges).
        # Hole edges have both verts with x in [0.25, 0.75] and z in [0.25, 0.75].
        def is_hole_boundary(e):
            if not e.is_boundary:
                return False
            for v in e.verts:
                x, z = v.co.x, v.co.z
                x_on_hole = abs(x - 0.25) < 0.01 or abs(x - 0.75) < 0.01
                z_on_hole = abs(z - 0.25) < 0.01 or abs(z - 0.75) < 0.01
                if not (x_on_hole or z_on_hole):
                    return False
                # Must be within the hole rectangle
                if x < 0.24 or x > 0.76 or z < 0.24 or z > 0.76:
                    return False
            return True

        boundary_count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=is_hole_boundary,
        )
        self.assertEqual(boundary_count, 8,
                         f"Expected 8 hole boundary edges (4 per hole), got {boundary_count}")

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops()

        yield 0.5

        # 5. Select the top two edges of the corridor for beveling
        # The hole top is at z=0.75, corridor runs along Y from y=0 to y=2
        bm = bmesh.from_edit_mesh(obj.data)

        count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=lambda e: (
                all(abs(v.co.z - 0.75) < 0.05 for v in e.verts)
                and abs(e.verts[0].co.y - e.verts[1].co.y) > 0.1  # runs along Y
            ),
        )

        self.assertEqual(count, 2,
                         f"Expected 2 top corridor edges selected, got {count}")

        # 6. Bevel the selected top edges into 3 segments
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bevel(
                offset=0.1,
                offset_pct=0,
                segments=3,
                affect='EDGES',
            )

        yield 0.5

        # 7. Assert UV transforms on all faces
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv[0]
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        def _face_key(face):
            n = face.normal
            c = face.calc_center_median()
            return (round(n.x, 2), round(n.y, 2), round(n.z, 2),
                    round(c.x, 2), round(c.y, 2), round(c.z, 2))

        # (scale_u, scale_v, rotation, offset_x, offset_y)
        expected = {
            # --- Bottom strip faces (z~0.12) on each plane ---
            (0.0, -1.0, 0.0, 0.5, 0.0, 0.12):   (1.0, 1.0, 180.0, 0.75, 0.25),
            (0.0, 1.0, -0.0, 0.5, 2.0, 0.12):    (1.0, 1.0, 180.0, 0.75, 0.25),
            # --- Corridor bottom face ---
            (0.0, -0.0, 1.0, 0.5, 1.0, 0.25):    (1.0, 1.0, 0.0, 0.25, 0.25),
            # --- Bevel faces on plane_a (y=0) right side ---
            (-0.0, -1.0, 0.0, 0.72, 0.0, 0.77):  (1.0, 1.0, -15.0, 0.65, 0.75),
            (-0.0, -1.0, 0.0, 0.75, 0.0, 0.75):  (1.0, 1.0, -45.0, 0.70, 0.74),
            (-0.0, -1.0, 0.0, 0.77, 0.0, 0.72):  (1.0, 1.0, -75.0, 0.74, 0.70),
            # --- Bevel faces on plane_a (y=0) left side ---
            (0.0, -1.0, -0.0, 0.23, 0.0, 0.72):  (1.0, 1.0, 75.0, 0.25, 0.65),
            (0.0, -1.0, -0.0, 0.25, 0.0, 0.75):  (1.0, 1.0, 45.0, 0.26, 0.70),
            (0.0, -1.0, -0.0, 0.28, 0.0, 0.77):  (1.0, 1.0, 15.0, 0.30, 0.74),
            # --- Bevel faces on plane_b (y=2) right side ---
            (0.0, 1.0, 0.0, 0.77, 2.0, 0.72):    (1.0, 1.0, 75.0, 0.25, 0.65),
            (0.0, 1.0, 0.0, 0.75, 2.0, 0.75):    (1.0, 1.0, 45.0, 0.26, 0.70),
            (0.0, 1.0, 0.0, 0.72, 2.0, 0.77):    (1.0, 1.0, 15.0, 0.30, 0.74),
            # --- Bevel faces on plane_b (y=2) left side ---
            (-0.0, 1.0, 0.0, 0.28, 2.0, 0.77):   (1.0, 1.0, -15.0, 0.65, 0.75),
            (-0.0, 1.0, 0.0, 0.25, 2.0, 0.75):   (1.0, 1.0, -45.0, 0.70, 0.74),
            (-0.0, 1.0, 0.0, 0.23, 2.0, 0.72):   (1.0, 1.0, -75.0, 0.74, 0.70),
            # --- Bevel curved faces (corridor ceiling) ---
            (-0.26, 0.0, -0.97, 0.68, 1.0, 0.74): (1.0, 1.0, 90.0, 0.35, 0.75),
            (-0.71, 0.0, -0.71, 0.72, 1.0, 0.72): (1.0, 1.0, 180.0, 0.25, 0.75),
            (-0.97, 0.0, -0.26, 0.74, 1.0, 0.67): (1.0, 1.0, 180.0, 0.25, 0.70),
            (0.26, 0.0, -0.97, 0.32, 1.0, 0.74):  (1.0, 1.0, 180.0, 0.75, 0.81),
            (0.71, 0.0, -0.71, 0.28, 1.0, 0.72):  (1.0, 1.0, 180.0, 0.75, 0.75),
            (0.97, 0.0, -0.26, 0.26, 1.0, 0.67):  (1.0, 1.0, 180.0, 0.75, 0.70),
            (0.0, 0.0, -1.0, 0.5, 1.0, 0.75):     (1.0, 1.0, 180.0, 0.65, 0.75),
            # --- Side strips on plane_a (y=0) ---
            (0.0, -1.0, 0.0, 0.86, 0.0, 0.54):   (1.0, 1.0, -112.5, 0.82, 0.82),
            (0.0, -1.0, 0.0, 0.5, 0.0, 0.86):    (1.0, 1.0, -22.5, 0.18, 0.82),
            (0.0, -1.0, 0.0, 0.14, 0.0, 0.54):   (1.0, 1.0, 90.0, 0.25, 0.25),
            # --- Corridor side walls ---
            (-1.0, 0.0, 0.0, 0.75, 1.0, 0.45):   (1.0, 1.0, -90.0, 0.25, 0.65),
            (1.0, -0.0, 0.0, 0.25, 1.0, 0.45):   (1.0, 1.0, 90.0, 0.75, 0.25),
            # --- Side strips on plane_b (y=2) ---
            (0.0, 1.0, 0.0, 0.5, 2.0, 0.86):     (1.0, 1.0, -22.5, 0.18, 0.82),
            (0.0, 1.0, 0.0, 0.86, 2.0, 0.54):    (1.0, 1.0, 90.0, 0.25, 0.25),
            (0.0, 1.0, 0.0, 0.14, 2.0, 0.54):    (1.0, 1.0, -112.5, 0.82, 0.82),
        }

        # Collect all failures, then select the bad faces at the end
        failed_faces = []
        errors = []
        for face in bm.faces:
            key = _face_key(face)
            if key not in expected:
                errors.append(f"Unexpected face {key}")
                failed_faces.append(face.index)
                continue
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            su, sv, rot, ox, oy = expected[key]
            label = (f"n=({key[0]},{key[1]},{key[2]}) "
                     f"c=({key[3]},{key[4]},{key[5]})")
            face_ok = True
            if abs(t['scale_u'] - su) > 0.005:
                errors.append(f"{label} scale_u={t['scale_u']:.4f}, expected {su}")
                face_ok = False
            if abs(t['scale_v'] - sv) > 0.005:
                errors.append(f"{label} scale_v={t['scale_v']:.4f}, expected {sv}")
                face_ok = False
            if abs(t['rotation'] - rot) > 0.05:
                errors.append(f"{label} rotation={t['rotation']:.2f}, expected {rot}")
                face_ok = False
            if abs(t['offset_x'] - ox) > 0.05:
                errors.append(f"{label} offset_x={t['offset_x']:.2f}, expected {ox}")
                face_ok = False
            if abs(t['offset_y'] - oy) > 0.05:
                errors.append(f"{label} offset_y={t['offset_y']:.2f}, expected {oy}")
                face_ok = False
            if not face_ok:
                failed_faces.append(face.index)

        # Select the failed faces so they're visible in the saved .blend
        if failed_faces:
            bm.select_mode = {'FACE'}
            for v in bm.verts:
                v.select_set(False)
            for e in bm.edges:
                e.select_set(False)
            for f in bm.faces:
                f.select_set(False)
            bm.faces.ensure_lookup_table()
            for idx in failed_faces:
                bm.faces[idx].select_set(True)
            bm.select_flush_mode()
            bmesh.update_edit_mesh(obj.data)

        if errors:
            self.fail(f"[2-edge] {len(failed_faces)} face(s) with wrong UVs:\n"
                      + "\n".join(errors))

    def test_bridge_and_bevel_corridor_1_segment(self):
        # 1. Create two vertical planes facing each other, 2 units apart
        plane_a = create_vertical_plane("corridor_1s_a")
        plane_b = create_vertical_plane("corridor_1s_b")

        ctx = _get_context_override()

        # plane_a already faces +Y (outward), no rotation needed.

        # Rotate plane_b 180 degrees so it faces -Y (outward)
        plane_b.select_set(True)
        plane_a.select_set(False)
        bpy.context.view_layer.objects.active = plane_b
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        plane_b.rotation_euler.z = math.pi
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(rotation=True)

        # Move plane_b to Y=2
        plane_b.location.y = 2.0
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(location=True)

        # 2. Cut a hole in each plane using execute_cube_cut
        for plane in [plane_a, plane_b]:
            plane_a.select_set(plane is plane_a)
            plane_b.select_set(plane is plane_b)
            bpy.context.view_layer.objects.active = plane
            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='EDIT')

            bm = bmesh.from_edit_mesh(plane.data)
            bm.select_mode = {'FACE'}
            for f in bm.faces:
                f.select = True
            bmesh.update_edit_mesh(plane.data)

            bm.verts.ensure_lookup_table()
            plane_y = bm.verts[0].co.y

            with bpy.context.temp_override(**ctx):
                success, msg = execute_cube_cut(
                    bpy.context,
                    Vector((0.25, plane_y - 0.25, 0.25)),
                    Vector((0.75, plane_y - 0.25, 0.75)),
                    0.5,
                    Vector((1, 0, 0)),
                    Vector((0, 0, 1)),
                    Vector((0, 1, 0)),
                )
            assert success, f"Cube cut failed on {plane.name}: {msg}"

            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='OBJECT')

        yield 0.5

        # 3. Join both planes into one object
        plane_a.select_set(True)
        plane_b.select_set(True)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.join()
        obj = plane_a

        # 4. Enter edit mode, select hole boundary edges, and bridge
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)

        def is_hole_boundary(e):
            if not e.is_boundary:
                return False
            for v in e.verts:
                x, z = v.co.x, v.co.z
                x_on_hole = abs(x - 0.25) < 0.01 or abs(x - 0.75) < 0.01
                z_on_hole = abs(z - 0.25) < 0.01 or abs(z - 0.75) < 0.01
                if not (x_on_hole or z_on_hole):
                    return False
                if x < 0.24 or x > 0.76 or z < 0.24 or z > 0.76:
                    return False
            return True

        boundary_count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=is_hole_boundary,
        )
        self.assertEqual(boundary_count, 8,
                         f"Expected 8 hole boundary edges (4 per hole), got {boundary_count}")

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops()

        yield 0.5

        # 5. Select the top two edges of the corridor for beveling
        bm = bmesh.from_edit_mesh(obj.data)

        count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=lambda e: (
                all(abs(v.co.z - 0.75) < 0.05 for v in e.verts)
                and abs(e.verts[0].co.y - e.verts[1].co.y) > 0.1
            ),
        )

        self.assertEqual(count, 2,
                         f"Expected 2 top corridor edges selected, got {count}")

        # 6. Bevel the selected top edges into 1 segment
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bevel(
                offset=0.1,
                offset_pct=0,
                segments=1,
                affect='EDGES',
            )

        yield 0.5

        # No assertions - visual inspection only

    def test_bridge_and_bevel_corridor_2_segments(self):
        """Non-interactive bevel with 2 segments to test projection logic."""
        # 1-4: Same setup as 1-segment test
        plane_a = create_vertical_plane("corridor_2seg_api_a")
        plane_b = create_vertical_plane("corridor_2seg_api_b")

        ctx = _get_context_override()

        plane_b.select_set(True)
        plane_a.select_set(False)
        bpy.context.view_layer.objects.active = plane_b
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        plane_b.rotation_euler.z = math.pi
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(rotation=True)

        plane_b.location.y = 2.0
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(location=True)

        for plane in [plane_a, plane_b]:
            plane_a.select_set(plane is plane_a)
            plane_b.select_set(plane is plane_b)
            bpy.context.view_layer.objects.active = plane
            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='EDIT')

            bm = bmesh.from_edit_mesh(plane.data)
            bm.select_mode = {'FACE'}
            for f in bm.faces:
                f.select = True
            bmesh.update_edit_mesh(plane.data)

            bm.verts.ensure_lookup_table()
            plane_y = bm.verts[0].co.y

            with bpy.context.temp_override(**ctx):
                success, msg = execute_cube_cut(
                    bpy.context,
                    Vector((0.25, plane_y - 0.25, 0.25)),
                    Vector((0.75, plane_y - 0.25, 0.75)),
                    0.5,
                    Vector((1, 0, 0)),
                    Vector((0, 0, 1)),
                    Vector((0, 1, 0)),
                )
            assert success, f"Cube cut failed on {plane.name}: {msg}"

            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='OBJECT')

        yield 0.5

        plane_a.select_set(True)
        plane_b.select_set(True)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.join()
        obj = plane_a

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)

        def is_hole_boundary(e):
            if not e.is_boundary:
                return False
            for v in e.verts:
                x, z = v.co.x, v.co.z
                x_on_hole = abs(x - 0.25) < 0.01 or abs(x - 0.75) < 0.01
                z_on_hole = abs(z - 0.25) < 0.01 or abs(z - 0.75) < 0.01
                if not (x_on_hole or z_on_hole):
                    return False
                if x < 0.24 or x > 0.76 or z < 0.24 or z > 0.76:
                    return False
            return True

        boundary_count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=is_hole_boundary,
        )
        self.assertEqual(boundary_count, 8)

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops()

        yield 0.5

        # 5. Select top corridor edges and bevel with 2 segments
        bm = bmesh.from_edit_mesh(obj.data)
        count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=lambda e: (
                all(abs(v.co.z - 0.75) < 0.05 for v in e.verts)
                and abs(e.verts[0].co.y - e.verts[1].co.y) > 0.1
            ),
        )
        self.assertEqual(count, 2)

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bevel(
                offset=0.1,
                offset_pct=0,
                segments=2,
                affect='EDGES',
            )

        yield 0.5

        # 6. Dump face transforms
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv[0]
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        for face in bm.faces:
            n = face.normal
            c = face.calc_center_median()
            key = (round(n.x, 2), round(n.y, 2), round(n.z, 2),
                   round(c.x, 2), round(c.y, 2), round(c.z, 2))
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            print(f"FACE {key}: scale_u={t['scale_u']:.4f} scale_v={t['scale_v']:.4f} "
                  f"rot={t['rotation']:.2f} ox={t['offset_x']:.2f} oy={t['offset_y']:.2f}")

    def test_bridge_and_bevel_corridor_1_segment_interactive(self):
        """Same as 1-segment test but uses interactive Ctrl+B bevel modal."""
        # 1. Create two vertical planes facing each other, 2 units apart
        plane_a = create_vertical_plane("corridor_int_a")
        plane_b = create_vertical_plane("corridor_int_b")

        ctx = _get_context_override()

        plane_b.select_set(True)
        plane_a.select_set(False)
        bpy.context.view_layer.objects.active = plane_b
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        plane_b.rotation_euler.z = math.pi
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(rotation=True)

        plane_b.location.y = 2.0
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(location=True)

        # 2. Cut holes
        for plane in [plane_a, plane_b]:
            plane_a.select_set(plane is plane_a)
            plane_b.select_set(plane is plane_b)
            bpy.context.view_layer.objects.active = plane
            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='EDIT')

            bm = bmesh.from_edit_mesh(plane.data)
            bm.select_mode = {'FACE'}
            for f in bm.faces:
                f.select = True
            bmesh.update_edit_mesh(plane.data)

            bm.verts.ensure_lookup_table()
            plane_y = bm.verts[0].co.y

            with bpy.context.temp_override(**ctx):
                success, msg = execute_cube_cut(
                    bpy.context,
                    Vector((0.25, plane_y - 0.25, 0.25)),
                    Vector((0.75, plane_y - 0.25, 0.75)),
                    0.5,
                    Vector((1, 0, 0)),
                    Vector((0, 0, 1)),
                    Vector((0, 1, 0)),
                )
            assert success, f"Cube cut failed on {plane.name}: {msg}"

            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='OBJECT')

        yield 0.5

        # 3. Join
        plane_a.select_set(True)
        plane_b.select_set(True)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.join()
        obj = plane_a

        # 4. Bridge hole edges
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)

        def is_hole_boundary(e):
            if not e.is_boundary:
                return False
            for v in e.verts:
                x, z = v.co.x, v.co.z
                x_on_hole = abs(x - 0.25) < 0.01 or abs(x - 0.75) < 0.01
                z_on_hole = abs(z - 0.25) < 0.01 or abs(z - 0.75) < 0.01
                if not (x_on_hole or z_on_hole):
                    return False
                if x < 0.24 or x > 0.76 or z < 0.24 or z > 0.76:
                    return False
            return True

        boundary_count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=is_hole_boundary,
        )
        self.assertEqual(boundary_count, 8)

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops()

        yield 0.5

        # 5. Select the top two corridor edges
        bm = bmesh.from_edit_mesh(obj.data)
        count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=lambda e: (
                all(abs(v.co.z - 0.75) < 0.05 for v in e.verts)
                and abs(e.verts[0].co.y - e.verts[1].co.y) > 0.1
            ),
        )
        self.assertEqual(count, 2)

        # 6. Interactive bevel: Ctrl+B, move mouse to set offset, type 0.1, Enter
        window = bpy.context.window or bpy.context.window_manager.windows[0]
        mx, my = self._get_3d_viewport_center()

        # Ctrl+B to start bevel modal
        window.event_simulate(type='B', value='PRESS', x=mx, y=my, ctrl=True)
        yield
        window.event_simulate(type='B', value='RELEASE', x=mx, y=my)
        yield

        # Mouse move to kick the modal into tracking
        window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=mx, y=my)
        yield

        # Move mouse a bit to simulate dragging (triggers intermediate depsgraph updates)
        for i in range(5):
            window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=mx + (i + 1) * 10, y=my)
            yield

        # Type exact value and confirm
        yield from self._simulate_number(0.1)
        yield from self._simulate_key_tap('RET')
        yield 0.5

        # 7. Dump face transforms
        self._dump_face_transforms(obj)

    def test_bridge_and_bevel_corridor_interactive_2_segments(self):
        """Interactive Ctrl+B bevel with 2 segments (scroll wheel to add segment)."""
        # 1. Create two vertical planes facing each other, 2 units apart
        plane_a = create_vertical_plane("corridor_2seg_a")
        plane_b = create_vertical_plane("corridor_2seg_b")

        ctx = _get_context_override()

        plane_b.select_set(True)
        plane_a.select_set(False)
        bpy.context.view_layer.objects.active = plane_b
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        plane_b.rotation_euler.z = math.pi
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(rotation=True)

        plane_b.location.y = 2.0
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(location=True)

        # 2. Cut holes
        for plane in [plane_a, plane_b]:
            plane_a.select_set(plane is plane_a)
            plane_b.select_set(plane is plane_b)
            bpy.context.view_layer.objects.active = plane
            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='EDIT')

            bm = bmesh.from_edit_mesh(plane.data)
            bm.select_mode = {'FACE'}
            for f in bm.faces:
                f.select = True
            bmesh.update_edit_mesh(plane.data)

            bm.verts.ensure_lookup_table()
            plane_y = bm.verts[0].co.y

            with bpy.context.temp_override(**ctx):
                success, msg = execute_cube_cut(
                    bpy.context,
                    Vector((0.25, plane_y - 0.25, 0.25)),
                    Vector((0.75, plane_y - 0.25, 0.75)),
                    0.5,
                    Vector((1, 0, 0)),
                    Vector((0, 0, 1)),
                    Vector((0, 1, 0)),
                )
            assert success, f"Cube cut failed on {plane.name}: {msg}"

            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='OBJECT')

        yield 0.5

        # 3. Join
        plane_a.select_set(True)
        plane_b.select_set(True)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.join()
        obj = plane_a

        # 4. Bridge hole edges
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)

        def is_hole_boundary(e):
            if not e.is_boundary:
                return False
            for v in e.verts:
                x, z = v.co.x, v.co.z
                x_on_hole = abs(x - 0.25) < 0.01 or abs(x - 0.75) < 0.01
                z_on_hole = abs(z - 0.25) < 0.01 or abs(z - 0.75) < 0.01
                if not (x_on_hole or z_on_hole):
                    return False
                if x < 0.24 or x > 0.76 or z < 0.24 or z > 0.76:
                    return False
            return True

        boundary_count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=is_hole_boundary,
        )
        self.assertEqual(boundary_count, 8)

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops()

        yield 0.5

        # 5. Select the top two corridor edges
        bm = bmesh.from_edit_mesh(obj.data)
        count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=lambda e: (
                all(abs(v.co.z - 0.75) < 0.05 for v in e.verts)
                and abs(e.verts[0].co.y - e.verts[1].co.y) > 0.1
            ),
        )
        self.assertEqual(count, 2)

        # 6. Interactive bevel with 2 segments:
        #    Ctrl+B, small drag, scroll wheel up (add segment), drag more,
        #    type 0.1, Enter — mimics real user flow
        window = bpy.context.window or bpy.context.window_manager.windows[0]
        mx, my = self._get_3d_viewport_center()

        # Ctrl+B to start bevel modal
        window.event_simulate(type='B', value='PRESS', x=mx, y=my, ctrl=True)
        yield
        window.event_simulate(type='B', value='RELEASE', x=mx, y=my)
        yield

        # Mouse move to kick the modal into tracking
        window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=mx, y=my)
        yield

        # Small initial drag to establish the bevel
        window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=mx + 10, y=my)
        yield
        window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=mx + 20, y=my)
        yield

        # Scroll wheel up to add a segment (1 -> 2) — this changes topology
        window.event_simulate(type='WHEELUPMOUSE', value='PRESS', x=mx + 20, y=my)
        yield
        window.event_simulate(type='WHEELUPMOUSE', value='RELEASE', x=mx + 20, y=my)
        yield 0.1

        # Continue dragging after segment change (triggers multiple depsgraph
        # updates with the new topology)
        for i in range(8):
            window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=mx + 20 + (i + 1) * 10, y=my)
            yield

        # Type exact value and confirm
        yield from self._simulate_number(0.1)
        yield from self._simulate_key_tap('RET')
        yield 0.5

        # 7. Dump face transforms
        self._dump_face_transforms(obj)

    # --- Helpers ---

    def _setup_corridor(self, suffix):
        """Create two planes with holes, join, and bridge. Returns (obj, ctx)."""
        plane_a = create_vertical_plane(f"corr_{suffix}_a")
        plane_b = create_vertical_plane(f"corr_{suffix}_b")

        ctx = _get_context_override()

        plane_b.select_set(True)
        plane_a.select_set(False)
        bpy.context.view_layer.objects.active = plane_b
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        plane_b.rotation_euler.z = math.pi
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(rotation=True)

        plane_b.location.y = 2.0
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(location=True)

        for plane in [plane_a, plane_b]:
            plane_a.select_set(plane is plane_a)
            plane_b.select_set(plane is plane_b)
            bpy.context.view_layer.objects.active = plane
            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='EDIT')

            bm = bmesh.from_edit_mesh(plane.data)
            bm.select_mode = {'FACE'}
            for f in bm.faces:
                f.select = True
            bmesh.update_edit_mesh(plane.data)

            bm.verts.ensure_lookup_table()
            plane_y = bm.verts[0].co.y

            with bpy.context.temp_override(**ctx):
                success, msg = execute_cube_cut(
                    bpy.context,
                    Vector((0.25, plane_y - 0.25, 0.25)),
                    Vector((0.75, plane_y - 0.25, 0.75)),
                    0.5,
                    Vector((1, 0, 0)),
                    Vector((0, 0, 1)),
                    Vector((0, 1, 0)),
                )
            assert success, f"Cube cut failed on {plane.name}: {msg}"

            with bpy.context.temp_override(**ctx):
                bpy.ops.object.mode_set(mode='OBJECT')

        plane_a.select_set(True)
        plane_b.select_set(True)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.join()
        obj = plane_a

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)

        def is_hole_boundary(e):
            if not e.is_boundary:
                return False
            for v in e.verts:
                x, z = v.co.x, v.co.z
                x_on_hole = abs(x - 0.25) < 0.01 or abs(x - 0.75) < 0.01
                z_on_hole = abs(z - 0.25) < 0.01 or abs(z - 0.75) < 0.01
                if not (x_on_hole or z_on_hole):
                    return False
                if x < 0.24 or x > 0.76 or z < 0.24 or z > 0.76:
                    return False
            return True

        boundary_count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=is_hole_boundary,
        )
        assert boundary_count == 8

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops()

        return obj, ctx

    def _select_single_top_edge(self, obj):
        """Select only the RIGHT top corridor edge (x=0.75, z=0.75)."""
        bm = bmesh.from_edit_mesh(obj.data)
        count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=lambda e: (
                all(abs(v.co.z - 0.75) < 0.05 for v in e.verts)
                and all(abs(v.co.x - 0.75) < 0.05 for v in e.verts)
                and abs(e.verts[0].co.y - e.verts[1].co.y) > 0.1
            ),
        )
        assert count == 1, f"Expected 1 top-right corridor edge, got {count}"

    def _select_single_bottom_edge(self, obj):
        """Select only the LEFT bottom corridor edge (x=0.25, z=0.25)."""
        bm = bmesh.from_edit_mesh(obj.data)
        count = _select_edges_by_filter(
            bm, obj.data,
            edge_filter=lambda e: (
                all(abs(v.co.z - 0.25) < 0.05 for v in e.verts)
                and all(abs(v.co.x - 0.25) < 0.05 for v in e.verts)
                and abs(e.verts[0].co.y - e.verts[1].co.y) > 0.1
            ),
        )
        assert count == 1, f"Expected 1 bottom-left corridor edge, got {count}"

    def _dump_face_transforms(self, obj):
        """Print face transforms for visual inspection."""
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv[0]
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        for face in bm.faces:
            n = face.normal
            c = face.calc_center_median()
            key = (round(n.x, 2), round(n.y, 2), round(n.z, 2),
                   round(c.x, 2), round(c.y, 2), round(c.z, 2))
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            print(f"FACE {key}: scale_u={t['scale_u']:.4f} scale_v={t['scale_v']:.4f} "
                  f"rot={t['rotation']:.2f} ox={t['offset_x']:.2f} oy={t['offset_y']:.2f}")

    # --- Single-edge bevel tests ---

    def test_single_edge_bevel_1_segment_interactive(self):
        """Interactive Ctrl+B bevel of ONE corridor edge, 1 segment.

        The non-interactive equivalent produces all scale=1.0.  The interactive
        modal triggers restore/re-apply cycles that can corrupt UVs.
        """
        obj, ctx = self._setup_corridor("1e1s_int")
        yield 0.5
        self._select_single_top_edge(obj)

        window = bpy.context.window or bpy.context.window_manager.windows[0]
        mx, my = self._get_3d_viewport_center()

        window.event_simulate(type='B', value='PRESS', x=mx, y=my, ctrl=True)
        yield
        window.event_simulate(type='B', value='RELEASE', x=mx, y=my)
        yield
        window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=mx, y=my)
        yield
        # Scroll down to force 1 segment (Blender remembers last-used count
        # which may have been raised by a prior test)
        for _ in range(5):
            window.event_simulate(type='WHEELDOWNMOUSE', value='PRESS', x=mx, y=my)
            yield
        for i in range(5):
            window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=mx + (i + 1) * 10, y=my)
            yield
        yield from self._simulate_number(0.1)
        yield from self._simulate_key_tap('RET')
        yield 0.5

        # Expected values from the non-interactive 1-segment single-edge test
        expected = {
            (0.0, -1.0, 0.0, 0.5, 0.0, 0.12):     (1.0, 1.0, 180.0, 0.75, 0.25),
            (0.0, -1.0, 0.0, 0.12, 0.0, 0.5):      (1.0, 1.0, 90.0, 0.25, 0.25),
            (-0.0, 1.0, 0.0, 0.12, 2.0, 0.5):      (1.0, 1.0, -90.0, 0.75, 0.75),
            (0.0, 1.0, -0.0, 0.5, 2.0, 0.12):      (1.0, 1.0, 180.0, 0.75, 0.25),
            (0.0, -0.0, 1.0, 0.5, 1.0, 0.25):      (1.0, 1.0, 0.0, 0.25, 0.25),
            (1.0, -0.0, 0.0, 0.25, 1.0, 0.5):      (1.0, 1.0, 90.0, 0.75, 0.25),
            (-0.0, -1.0, 0.0, 0.74, 0.0, 0.74):    (1.0, 1.0, -45.0, 0.65, 0.75),
            (0.0, 1.0, 0.0, 0.74, 2.0, 0.74):      (1.0, 1.0, 45.0, 0.25, 0.65),
            (-0.71, 0.0, -0.71, 0.7, 1.0, 0.7):    (1.0, 1.0, 180.0, 0.25, 0.79),
            (0.0, 0.0, -1.0, 0.45, 1.0, 0.75):     (1.0, 1.0, 180.0, 0.75, 0.75),
            (0.0, -1.0, 0.0, 0.86, 0.0, 0.54):     (1.0, 1.0, -112.5, 0.82, 0.82),
            (0.0, -1.0, 0.0, 0.54, 0.0, 0.86):     (1.0, 1.0, 0.0, 0.25, 0.75),
            (-1.0, 0.0, 0.0, 0.75, 1.0, 0.45):     (1.0, 1.0, -90.0, 0.25, 0.65),
            (0.0, 1.0, 0.0, 0.54, 2.0, 0.86):      (1.0, 1.0, -22.5, 0.18, 0.82),
            (0.0, 1.0, 0.0, 0.86, 2.0, 0.54):      (1.0, 1.0, 90.0, 0.25, 0.25),
        }

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv[0]
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        failed_faces = []
        errors = []
        for face in bm.faces:
            n = face.normal
            c = face.calc_center_median()
            key = (round(n.x, 2), round(n.y, 2), round(n.z, 2),
                   round(c.x, 2), round(c.y, 2), round(c.z, 2))
            if key not in expected:
                errors.append(f"Unexpected face {key}")
                failed_faces.append(face.index)
                continue
            t = derive_transform_from_uvs(face, uv_layer, ppm, obj.data)
            su, sv, rot, ox, oy = expected[key]
            label = f"n=({key[0]},{key[1]},{key[2]}) c=({key[3]},{key[4]},{key[5]})"
            face_ok = True
            if abs(t['scale_u'] - su) > 0.01:
                errors.append(f"{label} scale_u={t['scale_u']:.4f}, expected {su}")
                face_ok = False
            if abs(t['scale_v'] - sv) > 0.01:
                errors.append(f"{label} scale_v={t['scale_v']:.4f}, expected {sv}")
                face_ok = False
            if abs(t['rotation'] - rot) > 0.1:
                errors.append(f"{label} rotation={t['rotation']:.2f}, expected {rot}")
                face_ok = False
            if abs(t['offset_x'] - ox) > 0.06:
                errors.append(f"{label} offset_x={t['offset_x']:.2f}, expected {ox}")
                face_ok = False
            if abs(t['offset_y'] - oy) > 0.06:
                errors.append(f"{label} offset_y={t['offset_y']:.2f}, expected {oy}")
                face_ok = False
            if not face_ok:
                failed_faces.append(face.index)

        if failed_faces:
            bm.select_mode = {'FACE'}
            for v in bm.verts:
                v.select_set(False)
            for e in bm.edges:
                e.select_set(False)
            for f in bm.faces:
                f.select_set(False)
            bm.faces.ensure_lookup_table()
            for idx in failed_faces:
                bm.faces[idx].select_set(True)
            bm.select_flush_mode()
            bmesh.update_edit_mesh(obj.data)

        if errors:
            self.fail(f"[1-edge] {len(failed_faces)} face(s) with wrong UVs:\n"
                      + "\n".join(errors))

    def test_single_edge_bevel_3_segments_noninteractive(self):
        """Non-interactive bevel of ONE corridor edge (bottom), 3 segments.

        Generates reference transforms for the interactive 3-segment test.
        """
        obj, ctx = self._setup_corridor("1e3s_api")
        yield 0.5
        self._select_single_bottom_edge(obj)

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bevel(
                offset=0.1,
                offset_pct=0,
                segments=3,
                affect='EDGES',
            )

        yield 0.5
        self._dump_face_transforms(obj)

