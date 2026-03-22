import bmesh
import bpy
import math
from mathutils import Vector

from ..operators.cube_cut.geometry import execute_cube_cut
from ..utils import derive_transform_from_uvs
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

    def test_bridge_and_bevel_corridor(self):
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
            self.fail(f"{len(failed_faces)} face(s) with wrong UVs:\n"
                      + "\n".join(errors))
