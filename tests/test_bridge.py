import bmesh
import bpy

from ..utils import derive_transform_from_uvs
from .base_test import AnvilTestCase
from .helpers import create_vertical_plane, _get_context_override


class BridgeEdgeLoopsTest(AnvilTestCase):
    """Test that bridge_edge_loops produces correctly UV-projected faces."""

    def test_bridge_two_planes(self):
        # Create two vertical planes 1 unit apart facing away from each other.
        # create_vertical_plane gives +Y normal. Flip plane A to face -Y.
        # Result: A at Y=0 facing -Y (outward), B at Y=1 facing +Y (outward).
        plane_a = create_vertical_plane("bridge_a")
        plane_b = create_vertical_plane("bridge_b")

        ctx = _get_context_override()

        # Move plane B to Y=1
        plane_b.location.y = 1.0
        with bpy.context.temp_override(**ctx):
            bpy.context.view_layer.objects.active = plane_b
            bpy.ops.object.transform_apply(location=True)

        # Flip plane A's normal so it faces -Y (away from plane B)
        plane_a.select_set(True)
        plane_b.select_set(False)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.flip_normals()
            bpy.ops.object.mode_set(mode='OBJECT')

        # Join both planes into one object
        plane_a.select_set(True)
        plane_b.select_set(True)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.join()
        obj = plane_a

        # Enter edit mode and select all edges
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'EDGE'}
        for e in bm.edges:
            e.select_set(True)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

        # Bridge edge loops
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops()

        # Let depsgraph handler fire
        yield 0.5

        # Read transforms from all faces
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv[0]
        bm.faces.ensure_lookup_table()

        transforms = {}
        for face in bm.faces:
            n = face.normal
            c = face.calc_center_median()
            key = (
                round(n.x, 2), round(n.y, 2), round(n.z, 2),
                round(c.x, 2), round(c.y, 2), round(c.z, 2),
            )
            transforms[key] = derive_transform_from_uvs(
                face, uv_layer, ppm, obj.data)

        # 6 faces total: 2 original (along Y) + 4 bridged (along X/Z)
        self.assertEqual(len(transforms), 6,
                         f"Expected 6 faces, got {len(transforms)}: "
                         f"{list(transforms.keys())}")

        # The 4 bridged side faces have normals not along Y
        bridged = {k: v for k, v in transforms.items()
                   if abs(k[1]) < 0.5}
        self.assertEqual(len(bridged), 4,
                         f"Expected 4 bridged faces, got {len(bridged)}")

        # Expected bridged faces (normal, centroid):
        expected = {
            (-1.0, 0.0, 0.0, 1.0, 0.5, 0.5): {
                'scale_u': 1.0, 'scale_v': 1.0,
                'rotation': -90.0, 'offset_x': 0.0, 'offset_y': 0.0,
            },
            (1.0, -0.0, 0.0, 0.0, 0.5, 0.5): {
                'scale_u': 1.0, 'scale_v': 1.0,
                'rotation': 90.0, 'offset_x': 0.0, 'offset_y': 0.0,
            },
            (0.0, 0.0, -1.0, 0.5, 0.5, 1.0): {
                'scale_u': 1.0, 'scale_v': 1.0,
                'rotation': 0.0, 'offset_x': 0.0, 'offset_y': 0.0,
            },
            (0.0, -0.0, 1.0, 0.5, 0.5, 0.0): {
                'scale_u': 1.0, 'scale_v': 1.0,
                'rotation': 180.0, 'offset_x': 0.0, 'offset_y': 0.0,
            },
        }

        for face_key, expected_t in expected.items():
            self.assertIn(face_key, bridged,
                          f"Missing bridged face with key {face_key}. "
                          f"Available: {list(bridged.keys())}")
            actual = bridged[face_key]
            for prop in ('scale_u', 'scale_v', 'rotation',
                         'offset_x', 'offset_y'):
                self.assertAlmostEqual(
                    actual[prop], expected_t[prop], places=2,
                    msg=f"Face {face_key}: {prop} = {actual[prop]}, "
                        f"expected {expected_t[prop]}",
                )
