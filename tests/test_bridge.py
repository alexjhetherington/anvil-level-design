import bmesh
import bpy

from ..core.uv_projection import derive_transform_from_uvs
from .base_test import AnvilTestCase
from .helpers import create_vertical_plane, _get_context_override


class BridgeEdgeLoopsTest(AnvilTestCase):
    """Test that bridge_edge_loops produces correctly UV-projected faces."""

    def test_bridge_two_planes(self):
        # Create two vertical planes 1 unit apart facing away from each other.
        # A at Y=0 facing -Y (outward), B at Y=1 facing +Y (outward).
        plane_a = create_vertical_plane("bridge_a")
        plane_b = create_vertical_plane("bridge_b")

        ctx = _get_context_override()

        # Move origin to geometry center, rotate 180° around Z so it faces -Y
        import math
        plane_a.select_set(True)
        plane_b.select_set(False)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        plane_a.rotation_euler.z = math.pi
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(rotation=True)

        # Move plane B to Y=1
        plane_b.location.y = 1.0
        with bpy.context.temp_override(**ctx):
            bpy.context.view_layer.objects.active = plane_b
            bpy.ops.object.transform_apply(location=True)

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


class BridgeUndoRedoTest(AnvilTestCase):
    """Test that undoing and redoing bridge_edge_loops works correctly.

    Verifies that after undo the mesh reverts to the pre-bridge state (2 faces),
    and after re-bridging the UV-projected faces are identical to the first bridge.
    """

    def _setup_joined_planes(self):
        """Create two vertical planes, rotate A to face -Y, move B to Y=1, join."""
        import math

        plane_a = create_vertical_plane("bridge_undo_a")
        plane_b = create_vertical_plane("bridge_undo_b")

        ctx = _get_context_override()

        plane_a.select_set(True)
        plane_b.select_set(False)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        plane_a.rotation_euler.z = math.pi
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(rotation=True)

        plane_b.location.y = 1.0
        with bpy.context.temp_override(**ctx):
            bpy.context.view_layer.objects.active = plane_b
            bpy.ops.object.transform_apply(location=True)

        plane_a.select_set(True)
        plane_b.select_set(True)
        bpy.context.view_layer.objects.active = plane_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.join()

        return plane_a.name  # Return name, not reference (survives undo)

    def _select_all_edges_and_bridge(self, obj):
        """Enter edit mode, select all edges, and bridge edge loops."""
        ctx = _get_context_override()

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'EDGE'}
        for e in bm.edges:
            e.select_set(True)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops()

    def _read_face_transforms(self, obj):
        """Read UV transforms from all faces. Returns dict keyed by (normal, centroid)."""
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
        return transforms

    def _undo_ctx(self):
        """Build a full context override for ed.undo (needs window + screen)."""
        window = bpy.context.window or bpy.context.window_manager.windows[0]
        screen = window.screen
        area = next(a for a in screen.areas if a.type == 'VIEW_3D')
        region = next(r for r in area.regions if r.type == 'WINDOW')
        return {"window": window, "screen": screen, "area": area, "region": region}

    def _select_all_edges(self, obj):
        """Set edge select mode and select all edges."""
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'EDGE'}
        for e in bm.edges:
            e.select_set(True)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

    def test_bridge_undo_and_rebridge_preserves_uvs(self):
        """Replicate the bug: bridge → undo → bridge again → UVs stretched.

        Real-world workflow:
        1. Select edges, bridge edge loops (correct UVs)
        2. Undo the bridge (edges still selected from undo state)
        3. Bridge again WITHOUT re-selecting → UVs are stretched (BUG)

        The edges should still be selected after undoing the bridge, so
        re-bridging should produce identical results.
        """
        obj_name = self._setup_joined_planes()
        obj = bpy.data.objects[obj_name]
        ctx = _get_context_override()
        undo_ctx = self._undo_ctx()

        # Enter edit mode
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        # --- Undo step: before edge selection ---
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="Before edge selection")

        # Select all edges
        self._select_all_edges(obj)

        # --- Undo step: edges selected (pre-bridge) ---
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="Edges selected")

        # Bridge edge loops
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops()

        # --- Undo step: post-bridge ---
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="Post-bridge")

        # Let depsgraph handler fire
        yield 0.5

        # Verify 6 faces after bridge and record the correct transforms
        first_transforms = self._read_face_transforms(obj)
        self.assertEqual(len(first_transforms), 6,
                         f"Expected 6 faces after bridge, got {len(first_transforms)}")

        # --- Undo the bridge (back to "Edges selected" state) ---
        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo()
        yield 0.5

        # Re-acquire references (invalidated by undo)
        obj = bpy.data.objects[obj_name]
        bpy.context.view_layer.objects.active = obj

        # Should be back to 2 faces with edges still selected
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        self.assertEqual(len(bm.faces), 2,
                         f"Expected 2 faces after undo, got {len(bm.faces)}")

        # --- Bridge again WITHOUT re-selecting edges ---
        # This is the real user workflow: undo bridge, then bridge again.
        # The edges should still be selected from the undo state.
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops()

        with bpy.context.temp_override(**undo_ctx):
            bpy.ops.ed.undo_push(message="Post-rebridge")

        # Let depsgraph handler fire
        yield 0.5

        # Re-acquire references
        obj = bpy.data.objects[obj_name]

        # Verify we get the same 6 faces with correct UVs (not stretched)
        second_transforms = self._read_face_transforms(obj)
        self.assertEqual(len(second_transforms), 6,
                         f"Expected 6 faces after re-bridge, got {len(second_transforms)}")

        for face_key, first_t in first_transforms.items():
            self.assertIn(face_key, second_transforms,
                          f"Missing face {face_key} after re-bridge. "
                          f"Available: {list(second_transforms.keys())}")
            second_t = second_transforms[face_key]
            for prop in ('scale_u', 'scale_v', 'rotation',
                         'offset_x', 'offset_y'):
                self.assertAlmostEqual(
                    first_t[prop], second_t[prop], places=2,
                    msg=f"Face {face_key}: {prop} = {second_t[prop]} after re-bridge, "
                        f"was {first_t[prop]} after first bridge",
                )
