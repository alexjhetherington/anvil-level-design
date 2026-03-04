import bmesh
import bpy

from ..utils import derive_transform_from_uvs
from .base_test import AnvilTestCase
from .helpers import create_vertical_plane, _get_context_override


def _select_edge_by_vert_filter(bm, me, vert_filter):
    """Select the edge whose both endpoints satisfy vert_filter.

    Flushes selection and updates the edit mesh so the operator poll sees it.
    """
    bm.select_mode = {'EDGE'}
    for v in bm.verts:
        v.select_set(False)
    for e in bm.edges:
        e.select_set(False)
    for f in bm.faces:
        f.select_set(False)

    bm.edges.ensure_lookup_table()
    for edge in bm.edges:
        if all(vert_filter(v) for v in edge.verts):
            edge.select_set(True)
            bm.select_flush_mode()
            bmesh.update_edit_mesh(me)
            return
    raise RuntimeError("No matching edge found")


class UVExtendTest(AnvilTestCase):

    def _setup_plane_and_select_edge(self, name, vert_filter):
        """Create a textured plane, enter edit mode, and select the target edge.

        Returns the object (in edit mode with edge selected).
        """
        obj = create_vertical_plane(name)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)

        _select_edge_by_vert_filter(bm, obj.data, vert_filter)
        return obj

    def _read_transforms(self, obj):
        """Read UV transforms from both faces (original and new).

        Returns (original_transform, new_transform).
        """
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        original_face = bm.faces[0]
        new_face = bm.faces[1]

        original_transform = derive_transform_from_uvs(
            original_face, uv_layer, ppm, obj.data)
        new_transform = derive_transform_from_uvs(
            new_face, uv_layer, ppm, obj.data)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

        return original_transform, new_transform

    def _assert_transform(self, transform, scale_u, scale_v, rotation,
                          offset_x, offset_y):
        self.assertAlmostEqual(transform['scale_u'], scale_u, places=3)
        self.assertAlmostEqual(transform['scale_v'], scale_v, places=3)
        self.assertAlmostEqual(transform['rotation'], rotation, places=3)
        self.assertAlmostEqual(transform['offset_x'], offset_x, places=3)
        self.assertAlmostEqual(transform['offset_y'], offset_y, places=3)

    def test_extend_up(self):
        obj = self._setup_plane_and_select_edge(
            "extend_up",
            vert_filter=lambda v: abs(v.co.z - 1.0) < 1e-5,
        )
        yield from self.simulate_extrude(axis='Z', value=1)
        orig, new = self._read_transforms(obj)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 0.0, 0.0, 0.0)

    def test_extend_down(self):
        obj = self._setup_plane_and_select_edge(
            "extend_down",
            vert_filter=lambda v: abs(v.co.z) < 1e-5,
        )
        yield from self.simulate_extrude(axis='Z', value=-1)
        orig, new = self._read_transforms(obj)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 180.0, 0.0, 0.0)

    def test_extend_left(self):
        obj = self._setup_plane_and_select_edge(
            "extend_left",
            vert_filter=lambda v: abs(v.co.x) < 1e-5,
        )
        yield from self.simulate_extrude(axis='X', value=-1)
        orig, new = self._read_transforms(obj)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 90.0, 0.0, 0.0)

    def test_extend_right(self):
        obj = self._setup_plane_and_select_edge(
            "extend_right",
            vert_filter=lambda v: abs(v.co.x - 1.0) < 1e-5,
        )
        yield from self.simulate_extrude(axis='X', value=1)
        orig, new = self._read_transforms(obj)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, -90.0, 0.0, 0.0)
