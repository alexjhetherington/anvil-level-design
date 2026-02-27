import bmesh
import bpy
from mathutils import Vector

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


def _setup_extrude(name, vert_filter, direction):
    """Shared setup: create plane, extrude edge, let depsgraph handler fire, read UVs.

    Uses the real addon flow via bpy.ops operators:
    1. Create textured plane, enter edit mode (depsgraph handler seeds cache)
    2. Select edge, extrude via bpy.ops.mesh.extrude_region_move()
    3. The operator triggers depsgraph → handler detects topology change → projects UVs
    4. Derive transforms from both faces

    Returns (object, original_face_transform, new_face_transform).
    """
    obj = create_vertical_plane(name)

    ppm = bpy.context.scene.level_design_props.pixels_per_meter
    ctx = _get_context_override()

    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)

    _select_edge_by_vert_filter(bm, obj.data, vert_filter)

    # Extrude via high-level operator — this triggers depsgraph updates naturally
    with bpy.context.temp_override(**ctx):
        bpy.ops.mesh.extrude_region_move(
            TRANSFORM_OT_translate={"value": direction}
        )

    # Re-fetch bmesh after operator (operator may have invalidated it)
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()

    # Derive transforms from both faces
    bm.faces.ensure_lookup_table()
    original_face = bm.faces[0]
    new_face = bm.faces[1]
    original_transform = derive_transform_from_uvs(
        original_face, uv_layer, ppm, obj.data)
    new_transform = derive_transform_from_uvs(
        new_face, uv_layer, ppm, obj.data)

    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')

    return obj, original_transform, new_transform


class UVExtendTest(AnvilTestCase):

    def _assert_transform(self, transform, scale_u, scale_v, rotation,
                          offset_x, offset_y):
        self.assertAlmostEqual(transform['scale_u'], scale_u, places=3)
        self.assertAlmostEqual(transform['scale_v'], scale_v, places=3)
        self.assertAlmostEqual(transform['rotation'], rotation, places=3)
        self.assertAlmostEqual(transform['offset_x'], offset_x, places=3)
        self.assertAlmostEqual(transform['offset_y'], offset_y, places=3)

    def test_extend_up(self):
        obj, orig, new = _setup_extrude(
            "extend_up",
            vert_filter=lambda v: abs(v.co.z - 1.0) < 1e-5,
            direction=Vector((0, 0, 1)),
        )
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 0.0, 0.0, 0.0)

    def test_extend_down(self):
        obj, orig, new = _setup_extrude(
            "extend_down",
            vert_filter=lambda v: abs(v.co.z) < 1e-5,
            direction=Vector((0, 0, -1)),
        )
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 180.0, 0.0, 0.0)

    def test_extend_left(self):
        obj, orig, new = _setup_extrude(
            "extend_left",
            vert_filter=lambda v: abs(v.co.x) < 1e-5,
            direction=Vector((-1, 0, 0)),
        )
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 90.0, 0.0, 0.0)

    def test_extend_right(self):
        obj, orig, new = _setup_extrude(
            "extend_right",
            vert_filter=lambda v: abs(v.co.x - 1.0) < 1e-5,
            direction=Vector((1, 0, 0)),
        )
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, -90.0, 0.0, 0.0)
