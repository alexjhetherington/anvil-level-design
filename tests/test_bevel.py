import bmesh
import bpy
from mathutils import Vector

from .base_test import AnvilTestCase
from .helpers import create_vertical_plane, _get_context_override


def _select_edge_by_vert_filter(bm, me, vert_filter):
    """Select the edge whose both endpoints satisfy vert_filter."""
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


class BevelTest(AnvilTestCase):
    """Test beveling an edge at the corner of two walls."""

    def test_corner_bevel_8_segments(self):
        # 1. Create a vertical plane in XZ (facing +Y)
        obj = create_vertical_plane("bevel_corner")

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)

        # 2. Select the right edge (x=1) and extrude it along +Y to make a corner
        _select_edge_by_vert_filter(
            bm, obj.data,
            vert_filter=lambda v: abs(v.co.x - 1.0) < 1e-5,
        )

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={"value": Vector((0, 1, 0))}
            )
        yield 0.5

        # 3. Select the middle edge (the shared corner edge at x=1, y=0)
        bm = bmesh.from_edit_mesh(obj.data)
        _select_edge_by_vert_filter(
            bm, obj.data,
            vert_filter=lambda v: abs(v.co.x - 1.0) < 1e-5 and abs(v.co.y) < 1e-5,
        )

        # 4. Bevel with 8 segments
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bevel(
                offset=0.2,
                offset_pct=0,
                segments=8,
                affect='EDGES',
            )
        yield 0.5
