"""Test that spin's degenerate axis-wall faces are cleaned up.

When the spin axis coincides with an edge of the selected face, spin emits
zero-area "wall" quads at the axis: all four loops end up referencing the two
merged axis verts once use_auto_merge welds the duplicates. Blender's spin
op does not remove these degenerate faces — the depsgraph handler does, by
detecting MESH_OT_spin as the active operator after a topology change and
deleting any zero-area faces.
"""

import math

import bmesh
import bpy

from .base_test import AnvilTestCase
from .helpers import create_vertical_plane, _get_context_override


def _select_all_faces(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.select_mode = {'FACE'}
    for f in bm.faces:
        f.select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)


def _zero_area_face_indices(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return [f.index for f in bm.faces if f.calc_area() < 1e-8]


class SpinDegenerateWallCleanupTest(AnvilTestCase):

    def test_spin_axis_on_selection_edge_removes_degenerate_walls(self):
        obj = create_vertical_plane("spin_axis_on_edge")

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        _select_all_faces(obj)

        # Vertices (0,0,0) and (0,0,1) of the plane lie on the Z axis, so
        # each spin step emits a zero-area axis-wall quad.
        # The 'undo=True' positional (2nd arg after execution_context) is
        # required. Per Blender's bpy.ops docs, Python operator calls bypass
        # undo by default — which also skips populating active_operator, which
        # our depsgraph handler uses to detect spin and clean up axis-wall
        # degeneracies. Omitting this flag makes the cleanup path unreachable
        # from tests (production path via the spin gizmo always pushes undo).
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.spin(
                'EXEC_DEFAULT',
                True,
                steps=4,
                dupli=False,
                angle=math.radians(90),
                use_auto_merge=True,
                use_normal_flip=False,
                center=(0.0, 0.0, 0.0),
                axis=(0.0, 0.0, 1.0),
            )
        yield 0.5

        zero_area = _zero_area_face_indices(obj)
        self.assertEqual(
            zero_area, [],
            msg=(
                f"Spin produced {len(zero_area)} zero-area axis-wall face(s) "
                f"that were not cleaned up: indices={zero_area}"
            ),
        )
