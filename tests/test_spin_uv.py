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

from ..core.uv_projection import apply_uv_to_face
from .base_test import AnvilTestCase
from .helpers import create_vertical_plane, _get_context_override
from .test_uv_extend import (
    _setup_cube_and_select_top_face,
    _read_all_face_transforms,
)


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


class SpinPreservesUVScaleTest(AnvilTestCase):
    """Spinning a selected face should preserve the face's own UV scale on
    the end cap. Side walls created by the spin get scale 2 from neighbors;
    the end cap (the rotated copy of the originally-selected face) should
    keep scale 3.
    """

    def test_cube_spin_preserves_scale(self):
        obj = _setup_cube_and_select_top_face("spin_cube", 2.0, 2.0)

        # Select the +Y side face instead of the top; re-UV it to scale 3.0
        # on layer 0. Spin axis is the cube's vertical edge at x=1, y=1 —
        # two verts of the selected face lie on the axis (same shape as
        # test_spin_axis_on_selection_edge_removes_degenerate_walls). The
        # x=1 edge (not x=0) is chosen so Blender's CW spin sweeps the face
        # outward in +Y instead of back into the cube's interior.
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select_set(False)
        target = None
        for f in bm.faces:
            if f.normal.y > 0.9:
                f.select_set(True)
                target = f
                break
        bm.select_flush_mode()

        uv_layer_0 = bm.loops.layers.uv[0]
        mat = obj.data.materials[0]
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        apply_uv_to_face(target, uv_layer_0, 3.0, 3.0, 0.0, 0.0, 0.0,
                         mat, ppm, obj.data)
        bmesh.update_edit_mesh(obj.data)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.spin(
                'EXEC_DEFAULT',
                True,
                steps=4,
                dupli=False,
                angle=math.radians(90),
                use_auto_merge=True,
                use_normal_flip=False,
                center=(1.0, 1.0, 0.0),
                axis=(0.0, 0.0, 1.0),
            )
        yield 0.5

        transforms = _read_all_face_transforms(obj, 0)
        scale_3 = [t for t in transforms
                   if abs(t['scale_u'] - 3.0) < 0.001
                   and abs(t['scale_v'] - 3.0) < 0.001]
        self.assertEqual(
            len(scale_3), 1,
            "Expected exactly one face at scale 3.0 (the spin end cap)")
