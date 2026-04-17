import bmesh
import bpy

from ..core.uv_projection import derive_transform_from_uvs
from .base_test import AnvilTestCase, _get_window
from .helpers import create_textured_cube, _get_context_override


def _face_key(face):
    """Stable identifier for a face based on its normal (cube faces are axis-aligned)."""
    n = face.normal
    return (round(n.x), round(n.y), round(n.z))


def _snapshot_all_face_transforms(bm, uv_layer, ppm, me):
    out = {}
    for face in bm.faces:
        out[_face_key(face)] = derive_transform_from_uvs(face, uv_layer, ppm, me)
    return out


class WorldScaleUVRestoreTest(AnvilTestCase):
    """Cancelling a transform modal with ESC must restore UVs to their pre-modal state."""

    def test_grab_modal_escape_cancel_restores_uvs(self):
        obj = create_textured_cube("cube", scale_u=1.0, scale_v=1.0)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select_set(False)

        bm.faces.ensure_lookup_table()
        target = next(f for f in bm.faces if f.normal.y > 0.9)
        target.select_set(True)
        bmesh.update_edit_mesh(obj.data)

        yield 0.3

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv[0]
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        me = obj.data

        orig_verts_by_key = {
            _face_key(f): [v.co.copy() for v in f.verts] for f in bm.faces
        }
        orig_transforms = _snapshot_all_face_transforms(bm, uv_layer, ppm, me)

        window = _get_window()
        mx, my = self._get_3d_viewport_center()

        yield from self._simulate_key_tap('G')
        window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=mx, y=my)
        yield 0.05
        yield from self._simulate_key_tap('X')
        yield from self._simulate_number(2)
        yield 0.2

        window.event_simulate(type='ESC', value='PRESS', x=mx, y=my)
        yield
        window.event_simulate(type='ESC', value='RELEASE', x=mx, y=my)
        yield 0.3

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        for f in bm.faces:
            key = _face_key(f)
            orig_verts = orig_verts_by_key[key]
            for orig_v, v in zip(orig_verts, f.verts):
                self.assertLess(
                    (v.co - orig_v).length, 1e-4,
                    f"Geometry for face {key} did not revert after ESC; "
                    f"cannot verify UV restore",
                )

        final_transforms = _snapshot_all_face_transforms(bm, uv_layer, ppm, me)

        for key, orig_t in orig_transforms.items():
            final_t = final_transforms[key]
            self.assertAlmostEqual(
                final_t['scale_u'], orig_t['scale_u'], places=3,
                msg=f"face {key} scale_u not restored: orig={orig_t} final={final_t}")
            self.assertAlmostEqual(
                final_t['scale_v'], orig_t['scale_v'], places=3,
                msg=f"face {key} scale_v not restored: orig={orig_t} final={final_t}")
            self.assertAlmostEqual(
                final_t['rotation'], orig_t['rotation'], places=2,
                msg=f"face {key} rotation not restored: orig={orig_t} final={final_t}")
            self.assertAlmostEqual(
                final_t['offset_x'], orig_t['offset_x'], places=3,
                msg=f"face {key} offset_x not restored: orig={orig_t} final={final_t}")
            self.assertAlmostEqual(
                final_t['offset_y'], orig_t['offset_y'], places=3,
                msg=f"face {key} offset_y not restored: orig={orig_t} final={final_t}")
