import os

import bmesh
import bpy

from .base_test import AnvilTestCase
from .helpers import _get_context_override
from ..core.materials import find_material_with_image, create_material_with_image
from ..hotspot_mapping.json_storage import (
    add_texture_as_hotspottable,
    add_line,
    load_hotspots,
    get_texture_hotspots,
)

HOTSPOT_TEXTURE_PATH = os.path.join(os.path.dirname(__file__), "dev_hotspot.png")


def _setup_hotspot_map():
    """Register dev_hotspot.png as hotspottable and add bisecting lines.

    Creates a 2x2 grid on the 2048x2048 image:
    - Horizontal line at y=512 (512px from top), spanning full width
    - Vertical line at x=1536 (512px from right), spanning full height
    """
    image = bpy.data.images.load(HOTSPOT_TEXTURE_PATH, check_existing=True)
    w, h = image.size[0], image.size[1]
    add_texture_as_hotspottable(image.name, w, h)
    add_line(image.name, "h", 512, 0, w)
    add_line(image.name, "v", w - 512, 0, h)


def _create_hotspot_cube(name):
    """Create a 1x1x1 cube textured with dev_hotspot.png.

    Returns the object in object mode.
    """
    image = bpy.data.images.load(HOTSPOT_TEXTURE_PATH, check_existing=True)

    mat = find_material_with_image(image)
    if not mat:
        mat = create_material_with_image(image)

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x += 0.5
        v.co.y += 0.5
        v.co.z += 0.5
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    obj.data.materials.append(mat)
    mat_index = obj.data.materials.find(mat.name)

    ppm = bpy.context.scene.level_design_props.pixels_per_meter

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()

    for face in bm.faces:
        face.material_index = mat_index
        # Simple planar UV: each face gets 0-1 range
        loops = list(face.loops)
        loops[0][uv_layer].uv = (0.0, 0.0)
        loops[1][uv_layer].uv = (1.0, 0.0)
        loops[2][uv_layer].uv = (1.0, 1.0)
        loops[3][uv_layer].uv = (0.0, 1.0)

    bmesh.update_edit_mesh(obj.data)
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')

    return obj


class HotspotApplyTest(AnvilTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _setup_hotspot_map()

    def test_apply_hotspot_to_cube_remaps_uvs(self):
        """Apply hotspots to a cube and verify UVs are remapped into hotspot cells."""
        obj = _create_hotspot_cube("hotspot_cube")

        # Verify hotspot map is set up correctly: 4 cells in a 2x2 grid
        image = bpy.data.images.get("dev_hotspot.png")
        self.assertIsNotNone(image, "dev_hotspot.png should be loaded")
        hotspots = get_texture_hotspots(image.name)
        self.assertEqual(len(hotspots), 4, "Should have 4 hotspot cells from 2x2 grid")

        # Apply hotspots in object mode
        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            result = bpy.ops.leveldesign.apply_hotspot()
        self.assertEqual(result, {'FINISHED'})

        # Verify UVs have been remapped: each face's UVs should now fall
        # within one of the four hotspot cells (not spanning the full 0-1 range)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()

        for face in bm.faces:
            us = [loop[uv_layer].uv.x for loop in face.loops]
            vs = [loop[uv_layer].uv.y for loop in face.loops]
            u_min, u_max = min(us), max(us)
            v_min, v_max = min(vs), max(vs)

            # Each face should be confined to a hotspot cell, not spanning
            # the full 0-1 UV range. The largest cell is 1536/2048 = 0.75.
            u_span = u_max - u_min
            v_span = v_max - v_min
            self.assertLessEqual(
                u_span, 0.76,
                f"Face UV u-span {u_span:.3f} should fit within a hotspot cell"
            )
            self.assertLessEqual(
                v_span, 0.76,
                f"Face UV v-span {v_span:.3f} should fit within a hotspot cell"
            )

        bm.free()
