import math
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

    Layout on the 1024x1024 image:
    - Horizontal lines at y=128, 384, 704 (from top), full width
    - Vertical line at x=512 spanning y=384..1024 (splits the bottom
      two strips into left/right halves)
    """
    image = bpy.data.images.load(HOTSPOT_TEXTURE_PATH, check_existing=True)
    w, h = image.size[0], image.size[1]
    add_texture_as_hotspottable(image.name, w, h)
    add_line(image.name, "h", 128, 0, w)
    add_line(image.name, "h", 384, 0, w)
    add_line(image.name, "h", 704, 0, w)
    add_line(image.name, "v", w // 2, 384, h)


def _apply_hotspot_material(obj):
    """Apply the dev_hotspot.png material to all faces of an object.

    Assigns the material and sets basic 0-1 UVs on each face.
    """
    image = bpy.data.images.load(HOTSPOT_TEXTURE_PATH, check_existing=True)

    mat = find_material_with_image(image)
    if not mat:
        mat = create_material_with_image(image)

    if mat.name not in [m.name for m in obj.data.materials if m]:
        obj.data.materials.append(mat)
    mat_index = obj.data.materials.find(mat.name)

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()

    for face in bm.faces:
        face.material_index = mat_index
        # Simple planar UV: map each face to 0-1 range
        loops = list(face.loops)
        n = len(loops)
        for i, loop in enumerate(loops):
            angle = 2.0 * math.pi * i / n
            loop[uv_layer].uv = (
                0.5 + 0.5 * math.cos(angle),
                0.5 + 0.5 * math.sin(angle),
            )

    bmesh.update_edit_mesh(obj.data)
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')


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

        # Verify hotspot map is set up correctly: 2 full-width strips on top
        # plus 2 half-width cells in each of the bottom two strips = 6 cells
        image = bpy.data.images.get("dev_hotspot.png")
        self.assertIsNotNone(image, "dev_hotspot.png should be loaded")
        hotspots = get_texture_hotspots(image.name)
        self.assertEqual(len(hotspots), 6, "Should have 6 hotspot cells")

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

    def test_apply_hotspot_to_90_degree_pipe(self):
        """Create two cylinders at 90 degrees, bridge end caps into a pipe, apply hotspots."""
        ctx = _get_context_override()

        # Cylinder A: vertical along Z, bottom at z=0, top at z=0.5
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=16, radius=0.5, depth=0.5,
                location=(0, 0, 0.25), end_fill_type='NGON',
            )
        cyl_a = bpy.context.active_object
        cyl_a.name = "pipe_vert"
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(location=True)

        # Cylinder B: horizontal along X, left end at x=1 z=1.5, right end at x=1.5 z=1.5
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=16, radius=0.5, depth=0.5,
                location=(1.25, 0, 1.5), rotation=(0, math.pi / 2, 0),
                end_fill_type='NGON',
            )
        cyl_b = bpy.context.active_object
        cyl_b.name = "pipe_horiz"
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.transform_apply(location=True, rotation=True)

        # Join into one object
        cyl_a.select_set(True)
        cyl_b.select_set(True)
        bpy.context.view_layer.objects.active = cyl_a
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.join()
        obj = cyl_a

        # Enter edit mode and select the two end cap faces to bridge:
        # - Cylinder A top cap: center near (0, 0, 0.5), normal ~+Z
        # - Cylinder B left cap: center near (1, 0, 1.5), normal ~-X
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select_set(False)

        caps_selected = 0
        for face in bm.faces:
            if len(face.verts) <= 4:
                continue
            center = face.calc_center_median()
            # Cylinder A top cap
            is_a_top = (abs(center.x) < 0.01 and abs(center.y) < 0.01
                        and abs(center.z - 0.5) < 0.01)
            # Cylinder B left cap
            is_b_left = (abs(center.x - 1.0) < 0.01 and abs(center.y) < 0.01
                         and abs(center.z - 1.5) < 0.01)
            if is_a_top or is_b_left:
                face.select_set(True)
                caps_selected += 1

        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)
        assert caps_selected == 2, f"Expected 2 end caps at junction, found {caps_selected}"

        # Delete the selected cap faces to open the ends
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.delete(type='FACE')

        # Select the two boundary edge loops at the junction
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        bm.select_mode = {'EDGE'}
        for v in bm.verts:
            v.select_set(False)
        for e in bm.edges:
            e.select_set(False)
        for f in bm.faces:
            f.select_set(False)

        boundary_count = 0
        for edge in bm.edges:
            if not edge.is_boundary:
                continue
            mid = (edge.verts[0].co + edge.verts[1].co) / 2
            # Cylinder A open end near (0, 0, 0.5)
            near_a = (abs(mid.x) < 0.6 and abs(mid.y) < 0.6
                      and abs(mid.z - 0.5) < 0.6)
            # Cylinder B open end near (1, 0, 1.5)
            near_b = (abs(mid.x - 1.0) < 0.6 and abs(mid.y) < 0.6
                      and abs(mid.z - 1.5) < 0.6)
            if near_a or near_b:
                edge.select_set(True)
                boundary_count += 1

        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)
        assert boundary_count == 32, f"Expected 32 boundary edges (2×16), got {boundary_count}"

        # Bridge edge loops to create the pipe bend (extra cuts for a smooth curve)
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.bridge_edge_loops(number_cuts=4, smoothness=0.5)

        # Back to object mode, apply hotspot material
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

        _apply_hotspot_material(obj)

        # Apply hotspots
        with bpy.context.temp_override(**ctx):
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            result = bpy.ops.leveldesign.apply_hotspot()
        self.assertEqual(result, {'FINISHED'})

        # Verify: all faces should have UVs within valid hotspot cell bounds
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()

        for face in bm.faces:
            us = [loop[uv_layer].uv.x for loop in face.loops]
            vs = [loop[uv_layer].uv.y for loop in face.loops]
            u_min, u_max = min(us), max(us)
            v_min, v_max = min(vs), max(vs)

            u_span = u_max - u_min
            v_span = v_max - v_min
            self.assertLessEqual(
                u_span, 0.76,
                f"Pipe face UV u-span {u_span:.3f} should fit within a hotspot cell"
            )
            self.assertLessEqual(
                v_span, 0.76,
                f"Pipe face UV v-span {v_span:.3f} should fit within a hotspot cell"
            )

        bm.free()

    def test_apply_hotspot_to_ring_with_hole(self):
        """Create a ring of quad faces with a hole, apply hotspots."""
        ctx = _get_context_override()

        # Create an empty mesh object
        mesh = bpy.data.meshes.new("ring")
        obj = bpy.data.objects.new("ring", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        # Add outer circle
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.primitive_circle_add(vertices=16, radius=1.0)

        # Deselect, then add inner circle
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.mesh.primitive_circle_add(vertices=16, radius=0.5)

        # Select all edges and bridge to form a ring of quads
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.bridge_edge_loops()
            bpy.ops.mesh.flip_normals()

        # Verify ring topology: 16 quad faces
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        self.assertEqual(len(bm.faces), 16, f"Expected 16 ring faces, got {len(bm.faces)}")
        for face in bm.faces:
            self.assertEqual(len(face.verts), 4, "All ring faces should be quads")

        # Back to object mode, apply hotspot material
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

        _apply_hotspot_material(obj)

        # Apply hotspots
        with bpy.context.temp_override(**ctx):
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            result = bpy.ops.leveldesign.apply_hotspot()
        self.assertEqual(result, {'FINISHED'})

        # Verify: all faces have UVs within hotspot cell bounds
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()

        for face in bm.faces:
            us = [loop[uv_layer].uv.x for loop in face.loops]
            vs = [loop[uv_layer].uv.y for loop in face.loops]
            u_min, u_max = min(us), max(us)
            v_min, v_max = min(vs), max(vs)

            u_span = u_max - u_min
            v_span = v_max - v_min
            self.assertLessEqual(
                u_span, 0.76,
                f"Ring face UV u-span {u_span:.3f} should fit within a hotspot cell"
            )
            self.assertLessEqual(
                v_span, 0.76,
                f"Ring face UV v-span {v_span:.3f} should fit within a hotspot cell"
            )

        bm.free()
