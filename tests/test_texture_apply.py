import math

import bmesh
import bpy
from mathutils import Vector

from ..core.uv_projection import derive_transform_from_uvs
from ..core.uv_projection import apply_uv_to_face
from .base_test import AnvilTestCase
from .helpers import _get_context_override, TEXTURE_PATH
from ..core.materials import find_material_with_image, create_material_with_image
from ..operators.texture_apply import (
    set_uv_from_other_face, stretch_uv_from_other_face,
    _apply_to_other_obj, _dispatch_set_uv_from_other_face,
)
from ..core.face_id import get_face_id_layer


def _make_vertical_face(bm, x_offset):
    """Add a 1x1 vertical quad to a bmesh at the given x offset."""
    v0 = bm.verts.new((x_offset, 0, 0))
    v1 = bm.verts.new((x_offset + 1, 0, 0))
    v2 = bm.verts.new((x_offset + 1, 0, 1))
    v3 = bm.verts.new((x_offset, 0, 1))
    return bm.faces.new((v0, v1, v2, v3))


def _get_material():
    """Get (or create once) the shared test material from dev_orange_wall.png."""
    image = bpy.data.images.load(TEXTURE_PATH, check_existing=True)
    mat = find_material_with_image(image)
    if not mat:
        mat = create_material_with_image(image)
    return mat


def _create_two_face_plane(name, source_scale_u, source_scale_v,
                           source_rotation, source_offset_x, source_offset_y):
    """Create a 2-face plane: two vertical quads sharing an edge.

    Face 0: (0,0,0), (1,0,0), (1,0,1), (0,0,1) — lower, default UVs
    Face 1: (0,0,1), (1,0,1), (1,0,2), (0,0,2) — upper, custom UVs

    Both faces share the edge at z=1. Returns the object in object mode.
    """
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    v0 = bm.verts.new((0, 0, 0))
    v1 = bm.verts.new((1, 0, 0))
    v2 = bm.verts.new((1, 0, 1))
    v3 = bm.verts.new((0, 0, 1))
    v4 = bm.verts.new((1, 0, 2))
    v5 = bm.verts.new((0, 0, 2))

    bm.faces.new((v0, v1, v2, v3))
    bm.faces.new((v3, v2, v4, v5))

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mat = _get_material()
    obj.data.materials.append(mat)
    mat_index = obj.data.materials.find(mat.name)

    ppm = bpy.context.scene.level_design_props.pixels_per_meter

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()
    bm.faces.ensure_lookup_table()

    for face in bm.faces:
        face.material_index = mat_index

    # Face 0: default transform (scale=1, offset=0, rotation=0)
    apply_uv_to_face(bm.faces[0], uv_layer, 1.0, 1.0, 0.0, 0.0, 0.0,
                     mat, ppm, obj.data)
    # Face 1: caller-specified transform
    apply_uv_to_face(bm.faces[1], uv_layer,
                     source_scale_u, source_scale_v, source_rotation,
                     source_offset_x, source_offset_y,
                     mat, ppm, obj.data)

    bmesh.update_edit_mesh(obj.data)
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')

    return obj


class TextureApplyTest(AnvilTestCase):
    """Test set_uv_from_other_face UV transfer logic."""

    def test_texture_apply_from_adjacent_face(self):
        """Applying from an adjacent face should transfer scale and rotation."""
        obj = _create_two_face_plane("apply_adjacent", 2.5, 1.5, 45.0, 0.1, 0.1)
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        # Verify faces start with different scales
        source_t = derive_transform_from_uvs(bm.faces[1], uv_layer, ppm, obj.data)
        target_t = derive_transform_from_uvs(bm.faces[0], uv_layer, ppm, obj.data)
        self.assertAlmostEqual(source_t['scale_u'], 2.5, places=3)
        self.assertAlmostEqual(target_t['scale_u'], 1.0, places=3)

        # Apply UV from face 1 (scale=2.5/1.5, rot=45) to face 0 (scale=1/1, rot=0)
        result = set_uv_from_other_face(
            bm.faces[1], bm.faces[0], uv_layer,
            ppm, obj.data, obj.matrix_world,
        )
        self.assertTrue(result)
        bmesh.update_edit_mesh(obj.data)

        target_t = derive_transform_from_uvs(bm.faces[0], uv_layer, ppm, obj.data)
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 2.5, places=3)
        self.assertAlmostEqual(target_t['scale_v'], 1.5, places=3)
        self.assertAlmostEqual(target_t['rotation'], 45.0, places=3)
        self.assertIsNotNone(target_t['offset_x'])
        self.assertIsNotNone(target_t['offset_y'])

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

    def test_texture_apply_different_params(self):
        """Applying from an adjacent face with different scale/rotation/offset."""
        obj = _create_two_face_plane("apply_params", 3.0, 2.0, 30.0, 0.25, 0.5)
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        # Verify faces start with different scales
        source_t = derive_transform_from_uvs(bm.faces[1], uv_layer, ppm, obj.data)
        target_t = derive_transform_from_uvs(bm.faces[0], uv_layer, ppm, obj.data)
        self.assertAlmostEqual(source_t['scale_u'], 3.0, places=3)
        self.assertAlmostEqual(source_t['scale_v'], 2.0, places=3)
        self.assertAlmostEqual(target_t['scale_u'], 1.0, places=3)
        self.assertAlmostEqual(target_t['scale_v'], 1.0, places=3)

        result = set_uv_from_other_face(
            bm.faces[1], bm.faces[0], uv_layer,
            ppm, obj.data, obj.matrix_world,
        )
        self.assertTrue(result)
        bmesh.update_edit_mesh(obj.data)

        target_t = derive_transform_from_uvs(bm.faces[0], uv_layer, ppm, obj.data)
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 3.0, places=3)
        self.assertAlmostEqual(target_t['scale_v'], 2.0, places=3)
        self.assertAlmostEqual(target_t['rotation'], 30.0, places=3)
        self.assertIsNotNone(target_t['offset_x'])
        self.assertIsNotNone(target_t['offset_y'])

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

    def test_texture_apply_cross_object(self):
        """Cross-object UV transfer while source object is in edit mode."""
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        mat = _get_material()
        ctx = _get_context_override()

        # --- Source object (will be in edit mode, like the real operator) ---
        mesh_a = bpy.data.meshes.new("src_mesh")
        mesh_a.materials.append(mat)
        obj_a = bpy.data.objects.new("src_obj", mesh_a)
        bpy.context.collection.objects.link(obj_a)

        # Build geometry via edit mode so bmesh is the edit-mode bmesh
        bpy.context.view_layer.objects.active = obj_a
        obj_a.select_set(True)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        bm_a = bmesh.from_edit_mesh(mesh_a)
        _make_vertical_face(bm_a, 0)
        bm_a.normal_update()
        uv_a = bm_a.loops.layers.uv.verify()
        bm_a.faces.ensure_lookup_table()
        bm_a.faces[0].material_index = 0
        apply_uv_to_face(bm_a.faces[0], uv_a, 2.5, 1.5, 45.0, 0.3, 0.7,
                         mat, ppm, mesh_a)

        # --- Target object (not in edit mode, standalone bmesh) ---
        mesh_b = bpy.data.meshes.new("tgt_mesh")
        mesh_b.materials.append(mat)
        bm_b = bmesh.new()
        _make_vertical_face(bm_b, 0)
        bm_b.normal_update()
        uv_b = bm_b.loops.layers.uv.verify()
        bm_b.faces.ensure_lookup_table()
        bm_b.faces[0].material_index = 0
        apply_uv_to_face(bm_b.faces[0], uv_b, 1.0, 1.0, 0.0, 0.0, 0.0,
                         mat, ppm, mesh_b)

        obj_b = bpy.data.objects.new("tgt_obj", mesh_b)
        bpy.context.collection.objects.link(obj_b)
        obj_b.location.x = 1
        bpy.context.view_layer.update()

        # Verify faces start with different scales
        source_t = derive_transform_from_uvs(bm_a.faces[0], uv_a, ppm, mesh_a)
        target_t = derive_transform_from_uvs(bm_b.faces[0], uv_b, ppm, mesh_b)
        self.assertAlmostEqual(source_t['scale_u'], 2.5, places=3)
        self.assertAlmostEqual(source_t['scale_v'], 1.5, places=3)
        self.assertAlmostEqual(target_t['scale_u'], 1.0, places=3)
        self.assertAlmostEqual(target_t['scale_v'], 1.0, places=3)

        # Objects side by side; source_to_target accounts for the offset
        source_to_target = obj_b.matrix_world.inverted() @ obj_a.matrix_world

        result = set_uv_from_other_face(
            bm_a.faces[0], bm_b.faces[0], uv_b,
            ppm, mesh_b, obj_b.matrix_world,
            source_uv_layer=uv_a, source_me=mesh_a,
            source_to_target=source_to_target,
        )
        self.assertTrue(result)

        target_t = derive_transform_from_uvs(bm_b.faces[0], uv_b, ppm, mesh_b)
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 2.5, places=3)
        self.assertAlmostEqual(target_t['scale_v'], 1.5, places=3)
        self.assertAlmostEqual(target_t['rotation'], 45.0, places=3)
        self.assertIsNotNone(target_t['offset_x'])
        self.assertIsNotNone(target_t['offset_y'])

        bm_b.to_mesh(mesh_b)
        bm_b.free()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')


def _create_stretch_test_object(name, source_scale_u, source_scale_v,
                                 source_rotation, source_offset_x, source_offset_y,
                                 target_verts):
    """Create an object with a 1x1 source face and a custom target face.

    Source face (index 0): (0,0,0), (1,0,0), (1,0,1), (0,0,1) — vertical quad
    Target face (index 1): from target_verts list of (x,y,z) tuples — separate

    Returns the object in object mode.
    """
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    # Source face: 1x1 vertical quad at y=0
    sv0 = bm.verts.new((0, 0, 0))
    sv1 = bm.verts.new((1, 0, 0))
    sv2 = bm.verts.new((1, 0, 1))
    sv3 = bm.verts.new((0, 0, 1))
    bm.faces.new((sv0, sv1, sv2, sv3))

    # Target face: custom geometry at y=1 (separate)
    tverts = [bm.verts.new(v) for v in target_verts]
    bm.faces.new(tverts)

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mat = _get_material()
    obj.data.materials.append(mat)
    mat_index = obj.data.materials.find(mat.name)

    ppm = bpy.context.scene.level_design_props.pixels_per_meter

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()
    bm.faces.ensure_lookup_table()

    for face in bm.faces:
        face.material_index = mat_index

    # Source: caller-specified transform
    apply_uv_to_face(bm.faces[0], uv_layer, source_scale_u, source_scale_v,
                     source_rotation, source_offset_x, source_offset_y,
                     mat, ppm, obj.data)
    # Target: default (will be overwritten by stretch apply)
    apply_uv_to_face(bm.faces[1], uv_layer, 1.0, 1.0, 0.0, 0.0, 0.0,
                     mat, ppm, obj.data)

    bmesh.update_edit_mesh(obj.data)
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')

    return obj


class StretchApplyTest(AnvilTestCase):
    """Test stretch_uv_from_other_face UV transfer logic."""

    def test_stretch_apply_double_size(self):
        """Stretch applying to a 2x2 face from a 1x1 source should double the scale."""
        # Target: 2x2 vertical quad at y=1
        target_verts = [(0, 1, 0), (2, 1, 0), (2, 1, 2), (0, 1, 2)]
        obj = _create_stretch_test_object(
            "stretch_double", 1.0, 1.0, 0.0, 0.0, 0.0, target_verts)
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        result = stretch_uv_from_other_face(
            bm.faces[0], bm.faces[1], uv_layer,
            ppm, obj.data, obj.matrix_world,
        )
        self.assertTrue(result)
        bmesh.update_edit_mesh(obj.data)

        target_t = derive_transform_from_uvs(bm.faces[1], uv_layer, ppm, obj.data)
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 2.0, places=2)
        self.assertAlmostEqual(target_t['scale_v'], 2.0, places=2)

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

    def test_stretch_pick_double_size(self):
        """Stretch pick to a 2x2 face with source scale=(2.0, 1.5) should give (4.0, 3.0)."""
        target_verts = [(0, 1, 0), (2, 1, 0), (2, 1, 2), (0, 1, 2)]
        obj = _create_stretch_test_object(
            "stretch_pick_double", 2.0, 1.5, 0.0, 0.1, 0.2, target_verts)
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        result = stretch_uv_from_other_face(
            bm.faces[0], bm.faces[1], uv_layer,
            ppm, obj.data, obj.matrix_world,
        )
        self.assertTrue(result)
        bmesh.update_edit_mesh(obj.data)

        target_t = derive_transform_from_uvs(bm.faces[1], uv_layer, ppm, obj.data)
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 4.0, places=2)
        self.assertAlmostEqual(target_t['scale_v'], 3.0, places=2)

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

    def test_stretch_apply_pentagon(self):
        """Stretch apply to a regular pentagonal face."""
        # Regular pentagon centered at (1, 1, 1), lying in y=1 plane, circumradius ~1
        target_verts = []
        for i in range(5):
            angle = 2 * math.pi * i / 5 + math.pi / 2  # start from top
            x = 1 + math.cos(angle)
            z = 1 + math.sin(angle)
            target_verts.append((x, 1, z))

        obj = _create_stretch_test_object(
            "stretch_pentagon", 1.0, 1.0, 0.0, 0.0, 0.0, target_verts)
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        result = stretch_uv_from_other_face(
            bm.faces[0], bm.faces[1], uv_layer,
            ppm, obj.data, obj.matrix_world,
        )
        self.assertTrue(result)
        bmesh.update_edit_mesh(obj.data)

        # Pentagon has different extents so scale will differ from source.
        # Just verify the function succeeded and UVs are sane.
        target_t = derive_transform_from_uvs(bm.faces[1], uv_layer, ppm, obj.data)
        self.assertIsNotNone(target_t)
        self.assertGreater(target_t['scale_u'], 0)
        self.assertGreater(target_t['scale_v'], 0)

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

    def test_stretch_apply_face_rotated_45(self):
        """Stretch apply to a 1x1 quad rotated 45 degrees around its face normal."""
        # Rotate a unit quad 45 degrees in the XZ plane, centered at (0.5, 1, 0.5)
        cx, cz = 0.5, 0.5
        cos_a = math.cos(math.radians(45))
        sin_a = math.sin(math.radians(45))
        base = [(0, 0), (1, 0), (1, 1), (0, 1)]
        target_verts = []
        for bx, bz in base:
            dx, dz = bx - cx, bz - cz
            rx = cx + dx * cos_a - dz * sin_a
            rz = cz + dx * sin_a + dz * cos_a
            target_verts.append((rx, 1, rz))

        obj = _create_stretch_test_object(
            "stretch_rot45", 1.0, 1.0, 0.0, 0.0, 0.0, target_verts)
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        result = stretch_uv_from_other_face(
            bm.faces[0], bm.faces[1], uv_layer,
            ppm, obj.data, obj.matrix_world,
        )
        self.assertTrue(result)
        bmesh.update_edit_mesh(obj.data)

        # Same physical size (1x1), just rotated. With edge-aligned UV,
        # scale should be close to 1.0 in both axes.
        target_t = derive_transform_from_uvs(bm.faces[1], uv_layer, ppm, obj.data)
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 1.0, places=2)
        self.assertAlmostEqual(target_t['scale_v'], 1.0, places=2)

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

    def test_stretch_apply_face_rotated_180(self):
        """Stretch apply to a 1x1 quad rotated 180 degrees around its face normal."""
        # Rotate a unit quad 180 degrees in the XZ plane, centered at (0.5, 1, 0.5)
        # This effectively reverses vertex order / flips face local axes.
        cx, cz = 0.5, 0.5
        cos_a = math.cos(math.radians(180))
        sin_a = math.sin(math.radians(180))
        base = [(0, 0), (1, 0), (1, 1), (0, 1)]
        target_verts = []
        for bx, bz in base:
            dx, dz = bx - cx, bz - cz
            rx = cx + dx * cos_a - dz * sin_a
            rz = cz + dx * sin_a + dz * cos_a
            target_verts.append((rx, 1, rz))

        obj = _create_stretch_test_object(
            "stretch_rot180", 1.0, 1.0, 0.0, 0.0, 0.0, target_verts)
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        result = stretch_uv_from_other_face(
            bm.faces[0], bm.faces[1], uv_layer,
            ppm, obj.data, obj.matrix_world,
        )
        self.assertTrue(result)
        bmesh.update_edit_mesh(obj.data)

        # Same physical size, just flipped. Scale should be close to 1.0.
        target_t = derive_transform_from_uvs(bm.faces[1], uv_layer, ppm, obj.data)
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 1.0, places=2)
        self.assertAlmostEqual(target_t['scale_v'], 1.0, places=2)

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

    def test_stretch_apply_uv_rotated(self):
        """Stretch apply with a 45-degree rotated source UV to a 2x2 target should preserve rotation."""
        # Source: 1x1, scale=1, rotation=45, offset=0
        # Target: 2x2 (separate, not adjacent)
        # Expected: scale doubles to (2.0, 2.0), rotation preserved at 45
        target_verts = [(0, 1, 0), (2, 1, 0), (2, 1, 2), (0, 1, 2)]
        obj = _create_stretch_test_object(
            "stretch_uv_rot", 1.0, 1.0, 45.0, 0.0, 0.0, target_verts)
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        result = stretch_uv_from_other_face(
            bm.faces[0], bm.faces[1], uv_layer,
            ppm, obj.data, obj.matrix_world,
        )
        self.assertTrue(result)
        bmesh.update_edit_mesh(obj.data)

        target_t = derive_transform_from_uvs(bm.faces[1], uv_layer, ppm, obj.data)
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 2.0, places=2)
        self.assertAlmostEqual(target_t['scale_v'], 2.0, places=2)
        self.assertAlmostEqual(target_t['rotation'], 45.0, places=2)

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')


def _get_second_material():
    """Get (or create once) a second test material distinct from the primary one."""
    mat_name = "UV_Transform_Test_Mat_B"
    mat = bpy.data.materials.get(mat_name)
    if mat:
        return mat
    # Create a new material with same image but different name
    image = bpy.data.images.load(TEXTURE_PATH, check_existing=True)
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = image
    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


def _create_two_face_two_material_plane(name, source_scale_u, source_scale_v,
                                         source_rotation, source_offset_x,
                                         source_offset_y):
    """Create a 2-face plane where each face has a DIFFERENT material.

    Face 0 (lower): mat_a, default UVs (scale=1, rotation=0, offset=0)
    Face 1 (upper): mat_b, caller-specified UV transform

    Both materials use the same image so derive_transform_from_uvs works,
    but they are distinct material slots so we can verify material_index
    is preserved.
    """
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    v0 = bm.verts.new((0, 0, 0))
    v1 = bm.verts.new((1, 0, 0))
    v2 = bm.verts.new((1, 0, 1))
    v3 = bm.verts.new((0, 0, 1))
    v4 = bm.verts.new((1, 0, 2))
    v5 = bm.verts.new((0, 0, 2))

    bm.faces.new((v0, v1, v2, v3))
    bm.faces.new((v3, v2, v4, v5))

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mat_a = _get_material()
    mat_b = _get_second_material()
    obj.data.materials.append(mat_a)
    obj.data.materials.append(mat_b)
    mat_a_index = obj.data.materials.find(mat_a.name)
    mat_b_index = obj.data.materials.find(mat_b.name)

    ppm = bpy.context.scene.level_design_props.pixels_per_meter

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()
    bm.faces.ensure_lookup_table()

    # Face 0: mat_a, default transform
    bm.faces[0].material_index = mat_a_index
    apply_uv_to_face(bm.faces[0], uv_layer, 1.0, 1.0, 0.0, 0.0, 0.0,
                     mat_a, ppm, obj.data)
    # Face 1: mat_b, caller-specified transform
    bm.faces[1].material_index = mat_b_index
    apply_uv_to_face(bm.faces[1], uv_layer,
                     source_scale_u, source_scale_v, source_rotation,
                     source_offset_x, source_offset_y,
                     mat_b, ppm, obj.data)

    bmesh.update_edit_mesh(obj.data)
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')

    return obj


class UVTransformApplyTest(AnvilTestCase):
    """Test UV-transform-only apply (no material change)."""

    def test_uv_transform_apply_transfers_scale_and_rotation(self):
        """Applying UV transform from adjacent face should transfer scale and rotation."""
        obj = _create_two_face_two_material_plane(
            "uv_xform_adjacent", 2.5, 1.5, 45.0, 0.1, 0.1)
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        # Verify starting state
        source_t = derive_transform_from_uvs(bm.faces[1], uv_layer, ppm, obj.data)
        target_t = derive_transform_from_uvs(bm.faces[0], uv_layer, ppm, obj.data)
        self.assertAlmostEqual(source_t['scale_u'], 2.5, places=3)
        self.assertAlmostEqual(target_t['scale_u'], 1.0, places=3)

        # Record material indices before
        mat_index_0_before = bm.faces[0].material_index
        mat_index_1_before = bm.faces[1].material_index
        self.assertNotEqual(mat_index_0_before, mat_index_1_before)

        # Apply UV transform from face 1 to face 0
        result = set_uv_from_other_face(
            bm.faces[1], bm.faces[0], uv_layer,
            ppm, obj.data, obj.matrix_world,
        )
        self.assertTrue(result)
        bmesh.update_edit_mesh(obj.data)

        # UV transform should be transferred
        target_t = derive_transform_from_uvs(bm.faces[0], uv_layer, ppm, obj.data)
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 2.5, places=3)
        self.assertAlmostEqual(target_t['scale_v'], 1.5, places=3)
        self.assertAlmostEqual(target_t['rotation'], 45.0, places=3)

        # Material index must NOT have changed
        self.assertEqual(bm.faces[0].material_index, mat_index_0_before)
        self.assertEqual(bm.faces[1].material_index, mat_index_1_before)

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

    def test_uv_transform_apply_preserves_material_with_different_params(self):
        """UV transform apply with different params should preserve each face's material."""
        obj = _create_two_face_two_material_plane(
            "uv_xform_params", 3.0, 2.0, 30.0, 0.25, 0.5)
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        mat_index_0_before = bm.faces[0].material_index
        mat_index_1_before = bm.faces[1].material_index
        self.assertNotEqual(mat_index_0_before, mat_index_1_before)

        result = set_uv_from_other_face(
            bm.faces[1], bm.faces[0], uv_layer,
            ppm, obj.data, obj.matrix_world,
        )
        self.assertTrue(result)
        bmesh.update_edit_mesh(obj.data)

        target_t = derive_transform_from_uvs(bm.faces[0], uv_layer, ppm, obj.data)
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 3.0, places=3)
        self.assertAlmostEqual(target_t['scale_v'], 2.0, places=3)
        self.assertAlmostEqual(target_t['rotation'], 30.0, places=3)

        # Material must be preserved
        self.assertEqual(bm.faces[0].material_index, mat_index_0_before)
        self.assertEqual(bm.faces[1].material_index, mat_index_1_before)

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

    def test_uv_transform_apply_cross_object_preserves_material(self):
        """Cross-object UV transform should transfer UVs but not change materials."""
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        mat_a = _get_material()
        mat_b = _get_second_material()
        ctx = _get_context_override()

        # --- Source object: mat_a with custom UVs ---
        mesh_a = bpy.data.meshes.new("uv_xform_src_mesh")
        mesh_a.materials.append(mat_a)
        obj_a = bpy.data.objects.new("uv_xform_src_obj", mesh_a)
        bpy.context.collection.objects.link(obj_a)

        bpy.context.view_layer.objects.active = obj_a
        obj_a.select_set(True)
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        bm_a = bmesh.from_edit_mesh(mesh_a)
        _make_vertical_face(bm_a, 0)
        bm_a.normal_update()
        uv_a = bm_a.loops.layers.uv.verify()
        bm_a.faces.ensure_lookup_table()
        bm_a.faces[0].material_index = 0
        apply_uv_to_face(bm_a.faces[0], uv_a, 2.5, 1.5, 45.0, 0.3, 0.7,
                         mat_a, ppm, mesh_a)

        # --- Target object: mat_b with default UVs ---
        mesh_b = bpy.data.meshes.new("uv_xform_tgt_mesh")
        mesh_b.materials.append(mat_b)
        bm_b = bmesh.new()
        _make_vertical_face(bm_b, 0)
        bm_b.normal_update()
        uv_b = bm_b.loops.layers.uv.verify()
        bm_b.faces.ensure_lookup_table()
        bm_b.faces[0].material_index = 0
        apply_uv_to_face(bm_b.faces[0], uv_b, 1.0, 1.0, 0.0, 0.0, 0.0,
                         mat_b, ppm, mesh_b)

        obj_b = bpy.data.objects.new("uv_xform_tgt_obj", mesh_b)
        bpy.context.collection.objects.link(obj_b)
        obj_b.location.x = 1
        bpy.context.view_layer.update()

        # Record target material before
        target_mat_before = mesh_b.materials[0].name

        source_to_target = obj_b.matrix_world.inverted() @ obj_a.matrix_world

        result = set_uv_from_other_face(
            bm_a.faces[0], bm_b.faces[0], uv_b,
            ppm, mesh_b, obj_b.matrix_world,
            source_uv_layer=uv_a, source_me=mesh_a,
            source_to_target=source_to_target,
        )
        self.assertTrue(result)

        # UV transform should be transferred
        target_t = derive_transform_from_uvs(bm_b.faces[0], uv_b, ppm, mesh_b)
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 2.5, places=3)
        self.assertAlmostEqual(target_t['scale_v'], 1.5, places=3)
        self.assertAlmostEqual(target_t['rotation'], 45.0, places=3)

        # Material must still be mat_b
        self.assertEqual(bm_b.faces[0].material_index, 0)
        self.assertEqual(mesh_b.materials[0].name, target_mat_before)

        bm_b.to_mesh(mesh_b)
        bm_b.free()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

    def test_apply_to_other_obj_does_not_modify_source_faces(self):
        """Cross-object dispatch must not mutate any source-obj face.

        The source obj is in edit mode and unrelated to the click target. If
        the dispatch passes the source bm where the target bm is expected
        (e.g. as the cache_single_face target), the source obj's face_id layer
        and UVs get touched — the user-visible symptom is faces on the
        selected object (none of which are under the cursor) changing during
        an alt-click on a different object.

        Verified by capturing UVs and material indices on multiple source
        faces before and after a synthetic cross-object hit.
        """
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        mat = _get_material()
        ctx = _get_context_override()

        # Source obj with TWO faces:
        #   F0 — the selected source face (op._source_face_index = 0).
        #   F1 — NOT under the cursor; we'll verify it's untouched.
        mesh_a = bpy.data.meshes.new("src_two_face_mesh")
        mesh_a.materials.append(mat)
        obj_a = bpy.data.objects.new("src_two_face_obj", mesh_a)
        bpy.context.collection.objects.link(obj_a)
        bpy.context.view_layer.objects.active = obj_a
        obj_a.select_set(True)

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        bm_a = bmesh.from_edit_mesh(mesh_a)
        _make_vertical_face(bm_a, 0)
        _make_vertical_face(bm_a, 2)
        bm_a.normal_update()
        uv_a = bm_a.loops.layers.uv.verify()
        # Pre-create the face_id layer on source bm. Without this, an
        # incorrectly-routed cache_single_face(target_face=other, bm=source)
        # would silently create the layer on source and assign an ID to a
        # source face. With it, the same wrong call raises ValueError because
        # the target face and id_layer come from different meshes — which is
        # how the user originally encountered this bug in production.
        id_layer_a = get_face_id_layer(bm_a)
        bm_a.faces.ensure_lookup_table()
        bm_a.faces[0].material_index = 0
        bm_a.faces[1].material_index = 0
        apply_uv_to_face(bm_a.faces[0], uv_a, 2.5, 1.5, 45.0, 0.3, 0.7,
                         mat, ppm, mesh_a)
        apply_uv_to_face(bm_a.faces[1], uv_a, 3.0, 2.0, 30.0, 0.1, 0.2,
                         mat, ppm, mesh_a)

        # Other obj: target face with default UVs.
        mesh_b = bpy.data.meshes.new("tgt_other_mesh")
        mesh_b.materials.append(mat)
        bm_b = bmesh.new()
        _make_vertical_face(bm_b, 0)
        bm_b.normal_update()
        uv_b = bm_b.loops.layers.uv.verify()
        bm_b.faces.ensure_lookup_table()
        bm_b.faces[0].material_index = 0
        apply_uv_to_face(bm_b.faces[0], uv_b, 1.0, 1.0, 0.0, 0.0, 0.0,
                         mat, ppm, mesh_b)
        bm_b.to_mesh(mesh_b)
        bm_b.free()

        obj_b = bpy.data.objects.new("tgt_other_obj", mesh_b)
        bpy.context.collection.objects.link(obj_b)
        obj_b.location.x = 5
        bpy.context.view_layer.update()

        # Capture source state BEFORE the cross-object dispatch.
        def snapshot_source():
            return [
                {
                    'uvs': [loop[uv_a].uv.copy() for loop in face.loops],
                    'material_index': face.material_index,
                    'face_id': face[id_layer_a],
                }
                for face in bm_a.faces
            ]
        before = snapshot_source()

        # Build a minimal fake op with the state _apply_to_other_obj reads.
        class _FakeOp:
            pass
        op = _FakeOp()
        op._paint_obj = obj_a
        op._source_face_index = 0
        op._mat = mat
        op._ppm = ppm
        op._other_bmeshes = {}
        op._paint_visited_other = set()

        processed = _apply_to_other_obj(
            op, bm_a, mesh_a, obj_b, 0, _dispatch_set_uv_from_other_face
        )
        self.assertTrue(processed)

        # Source must be untouched on every face.
        after = snapshot_source()
        for i, (b, a) in enumerate(zip(before, after)):
            self.assertEqual(
                b['material_index'], a['material_index'],
                f"Source face {i} material_index changed",
            )
            self.assertEqual(
                b['face_id'], a['face_id'],
                f"Source face {i} face_id changed",
            )
            for j, (b_uv, a_uv) in enumerate(zip(b['uvs'], a['uvs'])):
                self.assertAlmostEqual(
                    b_uv.x, a_uv.x, places=6,
                    msg=f"Source face {i} loop {j} UV.x changed",
                )
                self.assertAlmostEqual(
                    b_uv.y, a_uv.y, places=6,
                    msg=f"Source face {i} loop {j} UV.y changed",
                )

        # Sanity: target on other obj should have received the source UVs.
        other_bm = op._other_bmeshes[id(obj_b)]['bm']
        other_bm.faces.ensure_lookup_table()
        target_uv_layer = other_bm.loops.layers.uv.verify()
        target_t = derive_transform_from_uvs(
            other_bm.faces[0], target_uv_layer, ppm, mesh_b
        )
        self.assertIsNotNone(target_t)
        self.assertAlmostEqual(target_t['scale_u'], 2.5, places=3)
        self.assertAlmostEqual(target_t['scale_v'], 1.5, places=3)

        other_bm.free()
        op._other_bmeshes.clear()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')

