import os

import bmesh
import bpy
from mathutils import Vector

from ..core.uv_projection import apply_uv_to_face
from ..core.uv_projection import derive_transform_from_uvs
from ..core.materials import find_material_with_image, create_material_with_image
from ..hotspot_mapping.json_storage import add_texture_as_hotspottable
from .base_test import AnvilTestCase
from .helpers import (
    create_vertical_plane, create_textured_cube, add_uv_layer,
    _get_context_override, TEXTURE_PATH,
)


HOTSPOT_TEXTURE_PATH = os.path.join(os.path.dirname(__file__), "dev_hotspot.png")


def _setup_cube_and_select_top_face(name, scale_u, scale_v):
    """Create a textured cube, enter edit mode, select the top face.

    Returns the object (in edit mode with top face selected).
    """
    obj = create_textured_cube(name, scale_u, scale_v)
    add_uv_layer(obj, "UVMap.001", 0.5, 0.5)

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
    bm = bmesh.from_edit_mesh(obj.data)

    # Select only the top face (normal pointing +Z)
    bm.select_mode = {'FACE'}
    for f in bm.faces:
        f.select_set(False)
    for f in bm.faces:
        if f.normal.z > 0.9:
            f.select_set(True)
            break
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)

    return obj


def _read_all_face_transforms(obj, uv_layer_index):
    """Read UV transforms from all faces on the UV layer at the given index.

    Returns a list of transform dicts.
    """
    ppm = bpy.context.scene.level_design_props.pixels_per_meter

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv[uv_layer_index]
    bm.faces.ensure_lookup_table()

    transforms = []
    for face in bm.faces:
        transforms.append(derive_transform_from_uvs(face, uv_layer, ppm, obj.data))

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')

    return transforms


def _select_edge_by_vert_filter(bm, me, vert_filter):
    """Select the edge whose both endpoints satisfy vert_filter.

    Flushes selection and updates the edit mesh so the operator poll sees it.
    """
    with bpy.context.temp_override(**_get_context_override()):
        bpy.ops.mesh.select_mode(type='EDGE')
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


def _get_or_create_material_for_image(image):
    mat = find_material_with_image(image)
    if mat is None:
        mat = create_material_with_image(image)
    return mat


def _create_connected_dev_textured_wall_with_hotspot_slot_zero(name):
    """Create two connected quads with slot 0 reserved for a non-dev texture."""
    hotspot_image = bpy.data.images.load(HOTSPOT_TEXTURE_PATH, check_existing=True)
    add_texture_as_hotspottable(
        hotspot_image.name, hotspot_image.size[0], hotspot_image.size[1])
    hotspot_mat = _get_or_create_material_for_image(hotspot_image)

    dev_image = bpy.data.images.load(TEXTURE_PATH, check_existing=True)
    dev_mat = _get_or_create_material_for_image(dev_image)

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh.materials.append(hotspot_mat)
    mesh.materials.append(dev_mat)
    dev_mat_index = mesh.materials.find(dev_mat.name)

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(mesh)
    v0 = bm.verts.new((0, 0, 0))
    v1 = bm.verts.new((1, 0, 0))
    v2 = bm.verts.new((1, 0, 1))
    v3 = bm.verts.new((0, 0, 1))
    face = bm.faces.new((v0, v1, v2, v3))
    face.material_index = 0
    bm.normal_update()

    _select_edge_by_vert_filter(
        bm, mesh, lambda v: abs(v.co.x - 1.0) < 1e-5)

    with bpy.context.temp_override(**ctx):
        bpy.ops.mesh.extrude_region_move(
            TRANSFORM_OT_translate={"value": Vector((1, 0, 0))}
        )

    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.verify()
    bm.faces.ensure_lookup_table()
    ppm = bpy.context.scene.level_design_props.pixels_per_meter

    # Paint-equivalent setup: both connected quads use the dev texture in
    # material slot 1, leaving slot 0 as the visible bad-fallback texture.
    for wall_face in bm.faces:
        wall_face.material_index = dev_mat_index
        apply_uv_to_face(
            wall_face, uv_layer, 1.0, 1.0, 0.0, 0.0, 0.0,
            dev_mat, ppm, mesh,
        )

    return obj, dev_mat.name, hotspot_mat.name


def _select_rightmost_face(obj):
    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.mesh.select_mode(type='FACE')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.select_mode = {'FACE'}
    target = None
    best_x = -float('inf')
    for vert in bm.verts:
        vert.select_set(False)
    for edge in bm.edges:
        edge.select_set(False)
    for face in bm.faces:
        face.select_set(False)
        center_x = face.calc_center_median().x
        if center_x > best_x:
            best_x = center_x
            target = face
    target.select_set(True)
    bm.faces.active = target
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)


def _extrude_selected_region(direction):
    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.mesh.extrude_region_move(
            TRANSFORM_OT_translate={"value": direction}
        )


def _move_selected_region(direction):
    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.transform.translate(value=direction)


def _face_material_name(me, face):
    if face.material_index < 0 or face.material_index >= len(me.materials):
        return f"<invalid:{face.material_index}>"
    mat = me.materials[face.material_index]
    if mat is None:
        return "<empty>"
    return mat.name


def _describe_faces_with_materials(bm, me):
    parts = []
    for face in bm.faces:
        center = face.calc_center_median()
        parts.append(
            f"face={face.index} mat={face.material_index}:"
            f"{_face_material_name(me, face)} "
            f"center=({center.x:.2f},{center.y:.2f},{center.z:.2f}) "
            f"normal=({face.normal.x:.0f},{face.normal.y:.0f},{face.normal.z:.0f})"
        )
    return "; ".join(parts)


def _setup_plane_and_select_edge(name, vert_filter):
    """Create a textured plane, enter edit mode, and select the target edge.

    Returns the object (in edit mode with edge selected).
    """
    obj = create_vertical_plane(name)
    add_uv_layer(obj, "UVMap.001", 0.5, 0.5)

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)

    _select_edge_by_vert_filter(bm, obj.data, vert_filter)
    return obj


def _read_transforms(obj, uv_layer_index):
    """Read UV transforms from both faces (original and new) on the UV layer at the given index.

    Returns (original_transform, new_transform).
    """
    ppm = bpy.context.scene.level_design_props.pixels_per_meter

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv[uv_layer_index]
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


class _UVExtendBase(AnvilTestCase):
    """Shared assertion helpers for UV extend tests."""

    def _assert_transform(self, transform, scale_u, scale_v, rotation,
                          offset_x, offset_y):
        self.assertAlmostEqual(transform['scale_u'], scale_u, places=3)
        self.assertAlmostEqual(transform['scale_v'], scale_v, places=3)
        self.assertAlmostEqual(transform['rotation'], rotation, places=3)
        self.assertAlmostEqual(transform['offset_x'], offset_x, places=3)
        self.assertAlmostEqual(transform['offset_y'], offset_y, places=3)


class ExtrudeConnectedFaceMaterialTest(AnvilTestCase):
    """Regression coverage for Blender's connected face-region extrude path."""

    def test_extrude_connected_dev_textured_face_zero_area_stage_preserves_material_on_side_walls(self):
        obj, dev_mat_name, hotspot_mat_name = (
            _create_connected_dev_textured_wall_with_hotspot_slot_zero(
                "connected_extrude_material_fallback"
            )
        )

        self.assertEqual(obj.data.materials[0].name, hotspot_mat_name)
        self.assertNotEqual(obj.data.materials[0].name, dev_mat_name)

        _select_rightmost_face(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        extrude_direction = bm.faces.active.normal.copy()
        _extrude_selected_region(Vector((0, 0, 0)))
        yield 0.5

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        self.assertEqual(
            len(bm.faces), 6,
            "Extruding one quad from a two-quad wall should leave six faces",
        )

        zero_area_faces = [
            face for face in bm.faces
            if face.calc_area() < 1e-8
        ]
        self.assertGreater(
            len(zero_area_faces), 0,
            "Zero-distance extrude should expose collapsed side walls",
        )

        bad_zero_area_faces = [
            face for face in zero_area_faces
            if _face_material_name(obj.data, face) != dev_mat_name
        ]
        bad_zero_area_descriptions = [
            f"{face.index}:{_face_material_name(obj.data, face)}"
            for face in bad_zero_area_faces
        ]
        self.assertEqual(
            bad_zero_area_descriptions, [],
            "Collapsed zero-area extrusion faces should copy the selected "
            f"dev material before the moved cap creates area. Faces: "
            f"{_describe_faces_with_materials(bm, obj.data)}",
        )

        moved_face_indices = {face.index for face in zero_area_faces}
        _move_selected_region(extrude_direction)
        yield 0.5

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        bad_faces = [
            face for face in bm.faces
            if _face_material_name(obj.data, face) != dev_mat_name
        ]

        # Leave the extrusion side walls selected in --save outputs.
        for face in bm.faces:
            face.select_set(face.index in moved_face_indices)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

        bad_descriptions = [
            f"{face.index}:{_face_material_name(obj.data, face)}"
            for face in bad_faces
        ]
        self.assertEqual(
            bad_descriptions, [],
            "Connected face extrusion should keep the dev material on every "
            f"generated side wall. Faces: {_describe_faces_with_materials(bm, obj.data)}",
        )


class UVExtendKeyboardTest(_UVExtendBase):
    """Test UV projection during modal extrude workflows.

    Face-normal extrudes use the E key path. Axis-constrained edge extrudes
    invoke the same modal extrude operator directly so timer-driven tests do
    not depend on raw keymap focus. In both cases the depsgraph handler sees a
    modal topology operation via window.modal_operators.
    """

    def test_uv_extend_up(self):
        obj = _setup_plane_and_select_edge(
            "kb_extend_up",
            vert_filter=lambda v: abs(v.co.z - 1.0) < 1e-5,
        )
        yield from self.simulate_extrude(axis='Z', value=1)
        orig, new = _read_transforms(obj, 0)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 0.0, 0.0, 0.0)
        orig2, new2 = _read_transforms(obj, 1)
        self._assert_transform(orig2, 0.5, 0.5, 0.0, 0.0, 0.0)
        self._assert_transform(new2, 0.5, 0.5, 0.0, 0.0, 0.0)

    def test_uv_extend_down(self):
        obj = _setup_plane_and_select_edge(
            "kb_extend_down",
            vert_filter=lambda v: abs(v.co.z) < 1e-5,
        )
        yield from self.simulate_extrude(axis='Z', value=-1)
        orig, new = _read_transforms(obj, 0)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 180.0, 0.0, 0.0)
        orig2, new2 = _read_transforms(obj, 1)
        self._assert_transform(orig2, 0.5, 0.5, 0.0, 0.0, 0.0)
        self._assert_transform(new2, 0.5, 0.5, 180.0, 0.0, 0.0)

    def test_uv_extend_left(self):
        obj = _setup_plane_and_select_edge(
            "kb_extend_left",
            vert_filter=lambda v: abs(v.co.x) < 1e-5,
        )
        yield from self.simulate_extrude(axis='X', value=-1)
        orig, new = _read_transforms(obj, 0)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 90.0, 0.0, 0.0)
        orig2, new2 = _read_transforms(obj, 1)
        self._assert_transform(orig2, 0.5, 0.5, 0.0, 0.0, 0.0)
        self._assert_transform(new2, 0.5, 0.5, 90.0, 0.0, 0.0)

    def test_uv_extend_right(self):
        obj = _setup_plane_and_select_edge(
            "kb_extend_right",
            vert_filter=lambda v: abs(v.co.x - 1.0) < 1e-5,
        )
        yield from self.simulate_extrude(axis='X', value=1)
        orig, new = _read_transforms(obj, 0)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, -90.0, 0.0, 0.0)
        orig2, new2 = _read_transforms(obj, 1)
        self._assert_transform(orig2, 0.5, 0.5, 0.0, 0.0, 0.0)
        self._assert_transform(new2, 0.5, 0.5, -90.0, 0.0, 0.0)

    def test_cube_extrude_preserves_scale(self):
        obj = _setup_cube_and_select_top_face("kb_cube_extrude", 2.0, 2.0)

        # Re-UV the selected top face to scale 3.0 on layer 0
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer_0 = bm.loops.layers.uv[0]
        mat = obj.data.materials[0]
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        for f in bm.faces:
            if f.normal.z > 0.9:
                apply_uv_to_face(f, uv_layer_0, 3.0, 3.0, 0.0, 0.0, 0.0,
                                 mat, ppm, obj.data)
                break
        bmesh.update_edit_mesh(obj.data)
        self.refresh_face_cache()

        yield from self.simulate_extrude(value=1)

        transforms = _read_all_face_transforms(obj, 0)
        scale_3 = [t for t in transforms
                   if abs(t['scale_u'] - 3.0) < 0.001
                   and abs(t['scale_v'] - 3.0) < 0.001]
        scale_2 = [t for t in transforms
                   if abs(t['scale_u'] - 2.0) < 0.001
                   and abs(t['scale_v'] - 2.0) < 0.001]
        self.assertEqual(len(scale_3), 1,
                         "Expected exactly one face at scale 3.0")
        self.assertEqual(len(scale_2), len(transforms) - 1,
                         "Expected all other faces at scale 2.0")

        transforms2 = _read_all_face_transforms(obj, 1)
        for t in transforms2:
            self.assertAlmostEqual(t['scale_u'], 0.5, places=3)
            self.assertAlmostEqual(t['scale_v'], 0.5, places=3)


class UVExtendToolTest(_UVExtendBase):
    """Test UV projection during extrude using the menu/tool (non-modal workflow).

    Replicates what happens when the user extrudes via:
      - Mesh menu > Extrude > Extrude Region
      - The Extrude Region toolbar tool
      - A script calling bpy.ops.mesh.extrude_region_move(...)

    These all complete instantly — the operator never appears in
    window.modal_operators. The handler must still apply correct UVs.
    """

    def _extrude_non_modal(self, obj, direction):
        """Run extrude_region_move as a non-modal operator call.

        This is what happens when the user uses the menu or toolbar tool:
        the operator runs to completion in a single call, with no modal state.
        Then we yield to let the depsgraph handler fire.
        """
        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={"value": direction}
            )
        # Yield to let Blender process depsgraph updates / handler callbacks
        yield 0.5

    def test_uv_extend_up(self):
        obj = _setup_plane_and_select_edge(
            "tool_extend_up",
            vert_filter=lambda v: abs(v.co.z - 1.0) < 1e-5,
        )
        yield from self._extrude_non_modal(obj, Vector((0, 0, 1)))
        orig, new = _read_transforms(obj, 0)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 0.0, 0.0, 0.0)
        orig2, new2 = _read_transforms(obj, 1)
        self._assert_transform(orig2, 0.5, 0.5, 0.0, 0.0, 0.0)
        self._assert_transform(new2, 0.5, 0.5, 0.0, 0.0, 0.0)

    def test_uv_extend_down(self):
        obj = _setup_plane_and_select_edge(
            "tool_extend_down",
            vert_filter=lambda v: abs(v.co.z) < 1e-5,
        )
        yield from self._extrude_non_modal(obj, Vector((0, 0, -1)))
        orig, new = _read_transforms(obj, 0)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 180.0, 0.0, 0.0)
        orig2, new2 = _read_transforms(obj, 1)
        self._assert_transform(orig2, 0.5, 0.5, 0.0, 0.0, 0.0)
        self._assert_transform(new2, 0.5, 0.5, 180.0, 0.0, 0.0)

    def test_uv_extend_left(self):
        obj = _setup_plane_and_select_edge(
            "tool_extend_left",
            vert_filter=lambda v: abs(v.co.x) < 1e-5,
        )
        yield from self._extrude_non_modal(obj, Vector((-1, 0, 0)))
        orig, new = _read_transforms(obj, 0)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, 90.0, 0.0, 0.0)
        orig2, new2 = _read_transforms(obj, 1)
        self._assert_transform(orig2, 0.5, 0.5, 0.0, 0.0, 0.0)
        self._assert_transform(new2, 0.5, 0.5, 90.0, 0.0, 0.0)

    def test_uv_extend_right(self):
        obj = _setup_plane_and_select_edge(
            "tool_extend_right",
            vert_filter=lambda v: abs(v.co.x - 1.0) < 1e-5,
        )
        yield from self._extrude_non_modal(obj, Vector((1, 0, 0)))
        orig, new = _read_transforms(obj, 0)
        self._assert_transform(orig, 1.0, 1.0, 0.0, 0.0, 0.0)
        self._assert_transform(new, 1.0, 1.0, -90.0, 0.0, 0.0)
        orig2, new2 = _read_transforms(obj, 1)
        self._assert_transform(orig2, 0.5, 0.5, 0.0, 0.0, 0.0)
        self._assert_transform(new2, 0.5, 0.5, -90.0, 0.0, 0.0)

    def test_cube_extrude_preserves_scale(self):
        obj = _setup_cube_and_select_top_face("tool_cube_extrude", 2.0, 2.0)

        # Re-UV the selected top face to scale 3.0 on layer 0
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer_0 = bm.loops.layers.uv[0]
        mat = obj.data.materials[0]
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        for f in bm.faces:
            if f.normal.z > 0.9:
                apply_uv_to_face(f, uv_layer_0, 3.0, 3.0, 0.0, 0.0, 0.0,
                                 mat, ppm, obj.data)
                break
        bmesh.update_edit_mesh(obj.data)
        self.refresh_face_cache()

        yield from self._extrude_non_modal(obj, Vector((0, 0, 1)))

        transforms = _read_all_face_transforms(obj, 0)
        scale_3 = [t for t in transforms
                   if abs(t['scale_u'] - 3.0) < 0.001
                   and abs(t['scale_v'] - 3.0) < 0.001]
        scale_2 = [t for t in transforms
                   if abs(t['scale_u'] - 2.0) < 0.001
                   and abs(t['scale_v'] - 2.0) < 0.001]
        self.assertEqual(len(scale_3), 1,
                         "Expected exactly one face at scale 3.0")
        self.assertEqual(len(scale_2), len(transforms) - 1,
                         "Expected all other faces at scale 2.0")

        transforms2 = _read_all_face_transforms(obj, 1)
        for t in transforms2:
            self.assertAlmostEqual(t['scale_u'], 0.5, places=3)
            self.assertAlmostEqual(t['scale_v'], 0.5, places=3)
