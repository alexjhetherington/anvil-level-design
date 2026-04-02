import bmesh
import bpy
from mathutils import Vector

from ..core.uv_projection import apply_uv_to_face
from ..core.uv_projection import derive_transform_from_uvs
from .base_test import AnvilTestCase
from .helpers import create_vertical_plane, create_textured_cube, add_uv_layer, _get_context_override


def _setup_cube_and_select_top_face(name, scale_u, scale_v):
    """Create a textured cube, enter edit mode, select the top face.

    Returns the object (in edit mode with top face selected).
    """
    obj = create_textured_cube(name, scale_u, scale_v)
    add_uv_layer(obj, "UVMap.001", 0.5, 0.5)

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
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


class UVExtendKeyboardTest(_UVExtendBase):
    """Test UV projection during extrude using the E key (modal workflow).

    Simulates the real user workflow: press E, constrain to axis, type value,
    press Enter. The extrude runs as a modal operator, so the depsgraph
    handler detects it via window.modal_operators.
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
