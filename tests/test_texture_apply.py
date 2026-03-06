import bmesh
import bpy
from mathutils import Quaternion, Vector
from bpy_extras import view3d_utils

from ..utils import derive_transform_from_uvs
from ..properties import apply_uv_to_face
from .base_test import AnvilTestCase, _get_window
from .helpers import _get_context_override, TEXTURE_PATH
from ..utils import create_material_with_image, find_material_with_image


def _get_3d_view():
    """Return (area, space, region, rv3d) for the first 3D viewport."""
    window = _get_window()
    for area in window.screen.areas:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            for region in area.regions:
                if region.type == 'WINDOW':
                    rv3d = space.region_3d
                    return area, space, region, rv3d
    raise RuntimeError("No 3D viewport found")


def _setup_front_ortho_view(center, width):
    """Set up a front orthographic view looking at center from +Y.

    Args:
        center: 3D point the camera looks at
        width: orthographic width (how many meters fit in the viewport)
    """
    _, _, _, rv3d = _get_3d_view()
    # Front view: looking from +Y toward -Y
    rv3d.view_rotation = Quaternion((0.7071, 0.7071, 0.0, 0.0))
    rv3d.view_location = Vector(center)
    rv3d.view_distance = 5.0
    rv3d.view_perspective = 'ORTHO'
    rv3d.window_matrix  # force update
    # Set ortho scale so the geometry fills the viewport
    rv3d.view_distance = width / 2


def _world_to_pixel(point_3d):
    """Convert a 3D world point to 2D pixel coordinates in the 3D viewport."""
    area, _, region, rv3d = _get_3d_view()
    coord_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(point_3d))
    if coord_2d is None:
        raise RuntimeError(f"Point {point_3d} is not visible in viewport")
    # location_3d_to_region_2d returns region-local coords; event_simulate
    # needs window-absolute coords
    return int(area.x + coord_2d.x), int(area.y + coord_2d.y)


def _create_two_face_plane(name):
    """Create a 2-face plane: two vertical quads sharing an edge.

    Face 0: (0,0,0), (1,0,0), (1,0,1), (0,0,1) — lower
    Face 1: (0,0,1), (1,0,1), (1,0,2), (0,0,2) — upper

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

    # Apply material to both faces
    image = bpy.data.images.load(TEXTURE_PATH, check_existing=True)
    mat = find_material_with_image(image)
    if not mat:
        mat = create_material_with_image(image)
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
    # Face 1: non-default transform (scale=2, offset=0.1, rotation=45)
    apply_uv_to_face(bm.faces[1], uv_layer, 2.0, 2.0, 45.0, 0.1, 0.1,
                     mat, ppm, obj.data)

    bmesh.update_edit_mesh(obj.data)
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')

    return obj


class TextureApplyTest(AnvilTestCase):
    """Test alt+click texture apply with depsgraph interaction.

    Uses a controlled front orthographic viewport and event_simulate to
    perform real alt+click operations, ensuring the depsgraph handler
    doesn't revert the applied UVs.
    """

    def test_apply_from_adjacent_face(self):
        """Alt+clicking an adjacent face should preserve the source face's
        scale and rotation even after depsgraph updates."""
        obj = _create_two_face_plane("apply_adjacent")
        ppm = bpy.context.scene.level_design_props.pixels_per_meter

        # Set up front orthographic view looking at the plane
        # Plane spans x=[0,1], z=[0,2], so center at (0.5, 0, 1)
        _setup_front_ortho_view(center=(0.5, 0, 1), width=4.0)

        ctx = _get_context_override()
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')

        # Switch to face select mode
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)

        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        # Select source face (face 1, upper) — this is the face we apply FROM
        for f in bm.faces:
            f.select_set(False)
        bm.faces[1].select_set(True)
        bm.faces.active = bm.faces[1]
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

        # Let depsgraph fire to populate the face data cache
        yield 0.5

        # Force a redraw so the viewport matrices are up to date
        window = _get_window()
        with bpy.context.temp_override(window=window):
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        yield

        # Compute pixel coords for face 0 center (lower face, center at 0.5, 0, 0.5)
        face0_px, face0_py = _world_to_pixel((0.5, 0, 0.5))

        # Alt+click on face 0 (the target) to apply texture from selected face 1
        window.event_simulate(
            type='LEFTMOUSE', value='PRESS',
            x=face0_px, y=face0_py, alt=True
        )
        yield
        window.event_simulate(
            type='LEFTMOUSE', value='RELEASE',
            x=face0_px, y=face0_py, alt=True
        )

        # Let depsgraph handler fire
        yield 0.5

        # Re-acquire bmesh after yields
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        # Verify target face (face 0) has the source's scale and rotation.
        # Offset is computed for seamless tiling at the shared edge, not copied
        # directly, so we only check scale and rotation here.
        target_t = derive_transform_from_uvs(bm.faces[0], uv_layer, ppm, obj.data)
        self.assertAlmostEqual(target_t['scale_u'], 2.0, places=3)
        self.assertAlmostEqual(target_t['scale_v'], 2.0, places=3)
        self.assertAlmostEqual(target_t['rotation'], 45.0, places=3)

        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')
