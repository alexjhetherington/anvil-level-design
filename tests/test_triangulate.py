import bmesh
import bpy

from ..core.uv_projection import derive_transform_from_uvs
from ..core.uv_projection import apply_uv_to_face
from .base_test import AnvilTestCase
from .helpers import _get_context_override, TEXTURE_PATH


def _create_three_face_horizontal_plane(name, rot_left, rot_center, rot_right):
    """Create a 3x1 horizontal plane with 3 coplanar quads facing +Z.

    Vertices layout (top view):
        (-1,0,0) -- (0,0,0) -- (1,0,0) -- (2,0,0)
            |          |          |          |
        (-1,1,0) -- (0,1,0) -- (1,1,0) -- (2,1,0)

    Returns the object in object mode with a textured material applied.
    """
    from ..core.materials import create_material_with_image, find_material_with_image

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    v0 = bm.verts.new((-1, 0, 0))
    v1 = bm.verts.new((0, 0, 0))
    v2 = bm.verts.new((1, 0, 0))
    v3 = bm.verts.new((2, 0, 0))
    v4 = bm.verts.new((-1, 1, 0))
    v5 = bm.verts.new((0, 1, 0))
    v6 = bm.verts.new((1, 1, 0))
    v7 = bm.verts.new((2, 1, 0))

    bm.faces.new((v0, v1, v5, v4))  # left face
    bm.faces.new((v1, v2, v6, v5))  # center face
    bm.faces.new((v2, v3, v7, v6))  # right face

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Apply material
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

    apply_uv_to_face(bm.faces[0], uv_layer, 1.0, 1.0, rot_left, 0.0, 0.0,
                     mat, ppm, obj.data)
    apply_uv_to_face(bm.faces[1], uv_layer, 1.0, 1.0, rot_center, 0.0, 0.0,
                     mat, ppm, obj.data)
    apply_uv_to_face(bm.faces[2], uv_layer, 1.0, 1.0, rot_right, 0.0, 0.0,
                     mat, ppm, obj.data)

    bmesh.update_edit_mesh(obj.data)
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')

    return obj


def _face_key(face):
    c = face.calc_center_median()
    return (round(c.x, 2), round(c.y, 2), round(c.z, 2))


class TriangulateTest(AnvilTestCase):
    """Test that triangulating faces preserves UV transforms."""

    def test_triangulate_three_coplanar_faces(self):
        """Triangulate 3 coplanar faces with different rotations.

        Verifies that triangulation preserves the correct UV transforms.
        """
        obj = _create_three_face_horizontal_plane("tri_test",
                                                  -45.0, 0.0, 45.0)
        ctx = _get_context_override()

        # Enter edit mode and let cache build
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
        yield 0.5

        # Select all faces and triangulate
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            f.select_set(True)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

        with bpy.context.temp_override(**ctx):
            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY',
                                               ngon_method='BEAUTY')
        yield 0.5

        # Read transforms
        ppm = bpy.context.scene.level_design_props.pixels_per_meter
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv[0]
        bm.faces.ensure_lookup_table()

        self.assertEqual(len(bm.faces), 6,
                         f"Expected 6 triangles, got {len(bm.faces)}")

        transforms = {}
        for face in bm.faces:
            key = _face_key(face)
            transforms[key] = derive_transform_from_uvs(
                face, uv_layer, ppm, obj.data)

        print(f"\n=== Triangulate Test Results ===")
        for key, t in sorted(transforms.items()):
            print(f"  {key}: scale=({t['scale_u']:.4f}, {t['scale_v']:.4f}) "
                  f"rot={t['rotation']:.4f} "
                  f"off=({t['offset_x']:.4f}, {t['offset_y']:.4f})")
        print(f"=== End Results ===\n")

        expected = {
            (-0.67, 0.33, 0.0): {
                'scale_u': 1.0, 'scale_v': 1.0,
                'rotation': 90.0, 'offset_x': 0.7071, 'offset_y': 0.2929,
            },
            (-0.33, 0.67, 0.0): {
                'scale_u': 1.0, 'scale_v': 1.0,
                'rotation': 45.0, 'offset_x': 0.7071, 'offset_y': 0.2929,
            },
            (0.33, 0.33, 0.0): {
                'scale_u': 1.0, 'scale_v': 1.0,
                'rotation': 135.0, 'offset_x': 1.0, 'offset_y': 0.0,
            },
            (0.67, 0.67, 0.0): {
                'scale_u': 1.0, 'scale_v': 1.0,
                'rotation': 90.0, 'offset_x': 0.0, 'offset_y': 0.0,
            },
            (1.33, 0.33, 0.0): {
                'scale_u': 1.0, 'scale_v': 1.0,
                'rotation': 180.0, 'offset_x': 0.7071, 'offset_y': 0.7071,
            },
            (1.67, 0.67, 0.0): {
                'scale_u': 1.0, 'scale_v': 1.0,
                'rotation': 135.0, 'offset_x': 0.7071, 'offset_y': 0.7071,
            },
        }

        for face_key, expected_t in expected.items():
            self.assertIn(face_key, transforms,
                          f"Missing face with centroid {face_key}. "
                          f"Available: {list(transforms.keys())}")
            actual = transforms[face_key]
            for prop in ('scale_u', 'scale_v', 'rotation',
                         'offset_x', 'offset_y'):
                self.assertAlmostEqual(
                    actual[prop], expected_t[prop], places=2,
                    msg=f"Face {face_key}: {prop} = {actual[prop]}, "
                        f"expected {expected_t[prop]}",
                )
