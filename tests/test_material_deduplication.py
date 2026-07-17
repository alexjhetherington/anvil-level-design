import bmesh
import bpy

from .base_test import AnvilTestCase
from .helpers import TEXTURE_PATH
from ..core.materials import (
    create_material_with_image,
    get_primary_image_from_material,
)
from ..operators.texture_apply import (
    _apply_to_other_obj,
    _dispatch_set_uv_from_other_face,
    _flush_other_bmeshes,
)


class MaterialDeduplicationTest(AnvilTestCase):

    def test_textured_object_duplication_preserves_single_shared_anvil_material(self):
        bpy.ops.mesh.primitive_cube_add()
        original = bpy.context.object
        original.name = "TexturedOriginal"

        result = bpy.ops.leveldesign.texture_browser_apply_file(
            filepath=TEXTURE_PATH,
        )

        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(len(original.data.materials), 1)
        original_material = original.data.materials[0]
        material_count_before_duplicate = len(bpy.data.materials)

        duplicate = original.copy()
        duplicate.data = original.data.copy()
        duplicate.name = "TexturedDuplicate"
        bpy.context.collection.objects.link(duplicate)

        self.assertNotEqual(duplicate, original)
        self.assertEqual(len(duplicate.data.materials), 1)
        self.assertEqual(duplicate.data.materials[0], original_material)
        self.assertEqual(len(bpy.data.materials), material_count_before_duplicate)

    def test_two_objects_textured_through_texture_browser_share_single_anvil_material(self):
        bpy.ops.mesh.primitive_plane_add(location=(0.0, 0.0, 0.0))
        first_object = bpy.context.object
        first_object.name = "FirstTextureBrowserObject"

        result = bpy.ops.leveldesign.texture_browser_apply_file(
            filepath=TEXTURE_PATH,
        )

        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(len(first_object.data.materials), 1)
        first_material = first_object.data.materials[0]

        first_object.select_set(False)
        bpy.ops.mesh.primitive_plane_add(location=(2.0, 0.0, 0.0))
        second_object = bpy.context.object
        second_object.name = "SecondTextureBrowserObject"

        result = bpy.ops.leveldesign.texture_browser_apply_file(
            filepath=TEXTURE_PATH,
        )

        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(len(second_object.data.materials), 1)
        self.assertEqual(second_object.data.materials[0], first_material)
        self.assertEqual(
            len([
                mat for mat in bpy.data.materials
                if get_primary_image_from_material(mat) is not None
            ]),
            1,
        )

    def test_two_objects_textured_through_texture_browser_then_alt_click_share_single_anvil_material(self):
        bpy.ops.mesh.primitive_plane_add(location=(0.0, 0.0, 0.0))
        first_object = bpy.context.object
        first_object.name = "TextureBrowserSourceObject"

        result = bpy.ops.leveldesign.texture_browser_apply_file(
            filepath=TEXTURE_PATH,
        )

        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(len(first_object.data.materials), 1)
        first_material = first_object.data.materials[0]

        first_object.select_set(False)
        bpy.ops.mesh.primitive_plane_add(location=(2.0, 0.0, 0.0))
        second_object = bpy.context.object
        second_object.name = "AltClickTargetObject"

        second_object.select_set(False)
        first_object.select_set(True)
        bpy.context.view_layer.objects.active = first_object
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.object.mode_set(mode='EDIT')

        first_bmesh = bmesh.from_edit_mesh(first_object.data)
        first_bmesh.faces.ensure_lookup_table()
        for face in first_bmesh.faces:
            face.select = False
        first_bmesh.faces[0].select = True
        first_bmesh.faces.active = first_bmesh.faces[0]
        bmesh.update_edit_mesh(first_object.data)

        class FakeAltClickOperator:
            pass

        operator = FakeAltClickOperator()
        operator._paint_obj = first_object
        operator._source_face_index = 0
        operator._mat = first_material
        operator._unassign_material = False
        operator._ppm = bpy.context.scene.level_design_props.pixels_per_meter
        operator._other_bmeshes = {}
        operator._paint_visited_other = set()
        operator._cross_object_undo_transaction_id = None

        processed = _apply_to_other_obj(
            operator,
            first_bmesh,
            first_object.data,
            second_object,
            0,
            _dispatch_set_uv_from_other_face,
        )
        self.assertTrue(processed)
        _flush_other_bmeshes(operator)

        second_face = second_object.data.polygons[0]
        self.assertEqual(
            second_object.data.materials[second_face.material_index],
            first_material,
        )
        self.assertEqual(
            len([
                mat for mat in bpy.data.materials
                if get_primary_image_from_material(mat) is not None
            ]),
            1,
        )

    def test_cleanup_unused_materials_removes_unused_anvil_duplicate_and_preserves_non_anvil_materials(self):
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.object
        obj.name = "CleanupMaterialObject"

        image = bpy.data.images.load(TEXTURE_PATH, check_existing=True)
        anvil_material = create_material_with_image(image)
        obj.data.materials.append(anvil_material)
        anvil_duplicate = anvil_material.copy()
        anvil_duplicate.name = "CleanupMaterialCopy"
        anvil_duplicate_name = anvil_duplicate.name

        non_anvil_material = bpy.data.materials.new("NonAnvilMaterial")
        non_anvil_duplicate = non_anvil_material.copy()
        non_anvil_duplicate.name = "NonAnvilMaterialCopy"
        non_anvil_material_name = non_anvil_material.name
        non_anvil_duplicate_name = non_anvil_duplicate.name

        self.assertEqual(anvil_duplicate.users, 0)
        self.assertEqual(non_anvil_material.users, 0)
        self.assertEqual(non_anvil_duplicate.users, 0)

        result = bpy.ops.leveldesign.cleanup_unused_materials()

        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(obj.data.materials[0], anvil_material)
        self.assertIsNone(bpy.data.materials.get(anvil_duplicate_name))
        self.assertIsNotNone(bpy.data.materials.get(non_anvil_material_name))
        self.assertIsNotNone(bpy.data.materials.get(non_anvil_duplicate_name))
