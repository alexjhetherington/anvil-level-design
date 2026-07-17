import bmesh
import bpy
from unittest.mock import patch

from .base_test import AnvilTestCase
from ..core import materials as materials_module
from ..core.material_shader import (
    build_canonical_material_shader,
    validate_material_shader,
)
from ..core.materials import (
    DEFAULT_MATERIAL_NAME_PATTERN,
    create_material_with_image,
    find_material_with_image,
    get_default_material_settings,
    get_primary_image_from_material,
    is_texture_alpha_connected,
    material_name_for_image,
    repair_material_shader,
)
from ..core.material_mappings import (
    analyze_material_mapping,
    apply_material_mapping_choices,
)
from ..handlers.active_image import set_active_image
from ..handlers.active_image import set_previous_image
from ..operators.material_mappings import finish_browser_mapping_selection
from ..panels.uv_panel import _texture_preview_state


def _test_image(name):
    image = bpy.data.images.new(name, width=2, height=2, alpha=True)
    image.pixels[:] = [1.0, 1.0, 1.0, 1.0] * 4
    return image


class MaterialLookupTest(AnvilTestCase):

    def test_material_primary_image_lookup_survives_material_rename(self):
        image = _test_image("lookup.png")
        material = create_material_with_image(image)

        material.name = "Completely Custom Material Name"

        self.assertEqual(find_material_with_image(image), material)

    def test_material_name_pattern_formats_filename_and_extension_without_defining_identity(self):
        image = _test_image("wall.png")

        name = material_name_for_image(image, "IMG_{filename}{extension}")

        self.assertEqual(name, "IMG_wall.png")

    def test_material_name_pattern_uses_blank_relative_path_when_relative_path_cannot_be_computed(self):
        with patch.object(
                materials_module.os.path,
                "relpath",
                side_effect=ValueError("path is on a different drive")):
            relative_path = materials_module._material_relative_path(
                "D:\\textures",
                "C:\\project\\level.blend",
            )

        self.assertEqual(relative_path, "")


class MaterialPreviewTest(AnvilTestCase):

    def test_texture_preview_selected_unmanaged_material_suppresses_previous_image_and_reports_material_state(self):
        previous_image = _test_image("previous_preview.png")
        set_previous_image(previous_image)
        unmanaged_material = bpy.data.materials.new("LegacyUnmanagedMaterial")
        mesh = bpy.data.meshes.new("UnmanagedPreviewMesh")
        mesh.from_pydata(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
            [],
            [(0, 1, 2, 3)],
        )
        mesh.materials.append(unmanaged_material)
        obj = bpy.data.objects.new("UnmanagedPreviewObject", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        with bpy.context.temp_override(**self._get_3d_view_context()):
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(mesh)
        bm.faces.ensure_lookup_table()
        bm.faces[0].select = True
        bm.faces.active = bm.faces[0]
        bmesh.update_edit_mesh(mesh)

        image, material, mixed = _texture_preview_state(
            'EDIT_MESH',
            (False, False, True),
            obj,
        )

        self.assertIsNone(image)
        self.assertEqual(material, unmanaged_material)
        self.assertFalse(mixed)


class MaterialShaderTest(AnvilTestCase):

    def test_material_creation_enables_texture_alpha_when_any_image_pixel_is_transparent(self):
        image = _test_image("transparent.png")
        pixels = list(image.pixels[:])
        pixels[3] = 0.5
        image.pixels[:] = pixels
        bpy.context.scene.level_design_props.default_texture_as_alpha = False

        material = create_material_with_image(image)

        self.assertTrue(is_texture_alpha_connected(material))

    def test_material_shader_builder_creates_every_supported_canonical_variant(self):
        image = _test_image("variants.png")
        defaults = get_default_material_settings()

        for texture_as_alpha, vertex_colors in (
                (False, False), (True, False), (False, True), (True, True)):
            with self.subTest(
                    texture_as_alpha=texture_as_alpha,
                    vertex_colors=vertex_colors):
                material = bpy.data.materials.new(
                    f"Variant_{texture_as_alpha}_{vertex_colors}"
                )
                settings = dict(defaults)
                settings["texture_as_alpha"] = texture_as_alpha
                settings["vertex_colors"] = vertex_colors

                result = build_canonical_material_shader(material, image, settings)

                self.assertTrue(result.is_canonical, result.issues)
                self.assertEqual(result.texture_as_alpha, texture_as_alpha)
                self.assertEqual(result.vertex_colors, vertex_colors)

    def test_material_shader_toggle_operators_keep_shader_canonical(self):
        image = _test_image("toggles.png")
        material = create_material_with_image(image)
        set_active_image(image)

        bpy.ops.leveldesign.toggle_texture_alpha()
        alpha_result = validate_material_shader(material, image)
        self.assertTrue(alpha_result.is_canonical, alpha_result.issues)

        bpy.ops.leveldesign.toggle_vertex_colors()
        combined_result = validate_material_shader(material, image)
        self.assertTrue(combined_result.is_canonical, combined_result.issues)

    def test_material_shader_validator_detects_extra_disconnected_node(self):
        image = _test_image("customized.png")
        material = create_material_with_image(image)
        material.node_tree.nodes.new("ShaderNodeValue")

        result = validate_material_shader(material, image)

        self.assertFalse(result.is_canonical)
        self.assertTrue(any("Unexpected" in issue for issue in result.issues))

    def test_material_shader_validator_warns_only_when_other_shader_images_exist(self):
        primary = _test_image("primary.png")
        other = _test_image("other.png")
        material = create_material_with_image(primary)
        texture = next(
            node for node in material.node_tree.nodes if node.type == 'TEX_IMAGE'
        )

        texture.image = other
        wrong_image = validate_material_shader(material, primary)
        self.assertTrue(wrong_image.has_shader_images)
        self.assertFalse(wrong_image.primary_image_in_shader)

        texture.image = None
        no_images = validate_material_shader(material, primary)
        self.assertFalse(no_images.has_shader_images)
        self.assertFalse(no_images.primary_image_in_shader)

    def test_material_shader_repair_rebuilds_custom_shader_and_restores_primary_image(self):
        image = _test_image("repair.png")
        material = create_material_with_image(image)
        material.node_tree.nodes.new("ShaderNodeValue")

        result = repair_material_shader(material)

        self.assertTrue(result.is_canonical, result.issues)
        self.assertTrue(result.primary_image_in_shader)

    def test_material_shader_repair_operator_undo_and_redo_restore_each_shader_state(self):
        image = _test_image("repair_undo.png")
        material = create_material_with_image(image)
        material.node_tree.nodes.new("ShaderNodeValue")
        material_name = material.name
        image_name = image.name
        bpy.ops.ed.undo_push(message="Before Material Shader Repair")

        result = bpy.ops.leveldesign.repair_material_shader(
            material_name=material_name,
        )

        self.assertEqual(result, {'FINISHED'})
        repaired = bpy.data.materials.get(material_name)
        repaired_image = bpy.data.images.get(image_name)
        self.assertTrue(
            validate_material_shader(repaired, repaired_image).is_canonical
        )

        bpy.ops.ed.undo()
        undone = bpy.data.materials.get(material_name)
        undone_image = bpy.data.images.get(image_name)
        self.assertFalse(
            validate_material_shader(undone, undone_image).is_canonical
        )

        bpy.ops.ed.redo()
        redone = bpy.data.materials.get(material_name)
        redone_image = bpy.data.images.get(image_name)
        self.assertTrue(
            validate_material_shader(redone, redone_image).is_canonical
        )


class FixMaterialMappingsTest(AnvilTestCase):

    def test_fix_material_mappings_browser_selection_updates_existing_review_rows_without_rebuilding_list(self):
        first_material = bpy.data.materials.new("BrowserMappingFirst")
        second_material = bpy.data.materials.new("BrowserMappingSecond")
        image = _test_image("browser_mapping_choice.png")
        rows = bpy.context.window_manager.anvil_material_mapping_rows
        rows.clear()
        first_row = rows.add()
        first_row.material = first_material
        second_row = rows.add()
        second_row.material = second_material

        finish_browser_mapping_selection(first_material.name, image)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].material, first_material)
        self.assertEqual(rows[0].selected_image, image)
        self.assertEqual(rows[1].material, second_material)
        self.assertIsNone(rows[1].selected_image)

    def test_fix_material_mappings_analysis_prefers_image_feeding_base_color_when_shader_has_multiple_images(self):
        primary = _test_image("mapping_primary.png")
        unused = _test_image("mapping_unused.png")
        material = create_material_with_image(primary)
        material.anvil_primary_image = None
        material.anvil_material_schema_version = 0
        extra = material.node_tree.nodes.new("ShaderNodeTexImage")
        extra.image = unused

        analysis = analyze_material_mapping(material)

        self.assertEqual(len(analysis.shader_images), 2)
        self.assertEqual(analysis.suggested_image, primary)

    def test_fix_material_mappings_apply_maps_only_chosen_material_without_reassigning_scene_slots(self):
        image = _test_image("shared_candidate.png")
        first = bpy.data.materials.new("wood_clean")
        second = bpy.data.materials.new("wood_rough")
        mesh = bpy.data.meshes.new("MappingSlotsMesh")
        mesh.materials.append(first)
        mesh.materials.append(second)

        mapped_count, renamed_count = apply_material_mapping_choices(
            [(first, image), (second, None)],
            False,
            DEFAULT_MATERIAL_NAME_PATTERN,
        )

        self.assertEqual(mapped_count, 1)
        self.assertEqual(renamed_count, 0)
        self.assertEqual(get_primary_image_from_material(first), image)
        self.assertIsNone(get_primary_image_from_material(second))
        self.assertEqual(mesh.materials[0], first)
        self.assertEqual(mesh.materials[1], second)

    def test_fix_material_mappings_apply_rejects_two_materials_using_same_primary_image(self):
        image = _test_image("conflict.png")
        first = bpy.data.materials.new("ConflictFirst")
        second = bpy.data.materials.new("ConflictSecond")

        with self.assertRaises(ValueError):
            apply_material_mapping_choices(
                [(first, image), (second, image)],
                False,
                DEFAULT_MATERIAL_NAME_PATTERN,
            )
