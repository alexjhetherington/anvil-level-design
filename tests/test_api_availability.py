import bpy
import gpu
from mathutils.bvhtree import BVHTree
from .base_test import AnvilTestCase


# Every bpy.ops.* operator the addon depends on (excluding custom leveldesign/hotspot ops)
_REQUIRED_OPERATORS = [
    "collection.exporter_add",
    "collection.exporter_export",
    "ed.redo",
    "ed.undo",
    "ed.undo_push",
    "export_scene.gltf",
    "file.make_paths_relative",
    "mesh.bridge_edge_loops",
    "mesh.separate",
    "mesh.edge_face_add",
    "mesh.flip_normals",
    "mesh.select_all",
    "mesh.select_edge_loop_multi",
    "mesh.select_edge_ring_multi",
    "mesh.select_linked",
    "mesh.shortest_path_pick",
    "mesh.shortest_path_select",
    "object.material_slot_remove",
    "scene.new",
    "object.mode_set",
    "object.modifier_apply",
    "object.select_all",
    "render.opengl",
    "screen.region_flip",
    "screen.userpref_show",
    "uv.follow_active_quads",
    "uv.unwrap",
    "view3d.view_axis",
    "view3d.view_persportho",
    "view3d.view_selected",
    "view3d.walk",
    "wm.call_menu",
    "wm.call_panel",
    "wm.lib_reload",
    "wm.save_mainfile",
    "wm.window_close",
    "workspace.reorder_to_front",
]


class APIAvailabilityTest(AnvilTestCase):

    def test_related_blender_default_perspective_orthographic_keymap_item_is_available_for_preferences_display(self):
        keymap = bpy.context.window_manager.keyconfigs.default.keymaps.get("3D View")
        found = False
        if keymap:
            for keymap_item in keymap.keymap_items:
                if keymap_item.idname == "view3d.view_persportho":
                    found = True
                    break

        self.assertTrue(
            found,
            "Missing related Blender default keymap item: 3D View -> view3d.view_persportho"
        )

    def test_toolbar_tool_select_panel_helper_api_is_available_for_visible_select_toolbar_placement(self):
        missing = []
        try:
            from bl_ui.space_toolsystem_common import ToolDef, ToolSelectPanelHelper
        except ImportError:
            missing.append("bl_ui.space_toolsystem_common")
        else:
            if not hasattr(ToolSelectPanelHelper, "_tool_class_from_space_type"):
                missing.append("ToolSelectPanelHelper._tool_class_from_space_type")
            if not issubclass(ToolDef, tuple):
                missing.append("ToolDef tuple behavior")

        self.assertEqual(
            missing, [],
            f"Missing toolbar tool API symbols: {', '.join(missing)}"
        )

    def test_all_required_operators_exist(self):
        missing = []
        for op_path in _REQUIRED_OPERATORS:
            parts = op_path.split(".")
            op = getattr(getattr(bpy.ops, parts[0], None), parts[1], None)
            if op is None:
                missing.append(f"bpy.ops.{op_path}")
                continue
            try:
                op.get_rna_type()
            except KeyError:
                missing.append(f"bpy.ops.{op_path}")

        self.assertEqual(
            missing, [],
            f"Missing Blender operators: {', '.join(missing)}"
        )

    def test_all_required_gpu_apis_exist(self):
        missing = []
        # gpu.types classes used by grid_overlay
        for cls_name in ("GPUShaderCreateInfo", "GPUStageInterfaceInfo", "GPUTexture"):
            if not hasattr(gpu.types, cls_name):
                missing.append(f"gpu.types.{cls_name}")
        # gpu.shader functions used by grid_overlay
        if not hasattr(gpu.shader, "create_from_info"):
            missing.append("gpu.shader.create_from_info")
        # gpu.texture functions used by texture-backed modal previews.
        if not hasattr(gpu, "texture"):
            missing.append("gpu.texture")
        elif not hasattr(gpu.texture, "from_image"):
            missing.append("gpu.texture.from_image")
        # gpu.state functions used by grid_overlay
        if not hasattr(gpu.state, "active_framebuffer_get"):
            missing.append("gpu.state.active_framebuffer_get")
        # GPUTexture.filter_mode used by the UV transform modal's
        # ghost texture preview to match the material's interpolation.
        if not hasattr(gpu.types.GPUTexture, "filter_mode"):
            missing.append("gpu.types.GPUTexture.filter_mode")
        self.assertEqual(
            missing, [],
            f"Missing GPU API symbols: {', '.join(missing)}"
        )

    def test_all_required_mathutils_bvh_apis_exist(self):
        missing = []
        if not hasattr(BVHTree, "FromPolygons"):
            missing.append("BVHTree.FromPolygons")
        else:
            try:
                BVHTree.FromPolygons(
                    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                    [(0, 1, 2)],
                )
            except TypeError:
                missing.append("BVHTree.FromPolygons(vertices, polygons)")

        self.assertEqual(
            missing, [],
            f"Missing mathutils BVH API symbols: {', '.join(missing)}"
        )

    def test_all_required_blender_data_apis_exist(self):
        missing = []
        if not hasattr(bpy.context.window, "screen"):
            missing.append("Window.screen")
        if not hasattr(bpy.context.window, "cursor_warp"):
            missing.append("Window.cursor_warp")
        screen = bpy.context.window.screen
        if not any(hasattr(area, "ui_type") for area in screen.areas):
            missing.append("Area.ui_type")
        area = next((candidate for candidate in screen.areas if hasattr(candidate, "spaces")), None)
        if area is None:
            missing.append("Area.spaces")
        else:
            space = area.spaces.active
            if not hasattr(space, "show_region_header"):
                missing.append("Space.show_region_header")
            header_region = next(
                (region for region in area.regions if region.type == 'HEADER'),
                None,
            )
            if header_region is None:
                missing.append("Area.regions HEADER")
            elif not hasattr(header_region, "alignment"):
                missing.append("Region.alignment")
        if not hasattr(bpy.context.preferences, "active_section"):
            missing.append("Preferences.active_section")
        if not hasattr(bpy.context.preferences, "system"):
            missing.append("Preferences.system")
        else:
            if not hasattr(bpy.context.preferences.system, "ui_scale"):
                missing.append("PreferencesSystem.ui_scale")
            if not hasattr(bpy.context.preferences.system, "pixel_size"):
                missing.append("PreferencesSystem.pixel_size")
        if not hasattr(bpy.context.preferences, "themes"):
            missing.append("Preferences.themes")
        elif len(bpy.context.preferences.themes) == 0:
            missing.append("Preferences.themes populated")
        elif not hasattr(bpy.context.preferences.themes[0], "view_3d"):
            missing.append("Theme.view_3d")
        elif not hasattr(bpy.context.preferences.themes[0].view_3d, "extra_edge_len"):
            missing.append("ThemeView3D.extra_edge_len")
        if not hasattr(bpy.context.scene, "cursor"):
            missing.append("Scene.cursor")
        elif not hasattr(bpy.context.scene.cursor, "location"):
            missing.append("Scene.cursor.location")
        if not hasattr(bpy.context.scene.unit_settings, "use_separate"):
            missing.append("UnitSettings.use_separate")
        if not hasattr(bpy.utils, "units"):
            missing.append("bpy.utils.units")
        elif not hasattr(bpy.utils.units, "to_string"):
            missing.append("bpy.utils.units.to_string")
        if not hasattr(bpy.utils, "user_resource"):
            missing.append("bpy.utils.user_resource")
        if not hasattr(bpy.types, "UILayout"):
            missing.append("bpy.types.UILayout")
        elif "operator_context" not in bpy.types.UILayout.bl_rna.properties:
            missing.append("UILayout.operator_context")
        for userpref_type_name in ("USERPREF_HT_header", "USERPREF_PT_navigation_bar", "USERPREF_PT_addons"):
            if not hasattr(bpy.types, userpref_type_name):
                missing.append(f"bpy.types.{userpref_type_name}")
        if not hasattr(bpy.types, "SpacePreferences"):
            missing.append("bpy.types.SpacePreferences")
        else:
            if not hasattr(bpy.types.SpacePreferences, "draw_handler_add"):
                missing.append("bpy.types.SpacePreferences.draw_handler_add")
            if not hasattr(bpy.types.SpacePreferences, "draw_handler_remove"):
                missing.append("bpy.types.SpacePreferences.draw_handler_remove")

        mesh = bpy.data.meshes.new("api_availability_mesh")
        material = bpy.data.materials.new("api_availability_material")
        obj = bpy.data.objects.new("api_availability_object", mesh)
        collection = bpy.data.collections.new("api_availability_collection")
        try:
            mesh.materials.append(material)
            if not hasattr(mesh, "calc_loop_triangles"):
                missing.append("Mesh.calc_loop_triangles")
            if not hasattr(mesh, "loop_triangles"):
                missing.append("Mesh.loop_triangles")
            if hasattr(mesh, "calc_loop_triangles") and hasattr(mesh, "loop_triangles"):
                mesh.from_pydata(
                    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                    [],
                    [(0, 1, 2)],
                )
                mesh.update()
                mesh.calc_loop_triangles()
                if len(mesh.loop_triangles) == 0:
                    missing.append("Mesh.loop_triangles populated triangles")
                elif not hasattr(mesh.loop_triangles[0], "loops"):
                    missing.append("MeshLoopTriangle.loops")
            if not hasattr(mesh.materials, "clear"):
                missing.append("Mesh.materials.clear")
            if not hasattr(obj, "material_slots"):
                missing.append("Object.material_slots")
            elif len(obj.material_slots) == 0:
                missing.append("Object.material_slots populated from mesh materials")
            else:
                material_slot = obj.material_slots[0]
                if not hasattr(material_slot, "link"):
                    missing.append("MaterialSlot.link")
                if not hasattr(material_slot, "material"):
                    missing.append("MaterialSlot.material")
            if not hasattr(material, "copy"):
                missing.append("Material.copy")
            if not hasattr(bpy.data.collections, "get"):
                missing.append("BlendDataCollections.get")
            if not hasattr(bpy.data.collections, "remove"):
                missing.append("BlendDataCollections.remove")
            if not hasattr(bpy.data.materials, "remove"):
                missing.append("BlendDataMaterials.remove")
            if not hasattr(bpy.data.objects, "remove"):
                missing.append("BlendDataObjects.remove")
            if not hasattr(bpy.data.meshes, "remove"):
                missing.append("BlendDataMeshes.remove")
            if not hasattr(bpy.data.meshes, "new_from_object"):
                missing.append("BlendDataMeshes.new_from_object")
            if not hasattr(bpy.data.curves, "remove"):
                missing.append("BlendDataCurves.remove")
            if not hasattr(collection, "children"):
                missing.append("Collection.children")
            if not hasattr(collection, "exporters"):
                missing.append("Collection.exporters")
        finally:
            bpy.data.objects.remove(obj)
            bpy.data.meshes.remove(mesh)
            bpy.data.materials.remove(material)
            bpy.data.collections.remove(collection)

        self.assertEqual(
            missing, [],
            f"Missing Blender data APIs: {', '.join(missing)}"
        )

    def test_all_required_asset_apis_exist(self):
        missing = []
        mesh = bpy.data.meshes.new("api_asset_mesh")
        obj = bpy.data.objects.new("api_asset_object", mesh)
        collection = bpy.data.collections.new("api_asset_collection")
        try:
            for data_name, data_block in (
                    ("bpy.types.Object", obj),
                    ("bpy.types.Collection", collection)):
                for fn_name in ("asset_mark", "asset_clear", "preview_ensure"):
                    if not hasattr(data_block, fn_name):
                        missing.append(f"{data_name}.{fn_name}")
            if not hasattr(obj, "make_local"):
                missing.append("bpy.types.Object.make_local")
        finally:
            bpy.data.objects.remove(obj)
            bpy.data.meshes.remove(mesh)
            bpy.data.collections.remove(collection)
        try:
            bpy.data.libraries.load(
                "//anvil_missing_asset_api_check.blend",
                link=False,
                recursive=True,
                reuse_local_id=False,
                assets_only=True,
                clear_asset_data=True,
            )
        except TypeError:
            missing.append("bpy.data.libraries.load(..., assets_only=True)")
        if not hasattr(bpy.data.libraries, "write"):
            missing.append("bpy.data.libraries.write")

        self.assertEqual(
            missing, [],
            f"Missing asset API symbols: {', '.join(missing)}"
        )
