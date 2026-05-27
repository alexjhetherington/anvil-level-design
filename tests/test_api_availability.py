import bpy
import gpu
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
    "uv.follow_active_quads",
    "uv.unwrap",
    "view3d.view_axis",
    "view3d.view_persportho",
    "view3d.walk",
    "wm.call_menu",
    "wm.call_panel",
    "wm.save_mainfile",
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

    def test_all_required_blender_data_apis_exist(self):
        missing = []
        screen = bpy.context.window.screen
        if not any(hasattr(area, "ui_type") for area in screen.areas):
            missing.append("Area.ui_type")

        mesh = bpy.data.meshes.new("api_availability_mesh")
        collection = bpy.data.collections.new("api_availability_collection")
        try:
            if not hasattr(mesh.materials, "clear"):
                missing.append("Mesh.materials.clear")
            if not hasattr(bpy.data.collections, "get"):
                missing.append("BlendDataCollections.get")
            if not hasattr(bpy.data.collections, "remove"):
                missing.append("BlendDataCollections.remove")
            if not hasattr(bpy.data.objects, "remove"):
                missing.append("BlendDataObjects.remove")
            if not hasattr(bpy.data.meshes, "remove"):
                missing.append("BlendDataMeshes.remove")
            if not hasattr(bpy.data.curves, "remove"):
                missing.append("BlendDataCurves.remove")
            if not hasattr(collection, "children"):
                missing.append("Collection.children")
            if not hasattr(collection, "exporters"):
                missing.append("Collection.exporters")
        finally:
            bpy.data.meshes.remove(mesh)
            bpy.data.collections.remove(collection)

        self.assertEqual(
            missing, [],
            f"Missing Blender data APIs: {', '.join(missing)}"
        )
