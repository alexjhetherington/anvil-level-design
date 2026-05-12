import bpy
import gpu
from .base_test import AnvilTestCase


# Every bpy.ops.* operator the addon depends on (excluding custom leveldesign/hotspot ops)
_REQUIRED_OPERATORS = [
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
    "object.material_slot_remove",
    "scene.new",
    "object.mode_set",
    "object.modifier_apply",
    "object.select_all",
    "uv.follow_active_quads",
    "uv.unwrap",
    "view3d.view_axis",
    "view3d.walk",
    "wm.call_menu",
    "wm.call_panel",
    "wm.save_mainfile",
    "workspace.reorder_to_front",
]


class APIAvailabilityTest(AnvilTestCase):

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
        # GPU platform fields used by Eevee crash diagnostics.
        for fn_name in ("vendor_get", "renderer_get", "version_get"):
            if not hasattr(gpu.platform, fn_name):
                missing.append(f"gpu.platform.{fn_name}")

        self.assertEqual(
            missing, [],
            f"Missing GPU API symbols: {', '.join(missing)}"
        )

    def test_all_required_bpy_utility_apis_exist(self):
        missing = []
        if not hasattr(bpy.utils, "script_paths"):
            missing.append("bpy.utils.script_paths")

        self.assertEqual(
            missing, [],
            f"Missing bpy utility API symbols: {', '.join(missing)}"
        )

    def test_all_required_blender_data_apis_exist(self):
        missing = []
        mesh = bpy.data.meshes.new("api_availability_mesh")
        try:
            if not hasattr(mesh.materials, "clear"):
                missing.append("Mesh.materials.clear")
        finally:
            bpy.data.meshes.remove(mesh)

        self.assertEqual(
            missing, [],
            f"Missing Blender data APIs: {', '.join(missing)}"
        )
