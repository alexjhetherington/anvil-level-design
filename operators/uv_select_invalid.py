import bpy
import bmesh
from bpy.types import Operator

from ..core.workspace_check import is_level_design_workspace
from ..core.uv_layers import get_render_active_uv_layer


class LEVELDESIGN_OT_select_invalid_uvs(Operator):
    """Select all faces with zero-area UVs"""
    bl_idname = "leveldesign.select_invalid_uvs"
    bl_label = "Select Invalid UVs"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() and context.object and context.object.type == 'MESH' and context.mode in {'EDIT_MESH', 'OBJECT'}

    def execute(self, context):
        obj = context.object
        me = obj.data

        was_object_mode = context.mode == 'OBJECT'
        if was_object_mode:
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(me)
        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

        invalid_count = 0
        for face in bm.faces:
            uv_area = self._calc_uv_area(face, uv_layer)
            if uv_area == 0.0:
                face.select = True
                invalid_count += 1
            else:
                face.select = False

        bmesh.update_edit_mesh(me)

        if invalid_count > 0:
            self.report({'WARNING'}, f"Selected {invalid_count} face(s) with invalid UVs")
        else:
            self.report({'INFO'}, "No invalid UVs found")

        return {'FINISHED'}

    def _calc_uv_area(self, face, uv_layer):
        uvs = [loop[uv_layer].uv for loop in face.loops]
        n = len(uvs)
        if n < 3:
            return 0.0
        # Shoelace formula
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += uvs[i].x * uvs[j].y
            area -= uvs[j].x * uvs[i].y
        return abs(area) * 0.5


classes = (
    LEVELDESIGN_OT_select_invalid_uvs,
)

addon_keymaps = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Mesh', space_type='EMPTY')
        kmi = km.keymap_items.new(
            "leveldesign.select_invalid_uvs",
            'NONE', 'PRESS',
            head=True
        )
        addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
