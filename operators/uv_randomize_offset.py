import random

import bmesh
import bpy
from bpy.types import Operator

from ..core.face_id import get_face_id_layer
from ..core.hotspot_queries import face_has_hotspot_material
from ..core.uv_layers import get_render_active_uv_layer
from ..core.uv_projection import apply_uv_to_face, derive_transform_from_uvs
from ..core.workspace_check import is_level_design_workspace
from ..handlers import (
    cache_single_face,
    mark_multi_face_set_offset,
    mark_multi_face_unset_offset,
)
from ..properties import set_updating_from_selection


def randomize_uv_offset_on_selected_faces(obj, props, axis, random_value_fn):
    """Randomise one UV offset axis on selected non-hotspot faces."""
    if not obj or obj.type != 'MESH':
        return 0, None

    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    uv_layer = get_render_active_uv_layer(bm, me)
    if uv_layer is None:
        return 0, None

    get_face_id_layer(bm)
    selected_faces = [face for face in bm.faces if face.select]
    if not selected_faces:
        return 0, None

    ppm = props.pixels_per_meter
    affected_count = 0
    first_offsets = None

    for face in selected_faces:
        if face_has_hotspot_material(face, me):
            continue

        current = derive_transform_from_uvs(face, uv_layer, ppm, me)
        if current is None:
            continue

        offset_x = current['offset_x']
        offset_y = current['offset_y']
        random_offset = random_value_fn()
        if axis == 'X':
            offset_x = random_offset
        else:
            offset_y = random_offset

        mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
        apply_uv_to_face(
            face, uv_layer,
            current['scale_u'], current['scale_v'], current['rotation'],
            offset_x, offset_y,
            mat, ppm, me,
        )
        cache_single_face(face, bm, ppm, me)

        affected_count += 1
        if first_offsets is None:
            first_offsets = (offset_x, offset_y)

    return affected_count, first_offsets


class LEVELDESIGN_OT_randomize_uv_offset(Operator):
    """Randomise the selected faces' UV offset"""
    bl_idname = "leveldesign.randomize_uv_offset"
    bl_label = "Randomise UV Offset"
    bl_description = "Randomise the selected faces' UV offset"
    bl_options = {'REGISTER', 'UNDO'}

    axis: bpy.props.EnumProperty(
        name="Axis",
        items=[
            ('X', "X", "Randomise X offset"),
            ('Y', "Y", "Randomise Y offset"),
        ],
    )

    @classmethod
    def poll(cls, context):
        return (
            is_level_design_workspace()
            and context.object
            and context.object.type == 'MESH'
            and context.mode == 'EDIT_MESH'
            and context.tool_settings.mesh_select_mode[2]
        )

    @classmethod
    def description(cls, context, properties):
        if properties.axis == 'X':
            return "Randomise X offset for selected faces"
        return "Randomise Y offset for selected faces"

    def execute(self, context):
        obj = context.object
        props = context.scene.level_design_props
        affected_count, first_offsets = randomize_uv_offset_on_selected_faces(
            obj, props, self.axis, random.random,
        )

        if affected_count == 0:
            self.report({'WARNING'}, "No non-hotspot faces selected")
            return {'CANCELLED'}

        if affected_count > 1:
            mark_multi_face_unset_offset()
        else:
            mark_multi_face_set_offset()

        if first_offsets is not None:
            set_updating_from_selection(True)
            try:
                props.texture_offset_x = first_offsets[0]
                props.texture_offset_y = first_offsets[1]
            finally:
                set_updating_from_selection(False)

        return {'FINISHED'}


classes = (
    LEVELDESIGN_OT_randomize_uv_offset,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
