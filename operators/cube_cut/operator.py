"""
Cube Cut Tool - Main Modal Operator

Thin subclass of ModalDrawBase that executes cube cut geometry.
"""

import bpy

from . import geometry
from ..modal_draw.base_operator import ModalDrawBase
from ...utils import is_level_design_workspace


class MESH_OT_cube_cut(ModalDrawBase, bpy.types.Operator):
    """Cut a cuboid-shaped void from mesh geometry"""
    bl_idname = "leveldesign.cube_cut"
    bl_label = "Cube Cut"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (
            is_level_design_workspace() and
            context.active_object is not None and
            context.active_object.type == 'MESH' and
            context.mode == 'EDIT_MESH'
        )

    def _execute_action(self, context, first_vertex, second_vertex, depth,
                        local_x, local_y, local_z):
        return geometry.execute_cube_cut(
            context, first_vertex, second_vertex, depth,
            local_x, local_y, local_z
        )

    def _get_tool_name(self):
        return "Cube Cut"


def register():
    bpy.utils.register_class(MESH_OT_cube_cut)


def unregister():
    bpy.utils.unregister_class(MESH_OT_cube_cut)
