"""
Cube Cut Tool - Main Modal Operator

Thin subclass of ModalDrawBase that executes cube cut geometry.
"""

import bpy

from . import geometry
from ..modal_draw.base_operator import ModalDrawBase, MIN_RECTANGLE_SIZE
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

    def _confirm_second_vertex(self, context, event):
        """In ortho views, skip the depth step and execute immediately with infinite depth."""
        if not self._is_2d_view:
            return super()._confirm_second_vertex(context, event)

        if self._first_vertex is None or self._second_vertex is None:
            return {'RUNNING_MODAL'}

        # Check minimum size
        diff = self._second_vertex - self._first_vertex
        local_dx = abs(diff.dot(self._local_x))
        local_dy = abs(diff.dot(self._local_y))

        if local_dx < MIN_RECTANGLE_SIZE and local_dy < MIN_RECTANGLE_SIZE:
            return {'RUNNING_MODAL'}

        # Shift both vertices back 5000 units along -local_z so the cut
        # is centered on the geometry, then use depth=10000 to extend
        # 5000 units in each direction.
        offset = self._local_z * -5000
        self._first_vertex = self._first_vertex + offset
        self._second_vertex = self._second_vertex + offset
        self._depth = 10000

        success, message = self._execute_action(
            context,
            self._first_vertex,
            self._second_vertex,
            self._depth,
            self._local_x,
            self._local_y,
            self._local_z
        )

        if success:
            self.report({'INFO'}, message)
        else:
            self.report({'ERROR'}, message)

        self._cleanup(context)
        return {'FINISHED'}

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
