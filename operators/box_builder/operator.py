"""
Box Builder - Main Modal Operator

Thin subclass of ModalDrawBase that creates box geometry.
"""

import bpy
import bmesh
from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty, StringProperty
from mathutils import Vector

from . import geometry
from ..modal_draw.base_operator import ModalDrawBase
from ..modal_draw.default_grid_pivot import (
    DefaultGridPivotMixin,
    selected_vertex_world_coords,
)
from ...core.workspace_check import is_level_design_workspace
from ..weld import set_weld_from_box_builder, set_weld_from_box_builder_object_mode


_ADDON_PACKAGE = __package__.split(".")[0]


def _get_selected_vertex_world_coords(context):
    """Return world-space positions of selected vertices in the active mesh.

    Respects the current selection mode (vertex/edge/face). Returns an empty
    set when nothing is selected or not in edit mode.
    """
    return selected_vertex_world_coords(context.active_object, context.mode)


class MESH_OT_box_builder(DefaultGridPivotMixin, ModalDrawBase, bpy.types.Operator):
    """Create a box mesh using the 3-state draw workflow"""
    bl_idname = "leveldesign.box_builder"
    bl_label = "Box Builder"
    bl_options = {'REGISTER', 'UNDO'}

    action_first_vertex: FloatVectorProperty(
        size=3,
    )
    action_second_vertex: FloatVectorProperty(
        size=3,
    )
    action_depth: FloatProperty()
    action_local_x: FloatVectorProperty(
        size=3,
    )
    action_local_y: FloatVectorProperty(
        size=3,
    )
    action_local_z: FloatVectorProperty(
        size=3,
    )
    action_had_selection: BoolProperty()
    action_was_edit_mode: BoolProperty()
    action_object_name: StringProperty()
    action_name_suffix: StringProperty()
    keep_anti_parallel_coplanar_faces: BoolProperty(
        name="Keep Overlap Faces",
        description="Keep box faces that overlap existing faces",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        if not is_level_design_workspace():
            return False
        if context.mode == 'OBJECT':
            return True
        return (
            context.active_object is not None and
            context.active_object.type == 'MESH' and
            context.mode == 'EDIT_MESH'
        )

    def invoke(self, context, event):
        self._had_selection = bool(_get_selected_vertex_world_coords(context))
        return super().invoke(context, event)

    def _is_valid_mode(self, context):
        return context.mode in ('EDIT_MESH', 'OBJECT')

    def draw(self, context):
        self.layout.prop(self, "keep_anti_parallel_coplanar_faces")

    def _restore_edit_action_context(self, context, object_name):
        obj = bpy.data.objects.get(object_name)
        if obj is None or obj.type != 'MESH':
            return None

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        context.view_layer.objects.active = obj
        obj.select_set(True)

        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')

        return obj

    def _execute_action(self, context, first_vertex, second_vertex, depth,
                        local_x, local_y, local_z, action_was_edit_mode,
                        action_object_name, action_name_suffix):
        # The first clicked point becomes the box pivot.
        rv3d = context.region_data
        view_forward = rv3d.view_rotation @ Vector((0, 0, -1))

        props = context.scene.level_design_props
        ppm = props.pixels_per_meter

        if not action_was_edit_mode:
            result = geometry.execute_box_builder_object_mode(
                first_vertex, second_vertex, depth,
                local_x, local_y, local_z,
                ppm, view_forward, action_name_suffix
            )
            is_box = result[0] and result[1] == "Box object created"
            if is_box:
                set_weld_from_box_builder_object_mode(context.active_object)
                props.weld_mode = 'INVERT'
            return result

        obj = self._restore_edit_action_context(context, action_object_name)
        if obj is None:
            return (False, "No active mesh object")

        result = geometry.execute_box_builder(
            first_vertex, second_vertex, depth,
            local_x, local_y, local_z,
            obj, ppm, view_forward,
            self.keep_anti_parallel_coplanar_faces
        )

        is_box = result[0] and result[1] == "Box created"
        if is_box:
            new_face_verts = result[2] if len(result) > 2 else []
            set_weld_from_box_builder(context, new_face_verts)

        # If there was no selection before invoking, deselect everything
        if result[0] and not self._had_selection:
            me = obj.data
            bm = bmesh.from_edit_mesh(me)
            for f in bm.faces:
                f.select = False
            for e in bm.edges:
                e.select = False
            for v in bm.verts:
                v.select = False
            bm.select_flush(False)
            bmesh.update_edit_mesh(me)

        return result

    def _capture_action_properties(self, context, first_vertex, second_vertex,
                                   depth, local_x, local_y, local_z):
        self.action_first_vertex = first_vertex
        self.action_second_vertex = second_vertex
        self.action_depth = depth
        self.action_local_x = local_x
        self.action_local_y = local_y
        self.action_local_z = local_z
        self.action_had_selection = self._had_selection
        self.action_was_edit_mode = context.mode == 'EDIT_MESH'
        active_object = context.active_object
        self.action_object_name = active_object.name if active_object is not None else ""
        addon = context.preferences.addons.get(_ADDON_PACKAGE)
        if addon is None:
            self.action_name_suffix = ""
        else:
            self.action_name_suffix = addon.preferences.pref_box_builder_name_suffix

    def execute(self, context):
        self._had_selection = self.action_had_selection
        first_vertex = Vector(self.action_first_vertex)
        second_vertex = Vector(self.action_second_vertex)
        local_x = Vector(self.action_local_x)
        local_y = Vector(self.action_local_y)
        local_z = Vector(self.action_local_z)

        result = self._execute_action(
            context, first_vertex, second_vertex, self.action_depth,
            local_x, local_y, local_z,
            self.action_was_edit_mode, self.action_object_name,
            self.action_name_suffix
        )
        self._last_action_result = result

        success, message = result[0], result[1]
        if success:
            self.report({'INFO'}, message)
            self._action_reported = True
            return {'FINISHED'}

        self.report({'ERROR'}, message)
        self._action_reported = True
        return {'CANCELLED'}

    def _get_tool_name(self):
        return "Box Builder"


def register():
    bpy.utils.register_class(MESH_OT_box_builder)


def unregister():
    bpy.utils.unregister_class(MESH_OT_box_builder)
