"""Prism Builder modal operator."""

import json

import bpy
from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty, StringProperty
from mathutils import Vector

from . import geometry
from ..modal_draw.base_operator import ModalDrawBase
from ..modal_draw.default_grid_pivot import DefaultGridPivotMixin
from ..modal_draw.prism_profile import (
    PrismProfileDrawMixin,
    profile_from_json,
    profile_to_json,
)
from ..shape_builder_operator import ShapeBuilderOperatorMixin


class MESH_OT_prism_builder(
        ShapeBuilderOperatorMixin,
        DefaultGridPivotMixin,
        PrismProfileDrawMixin,
        ModalDrawBase,
        bpy.types.Operator):
    """Create an extruded freeform polygon prism."""

    bl_idname = "leveldesign.prism_builder"
    bl_label = "Prism Builder"
    bl_options = {'REGISTER', 'UNDO'}

    name_suffix: StringProperty(
        name="Suffix",
        description="Suffix appended after Blender numbering, e.g. Anvil.Prism.001-col",
        default="",
    )
    keep_anti_parallel_coplanar_faces: BoolProperty(
        name="Keep Overlap Faces",
        description="Keep prism faces that overlap existing faces",
        default=True,
    )
    prefer_quads: BoolProperty(
        name="Prefer Quads",
        description="Join cap triangles into valid quads where possible",
        default=True,
    )

    action_profile_json: StringProperty()
    action_depth: FloatProperty()
    action_local_z: FloatVectorProperty(size=3)
    action_view_forward: FloatVectorProperty(size=3)
    action_had_selection: BoolProperty()
    action_was_edit_mode: BoolProperty()
    action_object_name: StringProperty()

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = layout.column(align=True)
        settings.prop(self, "name_suffix")
        if context.mode == 'EDIT_MESH':
            settings.prop(self, "keep_anti_parallel_coplanar_faces")
        settings.prop(self, "prefer_quads")

    def _execute_action(self, context, first_vertex, second_vertex, depth,
                        local_x, local_y, local_z):
        view_forward = context.region_data.view_rotation @ Vector((0, 0, -1))
        return self._execute_prism_builder(
            context,
            self._profile_vertices,
            depth,
            local_z,
            view_forward,
            context.mode == 'EDIT_MESH',
            context.active_object.name if context.active_object is not None else "",
            self.name_suffix,
        )

    def _execute_prism_builder(
            self, context, profile_vertices, depth, local_z, view_forward,
            action_was_edit_mode, action_object_name, name_suffix):
        if not profile_vertices:
            return (False, "Prism profile must contain at least three points")
        pixels_per_meter = context.scene.level_design_props.pixels_per_meter

        if not action_was_edit_mode:
            result = geometry.execute_prism_builder_object_mode(
                profile_vertices,
                depth,
                local_z,
                view_forward,
                pixels_per_meter,
                self.prefer_quads,
                name_suffix,
            )
            return self._finish_object_builder_result(
                context.active_object,
                result,
                "Prism object created",
            )

        obj = self._restore_edit_action_context(context, action_object_name)
        if obj is None:
            return (False, "No active mesh object")
        result = geometry.execute_prism_builder_edit_mode(
            profile_vertices,
            depth,
            local_z,
            view_forward,
            obj,
            pixels_per_meter,
            self.keep_anti_parallel_coplanar_faces,
            self.prefer_quads,
        )
        return self._finish_edit_builder_result(
            obj,
            result,
            "Prism created",
        )

    def _capture_action_properties(self, context, first_vertex, second_vertex,
                                   depth, local_x, local_y, local_z):
        self.action_profile_json = profile_to_json(self._profile_vertices)
        self.action_depth = depth
        self.action_local_z = local_z
        self.action_view_forward = (
            context.region_data.view_rotation @ Vector((0, 0, -1))
        )
        self.action_had_selection = self._had_selection
        self.action_was_edit_mode = context.mode == 'EDIT_MESH'
        active_object = context.active_object
        self.action_object_name = (
            active_object.name if active_object is not None else ""
        )

    def execute(self, context):
        self._had_selection = self.action_had_selection
        try:
            profile_vertices = profile_from_json(self.action_profile_json)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            result = (False, f"Invalid captured Prism Builder profile: {error}")
        else:
            result = self._execute_prism_builder(
                context,
                profile_vertices,
                self.action_depth,
                Vector(self.action_local_z),
                Vector(self.action_view_forward),
                self.action_was_edit_mode,
                self.action_object_name,
                self.name_suffix,
            )
        return self._report_builder_result(result)

    def _get_tool_name(self):
        return "Prism Builder"


def register():
    bpy.utils.register_class(MESH_OT_prism_builder)


def unregister():
    bpy.utils.unregister_class(MESH_OT_prism_builder)
