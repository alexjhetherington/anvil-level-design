"""
Box Create Tool - Main Modal Operator

Thin subclass of ModalDrawBase that creates box geometry.
"""

import bpy

from . import geometry
from ..modal_draw.base_operator import ModalDrawBase
from ...utils import is_level_design_workspace


class MESH_OT_box_create(ModalDrawBase, bpy.types.Operator):
    """Create a box mesh using the 3-state draw workflow"""
    bl_idname = "leveldesign.box_create"
    bl_label = "Box"
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
        from mathutils import Vector
        from ..modal_draw import utils

        obj = context.active_object
        props = context.scene.level_design_props
        ppm = props.pixels_per_meter

        # Determine plane normal direction from camera view.
        # The camera looks along its local -Z; rotate into world space to
        # get a reliable forward direction in both perspective and ortho.
        # Default winding gives normal in +lz. Reverse when +lz points
        # into the scene (aligns with view_forward) so the plane faces
        # toward the camera.
        # Determine plane normal direction from camera view.
        # Default winding [v0,v1,v2,v3] produces normal in lx×ly direction.
        # This equals local_z only when the axes are right-handed; ortho
        # views can be left-handed, so use the actual cross product.
        # Reverse when that geometric normal points into the scene (aligns
        # with view_forward) so the plane faces toward the camera.
        rv3d = context.region_data
        view_forward = rv3d.view_rotation @ Vector((0, 0, -1))
        geometric_normal = local_x.cross(local_y)
        dot = geometric_normal.dot(view_forward)
        reverse_plane_normal = dot > 0

        return geometry.execute_box_create(
            first_vertex, second_vertex, depth,
            local_x, local_y, local_z,
            obj, ppm, reverse_plane_normal
        )

    def _get_tool_name(self):
        return "Box"


def register():
    bpy.utils.register_class(MESH_OT_box_create)


def unregister():
    bpy.utils.unregister_class(MESH_OT_box_create)
