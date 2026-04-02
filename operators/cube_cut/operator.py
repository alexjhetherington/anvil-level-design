"""
Cube Cut Tool - Main Modal Operator

Thin subclass of ModalDrawBase that executes cube cut geometry.
"""

import bpy

from . import geometry
from ..modal_draw.base_operator import ModalDrawBase, MIN_RECTANGLE_SIZE
from ...core.workspace_check import is_level_design_workspace
from ..weld import set_weld_from_edge_selection, snapshot_coplanar_sides


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

        result = self._execute_action(
            context,
            self._first_vertex,
            self._second_vertex,
            self._depth,
            self._local_x,
            self._local_y,
            self._local_z
        )
        success, message = result[0], result[1]

        if success:
            self.report({'INFO'}, message)
        else:
            self.report({'ERROR'}, message)

        self._cleanup(context)
        return {'FINISHED'}

    def _execute_action(self, context, first_vertex, second_vertex, depth,
                        local_x, local_y, local_z):
        # Snapshot coplanar faces BEFORE the cut modifies geometry
        from mathutils import Vector
        obj = context.active_object
        coplanar_blocked = 0
        if obj and obj.type == 'MESH':
            import bmesh as _bmesh
            bm_snap = _bmesh.from_edit_mesh(obj.data)
            w2l = obj.matrix_world.inverted()
            w2l_rot = w2l.to_3x3()
            snap_origin = w2l @ first_vertex
            snap_lx = (w2l_rot @ Vector(local_x)).normalized()
            snap_ly = (w2l_rot @ Vector(local_y)).normalized()
            diff_world = Vector(second_vertex) - Vector(first_vertex)
            snap_cdx = abs(diff_world.dot(Vector(local_x))) * (w2l_rot @ Vector(local_x)).length
            snap_cdy = abs(diff_world.dot(Vector(local_y))) * (w2l_rot @ Vector(local_y)).length
            snap_diff = (w2l @ Vector(second_vertex)) - snap_origin
            if snap_diff.dot(snap_lx) < 0:
                snap_lx = -snap_lx
            if snap_diff.dot(snap_ly) < 0:
                snap_ly = -snap_ly
            coplanar_blocked = snapshot_coplanar_sides(
                bm_snap, (snap_origin, snap_lx, snap_ly, snap_cdx, snap_cdy))

        result = geometry.execute_cube_cut(
            context, first_vertex, second_vertex, depth,
            local_x, local_y, local_z
        )
        success, message = result

        if success:
            extrude_dir = -local_z
            back_point = first_vertex + local_z * depth
            back_plane_offset = back_point.dot(extrude_dir.normalized())
            from ...core.logging import debug_log
            debug_log(f"[CubeCut] Corridor depth setup: depth={depth:.4f}, abs_depth={abs(depth):.4f}")
            debug_log(f"[CubeCut]   first_vertex={first_vertex}, second_vertex={second_vertex}")
            debug_log(f"[CubeCut]   local_z={local_z}, extrude_dir={extrude_dir}")
            debug_log(f"[CubeCut]   back_point={back_point} (first_vertex + local_z * depth)")
            debug_log(f"[CubeCut]   back_plane_offset={back_plane_offset:.4f} (back_point dot extrude_dir)")
            set_weld_from_edge_selection(
                context, abs(depth), extrude_dir, back_plane_offset,
                first_vertex, second_vertex, local_x, local_y,
                coplanar_blocked,
            )

        return result

    def _get_tool_name(self):
        return "Cube Cut"


def register():
    bpy.utils.register_class(MESH_OT_cube_cut)


def unregister():
    bpy.utils.unregister_class(MESH_OT_cube_cut)
