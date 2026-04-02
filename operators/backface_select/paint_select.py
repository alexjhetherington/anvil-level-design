import bpy
import bmesh
from bpy.types import Operator

from ...core.workspace_check import is_level_design_workspace
from .paint_base import ModalPaintBase
from .raycast import raycast_bvh_skip_backfaces
from .operator import (
    _raycast_element_aware,
    _nearest_edge_on_face,
    _nearest_vert_on_face,
)


class LEVELDESIGN_OT_backface_paint_select(ModalPaintBase, Operator):
    """Paint-select geometry by dragging, ignoring backface-culled faces"""
    bl_idname = "leveldesign.backface_paint_select"
    bl_label = "Paint Select"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not is_level_design_workspace():
            return False
        obj = context.object
        return (obj is not None
                and obj.type == 'MESH'
                and context.mode == 'EDIT_MESH')

    def invoke(self, context, event):
        return self._invoke_paint(context, event)

    def modal(self, context, event):
        return self._modal_paint(context, event)

    def paint_begin(self, context, event):
        bm = bmesh.from_edit_mesh(self._paint_obj.data)
        self._saved_vert_sel = {v.index for v in bm.verts if v.select}
        self._saved_edge_sel = {e.index for e in bm.edges if e.select}
        self._saved_face_sel = {f.index for f in bm.faces if f.select}
        self._select_mode = tuple(context.tool_settings.mesh_select_mode)
        return True

    def paint_sample(self, context, mouse_2d, region, rv3d):
        obj = self._paint_obj
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()

        origin_local, dir_local = self._paint_ray_local(region, rv3d, mouse_2d)

        is_vert_mode = self._select_mode[0]
        is_edge_mode = self._select_mode[1]
        is_face_mode = self._select_mode[2]

        if is_face_mode:
            location, normal, face_index, distance = raycast_bvh_skip_backfaces(
                self._paint_bvh, origin_local, dir_local,
                bm, me.materials, max_iterations=64
            )
            if face_index is None:
                return
            if face_index in self._paint_visited:
                return
            self._paint_visited.add(face_index)

            bm.faces[face_index].select = True
        else:
            face, location, culled_element = _raycast_element_aware(
                self._paint_bvh, origin_local, dir_local,
                bm, me.materials, region, rv3d, obj.matrix_world, mouse_2d,
                is_edge_mode, max_iterations=64
            )
            if face is None:
                return

            if culled_element is not None:
                elem_key = ('culled', culled_element.index)
                if elem_key in self._paint_visited:
                    return
                self._paint_visited.add(elem_key)

                if is_edge_mode:
                    culled_element.select = True
                    for v in culled_element.verts:
                        v.select = True
                else:
                    culled_element.select = True
            else:
                if is_edge_mode:
                    edge = _nearest_edge_on_face(location, face)
                    if edge is None:
                        return
                    elem_key = ('edge', edge.index)
                    if elem_key in self._paint_visited:
                        return
                    self._paint_visited.add(elem_key)
                    edge.select = True
                    for v in edge.verts:
                        v.select = True
                else:
                    vert = _nearest_vert_on_face(location, face)
                    if vert is None:
                        return
                    elem_key = ('vert', vert.index)
                    if elem_key in self._paint_visited:
                        return
                    self._paint_visited.add(elem_key)
                    vert.select = True

    def paint_cancel(self, context):
        bm = bmesh.from_edit_mesh(self._paint_obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        for v in bm.verts:
            v.select = v.index in self._saved_vert_sel
        for e in bm.edges:
            e.select = e.index in self._saved_edge_sel
        for f in bm.faces:
            f.select = f.index in self._saved_face_sel


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_backface_paint_select)


def unregister():
    bpy.utils.unregister_class(LEVELDESIGN_OT_backface_paint_select)
