"""
Shared base for modal paint-over-geometry operators.

Provides the modal loop with mouse interpolation between events.
Subclasses implement paint_begin, paint_sample, paint_finish, paint_cancel.
"""

import math

import bmesh
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.bvhtree import BVHTree


class ModalPaintBase:
    """Mixin for modal operators that paint over geometry by dragging.

    Call _invoke_paint from invoke and _modal_paint from modal.
    Override paint_begin, paint_sample, paint_finish, paint_cancel.
    """

    _paint_obj = None
    _paint_bvh = None
    _paint_prev_mouse = None
    _paint_visited = None
    _paint_sample_spacing = 10  # pixels between interpolated samples

    def paint_begin(self, context, event):
        """Called before entering modal. Return True to proceed."""
        return True

    def paint_sample(self, context, mouse_2d, region, rv3d):
        """Called for each interpolated sample point during drag."""
        pass

    def paint_finish(self, context):
        """Called on LMB release."""
        pass

    def paint_cancel(self, context):
        """Called on ESC/RMB. Should restore original state."""
        pass

    def _build_paint_state(self, context, event):
        """Build shared state for the paint session."""
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()

        self._paint_obj = obj
        self._paint_bvh = BVHTree.FromBMesh(bm)
        self._paint_visited = set()
        self._paint_prev_mouse = (event.mouse_region_x, event.mouse_region_y)

    def _invoke_paint(self, context, event):
        """Start the paint modal. Call from subclass invoke."""
        self._build_paint_state(context, event)

        if not self.paint_begin(context, event):
            return {'CANCELLED'}

        region = context.region
        rv3d = context.region_data

        # Process the initial click point
        mouse_2d = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
        self.paint_sample(context, mouse_2d, region, rv3d)

        me = self._paint_obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(me)
        context.area.tag_redraw()

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _modal_paint(self, context, event):
        """Handle modal events. Call from subclass modal."""
        if event.type == 'MOUSEMOVE':
            region = context.region
            rv3d = context.region_data

            curr = (event.mouse_region_x, event.mouse_region_y)
            prev = self._paint_prev_mouse

            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > 0:
                steps = max(1, int(dist / self._paint_sample_spacing))
                for i in range(1, steps + 1):
                    t = i / steps
                    mx = prev[0] + dx * t
                    my = prev[1] + dy * t
                    self.paint_sample(context, Vector((mx, my)), region, rv3d)

            self._paint_prev_mouse = curr

            me = self._paint_obj.data
            bm = bmesh.from_edit_mesh(me)
            bm.select_flush_mode()
            bmesh.update_edit_mesh(me)
            context.area.tag_redraw()

            return {'RUNNING_MODAL'}

        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self.paint_finish(context)
            me = self._paint_obj.data
            bm = bmesh.from_edit_mesh(me)
            bm.select_flush_mode()
            bmesh.update_edit_mesh(me)
            return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.paint_cancel(context)
            me = self._paint_obj.data
            bm = bmesh.from_edit_mesh(me)
            bm.select_flush_mode()
            bmesh.update_edit_mesh(me)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _paint_ray_local(self, region, rv3d, mouse_2d):
        """Convert a 2D mouse position to a local-space ray."""
        coord = (mouse_2d.x, mouse_2d.y)
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        matrix_inv = self._paint_obj.matrix_world.inverted()
        origin_local = matrix_inv @ ray_origin
        dir_local = (matrix_inv.to_3x3() @ view_vector).normalized()

        return origin_local, dir_local
