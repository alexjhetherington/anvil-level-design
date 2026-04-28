"""Shared pivot snapping with box-builder-style default grid fallback."""

import bmesh
from mathutils import Vector

from . import snapping
from . import utils


def selected_vertex_world_coords(active_object, mode):
    """Return world-space positions of selected vertices in the active mesh."""
    if active_object is None or active_object.type != 'MESH' or mode != 'EDIT_MESH':
        return set()

    me = active_object.data
    bm = bmesh.from_edit_mesh(me)
    bm.verts.ensure_lookup_table()
    world = active_object.matrix_world

    coords = set()
    for vert in bm.verts:
        if vert.select and not vert.hide:
            co = world @ vert.co
            coords.add((co.x, co.y, co.z))

    return coords


def calculate_default_grid_snap_3d(context, event, coords):
    """Perspective snap fallback to a horizontal grid plane."""
    if coords:
        z_height = min(co[2] for co in coords)
    else:
        z_height = 0.0

    plane_point = Vector((0, 0, z_height))
    plane_normal = Vector((0, 0, 1))

    point = utils.mouse_to_3d_on_plane(context, event, plane_point, plane_normal)
    if point is None:
        return (None, None, None, False)

    grid_size = utils.get_grid_size(context)
    if utils.is_snapping_enabled(context):
        snapped = snapping.snap_to_grid(point, grid_size)
        snapped.z = z_height
    else:
        snapped = point

    return (snapped, Vector((0, 0, 1)), None, False)


def calculate_default_grid_snap_2d(context, event, coords):
    """Ortho snap fallback using selected vertices to choose the fixed axis."""
    plane_data = utils.get_2d_view_plane(context)
    if plane_data is None:
        return (None, None)

    plane_point, plane_normal, _, _ = plane_data
    view_type = utils.get_view_type(context)

    if coords:
        if view_type == 'TOP':
            plane_point = Vector((0, 0, max(co[2] for co in coords)))
        elif view_type == 'BOTTOM':
            plane_point = Vector((0, 0, min(co[2] for co in coords)))
        elif view_type == 'FRONT':
            plane_point = Vector((0, max(co[1] for co in coords), 0))
        elif view_type == 'BACK':
            plane_point = Vector((0, min(co[1] for co in coords), 0))
        elif view_type == 'RIGHT':
            plane_point = Vector((max(co[0] for co in coords), 0, 0))
        elif view_type == 'LEFT':
            plane_point = Vector((min(co[0] for co in coords), 0, 0))

    point = utils.mouse_to_3d_on_plane(context, event, plane_point, plane_normal)
    if point is None:
        return (None, None)

    grid_size = utils.get_grid_size(context)
    if utils.is_snapping_enabled(context):
        snapped = snapping.snap_to_grid(point, grid_size)
    else:
        snapped = point.copy()

    if view_type in ('TOP', 'BOTTOM'):
        snapped.z = plane_point.z
    elif view_type in ('FRONT', 'BACK'):
        snapped.y = plane_point.y
    elif view_type in ('RIGHT', 'LEFT'):
        snapped.x = plane_point.x

    return (snapped, plane_normal)


class DefaultGridPivotMixin:
    """Pivot snap hooks with the default grid used by box builder."""

    def _selected_vertex_world_coords(self, context):
        return selected_vertex_world_coords(context.active_object, context.mode)

    def _calculate_first_vertex_snap_3d(self, context, event):
        """Perspective snap with fallback to horizontal plane when no face is hit."""
        result = super()._calculate_first_vertex_snap_3d(context, event)
        snapped, face_normal, obj, was_clamped = result

        if snapped is not None:
            return result

        return calculate_default_grid_snap_3d(
            context, event, self._selected_vertex_world_coords(context)
        )

    def _calculate_first_vertex_snap_2d(self, context, event):
        """Ortho snap that uses selected vertices to set the ambiguous axis value."""
        return calculate_default_grid_snap_2d(
            context, event, self._selected_vertex_world_coords(context)
        )
