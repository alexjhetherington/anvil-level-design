"""Detect and highlight overlapping coplanar faces in the mesh.

Provides a toggle in the Debug panel that scans the active edit-mode mesh
for faces that lie on the same plane and overlap (fully or partially),
then draws a semi-transparent red overlay on top of them.
"""

import bpy
import bmesh
import gpu
from bpy.types import Operator
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from ..core.logging import debug_log
from ..core.workspace_check import is_level_design_workspace


# ---------------------------------------------------------------------------
#  Module state
# ---------------------------------------------------------------------------

_draw_handler = None
# Tri lists for same-normal (red) and opposite-normal (blue) overlaps
_overlay_tris_same = []
_overlay_tris_opposite = []
# Count of overlapping faces found (for UI reporting)
_overlap_count = 0


# ---------------------------------------------------------------------------
#  2-D geometry helpers
# ---------------------------------------------------------------------------

def _polygon_area(poly):
    """Signed area of a 2-D polygon via the shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return area * 0.5


def _line_intersect(p1, p2, p3, p4):
    """Intersection point of lines (p1→p2) and (p3→p4), or None if parallel."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _sutherland_hodgman(subject, clip):
    """Clip *subject* polygon by *clip* polygon (Sutherland-Hodgman).

    Both are lists of (x, y) tuples.  Returns the clipped polygon (may be
    empty if there is no overlap).
    """
    output = list(subject)
    n_clip = len(clip)
    for i in range(n_clip):
        if len(output) == 0:
            return []
        edge_a = clip[i]
        edge_b = clip[(i + 1) % n_clip]
        ex, ey = edge_b[0] - edge_a[0], edge_b[1] - edge_a[1]

        inp = output
        output = []
        n_inp = len(inp)
        for j in range(n_inp):
            cur = inp[j]
            nxt = inp[(j + 1) % n_inp]
            # Signed distance from clip edge (positive = inside)
            d_cur = ex * (cur[1] - edge_a[1]) - ey * (cur[0] - edge_a[0])
            d_nxt = ex * (nxt[1] - edge_a[1]) - ey * (nxt[0] - edge_a[0])
            if d_cur >= 0:
                output.append(cur)
                if d_nxt < 0:
                    pt = _line_intersect(cur, nxt, edge_a, edge_b)
                    if pt is not None:
                        output.append(pt)
            elif d_nxt >= 0:
                pt = _line_intersect(cur, nxt, edge_a, edge_b)
                if pt is not None:
                    output.append(pt)
                output.append(nxt)
    return output


# Minimum intersection area (in square local units) to count as overlap.
# Adjacent faces sharing only an edge produce a degenerate strip with ~zero area.
_AREA_EPSILON = 1e-3


def _polygons_overlap_2d(poly_a, poly_b):
    """Return True if two 2-D polygons share non-trivial area."""
    clipped = _sutherland_hodgman(poly_a, poly_b)
    if len(clipped) < 3:
        return False
    return abs(_polygon_area(clipped)) > _AREA_EPSILON


# ---------------------------------------------------------------------------
#  Overlap detection
# ---------------------------------------------------------------------------

_PLANE_NORMAL_TOL = 0.01   # tolerance for normal comparison
_PLANE_DIST_TOL = 0.001    # tolerance for plane distance (metres)


def _canonicalize_normal(n):
    """Return a canonical direction for *n* so that opposite-facing coplanar
    faces end up in the same bucket."""
    if n.z < -_PLANE_NORMAL_TOL:
        return -n
    if abs(n.z) <= _PLANE_NORMAL_TOL:
        if n.y < -_PLANE_NORMAL_TOL:
            return -n
        if abs(n.y) <= _PLANE_NORMAL_TOL and n.x < 0:
            return -n
    return n.copy()


def _plane_key(normal, dist):
    """Quantised (normal, distance) key for grouping faces by plane."""
    # Round to coarse grid to handle floating-point noise
    def _r(v):
        return round(v / _PLANE_NORMAL_TOL) * _PLANE_NORMAL_TOL

    return (
        round(normal.x, 3),
        round(normal.y, 3),
        round(normal.z, 3),
        round(dist / _PLANE_DIST_TOL) * _PLANE_DIST_TOL,
    )


def _project_face_2d(face, u_axis, v_axis, origin):
    """Project a BMFace onto the (u, v) plane and return list of (x, y)."""
    pts = []
    for vert in face.verts:
        delta = vert.co - origin
        pts.append((delta.dot(u_axis), delta.dot(v_axis)))
    return pts


def find_overlapping_faces(bm):
    """Return two sets of face indices: (same_normal, opposite_normal).

    *same_normal* – faces overlapping another face with the same normal direction.
    *opposite_normal* – faces overlapping another face with the opposite normal.

    A face may appear in both sets if it overlaps faces of both kinds.
    Only detects coplanar overlap (faces on the same plane whose 2-D
    projections share area).
    """
    # --- Step 1: group faces by approximate plane ---
    plane_groups = {}
    face_canon = {}  # face index → canonical normal used for grouping
    for face in bm.faces:
        n = face.normal
        if n.length < 1e-6:
            continue
        n = n.normalized()
        canon = _canonicalize_normal(n)
        face_canon[face.index] = canon
        dist = canon.dot(face.calc_center_median())
        key = _plane_key(canon, dist)
        plane_groups.setdefault(key, []).append(face)

    # --- Step 2: pairwise overlap check within each plane group ---
    same_normal = set()
    opposite_normal = set()
    for faces in plane_groups.values():
        if len(faces) < 2:
            continue

        # Build 2-D projection axes from the first face in the group
        ref = faces[0]
        n = ref.normal.normalized()
        # Robust U axis: try using an edge, fall back to cross product
        edge_vec = ref.verts[1].co - ref.verts[0].co
        if edge_vec.length < 1e-8:
            u_axis = n.orthogonal().normalized()
        else:
            u_axis = edge_vec.normalized()
        v_axis = n.cross(u_axis).normalized()
        origin = ref.verts[0].co

        # Pre-project all faces
        projected = []
        for face in faces:
            projected.append(_project_face_2d(face, u_axis, v_axis, origin))

        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                if _polygons_overlap_2d(projected[i], projected[j]):
                    # Determine if normals point the same or opposite way
                    dot = faces[i].normal.dot(faces[j].normal)
                    if dot >= 0:
                        same_normal.add(faces[i].index)
                        same_normal.add(faces[j].index)
                    else:
                        opposite_normal.add(faces[i].index)
                        opposite_normal.add(faces[j].index)

    return same_normal, opposite_normal


# ---------------------------------------------------------------------------
#  GPU drawing
# ---------------------------------------------------------------------------

_COLOR_SAME = (1.0, 0.15, 0.15, 0.35)      # semi-transparent red
_COLOR_OPPOSITE = (0.15, 0.15, 1.0, 0.35)  # semi-transparent blue


def _draw_overlap_overlay():
    """POST_VIEW draw callback – renders the overlap overlays."""
    if (not _overlay_tris_same and not _overlay_tris_opposite) or not is_level_design_workspace():
        return

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        if _overlay_tris_same:
            shader.uniform_float("color", _COLOR_SAME)
            batch = batch_for_shader(shader, 'TRIS', {"pos": _overlay_tris_same})
            batch.draw(shader)
        if _overlay_tris_opposite:
            shader.uniform_float("color", _COLOR_OPPOSITE)
            batch = batch_for_shader(shader, 'TRIS', {"pos": _overlay_tris_opposite})
            batch.draw(shader)
    except Exception:
        pass
    finally:
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)


def _register_draw_handler():
    global _draw_handler
    _unregister_draw_handler()
    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        _draw_overlap_overlay, (), 'WINDOW', 'POST_VIEW'
    )


def _unregister_draw_handler():
    global _draw_handler
    if _draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        except Exception:
            pass
        _draw_handler = None


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _triangulate_face_world(face, matrix_world):
    """Fan-triangulate a face and return a list of world-space vertex tuples."""
    verts = [matrix_world @ v.co for v in face.verts]
    tris = []
    for i in range(1, len(verts) - 1):
        tris.append(verts[0][:])
        tris.append(verts[i][:])
        tris.append(verts[i + 1][:])
    return tris


def _build_overlay(bm, matrix_world, same_indices, opposite_indices):
    """Build the triangle lists for the GPU overlay."""
    global _overlay_tris_same, _overlay_tris_opposite
    _overlay_tris_same = []
    _overlay_tris_opposite = []
    bm.faces.ensure_lookup_table()
    for idx in same_indices:
        if idx < len(bm.faces):
            _overlay_tris_same.extend(_triangulate_face_world(bm.faces[idx], matrix_world))
    for idx in opposite_indices:
        if idx < len(bm.faces):
            _overlay_tris_opposite.extend(_triangulate_face_world(bm.faces[idx], matrix_world))


def is_overlap_check_active():
    """Return True if the overlap overlay is currently shown."""
    return _draw_handler is not None


# ---------------------------------------------------------------------------
#  Operator
# ---------------------------------------------------------------------------

class LEVELDESIGN_OT_toggle_overlap_check(Operator):
    """Toggle highlighting of overlapping coplanar faces"""

    bl_idname = "leveldesign.toggle_overlap_check"
    bl_label = "Toggle Overlap Check"

    @classmethod
    def poll(cls, context):
        if not is_level_design_workspace():
            return False
        obj = context.object
        return obj is not None and obj.type == 'MESH' and context.mode == 'EDIT_MESH'

    def execute(self, context):
        global _overlap_count

        if is_overlap_check_active():
            # Turn off
            _unregister_draw_handler()
            _overlap_count = 0
            _overlay_tris_same.clear()
            _overlay_tris_opposite.clear()
            context.area.tag_redraw()
            return {'FINISHED'}

        # Turn on – run detection
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()

        same_normal, opposite_normal = find_overlapping_faces(bm)
        all_overlapping = same_normal | opposite_normal
        _overlap_count = len(all_overlapping)

        if all_overlapping:
            _build_overlay(bm, obj.matrix_world, same_normal, opposite_normal)
            _register_draw_handler()
            self.report({'WARNING'},
                        f"Found {_overlap_count} overlapping faces"
                        f" ({len(same_normal)} same-normal, {len(opposite_normal)} opposite-normal)")
        else:
            self.report({'INFO'}, "No overlapping faces found")

        context.area.tag_redraw()
        return {'FINISHED'}


def get_overlap_count():
    """Return the number of overlapping faces found in the last scan."""
    return _overlap_count


# ---------------------------------------------------------------------------
#  Registration
# ---------------------------------------------------------------------------

classes = (
    LEVELDESIGN_OT_toggle_overlap_check,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    _unregister_draw_handler()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
