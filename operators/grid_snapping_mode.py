import bpy
import bmesh
import math
from bpy.types import Operator

from ..core.workspace_check import is_level_design_workspace
from ..core.materials import get_texture_dimensions_from_material
from ..core.uv_layers import get_render_active_uv_layer
from ..core.hotspot_queries import face_has_hotspot_material
from mathutils import Vector

from bpy_extras.view3d_utils import location_3d_to_region_2d

from ..handlers import cache_single_face
from .hotspot_apply import _bfs_propagate_grid_uvs


def validate_quad_grid(faces, require_rectangular):
    """Check if faces form a valid quad grid.

    Validates that all faces are quads and form a single connected component
    via shared edges. When require_rectangular is True (default), also checks
    that the grid has exactly 4 corner vertices (rectangular grid).

    Args:
        faces: list of BMFaces
        require_rectangular: if True, require exactly 4 corners (rectangular).
            Set to False for grid snapping which allows non-rectangular grids.

    Returns:
        dict with keys:
            'valid': bool
            'reason': str (only if invalid)
            'edge_faces': dict of edge -> list of faces (only if valid)
            'boundary_edges': list of edges with 1 adjacent face (only if valid)
    """
    if not faces:
        return {'valid': False, 'reason': 'No faces selected'}

    # All faces must be quads
    non_quads = [f for f in faces if len(f.loops) != 4]
    if non_quads:
        return {'valid': False, 'reason': 'All faces must be quads',
                'non_quad_faces': non_quads}

    face_set = set(faces)

    # Build edge -> faces lookup (only among selected faces)
    edge_faces = {}
    for face in faces:
        for edge in face.edges:
            edge_faces.setdefault(edge, []).append(face)

    # Flood-fill to check single connected component
    visited = {faces[0]}
    queue = [faces[0]]
    while queue:
        current = queue.pop(0)
        for edge in current.edges:
            neighbors = edge_faces.get(edge, [])
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

    if len(visited) != len(face_set):
        return {'valid': False, 'reason': 'Faces must form a single connected group'}

    # Boundary edges (edges with only 1 adjacent selected face)
    boundary_edges = [edge for edge, adj in edge_faces.items() if len(adj) == 1]

    # Corner vertex test: vertices touching exactly 1 selected face must be exactly 4
    if require_rectangular:
        vert_face_count = {}
        for face in faces:
            for vert in face.verts:
                vert_face_count[vert] = vert_face_count.get(vert, 0) + 1

        corner_count = sum(1 for count in vert_face_count.values() if count == 1)
        if corner_count != 4:
            return {'valid': False, 'reason': 'Faces must form a rectangular grid'}

    return {
        'valid': True,
        'edge_faces': edge_faces,
        'boundary_edges': boundary_edges,
    }


def build_grid_uvs(faces, uv_layer, edge_faces, tex_meters_u, tex_meters_v):
    """Build regular grid UVs for a rectangular quad grid.

    Uses BFS propagation from the first face. Rows share height,
    columns share width. UVs are normalized to texture space.

    Args:
        faces: list of BMFaces forming a rectangular grid
        uv_layer: BMesh UV layer
        edge_faces: dict of edge -> list of faces (from validate_quad_grid)
        tex_meters_u: texture width in meters (image_pixels / ppm)
        tex_meters_v: texture height in meters (image_pixels / ppm)
    """
    _bfs_propagate_grid_uvs(faces, uv_layer, edge_faces)

    # Normalize UVs from meters to texture space
    for face in faces:
        for loop in face.loops:
            loop[uv_layer].uv.x /= tex_meters_u
            loop[uv_layer].uv.y /= tex_meters_v


class LEVELDESIGN_OT_grid_snapping_mode(Operator):
    """Enter Grid Snapping mode for texture alignment on quad grids"""
    bl_idname = "leveldesign.grid_snapping_mode"
    bl_label = "Grid Snapping Mode"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (is_level_design_workspace() and context.object
                and context.object.type == 'MESH' and context.mode == 'EDIT_MESH')

    def _show_error(self, context, message):
        def draw(self_menu, context):
            self_menu.layout.label(text=message)
        context.window_manager.popup_menu(draw, title="Grid Snapping", icon='ERROR')

    def invoke(self, context, event):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        selected_faces = [f for f in bm.faces if f.select]
        if len(selected_faces) < 2:
            self._show_error(context, "Select at least 2 faces for grid snapping")
            return {'CANCELLED'}

        # Check for hotspot materials
        for face in selected_faces:
            if face_has_hotspot_material(face, me):
                self._show_error(context, "Cannot use Grid Snapping Mode on hotspottable faces")
                return {'CANCELLED'}

        # Validate grid topology (non-rectangular grids allowed for grid snapping)
        grid_result = validate_quad_grid(selected_faces, require_rectangular=False)
        if not grid_result['valid']:
            self._show_error(context, grid_result['reason'])
            return {'CANCELLED'}

        self.face_indices = [f.index for f in selected_faces]

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

        # Save initial UVs for ESC revert
        self.saved_uvs = {}
        for face in selected_faces:
            self.saved_uvs[face.index] = [
                (loop[uv_layer].uv.x, loop[uv_layer].uv.y) for loop in face.loops
            ]

        self.boundary_edge_indices = [e.index for e in grid_result['boundary_edges']]

        # Default state
        self.texture_edge = 'BOTTOM'
        self.fit_mode = None

        self.mouse_x = event.mouse_region_x
        self.mouse_y = event.mouse_region_y

        # Register vignette draw handler
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_vignette, (context,), 'WINDOW', 'POST_PIXEL'
        )

        context.window_manager.modal_handler_add(self)
        self._update_status_text(context)

        # Apply initial snap
        self._apply_grid_snap(context)

        return {'RUNNING_MODAL'}

    def _update_status_text(self, context):
        """Update status bar text based on current fit mode."""
        fit_indicator = ""
        if self.fit_mode == 'vertical':
            fit_indicator = " [V-Fit]"
        elif self.fit_mode == 'horizontal':
            fit_indicator = " [H-Fit]"
        context.workspace.status_text_set(
            f"W: Top    A: Left    S: Bottom    D: Right    Q: V-Fit    E: H-Fit    R: Reset Scale    LMB: Confirm    Esc: Cancel{fit_indicator}"
        )

    def _draw_vignette(self, context):
        """Draw a bottom-edge vignette bar with label to indicate Grid Snapping Mode."""
        space = context.space_data
        if not space or space.type != 'VIEW_3D' or space.region_3d.is_perspective == False:
            return

        import gpu
        import blf
        from gpu_extras.batch import batch_for_shader

        region = context.region
        w = region.width
        bar_height = 28

        gpu.state.blend_set('ALPHA')
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.uniform_float("color", (0.0, 0.0, 0.0, 0.7))

        verts = [(0, 0), (w, 0), (w, bar_height), (0, bar_height)]
        batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
        batch.draw(shader)

        gpu.state.blend_set('NONE')

        font_id = 0
        label = "Grid Snapping Mode"
        blf.size(font_id, 13)
        blf.color(font_id, 1.0, 1.0, 1.0, 0.8)
        tw, th = blf.dimensions(font_id, label)
        blf.position(font_id, (w - tw) / 2, (bar_height - th) / 2, 0)
        blf.draw(font_id, label)

    def _remove_draw_handler(self):
        """Remove the vignette draw handler if active."""
        if self._draw_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None

    def _get_closest_boundary_edge(self, context, faces, uv_layer):
        """Find the boundary edge closest to the mouse and its outward direction.

        Projects boundary edge midpoints to screen space, finds the closest,
        then determines its outward-facing direction from the adjacent face center.

        Args:
            context: Blender context
            faces: list of BMFaces in the grid
            uv_layer: UV layer

        Returns:
            tuple of (direction, edge_index) where direction is one of
            'bottom', 'top', 'left', 'right' and edge_index is the BMEdge index.
        """
        region = context.region
        rv3d = context.region_data
        mouse_pos = Vector((self.mouse_x, self.mouse_y))
        obj = context.object
        world_matrix = obj.matrix_world

        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()

        boundary_edges = [bm.edges[i] for i in self.boundary_edge_indices
                          if i < len(bm.edges)]

        closest_edge = None
        closest_dist = float('inf')

        for edge in boundary_edges:
            v0 = world_matrix @ edge.verts[0].co
            v1 = world_matrix @ edge.verts[1].co
            midpoint = (v0 + v1) / 2

            screen_pos = location_3d_to_region_2d(region, rv3d, midpoint)
            if screen_pos is None:
                continue

            dist = (screen_pos - mouse_pos).length
            if dist < closest_dist:
                closest_dist = dist
                closest_edge = edge

        if closest_edge is None:
            return ('bottom', -1)

        # Determine outward direction from the adjacent face center
        face_set = set(faces)
        adj_face = None
        for face in closest_edge.link_faces:
            if face in face_set:
                adj_face = face
                break

        if adj_face is None:
            return ('bottom', closest_edge.index)

        # Build vert -> UV lookup
        vert_uv = {}
        for face in faces:
            for loop in face.loops:
                if loop.vert not in vert_uv:
                    vert_uv[loop.vert] = loop[uv_layer].uv.copy()

        v0, v1 = closest_edge.verts
        uv0 = vert_uv.get(v0)
        uv1 = vert_uv.get(v1)
        if uv0 is None or uv1 is None:
            return ('bottom', closest_edge.index)

        edge_mid_u = (uv0.x + uv1.x) / 2
        edge_mid_v = (uv0.y + uv1.y) / 2

        face_center_u = 0.0
        face_center_v = 0.0
        for loop in adj_face.loops:
            face_center_u += loop[uv_layer].uv.x
            face_center_v += loop[uv_layer].uv.y
        face_center_u /= len(adj_face.loops)
        face_center_v /= len(adj_face.loops)

        out_u = edge_mid_u - face_center_u
        out_v = edge_mid_v - face_center_v

        if abs(out_u) > abs(out_v):
            direction = 'left' if out_u < 0 else 'right'
        else:
            direction = 'bottom' if out_v < 0 else 'top'

        return (direction, closest_edge.index)

    def _apply_grid_snap(self, context):
        """Build grid UVs and snap to the closest boundary side."""
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()

        faces = []
        for idx in self.face_indices:
            if idx >= len(bm.faces):
                return
            face = bm.faces[idx]
            if not face.is_valid:
                return
            faces.append(face)

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

        props = context.scene.level_design_props

        # Get texture dimensions
        mat = me.materials[faces[0].material_index] if faces[0].material_index < len(me.materials) else None
        tex_meters_u, tex_meters_v = get_texture_dimensions_from_material(mat, props.pixels_per_meter)

        # Build edge_faces lookup for selected faces
        edge_faces = {}
        for face in faces:
            for edge in face.edges:
                edge_faces.setdefault(edge, []).append(face)

        # Build grid UVs (in texture space)
        build_grid_uvs(faces, uv_layer, edge_faces, tex_meters_u, tex_meters_v)

        # Find the closest boundary edge and its outward direction
        closest_direction, snap_edge_idx = self._get_closest_boundary_edge(
            context, faces, uv_layer)

        # Get grid bounding box
        min_u = float('inf')
        max_u = float('-inf')
        min_v = float('inf')
        max_v = float('-inf')
        for face in faces:
            for loop in face.loops:
                u, v = loop[uv_layer].uv
                if u < min_u: min_u = u
                if u > max_u: max_u = u
                if v < min_v: min_v = v
                if v > max_v: max_v = v

        grid_width = max_u - min_u
        grid_height = max_v - min_v

        # Apply fit mode scaling if active
        if self.fit_mode == 'vertical' and grid_height > 0.0001:
            fit_scale = 1.0 / grid_height
            for face in faces:
                for loop in face.loops:
                    loop[uv_layer].uv.x = (loop[uv_layer].uv.x - min_u) * fit_scale
                    loop[uv_layer].uv.y = (loop[uv_layer].uv.y - min_v) * fit_scale
        elif self.fit_mode == 'horizontal' and grid_width > 0.0001:
            fit_scale = 1.0 / grid_width
            for face in faces:
                for loop in face.loops:
                    loop[uv_layer].uv.x = (loop[uv_layer].uv.x - min_u) * fit_scale
                    loop[uv_layer].uv.y = (loop[uv_layer].uv.y - min_v) * fit_scale
        else:
            # Shift grid so min is at origin (simplifies snap math)
            for face in faces:
                for loop in face.loops:
                    loop[uv_layer].uv.x -= min_u
                    loop[uv_layer].uv.y -= min_v

        # Rotate the grid so the selected edge's outward direction aligns with
        # the chosen texture edge direction.
        side_angle = {'bottom': 0, 'right': 90, 'top': 180, 'left': 270}
        tex_angle = {'BOTTOM': 0, 'RIGHT': 90, 'TOP': 180, 'LEFT': 270}

        rotation_deg = tex_angle[self.texture_edge] - side_angle[closest_direction]
        rotation_rad = math.radians(rotation_deg)
        cos_r = math.cos(rotation_rad)
        sin_r = math.sin(rotation_rad)

        # Recalculate bounds for rotation center
        min_u = float('inf')
        max_u = float('-inf')
        min_v = float('inf')
        max_v = float('-inf')
        for face in faces:
            for loop in face.loops:
                u, v = loop[uv_layer].uv
                if u < min_u: min_u = u
                if u > max_u: max_u = u
                if v < min_v: min_v = v
                if v > max_v: max_v = v

        center_u = (min_u + max_u) / 2
        center_v = (min_v + max_v) / 2

        for face in faces:
            for loop in face.loops:
                du = loop[uv_layer].uv.x - center_u
                dv = loop[uv_layer].uv.y - center_v
                loop[uv_layer].uv.x = du * cos_r - dv * sin_r + center_u
                loop[uv_layer].uv.y = du * sin_r + dv * cos_r + center_v

        # Find the snapped edge's UV position after rotation, then translate
        # so that specific edge sits on the unit square boundary.
        bm.edges.ensure_lookup_table()

        # Build vert -> UV lookup after rotation
        vert_uv = {}
        for face in faces:
            for loop in face.loops:
                if loop.vert not in vert_uv:
                    vert_uv[loop.vert] = loop[uv_layer].uv.copy()

        # Get the snap edge's UV coordinate along the snap axis
        snap_edge = bm.edges[snap_edge_idx] if snap_edge_idx >= 0 else None
        if snap_edge is not None:
            ev0, ev1 = snap_edge.verts
            suv0 = vert_uv.get(ev0)
            suv1 = vert_uv.get(ev1)
        else:
            suv0 = None
            suv1 = None

        # Recalculate bounds after rotation (needed for centering on the other axis)
        min_u = float('inf')
        max_u = float('-inf')
        min_v = float('inf')
        max_v = float('-inf')
        for face in faces:
            for loop in face.loops:
                u, v = loop[uv_layer].uv
                if u < min_u: min_u = u
                if u > max_u: max_u = u
                if v < min_v: min_v = v
                if v > max_v: max_v = v

        # Translate so the snap edge aligns with the chosen texture boundary.
        # The perpendicular axis is centered within the unit square.
        if suv0 is not None and suv1 is not None:
            if self.texture_edge == 'BOTTOM':
                edge_v = (suv0.y + suv1.y) / 2
                offset_u = 0.5 - (min_u + max_u) / 2
                offset_v = -edge_v
            elif self.texture_edge == 'TOP':
                edge_v = (suv0.y + suv1.y) / 2
                offset_u = 0.5 - (min_u + max_u) / 2
                offset_v = 1.0 - edge_v
            elif self.texture_edge == 'LEFT':
                edge_u = (suv0.x + suv1.x) / 2
                offset_u = -edge_u
                offset_v = 0.5 - (min_v + max_v) / 2
            elif self.texture_edge == 'RIGHT':
                edge_u = (suv0.x + suv1.x) / 2
                offset_u = 1.0 - edge_u
                offset_v = 0.5 - (min_v + max_v) / 2
        else:
            offset_u = 0.0
            offset_v = 0.0

        for face in faces:
            for loop in face.loops:
                loop[uv_layer].uv.x += offset_u
                loop[uv_layer].uv.y += offset_v

        # TODO: Properties panel will show derived (likely incorrect) values
        # for individual faces in the grid. Leave as-is for now.
        # TODO: Editing a single face after grid snapping will reset that face's
        # UVs to the (incorrect) derived values. Leave as-is for now.

        # Update the face data cache so the depsgraph handler doesn't
        # overwrite our UVs with stale cached values.
        props = context.scene.level_design_props
        ppm = props.pixels_per_meter
        for face in faces:
            cache_single_face(face, bm, ppm, me)

        bmesh.update_edit_mesh(me)

    def _revert_uvs(self, context):
        """Revert all grid face UVs to saved state on ESC."""
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

        for face_idx, uv_data in self.saved_uvs.items():
            if face_idx >= len(bm.faces):
                continue
            face = bm.faces[face_idx]
            if not face.is_valid:
                continue
            for i, loop in enumerate(face.loops):
                loop[uv_layer].uv.x = uv_data[i][0]
                loop[uv_layer].uv.y = uv_data[i][1]

        bmesh.update_edit_mesh(me)

    def modal(self, context, event):
        self.mouse_x = event.mouse_region_x
        self.mouse_y = event.mouse_region_y

        # WASD to set texture edge
        if event.type == 'W' and event.value == 'PRESS':
            self.texture_edge = 'TOP'
            self._apply_grid_snap(context)
            return {'RUNNING_MODAL'}

        if event.type == 'A' and event.value == 'PRESS':
            self.texture_edge = 'LEFT'
            self._apply_grid_snap(context)
            return {'RUNNING_MODAL'}

        if event.type == 'S' and event.value == 'PRESS':
            self.texture_edge = 'BOTTOM'
            self._apply_grid_snap(context)
            return {'RUNNING_MODAL'}

        if event.type == 'D' and event.value == 'PRESS':
            self.texture_edge = 'RIGHT'
            self._apply_grid_snap(context)
            return {'RUNNING_MODAL'}

        # Q for vertical fit toggle
        if event.type == 'Q' and event.value == 'PRESS':
            if self.fit_mode == 'vertical':
                self.fit_mode = None
            else:
                self.fit_mode = 'vertical'
            self._update_status_text(context)
            self._apply_grid_snap(context)
            return {'RUNNING_MODAL'}

        # E for horizontal fit toggle
        if event.type == 'E' and event.value == 'PRESS':
            if self.fit_mode == 'horizontal':
                self.fit_mode = None
            else:
                self.fit_mode = 'horizontal'
            self._update_status_text(context)
            self._apply_grid_snap(context)
            return {'RUNNING_MODAL'}

        # R for reset scale (revert to default ppm-based sizing)
        if event.type == 'R' and event.value == 'PRESS':
            self.fit_mode = None
            self._update_status_text(context)
            self._apply_grid_snap(context)
            return {'RUNNING_MODAL'}

        # Mouse move
        if event.type == 'MOUSEMOVE':
            self._apply_grid_snap(context)
            return {'RUNNING_MODAL'}

        # Confirm
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._remove_draw_handler()
            context.workspace.status_text_set(None)
            context.area.tag_redraw()
            return {'FINISHED'}

        # Cancel
        if event.type == 'ESC':
            self._revert_uvs(context)
            self._remove_draw_handler()
            context.workspace.status_text_set(None)
            context.area.tag_redraw()
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


classes = (
    LEVELDESIGN_OT_grid_snapping_mode,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
