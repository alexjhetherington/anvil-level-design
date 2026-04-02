import bpy
import bmesh
import math
from bpy.types import Operator

from ..core.logging import debug_log
from ..core.workspace_check import is_level_design_workspace
from ..core.geometry import align_2d_shape_to_square
from ..core.materials import get_texture_dimensions_from_material
from ..core.uv_projection import derive_transform_from_uvs, face_aligned_project, get_face_local_axes, apply_uv_to_face
from ..core.uv_layers import get_render_active_uv_layer
from ..core.hotspot_queries import face_has_hotspot_material
from mathutils import Vector

from bpy_extras.view3d_utils import location_3d_to_region_2d

from ..handlers import cache_face_data, cache_single_face
from ..properties import set_updating_from_selection, sync_scale_tracking


class LEVELDESIGN_OT_face_aligned_project(Operator):
    """Project UVs aligned to each face independently"""
    bl_idname = "leveldesign.face_aligned_project"
    bl_label = "Face-Aligned Projection"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() and context.object and context.object.type == 'MESH' and context.mode in {'EDIT_MESH', 'OBJECT'}

    def execute(self, context):
        obj = context.object
        me = obj.data

        # Handle object mode: enter edit mode, select all faces, operate, then return
        was_object_mode = context.mode == 'OBJECT'
        if was_object_mode:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')

        bm = bmesh.from_edit_mesh(me)

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            if was_object_mode:
                bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'WARNING'}, "No faces selected")
            return {'CANCELLED'}

        props = context.scene.level_design_props
        ppm = props.pixels_per_meter
        scale = props.projection_scale

        projected_count = 0
        for face in selected_faces:
            # Skip faces with hotspottable materials
            if face_has_hotspot_material(face, me):
                continue

            mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
            face_aligned_project(face, uv_layer, mat, ppm, scale)
            projected_count += 1

        bmesh.update_edit_mesh(me)
        self.report({'INFO'}, f"Projected {projected_count} faces")

        # Update properties to reflect projection
        set_updating_from_selection(True)
        try:
            props.texture_scale_u = scale
            props.texture_scale_v = scale
            props.texture_rotation = 0.0
            # Offset will be derived from the resulting UVs
        finally:
            set_updating_from_selection(False)
            sync_scale_tracking(context)

        # Update face cache so depsgraph handler doesn't overwrite our changes
        cache_face_data(context)

        # Return to object mode if we started there
        if was_object_mode:
            bpy.ops.object.mode_set(mode='OBJECT')

        return {'FINISHED'}


class LEVELDESIGN_OT_align_uv(Operator):
    """Align texture to face edges"""
    bl_idname = "leveldesign.align_uv"
    bl_label = "Align UV"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        items=[
            ('LEFT', "Left", ""),
            ('RIGHT', "Right", ""),
            ('TOP', "Top", ""),
            ('BOTTOM', "Bottom", ""),
            ('CENTER', "Center", ""),
        ]
    )

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() and context.object and context.object.type == 'MESH' and context.mode in {'EDIT_MESH', 'OBJECT'}

    def execute(self, context):
        obj = context.object
        me = obj.data

        # Handle object mode: enter edit mode, select all faces, operate, then return
        was_object_mode = context.mode == 'OBJECT'
        if was_object_mode:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')

        bm = bmesh.from_edit_mesh(me)

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()
        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            if was_object_mode:
                bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'WARNING'}, "No faces selected")
            return {'CANCELLED'}

        for face in selected_faces:
            # Skip faces with hotspottable materials
            if face_has_hotspot_material(face, me):
                continue

            uvs = [loop[uv_layer].uv.copy() for loop in face.loops]

            if not uvs:
                continue

            # Get UV bounds
            min_u = min(uv.x for uv in uvs)
            max_u = max(uv.x for uv in uvs)
            min_v = min(uv.y for uv in uvs)
            max_v = max(uv.y for uv in uvs)

            center_u = (min_u + max_u) / 2
            center_v = (min_v + max_v) / 2

            # Calculate offset based on direction
            offset_u = 0
            offset_v = 0

            if self.direction == 'LEFT':
                offset_u = -min_u
            elif self.direction == 'RIGHT':
                offset_u = 1.0 - max_u
            elif self.direction == 'TOP':
                offset_v = 1.0 - max_v
            elif self.direction == 'BOTTOM':
                offset_v = -min_v
            elif self.direction == 'CENTER':
                offset_u = 0.5 - center_u
                offset_v = 0.5 - center_v

            # Apply offset
            for loop in face.loops:
                loop[uv_layer].uv.x += offset_u
                loop[uv_layer].uv.y += offset_v

        bmesh.update_edit_mesh(me)

        # Update face cache so depsgraph handler doesn't overwrite our changes
        cache_face_data(context)

        # Return to object mode if we started there
        if was_object_mode:
            bpy.ops.object.mode_set(mode='OBJECT')

        return {'FINISHED'}


class LEVELDESIGN_OT_fit_to_face(Operator):
    """Fit texture to fill the entire face (0-1 UV space)"""
    bl_idname = "leveldesign.fit_to_face"
    bl_label = "Fit to Face"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() and context.object and context.object.type == 'MESH' and context.mode in {'EDIT_MESH', 'OBJECT'}

    def execute(self, context):
        obj = context.object
        me = obj.data

        # Handle object mode: enter edit mode, select all faces, operate, then return
        was_object_mode = context.mode == 'OBJECT'
        if was_object_mode:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')

        bm = bmesh.from_edit_mesh(me)

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()
        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            if was_object_mode:
                bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'WARNING'}, "No faces selected")
            return {'CANCELLED'}

        for face in selected_faces:
            # Skip faces with hotspottable materials
            if face_has_hotspot_material(face, me):
                continue

            uvs = [loop[uv_layer].uv.copy() for loop in face.loops]

            if not uvs:
                continue

            # Get UV bounds
            min_u = min(uv.x for uv in uvs)
            max_u = max(uv.x for uv in uvs)
            min_v = min(uv.y for uv in uvs)
            max_v = max(uv.y for uv in uvs)

            width = max_u - min_u
            height = max_v - min_v

            if width < 0.0001 or height < 0.0001:
                continue

            # Scale and offset to fit 0-1
            for loop in face.loops:
                u = (loop[uv_layer].uv.x - min_u) / width
                v = (loop[uv_layer].uv.y - min_v) / height
                loop[uv_layer].uv = (u, v)

        bmesh.update_edit_mesh(me)

        # Update face cache so depsgraph handler doesn't overwrite our changes
        cache_face_data(context)

        # Return to object mode if we started there
        if was_object_mode:
            bpy.ops.object.mode_set(mode='OBJECT')

        return {'FINISHED'}


class LEVELDESIGN_OT_face_uv_mode(Operator):
    """Enter Face UV mode for streamlined texture edge alignment"""
    bl_idname = "leveldesign.face_uv_mode"
    bl_label = "Face UV Mode"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() and context.object and context.object.type == 'MESH' and context.mode == 'EDIT_MESH'

    def invoke(self, context, event):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        selected_faces = [f for f in bm.faces if f.select]
        if len(selected_faces) != 1:
            self.report({'WARNING'}, "Select exactly one face")
            return {'CANCELLED'}

        # Skip faces with hotspottable materials
        if face_has_hotspot_material(selected_faces[0], me):
            self.report({'WARNING'}, "Cannot use Face UV Mode on hotspottable faces")
            return {'CANCELLED'}

        self.face_index = selected_faces[0].index
        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

        # Save initial transform for ESC revert
        props = context.scene.level_design_props
        self.saved_scale_u = props.texture_scale_u
        self.saved_scale_v = props.texture_scale_v
        self.saved_rotation = props.texture_rotation
        self.saved_offset_x = props.texture_offset_x
        self.saved_offset_y = props.texture_offset_y
        self.saved_edge_index = props.edge_index

        # Save initial UVs for revert
        face = selected_faces[0]
        self.saved_uvs = [(loop[uv_layer].uv.x, loop[uv_layer].uv.y) for loop in face.loops]

        # Default texture edge is BOTTOM
        self.texture_edge = 'BOTTOM'

        # Track last applied state to avoid redundant updates
        self.last_edge_index = -1
        self.last_texture_edge = None
        self.last_fit_mode = None

        # Fit mode state: None, 'vertical', or 'horizontal'
        self.fit_mode = None
        # Scales before fit mode was enabled (for reverting)
        self.pre_fit_scale_u = None
        self.pre_fit_scale_v = None

        # Initialize mouse position from invoke event
        self.mouse_x = event.mouse_region_x
        self.mouse_y = event.mouse_region_y

        # Register vignette draw handler
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_vignette, (context,), 'WINDOW', 'POST_PIXEL'
        )

        context.window_manager.modal_handler_add(self)
        self._update_status_text(context)

        # Apply initial snap based on cursor position at invoke time
        self._apply_snap(context)

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
        """Draw a bottom-edge vignette bar with label to indicate Face UV Mode."""
        # Only draw in perspective 3D views
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

        # Draw label text centered in the bar
        font_id = 0
        label = "Face Snapping Mode"
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

    def _calculate_aspect_locked_fit_scale(self, face, uv_layer, fit_mode, current_scale_u, current_scale_v):
        """Calculate uniform scale for aspect-locked fit.

        The fit scale makes the texture fit the face in one dimension (vertical or horizontal),
        with both scale_u and scale_v set to the same value to maintain aspect ratio.

        Args:
            face: The BMesh face with aligned UVs
            uv_layer: UV layer
            fit_mode: 'vertical' or 'horizontal'
            current_scale_u: Current horizontal scale
            current_scale_v: Current vertical scale

        Returns:
            The uniform scale value that achieves the fit
        """
        uvs = [loop[uv_layer].uv for loop in face.loops]
        min_u = min(uv.x for uv in uvs)
        max_u = max(uv.x for uv in uvs)
        min_v = min(uv.y for uv in uvs)
        max_v = max(uv.y for uv in uvs)

        width_uv = max_u - min_u
        height_uv = max_v - min_v

        if fit_mode == 'vertical':
            if height_uv < 0.0001:
                return current_scale_v
            return height_uv * current_scale_v
        else:  # horizontal
            if width_uv < 0.0001:
                return current_scale_u
            return width_uv * current_scale_u

    def _get_closest_edge_index(self, context, face):
        """Find which face edge is closest to the mouse cursor in screen space."""
        region = context.region
        rv3d = context.region_data
        mouse_pos = Vector((self.mouse_x, self.mouse_y))

        obj = context.object
        world_matrix = obj.matrix_world

        loops = list(face.loops)
        num_edges = len(loops)

        closest_idx = 0
        closest_dist = float('inf')

        for i in range(num_edges):
            # Get edge midpoint in world space
            v1 = world_matrix @ loops[i].vert.co
            v2 = world_matrix @ loops[(i + 1) % num_edges].vert.co
            midpoint = (v1 + v2) / 2

            # Project to screen space
            screen_pos = location_3d_to_region_2d(region, rv3d, midpoint)
            if screen_pos is None:
                continue

            dist = (screen_pos - mouse_pos).length
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i

        return closest_idx

    def _apply_snap(self, context):
        """Apply snap to edge based on current texture_edge, closest face edge, and fit mode."""
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        bm.faces.ensure_lookup_table()
        if self.face_index >= len(bm.faces):
            return

        face = bm.faces[self.face_index]
        if not face.is_valid:
            return

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()
        props = context.scene.level_design_props

        # Get closest edge to mouse
        edge_index = self._get_closest_edge_index(context, face)

        # Skip if nothing changed
        if (edge_index == self.last_edge_index and
            self.texture_edge == self.last_texture_edge and
            self.fit_mode == self.last_fit_mode):
            return

        self.last_edge_index = edge_index
        self.last_texture_edge = self.texture_edge
        self.last_fit_mode = self.fit_mode

        # Map texture_edge to target angle in UV space:
        # BOTTOM = +U direction (0°), RIGHT = +V direction (90°),
        # TOP = -U direction (180°), LEFT = -V direction (-90°)
        target_angle_map = {'BOTTOM': 0.0, 'RIGHT': 90.0, 'TOP': 180.0, 'LEFT': -90.0}
        target_angle_deg = target_angle_map[self.texture_edge]

        # Map texture_edge to square_edge_index for align_2d_shape_to_square
        square_edge_map = {'BOTTOM': 0, 'RIGHT': 1, 'TOP': 2, 'LEFT': 3}
        square_edge_index = square_edge_map[self.texture_edge]

        # Update edge_index property
        props.edge_index = edge_index

        # Compute rotation directly from face geometry.
        # The face edge direction in face-local 2D space tells us what rotation
        # of the projection axes will make that edge align with the target
        # texture direction. This avoids deriving rotation from potentially
        # distorted UVs.
        face_axes = get_face_local_axes(face)
        if not face_axes:
            return
        face_local_x, face_local_y = face_axes

        loops = list(face.loops)
        n = len(loops)
        v1 = loops[edge_index].vert.co
        v2 = loops[(edge_index + 1) % n].vert.co
        edge_3d = v2 - v1

        # Project edge into face-local 2D
        edge_local_x = edge_3d.dot(face_local_x)
        edge_local_y = edge_3d.dot(face_local_y)
        edge_angle_deg = math.degrees(math.atan2(edge_local_y, edge_local_x))

        # rotation is applied to projection axes: a face-local vector at
        # angle edge_angle should map to target_angle in UV space.
        # Forward transform: u component ~ cos(rot)*x - sin(rot)*y
        # An edge at angle A in face space appears at angle (A - rot) in UV space.
        # We want (A - rot) = target_angle  =>  rot = A - target_angle
        rotation = target_angle_deg - edge_angle_deg

        # Determine scale values
        if self.fit_mode:
            # We need UVs applied first to calculate fit scale; use current scale as starting point
            if self.pre_fit_scale_u is not None:
                scale_u = self.pre_fit_scale_u
                scale_v = self.pre_fit_scale_v
            else:
                scale_u = props.texture_scale_u
                scale_v = props.texture_scale_v
        else:
            if self.pre_fit_scale_u is not None:
                scale_u = self.pre_fit_scale_u
                scale_v = self.pre_fit_scale_v
            else:
                scale_u = props.texture_scale_u
                scale_v = props.texture_scale_v

        # Apply clean UVs from scratch with the computed rotation
        mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
        apply_uv_to_face(face, uv_layer, scale_u, scale_v, rotation,
                         0.0, 0.0, mat, props.pixels_per_meter, me)

        # Now align the edge to the texture edge (handles offset/positioning)
        shape = [(loop[uv_layer].uv.x, loop[uv_layer].uv.y) for loop in face.loops]
        aligned = align_2d_shape_to_square(shape, edge_index, square_edge_index)
        for i, loop in enumerate(face.loops):
            loop[uv_layer].uv.x = aligned[i][0]
            loop[uv_layer].uv.y = aligned[i][1]

        # Calculate fit scale if needed (now that UVs are applied)
        if self.fit_mode:
            fit_scale = self._calculate_aspect_locked_fit_scale(
                face, uv_layer, self.fit_mode, scale_u, scale_v
            )
            scale_u = fit_scale
            scale_v = fit_scale
            # Re-apply with fit scale
            apply_uv_to_face(face, uv_layer, scale_u, scale_v, rotation,
                             0.0, 0.0, mat, props.pixels_per_meter, me)
            shape = [(loop[uv_layer].uv.x, loop[uv_layer].uv.y) for loop in face.loops]
            aligned = align_2d_shape_to_square(shape, edge_index, square_edge_index)
            for i, loop in enumerate(face.loops):
                loop[uv_layer].uv.x = aligned[i][0]
                loop[uv_layer].uv.y = aligned[i][1]

        # Derive final offset from the aligned UVs
        final_transform = derive_transform_from_uvs(face, uv_layer, props.pixels_per_meter, me)
        if final_transform:
            set_updating_from_selection(True)
            try:
                props.texture_scale_u = scale_u
                props.texture_scale_v = scale_v
                props.texture_rotation = final_transform['rotation']
                props.texture_offset_x = final_transform['offset_x']
                props.texture_offset_y = final_transform['offset_y']
            finally:
                set_updating_from_selection(False)

        # Update cache
        cache_single_face(face, bm, props.pixels_per_meter, me)

        bmesh.update_edit_mesh(me)

    def _revert_transform(self, context):
        """Revert to saved transform on ESC."""
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        bm.faces.ensure_lookup_table()
        if self.face_index >= len(bm.faces):
            return

        face = bm.faces[self.face_index]
        if not face.is_valid:
            return

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()
        props = context.scene.level_design_props

        # Restore saved UVs directly
        loops = list(face.loops)
        for i, loop in enumerate(loops):
            loop[uv_layer].uv.x = self.saved_uvs[i][0]
            loop[uv_layer].uv.y = self.saved_uvs[i][1]

        # Restore saved properties
        set_updating_from_selection(True)
        try:
            props.texture_scale_u = self.saved_scale_u
            props.texture_scale_v = self.saved_scale_v
            props.texture_rotation = self.saved_rotation
            props.texture_offset_x = self.saved_offset_x
            props.texture_offset_y = self.saved_offset_y
            props.edge_index = self.saved_edge_index
        finally:
            set_updating_from_selection(False)

        # Update cache with reverted state
        cache_single_face(face, bm, props.pixels_per_meter, me)

        bmesh.update_edit_mesh(me)

    def modal(self, context, event):
        # Store mouse position for edge detection
        self.mouse_x = event.mouse_region_x
        self.mouse_y = event.mouse_region_y

        props = context.scene.level_design_props

        # WASD to set texture edge
        if event.type == 'W' and event.value == 'PRESS':
            self.texture_edge = 'TOP'
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        if event.type == 'A' and event.value == 'PRESS':
            self.texture_edge = 'LEFT'
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        if event.type == 'S' and event.value == 'PRESS':
            self.texture_edge = 'BOTTOM'
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        if event.type == 'D' and event.value == 'PRESS':
            self.texture_edge = 'RIGHT'
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        # Q for vertical fit toggle
        if event.type == 'Q' and event.value == 'PRESS':
            if self.fit_mode == 'vertical':
                # Disable fit mode
                self.fit_mode = None
            else:
                # Enable vertical fit (save scales if first time enabling any fit)
                if self.fit_mode is None:
                    self.pre_fit_scale_u = props.texture_scale_u
                    self.pre_fit_scale_v = props.texture_scale_v
                self.fit_mode = 'vertical'
            self._update_status_text(context)
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        # E for horizontal fit toggle
        if event.type == 'E' and event.value == 'PRESS':
            if self.fit_mode == 'horizontal':
                # Disable fit mode
                self.fit_mode = None
            else:
                # Enable horizontal fit (save scales if first time enabling any fit)
                if self.fit_mode is None:
                    self.pre_fit_scale_u = props.texture_scale_u
                    self.pre_fit_scale_v = props.texture_scale_v
                self.fit_mode = 'horizontal'
            self._update_status_text(context)
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        # R for reset scale to 1
        if event.type == 'R' and event.value == 'PRESS':
            # Disable any fit mode
            self.fit_mode = None
            # Set pre_fit scales to 1 so disabling fit returns to 1
            self.pre_fit_scale_u = 1.0
            self.pre_fit_scale_v = 1.0
            # Force re-apply by clearing last state
            self.last_fit_mode = 'force_update'
            self._update_status_text(context)
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        # Mouse move - update snap based on new cursor position
        if event.type == 'MOUSEMOVE':
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        # Left click - confirm and exit
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # Clear pre-fit scales on confirm (they're now the real scales)
            self.pre_fit_scale_u = None
            self.pre_fit_scale_v = None
            self._remove_draw_handler()
            context.workspace.status_text_set(None)
            context.area.tag_redraw()
            return {'FINISHED'}

        # Escape - revert and exit
        if event.type == 'ESC':
            self._revert_transform(context)
            self._remove_draw_handler()
            context.workspace.status_text_set(None)
            context.area.tag_redraw()
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


class LEVELDESIGN_OT_snapping_mode_dispatch(Operator):
    """Dispatch to Face UV Mode or Grid Snapping Mode based on selection"""
    bl_idname = "leveldesign.snapping_mode_dispatch"
    bl_label = "Snapping Mode"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return (is_level_design_workspace() and context.object
                and context.object.type == 'MESH' and context.mode == 'EDIT_MESH')

    def invoke(self, context, event):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        selected_faces = [f for f in bm.faces if f.select]

        if len(selected_faces) == 0:
            self.report({'WARNING'}, "Select at least one face")
            return {'CANCELLED'}

        if len(selected_faces) == 1:
            return bpy.ops.leveldesign.face_uv_mode('INVOKE_DEFAULT')

        # 2+ faces: try grid snapping
        return bpy.ops.leveldesign.grid_snapping_mode('INVOKE_DEFAULT')


classes = (
    LEVELDESIGN_OT_face_aligned_project,
    LEVELDESIGN_OT_align_uv,
    LEVELDESIGN_OT_fit_to_face,
    LEVELDESIGN_OT_face_uv_mode,
    LEVELDESIGN_OT_snapping_mode_dispatch,
)

addon_keymaps = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Register keymap for Snapping Mode dispatcher (T key)
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Mesh', space_type='EMPTY')
        kmi = km.keymap_items.new(
            "leveldesign.snapping_mode_dispatch",
            'T', 'PRESS',
            head=True
        )
        addon_keymaps.append((km, kmi))

        # UV shortcut keymaps (unbound by default)
        kmi = km.keymap_items.new(
            "leveldesign.face_aligned_project",
            'NONE', 'PRESS',
            head=True
        )
        addon_keymaps.append((km, kmi))

        kmi = km.keymap_items.new(
            "leveldesign.align_uv",
            'NONE', 'PRESS',
            head=True
        )
        kmi.properties.direction = 'CENTER'
        addon_keymaps.append((km, kmi))

        kmi = km.keymap_items.new(
            "leveldesign.fit_to_face",
            'NONE', 'PRESS',
            head=True
        )
        addon_keymaps.append((km, kmi))


def unregister():
    # Remove keymaps
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
