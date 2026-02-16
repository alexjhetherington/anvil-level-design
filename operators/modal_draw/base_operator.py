"""
Modal Draw - Base Operator

Shared state machine logic for 3-state modal draw operators
(first vertex → second vertex → depth).

Subclasses must override _execute_action() and _get_tool_name().
Subclasses may override the _calculate_* hook methods for custom snapping.
"""

from mathutils import Vector

from . import utils
from . import snapping
from . import preview


# Minimum rectangle size (world units) to prevent degenerate geometry
MIN_RECTANGLE_SIZE = 0.001


class _MousePosition:
    """Simple container for mouse position, used when re-snapping after grid change."""
    def __init__(self, x, y):
        self.mouse_region_x = x
        self.mouse_region_y = y


def _get_undo_redo_keys(context):
    """
    Get the keys bound to undo and redo operations.

    Returns:
        set: Set of (type, ctrl, shift, alt) tuples for undo/redo bindings
    """
    keys = set()
    wm = context.window_manager
    kc = wm.keyconfigs.user

    if kc is None:
        # Fallback to default Ctrl+Z
        return {('Z', True, False, False), ('Z', True, True, False)}

    for km in kc.keymaps:
        for kmi in km.keymap_items:
            if kmi.idname in ('ed.undo', 'ed.redo') and kmi.active:
                keys.add((kmi.type, kmi.ctrl, kmi.shift, kmi.alt))

    # If no bindings found, use defaults
    if not keys:
        keys = {('Z', True, False, False), ('Z', True, True, False)}

    return keys


class ModalDrawBase:
    """
    Mixin base class for 3-state modal draw operators.

    Provides the full state machine: invoke, modal, preview management,
    snapping, and cleanup. Not registered with Blender directly.

    Subclasses MUST override:
        _execute_action(context, first_vertex, second_vertex, depth,
                        local_x, local_y, local_z) -> (bool, str)
        _get_tool_name() -> str

    Subclasses MAY override (for custom snapping):
        _calculate_first_vertex_snap_2d(context, event)
        _calculate_first_vertex_snap_3d(context, event)
        _calculate_depth_from_mouse_2d(context, event, initial_mouse_x)
        _calculate_depth_from_mouse_3d(context, event, first_vertex,
                                       second_vertex, local_z, initial_mouse_pos)
    """

    # State machine states
    STATE_FIRST_VERTEX = 'FIRST_VERTEX'
    STATE_LINE_END = 'LINE_END'
    STATE_SECOND_VERTEX = 'SECOND_VERTEX'
    STATE_DEPTH = 'DEPTH'

    # --- Abstract methods (must override) ---

    def _execute_action(self, context, first_vertex, second_vertex, depth,
                        local_x, local_y, local_z):
        """Execute the tool's action after depth is confirmed.

        Returns:
            tuple: (success: bool, message: str)
        """
        raise NotImplementedError

    def _get_tool_name(self):
        """Return the display name for header text (e.g. 'Cube Cut', 'Box')."""
        raise NotImplementedError

    # --- Hook methods (may override) ---

    def _is_valid_mode(self, context):
        """Check if the current mode is valid for this operator. Override to allow other modes."""
        return context.mode == 'EDIT_MESH'

    def _calculate_first_vertex_snap_2d(self, context, event):
        return snapping.calculate_first_vertex_snap_2d(context, event)

    def _calculate_first_vertex_snap_3d(self, context, event):
        return snapping.calculate_first_vertex_snap_3d(context, event)

    def _calculate_depth_from_mouse_2d(self, context, event, initial_mouse_x):
        return snapping.calculate_depth_from_mouse_2d(context, event, initial_mouse_x)

    def _calculate_depth_from_mouse_3d(self, context, event, first_vertex,
                                       second_vertex, local_z, initial_mouse_pos):
        return snapping.calculate_depth_from_mouse_3d(
            context, event, first_vertex, second_vertex, local_z, initial_mouse_pos
        )

    # --- Operator lifecycle ---

    def invoke(self, context, event):
        # Initialize state
        self._state = self.STATE_FIRST_VERTEX

        # First vertex data
        self._first_vertex = None
        self._hit_face_normal = None
        self._hit_object = None

        # Rectangle axes
        self._local_x = None
        self._local_y = None
        self._local_z = None

        # Rectangle plane (for mouse projection)
        self._plane_point = None
        self._plane_normal = None

        # Line mode (rotated draw)
        self._line_mode = False
        self._line_end = None
        self._line_length = 0.0

        # Second vertex
        self._second_vertex = None

        # Depth phase
        self._depth = 0.0
        self._depth_start_mouse_pos = (0, 0)  # (x, y) for geometric depth calc

        # View tracking
        self._is_2d_view = utils.is_2d_view(context)

        # Grid size tracking (for detecting changes)
        self._last_grid_size = utils.get_grid_size(context)
        self._last_mouse_region_pos = None  # (x, y) tuple

        # Cache undo/redo key bindings for clean exit
        self._undo_redo_keys = _get_undo_redo_keys(context)

        # Set up preview drawing
        self._preview = preview.get_preview()
        self._preview.register_handlers()
        self._preview.set_state(self.STATE_FIRST_VERTEX)

        # Set cursor
        context.window.cursor_modal_set('CROSSHAIR')

        # Add modal handler
        context.window_manager.modal_handler_add(self)

        # Header text
        self._update_header(context)

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # Exit if user left a valid mode
        if not self._is_valid_mode(context):
            self._cleanup(context)
            return {'CANCELLED'}

        # Update 2D view check (user might switch views)
        self._is_2d_view = utils.is_2d_view(context)

        # Check if grid size changed (user may have custom hotkeys for this)
        current_grid_size = utils.get_grid_size(context)
        if current_grid_size != self._last_grid_size:
            self._last_grid_size = current_grid_size
            # Re-snap with new grid size using last known mouse position
            if self._last_mouse_region_pos is not None:
                fake_event = _MousePosition(*self._last_mouse_region_pos)
                self._handle_mouse_move(context, fake_event)
                utils.tag_redraw_all_3d_views()

        # ESC to cancel
        if event.type == 'ESC' and event.value == 'PRESS':
            self._cleanup(context)
            return {'CANCELLED'}

        # Mouse move - update previews
        if event.type == 'MOUSEMOVE':
            self._last_mouse_region_pos = (event.mouse_region_x, event.mouse_region_y)
            self._handle_mouse_move(context, event)
            utils.tag_redraw_all_3d_views()
            return {'RUNNING_MODAL'}

        # Left click - advance state
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            result = self._handle_click(context, event)
            utils.tag_redraw_all_3d_views()
            return result

        # Undo/redo - exit cleanly (uses cached key bindings)
        if event.value == 'PRESS':
            event_key = (event.type, event.ctrl, event.shift, event.alt)
            if event_key in self._undo_redo_keys:
                self._cleanup(context)
                return {'CANCELLED'}

        # Pass through all other events (navigation, user hotkeys, etc.)
        return {'PASS_THROUGH'}

    def _handle_mouse_move(self, context, event):
        """Handle mouse movement based on current state."""
        if self._state == self.STATE_FIRST_VERTEX:
            self._update_first_vertex_preview(context, event)
        elif self._state == self.STATE_LINE_END:
            self._update_line_end_preview(context, event)
        elif self._state == self.STATE_SECOND_VERTEX:
            self._update_second_vertex_preview(context, event)
        elif self._state == self.STATE_DEPTH:
            self._update_depth_preview(context, event)

    def _handle_click(self, context, event):
        """Handle left click based on current state."""
        if self._state == self.STATE_FIRST_VERTEX:
            return self._confirm_first_vertex(context, event)
        elif self._state == self.STATE_LINE_END:
            return self._confirm_line_end(context, event)
        elif self._state == self.STATE_SECOND_VERTEX:
            return self._confirm_second_vertex(context, event)
        elif self._state == self.STATE_DEPTH:
            return self._confirm_depth(context, event)

        return {'RUNNING_MODAL'}

    def _update_first_vertex_preview(self, context, event):
        """Update the snap preview for the first vertex."""
        if self._is_2d_view:
            snapped, plane_normal = self._calculate_first_vertex_snap_2d(context, event)
            if snapped is not None:
                # Get tangent axes for 2D view (aligned with view plane)
                tangent1, tangent2 = utils.get_2d_view_tangents(context)
                self._preview.update_snap_point(snapped, tangent1, tangent2)
                # No face grid for 2D views
                self._preview.clear_face_grid()
            else:
                self._preview.update_snap_point(None, None, None)
                self._preview.clear_face_grid()
        else:
            snapped, face_normal, obj, was_clamped = self._calculate_first_vertex_snap_3d(context, event)
            if snapped is not None:
                # Get snap-aligned tangent axes (match the grid snapping axes)
                tangent1, tangent2 = utils.get_snap_aligned_tangents(face_normal)
                self._preview.update_snap_point(snapped, tangent1, tangent2)
                # Update face grid overlay for 3D view
                # Grid shows world-aligned positions, independent of snap point
                grid_size = utils.get_grid_size(context)
                self._preview.update_face_grid(snapped, face_normal, grid_size, was_clamped)
            else:
                self._preview.update_snap_point(None, None, None)
                self._preview.clear_face_grid()

    def _update_line_end_preview(self, context, event):
        """Update the line end preview during line mode."""
        if self._first_vertex is None:
            return

        snapped = snapping.calculate_line_end_snap(
            context, event,
            self._first_vertex,
            self._plane_point, self._plane_normal,
            self._is_2d_view
        )

        if snapped is not None:
            self._line_end = snapped
            self._preview.update_line_end(snapped)
            # Update snap point for crosshair display
            if self._is_2d_view:
                tangent1, tangent2 = utils.get_2d_view_tangents(context)
            else:
                tangent1, tangent2 = utils.get_snap_aligned_tangents(self._hit_face_normal)
            self._preview.update_snap_point(snapped, tangent1, tangent2)

    def _update_second_vertex_preview(self, context, event):
        """Update the rectangle preview."""
        if self._first_vertex is None:
            return

        if self._line_mode:
            snapped = snapping.calculate_width_snap(
                context, event,
                self._first_vertex,
                self._line_length,
                self._local_x, self._local_y,
                self._plane_point, self._plane_normal
            )
        else:
            snapped = snapping.calculate_second_vertex_snap(
                context, event,
                self._first_vertex,
                self._local_x, self._local_y,
                self._plane_point, self._plane_normal
            )

        if snapped is not None:
            self._preview.update_second_vertex(snapped)
            self._second_vertex = snapped

    def _update_depth_preview(self, context, event):
        """Update the depth/cuboid preview."""
        if self._is_2d_view:
            depth = self._calculate_depth_from_mouse_2d(
                context, event, self._depth_start_mouse_pos[0]
            )
        else:
            depth = self._calculate_depth_from_mouse_3d(
                context, event,
                self._first_vertex, self._second_vertex,
                self._local_z, self._depth_start_mouse_pos
            )

        self._depth = depth
        self._preview.update_depth(depth)

        # Update header with current depth value
        self._update_header(context)

    def _confirm_first_vertex(self, context, event):
        """Confirm the first vertex and advance to second vertex state."""
        if self._is_2d_view:
            snapped, plane_normal = self._calculate_first_vertex_snap_2d(context, event)
            if snapped is None:
                return {'RUNNING_MODAL'}  # Ignore click

            self._first_vertex = snapped
            self._hit_face_normal = plane_normal

            # For 2D views, rectangle is on the view plane (NOT vertical)
            # local_x and local_y are the two axes of the view plane
            # local_z is the depth direction (perpendicular to view)
            view_type = utils.get_view_type(context)

            if view_type in ('TOP', 'BOTTOM'):
                # XY plane, depth along Z
                self._local_x = Vector((1, 0, 0))
                self._local_y = Vector((0, 1, 0))
                # "Outward" from TOP view is down (-Z), from BOTTOM is up (+Z)
                self._local_z = Vector((0, 0, -1)) if view_type == 'TOP' else Vector((0, 0, 1))
            elif view_type in ('FRONT', 'BACK'):
                # XZ plane (vertical rectangle), depth along Y
                self._local_x = Vector((1, 0, 0))
                self._local_y = Vector((0, 0, 1))
                # "Outward" from FRONT view is back (-Y), from BACK is forward (+Y)
                self._local_z = Vector((0, -1, 0)) if view_type == 'FRONT' else Vector((0, 1, 0))
            else:  # LEFT, RIGHT
                # YZ plane (vertical rectangle), depth along X
                self._local_x = Vector((0, 1, 0))
                self._local_y = Vector((0, 0, 1))
                # "Outward" from RIGHT view is left (-X), from LEFT is right (+X)
                self._local_z = Vector((-1, 0, 0)) if view_type == 'RIGHT' else Vector((1, 0, 0))

            # Set plane for rectangle drawing (the view plane through first vertex)
            self._plane_point = self._first_vertex.copy()
            self._plane_normal = self._local_z.copy()

        else:
            # 3D view
            snapped, face_normal, obj, _was_clamped = self._calculate_first_vertex_snap_3d(context, event)
            if snapped is None:
                return {'RUNNING_MODAL'}  # Ignore click - no face hit

            self._first_vertex = snapped
            self._hit_face_normal = face_normal
            self._hit_object = obj

            # Calculate rectangle orientation from face normal
            self._local_x, self._local_y, self._local_z = utils.get_rectangle_axes(
                face_normal, context
            )

            # Set plane for rectangle drawing (vertical plane through first vertex)
            self._plane_point = self._first_vertex.copy()
            self._plane_normal = self._local_z.copy()

        # Update preview state
        self._preview.set_first_vertex(
            self._first_vertex,
            self._local_x, self._local_y, self._local_z
        )

        # Check if line mode modifier is held
        if self._is_line_mode_key_held(context, event):
            self._line_mode = True
            self._preview.set_line_mode(True)
            self._preview.set_state(self.STATE_LINE_END)
            self._state = self.STATE_LINE_END
            self._line_end = self._first_vertex.copy()
        else:
            self._preview.set_state(self.STATE_SECOND_VERTEX)
            self._state = self.STATE_SECOND_VERTEX

            # Initialize second vertex to first (will update on mouse move)
            self._second_vertex = self._first_vertex.copy()
            self._preview.update_second_vertex(self._second_vertex)

        self._update_header(context)
        return {'RUNNING_MODAL'}

    def _confirm_line_end(self, context, event):
        """Confirm the line end point and advance to width (second vertex) state."""
        if self._first_vertex is None or self._line_end is None:
            return {'RUNNING_MODAL'}

        # Check minimum line length
        line_vec = self._line_end - self._first_vertex
        line_length = line_vec.length

        if line_length < MIN_RECTANGLE_SIZE:
            return {'RUNNING_MODAL'}

        # Compute new local axes from line direction
        line_dir = line_vec.normalized()
        local_z = self._local_z.copy()  # Depth direction stays the same

        # Line defines one edge: local_x = line direction
        local_x = line_dir
        # Perpendicular width axis
        local_y = local_z.cross(local_x)
        local_y.normalize()

        # Update axes
        self._local_x = local_x
        self._local_y = local_y
        self._line_length = line_length

        # Update preview with new axes
        self._preview.set_first_vertex(
            self._first_vertex,
            self._local_x, self._local_y, self._local_z
        )

        # Transition to second vertex (width) state
        self._preview.set_state(self.STATE_SECOND_VERTEX)
        self._state = self.STATE_SECOND_VERTEX

        # Initialize second vertex at line end (zero width)
        self._second_vertex = self._line_end.copy()
        self._preview.update_second_vertex(self._second_vertex)

        self._update_header(context)
        return {'RUNNING_MODAL'}

    def _is_line_mode_key_held(self, context, event):
        """Check if the configured line mode modifier key is held during the event."""
        wm = context.window_manager
        kc_user = wm.keyconfigs.user

        if kc_user:
            km = kc_user.keymaps.get("Mesh")
            if km:
                for kmi in km.keymap_items:
                    if kmi.idname == "leveldesign.line_mode_activate" and kmi.active:
                        key_type = kmi.type
                        if key_type in ('LEFT_SHIFT', 'RIGHT_SHIFT'):
                            return event.shift
                        elif key_type in ('LEFT_CTRL', 'RIGHT_CTRL'):
                            return event.ctrl
                        elif key_type in ('LEFT_ALT', 'RIGHT_ALT'):
                            return event.alt
                        # Non-modifier key — can't reliably detect "held" state
                        return False

        # Default: check shift
        return event.shift

    def _confirm_second_vertex(self, context, event):
        """Confirm the second vertex and advance to depth state."""
        if self._first_vertex is None or self._second_vertex is None:
            return {'RUNNING_MODAL'}

        # Check minimum size
        diff = self._second_vertex - self._first_vertex
        local_dx = abs(diff.dot(self._local_x))
        local_dy = abs(diff.dot(self._local_y))

        if local_dx < MIN_RECTANGLE_SIZE and local_dy < MIN_RECTANGLE_SIZE:
            # Rectangle too small, ignore click
            return {'RUNNING_MODAL'}

        # Advance to depth state
        self._preview.set_state(self.STATE_DEPTH)
        self._state = self.STATE_DEPTH

        # Store initial mouse position for depth calculation
        self._depth_start_mouse_pos = (event.mouse_region_x, event.mouse_region_y)
        self._depth = 0.0
        self._preview.update_depth(0.0)

        self._update_header(context)
        return {'RUNNING_MODAL'}

    def _confirm_depth(self, context, event):
        """Confirm the depth and execute the action."""
        success, message = self._execute_action(
            context,
            self._first_vertex,
            self._second_vertex,
            self._depth,
            self._local_x,
            self._local_y,
            self._local_z
        )

        if success:
            self.report({'INFO'}, message)
        else:
            self.report({'ERROR'}, message)

        self._cleanup(context)
        return {'FINISHED'}

    def _cleanup(self, context):
        """Clean up resources and restore state."""
        # Clear preview
        preview.cleanup_preview()

        # Force redraw all 3D views to clear any remaining visuals
        utils.tag_redraw_all_3d_views()

        # Restore cursor
        context.window.cursor_modal_restore()

        # Clear header
        context.area.header_text_set(None)

    def _update_header(self, context):
        """Update header text based on current state."""
        tool_name = self._get_tool_name()
        if self._state == self.STATE_FIRST_VERTEX:
            text = f"{tool_name}: Click to set first corner | ESC to cancel"
        elif self._state == self.STATE_LINE_END:
            text = f"{tool_name}: Click to set line end point | ESC to cancel"
        elif self._state == self.STATE_SECOND_VERTEX:
            if self._line_mode:
                text = f"{tool_name}: Click to set width | ESC to cancel"
            else:
                text = f"{tool_name}: Click to set opposite corner | ESC to cancel"
        elif self._state == self.STATE_DEPTH:
            text = f"{tool_name}: Move mouse to set depth ({self._depth:.3f}) | Click to confirm | ESC to cancel"
        else:
            text = f"{tool_name} | ESC to cancel"

        context.area.header_text_set(text)
