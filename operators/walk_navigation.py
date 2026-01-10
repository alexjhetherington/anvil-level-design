import bpy
from bpy.types import Operator
from mathutils import Vector, Euler
import math


def get_addon_prefs(context):
    """Get addon preferences."""
    addon = context.preferences.addons.get(__package__.rsplit('.', 1)[0])
    if addon:
        return addon.preferences
    return None


class LEVELDESIGN_OT_walk_navigation_hold(Operator):
    """First-person camera navigation while holding right mouse button."""
    bl_idname = "leveldesign.walk_navigation_hold"
    bl_label = "First Person Camera (Hold RMB)"
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            return {'CANCELLED'}

        rv3d = context.region_data
        if rv3d is None:
            return {'CANCELLED'}

        self.region_3d = rv3d

        # Save original settings to restore on exit
        self.original_view_distance = rv3d.view_distance
        self.original_smooth_view = context.preferences.view.smooth_view

        # Disable smooth view during freelook
        context.preferences.view.smooth_view = 0

        # Convert from orbit mode to freelook: move view_location to actual camera position
        if rv3d.view_distance > 0:
            rot_mat = rv3d.view_rotation.to_matrix()
            cam_offset = rot_mat @ Vector((0, 0, rv3d.view_distance))
            rv3d.view_location = rv3d.view_location + cam_offset
            rv3d.view_distance = 0

        # Extract current yaw/pitch from existing view rotation using forward vector
        # (more robust than euler decomposition for arbitrary rotations)
        rot_mat = rv3d.view_rotation.to_matrix()
        forward = rot_mat @ Vector((0, 0, -1))  # Camera looks down -Z

        # Yaw: horizontal angle from -Y axis (Blender's forward)
        self.yaw = math.atan2(-forward.x, forward.y)

        # Pitch: angle from horizontal plane
        horizontal_dist = math.sqrt(forward.x ** 2 + forward.y ** 2)
        self.pitch = math.atan2(forward.z, horizontal_dist)

        # Track which keys are held
        self.keys_held = set()

        # Capture mouse at current position
        context.window.cursor_set('NONE')
        center_x = context.region.x + context.region.width // 2
        center_y = context.region.y + context.region.height // 2
        context.window.cursor_warp(center_x, center_y)
        self.last_mouse_x = center_x
        self.last_mouse_y = center_y
        self._skip_first_mouse = True  # Skip first MOUSEMOVE to avoid warp-induced delta

        # Add timer for smooth movement
        self._timer = context.window_manager.event_timer_add(1/60, window=context.window)

        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()

        return {'RUNNING_MODAL'}

    def _yaw_pitch_to_rotation(self, yaw, pitch):
        """Convert yaw/pitch to view quaternion."""
        # pitch=0 means looking at horizon, so we add pi/2 to get Blender's euler.x
        euler = Euler((pitch + math.pi / 2, 0, yaw), 'XYZ')
        return euler.to_quaternion()

    def modal(self, context, event):
        rv3d = self.region_3d

        # Exit on RMB release or Escape
        if event.type == 'RIGHTMOUSE' and event.value == 'RELEASE':
            return self._finish(context)

        if event.type == 'ESC':
            return self._finish(context)

        # Mouse look
        if event.type == 'MOUSEMOVE':
            # Skip first MOUSEMOVE to avoid spurious delta from cursor warp
            if self._skip_first_mouse:
                self._skip_first_mouse = False
                center_x = context.region.x + context.region.width // 2
                center_y = context.region.y + context.region.height // 2
                self.last_mouse_x = center_x
                self.last_mouse_y = center_y
                return {'RUNNING_MODAL'}

            dx = event.mouse_x - self.last_mouse_x
            dy = event.mouse_y - self.last_mouse_y

            prefs = get_addon_prefs(context)
            sensitivity = prefs.mouse_sensitivity if prefs else 0.006
            self.yaw -= dx * sensitivity
            self.pitch += dy * sensitivity

            # Clamp pitch to prevent flipping (just under 90 degrees)
            max_pitch = math.pi / 2 - 0.01
            self.pitch = max(-max_pitch, min(max_pitch, self.pitch))

            # Apply rotation
            rv3d.view_rotation = self._yaw_pitch_to_rotation(self.yaw, self.pitch)

            # Warp cursor back to center
            center_x = context.region.x + context.region.width // 2
            center_y = context.region.y + context.region.height // 2
            context.window.cursor_warp(center_x, center_y)
            self.last_mouse_x = center_x
            self.last_mouse_y = center_y

            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Track key presses for movement
        movement_keys = {'W', 'A', 'S', 'D', 'Q', 'E', 'LEFT_SHIFT', 'RIGHT_SHIFT'}
        if event.type in movement_keys:
            if event.value == 'PRESS':
                self.keys_held.add(event.type)
            elif event.value == 'RELEASE':
                self.keys_held.discard(event.type)
            return {'RUNNING_MODAL'}

        # Scroll to adjust speed
        if event.type == 'WHEELUPMOUSE':
            prefs = get_addon_prefs(context)
            if prefs:
                prefs.move_speed *= 1.2
                self.report({'INFO'}, f"Speed: {prefs.move_speed:.3f}")
            return {'RUNNING_MODAL'}
        if event.type == 'WHEELDOWNMOUSE':
            prefs = get_addon_prefs(context)
            if prefs:
                prefs.move_speed /= 1.2
                self.report({'INFO'}, f"Speed: {prefs.move_speed:.3f}")
            return {'RUNNING_MODAL'}

        # Timer tick - apply movement
        if event.type == 'TIMER':
            self._apply_movement(context, rv3d)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def _apply_movement(self, context, rv3d):
        """Apply movement based on currently held keys."""
        if not self.keys_held:
            return

        prefs = get_addon_prefs(context)
        speed = prefs.move_speed if prefs else 0.1
        if 'LEFT_SHIFT' in self.keys_held or 'RIGHT_SHIFT' in self.keys_held:
            speed *= 3

        # Get view directions from rotation
        rot_mat = rv3d.view_rotation.to_matrix()
        forward = rot_mat @ Vector((0, 0, -1))  # Camera forward
        right = rot_mat @ Vector((1, 0, 0))     # Camera right
        world_up = Vector((0, 0, 1))            # World up for Q/E

        move_dir = Vector((0, 0, 0))

        # WASD: camera-relative movement
        if 'W' in self.keys_held:
            move_dir += forward
        if 'S' in self.keys_held:
            move_dir -= forward
        if 'D' in self.keys_held:
            move_dir += right
        if 'A' in self.keys_held:
            move_dir -= right
        # Q/E: world-relative vertical movement
        if 'E' in self.keys_held:
            move_dir += world_up
        if 'Q' in self.keys_held:
            move_dir -= world_up

        if move_dir.length > 0:
            move_dir.normalize()
            rv3d.view_location += move_dir * speed

    def _finish(self, context):
        """Clean up and exit modal."""
        rv3d = self.region_3d

        # Restore view_distance: move view_location back to orbit center
        if self.original_view_distance > 0:
            rot_mat = rv3d.view_rotation.to_matrix()
            cam_offset = rot_mat @ Vector((0, 0, self.original_view_distance))
            rv3d.view_location = rv3d.view_location - cam_offset
            rv3d.view_distance = self.original_view_distance

        # Restore smooth view setting
        context.preferences.view.smooth_view = self.original_smooth_view

        context.window.cursor_set('DEFAULT')
        context.window_manager.event_timer_remove(self._timer)
        context.area.tag_redraw()
        return {'FINISHED'}


class LEVELDESIGN_OT_context_menu(Operator):
    """Show the appropriate context menu for the current mode"""
    bl_idname = "leveldesign.context_menu"
    bl_label = "Context Menu"

    def execute(self, context):
        if context.mode == 'OBJECT':
            bpy.ops.wm.call_menu(name='VIEW3D_MT_object_context_menu')
        elif context.mode == 'EDIT_MESH':
            bpy.ops.wm.call_menu(name='VIEW3D_MT_edit_mesh_context_menu')
        elif context.mode == 'EDIT_CURVE':
            bpy.ops.wm.call_menu(name='VIEW3D_MT_edit_curve_context_menu')
        elif context.mode == 'EDIT_SURFACE':
            bpy.ops.wm.call_menu(name='VIEW3D_MT_edit_surface_context_menu')
        elif context.mode == 'EDIT_ARMATURE':
            bpy.ops.wm.call_menu(name='VIEW3D_MT_edit_armature_context_menu')
        elif context.mode == 'POSE':
            bpy.ops.wm.call_menu(name='VIEW3D_MT_pose_context_menu')
        elif context.mode == 'SCULPT':
            bpy.ops.wm.call_menu(name='VIEW3D_MT_sculpt_context_menu')
        elif context.mode == 'PAINT_WEIGHT':
            bpy.ops.wm.call_menu(name='VIEW3D_MT_weight_paint_context_menu')
        elif context.mode == 'PAINT_VERTEX':
            bpy.ops.wm.call_menu(name='VIEW3D_MT_vertex_paint_context_menu')
        elif context.mode == 'PAINT_TEXTURE':
            bpy.ops.wm.call_menu(name='VIEW3D_MT_texture_paint_context_menu')
        else:
            # Fallback to object context menu
            bpy.ops.wm.call_menu(name='VIEW3D_MT_object_context_menu')
        return {'FINISHED'}


classes = (
    LEVELDESIGN_OT_walk_navigation_hold,
    LEVELDESIGN_OT_context_menu,
)

addon_keymaps = []

# Keymaps to register our bindings in
KEYMAPS_TO_REGISTER = [
    ("Object Mode", 'EMPTY'),
    ("Mesh", 'EMPTY'),
    ("Curve", 'EMPTY'),
    ("Armature", 'EMPTY'),
    ("Pose", 'EMPTY'),
    ("Sculpt", 'EMPTY'),
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    for km_name, space_type in KEYMAPS_TO_REGISTER:
        km = kc.keymaps.new(name=km_name, space_type=space_type)

        # Add our walk navigation binding
        kmi = km.keymap_items.new(
            "leveldesign.walk_navigation_hold",
            'RIGHTMOUSE', 'PRESS',
            head=True
        )
        addon_keymaps.append((km, kmi))

        # Add Ctrl+Right-click for context menu
        kmi = km.keymap_items.new(
            "leveldesign.context_menu",
            'RIGHTMOUSE', 'PRESS',
            ctrl=True,
            head=True
        )
        addon_keymaps.append((km, kmi))


def unregister():
    # Remove our keymaps
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
