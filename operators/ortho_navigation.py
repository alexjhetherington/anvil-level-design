import bpy
from bpy.types import Operator

from ..utils import is_level_design_workspace


class LEVELDESIGN_OT_ortho_view(Operator):
    """Set orthographic view orientation, bypassing rotation lock."""
    bl_idname = "leveldesign.ortho_view"
    bl_label = "Ortho View (Bypass Lock)"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    view_type: bpy.props.EnumProperty(
        name="View Type",
        items=[
            ('FRONT', "Front", "Front view (-Y)"),
            ('BACK', "Back", "Back view (+Y)"),
            ('RIGHT', "Right", "Right view (+X)"),
            ('LEFT', "Left", "Left view (-X)"),
            ('TOP', "Top", "Top view (+Z)"),
            ('BOTTOM', "Bottom", "Bottom view (-Z)"),
        ],
        default='FRONT'
    )

    def execute(self, context):
        if context.area.type != 'VIEW_3D':
            return {'CANCELLED'}

        rv3d = context.region_data
        if rv3d is None:
            return {'CANCELLED'}

        # Store lock state and temporarily unlock
        was_locked = rv3d.lock_rotation
        rv3d.lock_rotation = False

        # Use Blender's built-in view_axis operator to properly set named orthographic view
        # This ensures the view is registered as "Top/Front/Right" etc., not "User Ortho"
        # which is required for proper grid display
        bpy.ops.view3d.view_axis(type=self.view_type, align_active=False)

        # Restore lock state
        rv3d.lock_rotation = was_locked

        context.area.tag_redraw()
        return {'FINISHED'}


class LEVELDESIGN_OT_ortho_pan(Operator):
    """Pan view with MMB in orthographic views, orbit in perspective views."""
    bl_idname = "leveldesign.ortho_pan"
    bl_label = "Ortho Pan"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            return {'PASS_THROUGH'}

        rv3d = context.region_data
        if rv3d is None:
            return {'PASS_THROUGH'}

        # Check if view is orthographic
        if rv3d.is_perspective:
            # Perspective view - pass through to default orbit
            return {'PASS_THROUGH'}

        # Orthographic view - handle panning ourselves
        self.region_3d = rv3d
        self.last_mouse_x = event.mouse_x
        self.last_mouse_y = event.mouse_y
        self.region = context.region

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        rv3d = self.region_3d

        if event.type == 'MIDDLEMOUSE' and event.value == 'RELEASE':
            return {'FINISHED'}

        if event.type == 'ESC':
            return {'CANCELLED'}

        if event.type == 'MOUSEMOVE':
            # Calculate mouse delta
            dx = event.mouse_x - self.last_mouse_x
            dy = event.mouse_y - self.last_mouse_y

            # Convert screen delta to world delta based on zoom level
            # view_distance controls zoom in ortho mode
            zoom_factor = rv3d.view_distance / self.region.height

            # Get view rotation to determine pan directions
            rot_mat = rv3d.view_rotation.to_matrix()

            # Pan in view-local X and Y directions
            # In ortho view: X is right, Y is up in screen space
            from mathutils import Vector
            view_right = rot_mat @ Vector((1, 0, 0))
            view_up = rot_mat @ Vector((0, 1, 0))

            # Apply pan (negative because dragging right should move view left)
            rv3d.view_location -= view_right * dx * zoom_factor
            rv3d.view_location -= view_up * dy * zoom_factor

            self.last_mouse_x = event.mouse_x
            self.last_mouse_y = event.mouse_y

            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}


classes = (
    LEVELDESIGN_OT_ortho_view,
    LEVELDESIGN_OT_ortho_pan,
)

addon_keymaps = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    # Register in 3D View (global) keymap
    km = kc.keymaps.new(name='3D View', space_type='VIEW_3D')

    # Add our MMB binding with head=True so it fires first
    kmi = km.keymap_items.new(
        "leveldesign.ortho_pan",
        'MIDDLEMOUSE', 'PRESS',
        head=True
    )
    addon_keymaps.append((km, kmi))

    # Numpad view bindings (bypass rotation lock)
    # Numpad 1 - Front
    kmi = km.keymap_items.new(
        "leveldesign.ortho_view",
        'NUMPAD_1', 'PRESS',
        head=True
    )
    kmi.properties.view_type = 'FRONT'
    addon_keymaps.append((km, kmi))

    # Ctrl+Numpad 1 - Back
    kmi = km.keymap_items.new(
        "leveldesign.ortho_view",
        'NUMPAD_1', 'PRESS',
        ctrl=True,
        head=True
    )
    kmi.properties.view_type = 'BACK'
    addon_keymaps.append((km, kmi))

    # Numpad 3 - Right
    kmi = km.keymap_items.new(
        "leveldesign.ortho_view",
        'NUMPAD_3', 'PRESS',
        head=True
    )
    kmi.properties.view_type = 'RIGHT'
    addon_keymaps.append((km, kmi))

    # Ctrl+Numpad 3 - Left
    kmi = km.keymap_items.new(
        "leveldesign.ortho_view",
        'NUMPAD_3', 'PRESS',
        ctrl=True,
        head=True
    )
    kmi.properties.view_type = 'LEFT'
    addon_keymaps.append((km, kmi))

    # Numpad 7 - Top
    kmi = km.keymap_items.new(
        "leveldesign.ortho_view",
        'NUMPAD_7', 'PRESS',
        head=True
    )
    kmi.properties.view_type = 'TOP'
    addon_keymaps.append((km, kmi))

    # Ctrl+Numpad 7 - Bottom
    kmi = km.keymap_items.new(
        "leveldesign.ortho_view",
        'NUMPAD_7', 'PRESS',
        ctrl=True,
        head=True
    )
    kmi.properties.view_type = 'BOTTOM'
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
