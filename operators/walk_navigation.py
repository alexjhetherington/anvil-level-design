import bpy
from bpy.types import Operator

from ..utils import is_level_design_workspace


# HACK: We manage CONFIRM keymap items in Blender's "View3D Walk Modal" keymap
# (user keyconfig). During our walk wrapper modal, we enable them (and disable
# conflicting items for the trigger key), then disable them on exit.
# Two CONFIRM items are created: one plain and one with shift, so that releasing
# the trigger key while holding shift (walk fast mode) also exits the modal.
#
# This addon avoids directly modifying user keyconfig keymaps wherever possible because
# it risks leaving behind stale or duplicate entries if anything goes wrong (crashes,
# partial unregister, etc.). We make an exception here because Blender's modal keymaps
# (like "View3D Walk Modal") cannot be extended via addon keyconfigs — there is no way
# to set defaults for modal keymaps through the addon keyconfig API. Directly touching
# the user keyconfig is the only option.
#
# Why not add/remove at runtime? Calling km.keymap_items.remove() while Blender's
# C-level modal event dispatch is still tearing down the walk operator causes a
# use-after-free crash. Even deferring the remove to the next event loop tick
# (bpy.app.timers with first_interval=0.0) is not enough — only a ~1s delay avoids
# the crash, which causes other problems. Toggling .active is safe because no memory
# is allocated or freed, just a flag flip on a stable pointer.
#
# Additionally, walk's modal exit shuffles the C-level keymap item array, silently
# invalidating Python kmi references (no ReferenceError — the wrappers just point at
# wrong data). Because of this, we NEVER cache kmi references across walk invocations.
# All deferred restore logic uses property-based lookup (re-scanning the keymap) instead
# of stored references.
#
# The item is kept disabled when not inside our modal so that:
# 1. Manually activating walk navigation (View > Navigation > Walk) is not affected
# 2. It serves as a visible marker in Blender's keymap editor showing which key
#    the addon controls
#
# Our CONFIRM item is identified by value='RELEASE' — no default Walk Modal items use
# RELEASE for CONFIRM (they all use PRESS or ANY), so this is a reliable marker.
#
# Lifecycle:
# - Created lazily on first operator invoke (or found if it already exists)
# - Stale duplicates from previous crashed sessions are cleaned up on each invoke
# - .type updated to match the current trigger key at invoke time
# - .active toggled True on enter, False on exit
# - Removed at addon unregister (all CONFIRM RELEASE items, covering stale ones too)
# - If the user changes the walk keybind in addon preferences, .type is synced
#   via sync_confirm_key() called from the preferences draw

_MOUSE_KEY_TYPES = frozenset({
    'LEFTMOUSE', 'MIDDLEMOUSE', 'RIGHTMOUSE',
    'BUTTON4MOUSE', 'BUTTON5MOUSE', 'BUTTON6MOUSE', 'BUTTON7MOUSE',
    'PEN', 'ERASER', 'MOUSEMOVE', 'TRACKPADPAN', 'TRACKPADZOOM',
    'MOUSEROTATE', 'MOUSESMARTZOOM',
    'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE',
    'WHEELLEFTMOUSE', 'WHEELRIGHTMOUSE',
})


def _map_type_for_key(key_type):
    """Return the correct map_type for a given key type."""
    if key_type in _MOUSE_KEY_TYPES:
        return 'MOUSE'
    return 'KEYBOARD'


def sync_confirm_key(key_type):
    """Update all CONFIRM RELEASE items' key type to match the current keybind."""
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.user
    if not kc:
        return
    km = kc.keymaps.get("View3D Walk Modal")
    if not km:
        return
    new_map_type = _map_type_for_key(key_type)
    for kmi in km.keymap_items:
        if kmi.propvalue == 'CONFIRM' and kmi.value == 'RELEASE':
            if kmi.type != key_type:
                kmi.map_type = new_map_type
                kmi.type = key_type


class LEVELDESIGN_OT_walk_navigation_hold(Operator):
    """First-person camera navigation using Blender's walk mode while holding the trigger key."""
    bl_idname = "leveldesign.walk_navigation_hold"
    bl_label = "Walk Navigation (Hold)"
    bl_options = {'REGISTER', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            return {'CANCELLED'}

        rv3d = context.region_data
        if rv3d is None:
            return {'CANCELLED'}

        # Don't activate in orthographic mode
        if not rv3d.is_perspective:
            return {'PASS_THROUGH'}

        self._trigger_key = event.type
        self._done = False
        self._saved_states = {}

        # Modify walk modal keybinds before invoking walk
        self._modify_walk_keybinds()

        # Invoke Blender's built-in walk navigation
        result = bpy.ops.view3d.walk('INVOKE_DEFAULT')
        if result != {'RUNNING_MODAL'}:
            self._restore_walk_keybinds()
            return {'CANCELLED'}

        # Register our modal handler AFTER walk's, so ours runs first
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if self._done:
            self._schedule_deferred_restore()
            context.window_manager.event_timer_remove(self._timer)
            return {'FINISHED'}

        if event.type == self._trigger_key and event.value == 'RELEASE':
            self._done = True
            return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}

    def cancel(self, context):
        self._schedule_deferred_restore()
        if hasattr(self, '_timer'):
            context.window_manager.event_timer_remove(self._timer)

    def _modify_walk_keybinds(self):
        """Disable conflicting keybinds and enable the CONFIRM item."""
        wm = bpy.context.window_manager
        kc = wm.keyconfigs.user
        if not kc:
            return

        km = kc.keymaps.get("View3D Walk Modal")
        if not km:
            return

        # Re-enable all items to fix any stale disabled state from previous
        # crashed sessions (the old code disabled items and never restored them).
        for kmi in km.keymap_items:
            if not kmi.active:
                kmi.active = True

        # Clean up all CONFIRM RELEASE items from previous sessions and recreate
        # exactly the ones we need. Safe to remove here — walk hasn't started yet.
        stale = [kmi for kmi in list(km.keymap_items)
                 if kmi.propvalue == 'CONFIRM' and kmi.value == 'RELEASE']
        for kmi in stale:
            km.keymap_items.remove(kmi)

        # Create CONFIRM RELEASE items — one plain, one with shift (for fast mode)
        map_type = _map_type_for_key(self._trigger_key)
        confirm_kmis = []
        for shift in (False, True):
            kmi = km.keymap_items.new_modal(
                'CONFIRM', self._trigger_key, 'RELEASE', shift=shift
            )
            confirm_kmis.append(kmi)
        # Save and disable items that use the same key as our trigger.
        # Keyed by (propvalue, value, shift) for reference-free restore after walk exits.
        self._saved_states = {}
        for kmi in km.keymap_items:
            if kmi in confirm_kmis:
                continue
            if kmi.type == self._trigger_key:
                self._saved_states[(kmi.propvalue, kmi.value, kmi.shift)] = kmi.active
                kmi.active = False

        for kmi in confirm_kmis:
            kmi.active = True

    def _schedule_deferred_restore(self):
        """Defer keybind restoration to next tick. Uses property-based lookup
        instead of cached references, since walk's exit corrupts Python kmi
        wrappers. Only toggles .active — never adds or removes items."""
        saved = dict(self._saved_states)
        self._saved_states = {}
        trigger_key = self._trigger_key

        def _deferred():
            wm = bpy.context.window_manager
            kc = wm.keyconfigs.user
            if not kc:
                return None
            km = kc.keymaps.get("View3D Walk Modal")
            if not km:
                return None

            for kmi in km.keymap_items:
                if kmi.type != trigger_key:
                    continue
                if kmi.propvalue == 'CONFIRM' and kmi.value == 'RELEASE':
                    kmi.active = False
                else:
                    key = (kmi.propvalue, kmi.value, kmi.shift)
                    if key in saved:
                        kmi.active = saved[key]
            return None

        bpy.app.timers.register(_deferred, first_interval=0.0)

    def _restore_walk_keybinds(self):
        """Synchronous restore for use during invoke (before walk starts).
        Safe to use direct references here since walk hasn't run yet."""
        wm = bpy.context.window_manager
        kc = wm.keyconfigs.user
        if not kc:
            return
        km = kc.keymaps.get("View3D Walk Modal")
        if not km:
            return

        for kmi in km.keymap_items:
            if kmi.type != self._trigger_key:
                continue
            if kmi.propvalue == 'CONFIRM' and kmi.value == 'RELEASE':
                kmi.active = False
            else:
                key = (kmi.propvalue, kmi.value, kmi.shift)
                if key in self._saved_states:
                    kmi.active = self._saved_states[key]
        self._saved_states = {}


class LEVELDESIGN_OT_context_menu(Operator):
    """Show the appropriate context menu for the current mode"""
    bl_idname = "leveldesign.context_menu"
    bl_label = "Context Menu"

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

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


def _remove_walk_confirm_items():
    """Remove all CONFIRM RELEASE items from Walk Modal (ours and any stale ones)."""
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.user
    if not kc:
        return

    km = kc.keymaps.get("View3D Walk Modal")
    if km:
        to_remove = [kmi for kmi in list(km.keymap_items)
                     if kmi.propvalue == 'CONFIRM' and kmi.value == 'RELEASE']
        for kmi in to_remove:
            try:
                km.keymap_items.remove(kmi)
            except (ReferenceError, RuntimeError):
                pass


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
    _remove_walk_confirm_items()

    # Remove our keymaps
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
