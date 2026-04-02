"""Active image tracking for texture operations."""

import bpy

from ..core.materials import get_image_from_material


# The currently active image for texture operations.
# Updated by: file browser selection, user clicking a face
# Used by: Alt+Click apply, UI panel preview
_active_image = None
# The previously active image name and filepath, shown as a disabled preview when
# no texture is selected. Stored as strings (not a reference) so they survive undo/redo.
# The filepath allows re-loading the image if undo removes it from bpy.data.images.
_previous_image_name = None
_previous_image_filepath = None
# Specifically for the case where faces start as selected e.g. initial cube on file creation.
# Guard flag: when True, depsgraph should not overwrite _active_image
# (set by apply_texture_from_file_browser which runs in a timer context
# where context.tool_settings may not reflect the 3D viewport state)
_active_image_just_set = False


def get_active_image():
    """Get the currently active image for texture operations.

    Returns None if the stored reference has been invalidated (e.g. by undo).
    """
    global _active_image
    if _active_image is None:
        return None
    try:
        _active_image.name
        return _active_image
    except ReferenceError:
        _active_image = None
        return None


def get_previous_image():
    """Get the previously active image for display when no texture is selected.

    Looks up the image by name each time so the reference survives undo/redo.
    If undo removed the image from bpy.data.images, re-loads it from the filepath.
    """
    if _previous_image_name is None:
        return None
    img = bpy.data.images.get(_previous_image_name)
    if img is None and _previous_image_filepath:
        try:
            img = bpy.data.images.load(_previous_image_filepath, check_existing=True)
        except RuntimeError:
            pass
    return img


def set_active_image(image):
    """Set the currently active image for texture operations.

    When a non-None image is set, it is also saved as the previous image
    so the panel can show a disabled preview when no texture is selected.

    Note: Does not call redraw_ui_panels here to avoid requiring a context parameter.
    Callers should call redraw_ui_panels(context) if an immediate UI update is needed.
    """
    global _active_image
    if image is not None:
        set_previous_image(image)
    _active_image = image


def set_previous_image(image):
    """Set the previous image for display when no texture is selected."""
    global _previous_image_name, _previous_image_filepath
    if image is not None:
        _previous_image_name = image.name
        _previous_image_filepath = image.filepath_from_user()
    else:
        _previous_image_name = None
        _previous_image_filepath = None


def get_active_image_just_set():
    """Check if active image was just set (guard for depsgraph)."""
    return _active_image_just_set


def set_active_image_just_set(value):
    """Set the active-image-just-set guard flag."""
    global _active_image_just_set
    _active_image_just_set = value


def update_active_image_from_face(context):
    """Update the active image based on the active face's material.

    Clears the active image if not in edit mode, no faces are selected,
    or the active face has no image material.
    """
    global _active_image_just_set
    if _active_image_just_set:
        _active_image_just_set = False
        return
    try:
        import bmesh

        obj = context.object
        if not obj or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
            set_active_image(None)
            return

        # Require face select mode
        if not context.tool_settings.mesh_select_mode[2]:
            set_active_image(None)
            return

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        # Check if any faces are selected
        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            set_active_image(None)
            return

        active_face = bm.faces.active
        if not active_face:
            set_active_image(None)
            return

        # Get the material on this face
        mat_index = active_face.material_index
        mat = obj.data.materials[mat_index] if mat_index < len(obj.data.materials) else None

        if mat:
            image = get_image_from_material(mat)
            if image:
                set_active_image(image)
            else:
                set_active_image(None)
        else:
            set_active_image(None)
    except Exception:
        pass  # Silently fail to avoid disrupting user workflow


def get_selected_faces_share_image(obj, bm, me):
    """Check if all selected faces share the same image texture.

    Returns (shared, image) where shared is True if all selected faces
    have the same image, and image is that shared image (or None).
    """
    selected_faces = [f for f in bm.faces if f.select]
    if not selected_faces:
        return False, None

    first_image = None
    for face in selected_faces:
        mat_index = face.material_index
        mat = me.materials[mat_index] if mat_index < len(me.materials) else None
        image = get_image_from_material(mat) if mat else None

        if first_image is None:
            first_image = image
        elif image != first_image:
            return False, None

    return True, first_image


def redraw_ui_panels(context):
    """Force redraw of UI panels to update texture preview"""
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def save_previous_image_state():
    """Return current previous image state for save/restore across undo."""
    return _previous_image_name, _previous_image_filepath


def restore_previous_image_state(name, filepath):
    """Restore previous image state after undo."""
    global _previous_image_name, _previous_image_filepath
    _previous_image_name = name
    _previous_image_filepath = filepath


def reset():
    """Reset all active image state."""
    global _active_image, _active_image_just_set, _previous_image_name, _previous_image_filepath
    _active_image = None
    _active_image_just_set = False
    _previous_image_name = None
    _previous_image_filepath = None
