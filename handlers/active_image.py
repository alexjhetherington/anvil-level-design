"""Active image tracking for texture operations."""

import bpy

from ..core.materials import get_image_from_material


# The currently active image for texture operations.
# Updated by: texture browser selection and face picking.
# Used by: Alt+Click apply, UI panel preview
_active_image = None
# The previously active image name and filepath, shown as a disabled preview when
# no texture is selected. Stored as strings (not a reference) so they survive undo/redo.
# The filepath allows re-loading the image if undo removes it from bpy.data.images.
_previous_image_name = None
_previous_image_filepath = None


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


def get_active_face_material(obj, mode, face_select_enabled):
    if (
            obj is None
            or obj.type != 'MESH'
            or mode != 'EDIT_MESH'
            or not face_select_enabled):
        return None

    import bmesh

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    if not any(face.select for face in bm.faces):
        return None

    active_face = bm.faces.active
    if active_face is None:
        return None

    material_index = active_face.material_index
    if material_index >= len(obj.data.materials):
        return None
    return obj.data.materials[material_index]


def update_active_image_from_face(context):
    """Update the active image based on the active face's material.

    Clears the active image if not in edit mode, no faces are selected,
    or the active face has no image material.
    """
    try:
        material = get_active_face_material(
            context.object,
            context.mode,
            context.tool_settings.mesh_select_mode[2],
        )
        image = get_image_from_material(material) if material is not None else None
        set_active_image(image)
    except Exception:
        pass  # Silently fail to avoid disrupting user workflow


def get_selected_faces_share_material(bm, me):
    selected_faces = [face for face in bm.faces if face.select]
    if not selected_faces:
        return False, None

    first_face = selected_faces[0]
    first_index = first_face.material_index
    first_material = me.materials[first_index] if first_index < len(me.materials) else None
    for face in selected_faces[1:]:
        material_index = face.material_index
        material = (
            me.materials[material_index]
            if material_index < len(me.materials)
            else None
        )
        if material != first_material:
            return False, None
    return True, first_material


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
    global _active_image, _previous_image_name, _previous_image_filepath
    _active_image = None
    _previous_image_name = None
    _previous_image_filepath = None
