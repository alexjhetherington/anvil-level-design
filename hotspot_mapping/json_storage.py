"""
Hotspot Mapping - JSON Storage

Handles reading/writing hotspot data.
Data is stored in scene properties for undo support.
Optionally synced to an external JSON file on blend save.
"""

import bpy
import json
import os

from ..utils import debug_log


# JSON schema version
HOTSPOTS_VERSION = "1.0"

# Orientation types for hotspots
ORIENTATION_TYPES = ('Any', 'Upwards', 'Floor', 'Ceiling')


def resolve_filepath():
    """Resolve the hotspots file path property to an absolute path.

    If the stored path is relative, resolves it against the .blend file directory.
    If the stored path is absolute, returns it as-is.

    Returns:
        Absolute path string, or None if no path is set.
    """
    scene = bpy.context.scene
    if not scene or not hasattr(scene, 'hotspot_mapping_props'):
        return None

    raw_path = scene.hotspot_mapping_props.hotspots_file_path
    if not raw_path:
        return None

    # Strip Blender's "//" relative path prefix
    if raw_path.startswith("//"):
        blend_filepath = bpy.data.filepath
        if not blend_filepath:
            return None
        blend_dir = os.path.dirname(blend_filepath)
        return os.path.normpath(os.path.join(blend_dir, raw_path[2:]))

    return os.path.normpath(raw_path)


def make_path_relative(absolute_path):
    """Convert an absolute path to a Blender-style relative path (//...).

    Returns the original path if the blend file is not saved or if a relative
    path cannot be computed (e.g. cross-drive on Windows).

    Args:
        absolute_path: Absolute file path.

    Returns:
        Blender-style relative path string, or the absolute path as fallback.
    """
    blend_filepath = bpy.data.filepath
    if not blend_filepath:
        return absolute_path

    blend_dir = os.path.dirname(blend_filepath)
    try:
        rel = os.path.relpath(absolute_path, blend_dir)
        return "//" + rel.replace("\\", "/")
    except ValueError:
        # Cross-drive on Windows - keep absolute
        return absolute_path


def load_hotspots():
    """Load hotspots data from scene property.

    Returns:
        Dict with hotspot data, or empty structure if not found.
    """
    scene = bpy.context.scene
    if not scene or not hasattr(scene, 'hotspot_mapping_props'):
        return _create_empty_data()

    props = scene.hotspot_mapping_props

    if props.hotspots_json:
        try:
            return json.loads(props.hotspots_json)
        except json.JSONDecodeError:
            debug_log("[Hotspots] Invalid JSON in scene property")

    return _create_empty_data()


def save_hotspots(data):
    """Save hotspots data to scene property.

    Args:
        data: Dict with hotspot data structure.

    Returns:
        True if save successful, False otherwise.
    """
    scene = bpy.context.scene
    if not scene or not hasattr(scene, 'hotspot_mapping_props'):
        return False

    scene.hotspot_mapping_props.hotspots_json = json.dumps(data)
    return True


def load_from_file(filepath):
    """Read hotspot data from a JSON file and store in scene property.

    If the file does not exist, creates it with current scene data (or empty data).

    Args:
        filepath: Absolute path to the JSON file.

    Returns:
        True if loaded successfully, False on error.
    """
    scene = bpy.context.scene
    if not scene or not hasattr(scene, 'hotspot_mapping_props'):
        return False

    if not os.path.exists(filepath):
        # File doesn't exist yet - create it with current scene data
        debug_log(f"[Hotspots] File not found, creating: {filepath}")
        data = load_hotspots()
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            debug_log(f"[Hotspots] Created: {filepath}")
            return True
        except IOError as e:
            print(f"Anvil Hotspots: Error creating {filepath}: {e}", flush=True)
            return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        scene.hotspot_mapping_props.hotspots_json = json.dumps(data)
        debug_log(f"[Hotspots] Loaded from: {filepath}")
        return True
    except (json.JSONDecodeError, IOError) as e:
        print(f"Anvil Hotspots: Error reading {filepath}: {e}", flush=True)
        return False


def save_to_file(filepath):
    """Write scene property hotspot data to a JSON file.

    Args:
        filepath: Absolute path to the JSON file.

    Returns:
        True if saved successfully, False on error.
    """
    scene = bpy.context.scene
    if not scene or not hasattr(scene, 'hotspot_mapping_props'):
        return False

    props = scene.hotspot_mapping_props
    if not props.hotspots_json:
        return True

    try:
        data = json.loads(props.hotspots_json)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        debug_log(f"[Hotspots] Saved to: {filepath}")
        return True
    except (json.JSONDecodeError, IOError) as e:
        print(f"Anvil Hotspots: Error writing {filepath}: {e}", flush=True)
        return False


def sync_from_file():
    """Load data from the configured JSON file into scene property.

    Called on file load. Does nothing if no file path is configured.
    """
    filepath = resolve_filepath()
    if filepath is None or not os.path.exists(filepath):
        return

    scene = bpy.context.scene
    if not scene or not hasattr(scene, 'hotspot_mapping_props'):
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        scene.hotspot_mapping_props.hotspots_json = json.dumps(data)
        debug_log(f"[Hotspots] Synced from file: {filepath}")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Anvil Hotspots: Error syncing from {filepath}: {e}", flush=True)


def sync_to_file():
    """Save data from scene property to the configured JSON file.

    Called on file save. Does nothing if no file path is configured.
    """
    filepath = resolve_filepath()
    if filepath is None:
        return

    scene = bpy.context.scene
    if not scene or not hasattr(scene, 'hotspot_mapping_props'):
        return

    props = scene.hotspot_mapping_props
    if not props.hotspots_json:
        return

    try:
        data = json.loads(props.hotspots_json)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        debug_log(f"[Hotspots] Synced to file: {filepath}")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Anvil Hotspots: Error syncing to {filepath}: {e}", flush=True)


def scene_has_hotspots():
    """Check if the current scene has any hotspot data.

    Returns:
        True if there are any textures with hotspots defined.
    """
    data = load_hotspots()
    textures = data.get("textures", {})
    for tex_data in textures.values():
        if tex_data.get("hotspots"):
            return True
    return False


def _create_empty_data():
    """Create empty hotspots data structure."""
    return {
        "version": HOTSPOTS_VERSION,
        "textures": {}
    }


def is_texture_hotspottable(texture_name):
    """Check if a texture is marked as hotspottable.

    Args:
        texture_name: Name of the texture (image filename).

    Returns:
        True if texture exists in hotspots data, False otherwise.
    """
    data = load_hotspots()
    return texture_name in data.get("textures", {})


def add_texture_as_hotspottable(texture_name, width, height):
    """Add a texture as hotspottable.

    Args:
        texture_name: Name of the texture (image filename).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        True if added successfully, False otherwise.
    """
    data = load_hotspots()

    if texture_name in data.get("textures", {}):
        debug_log(f"[Hotspots] Texture already exists: {texture_name}")
        return True

    data["textures"][texture_name] = {
        "image_width": width,
        "image_height": height,
        "hotspots": []
    }

    return save_hotspots(data)


def remove_texture_as_hotspottable(texture_name):
    """Remove a texture from hotspottable list.

    Args:
        texture_name: Name of the texture to remove.

    Returns:
        True if removed successfully, False otherwise.
    """
    data = load_hotspots()

    if texture_name not in data.get("textures", {}):
        return True

    del data["textures"][texture_name]
    return save_hotspots(data)


def get_texture_hotspots(texture_name, data=None):
    """Get all hotspots for a texture.

    Args:
        texture_name: Name of the texture.
        data: Pre-loaded hotspot data dict. If None, loads from storage.

    Returns:
        List of hotspot dicts, or empty list if texture not found.
    """
    if data is None:
        data = load_hotspots()
    texture_data = data.get("textures", {}).get(texture_name, {})
    return texture_data.get("hotspots", [])


def get_texture_dimensions(texture_name, data=None):
    """Get stored dimensions for a texture.

    Args:
        texture_name: Name of the texture.
        data: Pre-loaded hotspot data dict. If None, loads from storage.

    Returns:
        Tuple (width, height), or (0, 0) if not found.
    """
    if data is None:
        data = load_hotspots()
    texture_data = data.get("textures", {}).get(texture_name, {})
    return (
        texture_data.get("image_width", 0),
        texture_data.get("image_height", 0)
    )


def _generate_hotspot_id(hotspots):
    """Generate a unique hotspot ID.

    Args:
        hotspots: List of existing hotspots.

    Returns:
        String ID like "hotspot_0", "hotspot_1", etc.
    """
    existing_ids = {h.get("id", "") for h in hotspots}
    counter = 0
    while True:
        new_id = f"hotspot_{counter}"
        if new_id not in existing_ids:
            return new_id
        counter += 1


def add_hotspot(texture_name, x, y, width, height):
    """Add a new hotspot to a texture.

    Args:
        texture_name: Name of the texture.
        x: X position of top-left corner in pixels.
        y: Y position of top-left corner in pixels.
        width: Width of hotspot in pixels.
        height: Height of hotspot in pixels.

    Returns:
        The new hotspot ID, or None if failed.
    """
    data = load_hotspots()

    if texture_name not in data.get("textures", {}):
        debug_log(f"[Hotspots] Texture not found: {texture_name}")
        return None

    hotspots = data["textures"][texture_name].get("hotspots", [])
    new_id = _generate_hotspot_id(hotspots)

    hotspots.append({
        "id": new_id,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "orientation_type": "Any"
    })

    data["textures"][texture_name]["hotspots"] = hotspots

    if save_hotspots(data):
        return new_id
    return None


def update_hotspot(texture_name, hotspot_id, x, y, width, height):
    """Update an existing hotspot.

    Args:
        texture_name: Name of the texture.
        hotspot_id: ID of the hotspot to update.
        x: New X position.
        y: New Y position.
        width: New width.
        height: New height.

    Returns:
        True if updated successfully, False otherwise.
    """
    data = load_hotspots()

    if texture_name not in data.get("textures", {}):
        return False

    hotspots = data["textures"][texture_name].get("hotspots", [])

    for hotspot in hotspots:
        if hotspot.get("id") == hotspot_id:
            hotspot["x"] = x
            hotspot["y"] = y
            hotspot["width"] = width
            hotspot["height"] = height
            return save_hotspots(data)

    debug_log(f"[Hotspots] Hotspot not found: {hotspot_id}")
    return False


def delete_hotspot(texture_name, hotspot_id):
    """Delete a hotspot.

    Args:
        texture_name: Name of the texture.
        hotspot_id: ID of the hotspot to delete.

    Returns:
        True if deleted successfully, False otherwise.
    """
    data = load_hotspots()

    if texture_name not in data.get("textures", {}):
        return False

    hotspots = data["textures"][texture_name].get("hotspots", [])
    original_count = len(hotspots)

    data["textures"][texture_name]["hotspots"] = [
        h for h in hotspots if h.get("id") != hotspot_id
    ]

    if len(data["textures"][texture_name]["hotspots"]) < original_count:
        return save_hotspots(data)

    debug_log(f"[Hotspots] Hotspot not found for deletion: {hotspot_id}")
    return False


def get_hotspot_by_id(texture_name, hotspot_id):
    """Get a specific hotspot by ID.

    Args:
        texture_name: Name of the texture.
        hotspot_id: ID of the hotspot.

    Returns:
        Hotspot dict, or None if not found.
    """
    hotspots = get_texture_hotspots(texture_name)
    for hotspot in hotspots:
        if hotspot.get("id") == hotspot_id:
            return hotspot
    return None


def get_hotspot_orientation(texture_name, hotspot_id):
    """Get the orientation type of a hotspot.

    Args:
        texture_name: Name of the texture.
        hotspot_id: ID of the hotspot.

    Returns:
        Orientation type string, or 'Any' if not found.
    """
    hotspot = get_hotspot_by_id(texture_name, hotspot_id)
    if hotspot:
        return hotspot.get("orientation_type", "Any")
    return "Any"


def set_hotspot_orientation(texture_name, hotspot_id, orientation_type):
    """Set the orientation type of a hotspot.

    Args:
        texture_name: Name of the texture.
        hotspot_id: ID of the hotspot.
        orientation_type: One of ORIENTATION_TYPES.

    Returns:
        True if updated successfully, False otherwise.
    """
    if orientation_type not in ORIENTATION_TYPES:
        debug_log(f"[Hotspots] Invalid orientation type: {orientation_type}")
        return False

    data = load_hotspots()

    if texture_name not in data.get("textures", {}):
        return False

    hotspots = data["textures"][texture_name].get("hotspots", [])

    for hotspot in hotspots:
        if hotspot.get("id") == hotspot_id:
            hotspot["orientation_type"] = orientation_type
            return save_hotspots(data)

    debug_log(f"[Hotspots] Hotspot not found: {hotspot_id}")
    return False


def cycle_hotspot_orientation(texture_name, hotspot_id):
    """Cycle to the next orientation type for a hotspot.

    Args:
        texture_name: Name of the texture.
        hotspot_id: ID of the hotspot.

    Returns:
        The new orientation type, or None if failed.
    """
    current = get_hotspot_orientation(texture_name, hotspot_id)
    current_index = ORIENTATION_TYPES.index(current) if current in ORIENTATION_TYPES else 0
    next_index = (current_index + 1) % len(ORIENTATION_TYPES)
    next_type = ORIENTATION_TYPES[next_index]

    if set_hotspot_orientation(texture_name, hotspot_id, next_type):
        return next_type
    return None
