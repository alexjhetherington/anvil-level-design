"""
Hotspot Mapping - JSON Storage

Handles reading/writing hotspot data.
Data is stored in scene properties for undo support, synced to JSON file on save.
"""

import bpy
import json
import os

from ..utils import debug_log


# JSON schema version
HOTSPOTS_VERSION = "1.0"

# Orientation types for hotspots
ORIENTATION_TYPES = ('Any', 'Upwards', 'Floor', 'Ceiling')


def get_hotspots_filepath():
    """Get the path to hotspots.json next to the .blend file.

    Returns:
        Path string, or None if blend file is not saved.
    """
    blend_filepath = bpy.data.filepath
    if not blend_filepath:
        return None

    blend_dir = os.path.dirname(blend_filepath)
    return os.path.join(blend_dir, "hotspots.json")


_file_not_found_cache = set()


def invalidate_cache():
    """Clear scene property data, forcing next load to read from disk."""
    _file_not_found_cache.clear()
    scene = bpy.context.scene
    if scene and hasattr(scene, 'hotspot_mapping_props'):
        scene.hotspot_mapping_props.hotspots_json = ""


def load_hotspots():
    """Load hotspots data from scene property or JSON file.

    Uses scene property as primary source (for undo support).
    Falls back to JSON file if scene property is empty.

    Returns:
        Dict with hotspot data, or empty structure if not found.
    """
    scene = bpy.context.scene
    if not scene or not hasattr(scene, 'hotspot_mapping_props'):
        return _create_empty_data()

    props = scene.hotspot_mapping_props

    # Try scene property first (for undo support)
    if props.hotspots_json:
        try:
            return json.loads(props.hotspots_json)
        except json.JSONDecodeError:
            debug_log("[Hotspots] Invalid JSON in scene property, loading from file")

    # Fall back to JSON file
    filepath = get_hotspots_filepath()
    if filepath is None:
        debug_log("[Hotspots] Cannot load: blend file not saved")
        return _create_empty_data()

    if filepath in _file_not_found_cache:
        return _create_empty_data()

    if not os.path.exists(filepath):
        debug_log(f"[Hotspots] File not found: {filepath}")
        _file_not_found_cache.add(filepath)
        return _create_empty_data()

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        debug_log(f"[Hotspots] Loaded from: {filepath}")
        # Store in scene property for undo support
        props.hotspots_json = json.dumps(data)
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Anvil Hotspots: Error reading hotspots.json: {e}", flush=True)
        return _create_empty_data()


def save_hotspots(data, sync_to_disk=True):
    """Save hotspots data to scene property and optionally JSON file.

    Args:
        data: Dict with hotspot data structure.
        sync_to_disk: If True, also write to JSON file. Set False during
                      drag operations for performance (file syncs on blend save).

    Returns:
        True if save successful, False otherwise.
    """
    scene = bpy.context.scene
    if not scene or not hasattr(scene, 'hotspot_mapping_props'):
        return False

    # Save to scene property (this is tracked by undo)
    scene.hotspot_mapping_props.hotspots_json = json.dumps(data)

    if not sync_to_disk:
        return True

    # Also save to JSON file
    filepath = get_hotspots_filepath()
    if filepath is None:
        debug_log("[Hotspots] Cannot save to file: blend file not saved")
        return True  # Scene property was saved, file will sync later

    try:
        _file_not_found_cache.discard(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        debug_log(f"[Hotspots] Saved to: {filepath}")
        return True
    except IOError as e:
        print(f"Anvil Hotspots: Error writing hotspots.json: {e}", flush=True)
        return False


def sync_from_file():
    """Load data from JSON file into scene property.

    Called on file load to sync external changes.
    """
    filepath = get_hotspots_filepath()
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
        print(f"Anvil Hotspots: Error syncing from hotspots.json: {e}", flush=True)


def sync_to_file():
    """Save data from scene property to JSON file.

    Called on file save to ensure file is up to date.
    """
    scene = bpy.context.scene
    if not scene or not hasattr(scene, 'hotspot_mapping_props'):
        return

    props = scene.hotspot_mapping_props
    if not props.hotspots_json:
        return

    filepath = get_hotspots_filepath()
    if filepath is None:
        return

    try:
        data = json.loads(props.hotspots_json)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        debug_log(f"[Hotspots] Synced to file: {filepath}")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Anvil Hotspots: Error syncing to hotspots.json: {e}", flush=True)


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
        True if texture exists in hotspots.json, False otherwise.
    """
    data = load_hotspots()
    return texture_name in data.get("textures", {})


def add_texture_as_hotspottable(texture_name, width, height):
    """Add a texture to hotspots.json as hotspottable.

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
    """Remove a texture from hotspots.json.

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


def get_texture_hotspots(texture_name):
    """Get all hotspots for a texture.

    Args:
        texture_name: Name of the texture.

    Returns:
        List of hotspot dicts, or empty list if texture not found.
    """
    data = load_hotspots()
    texture_data = data.get("textures", {}).get(texture_name, {})
    return texture_data.get("hotspots", [])


def get_texture_dimensions(texture_name):
    """Get stored dimensions for a texture.

    Args:
        texture_name: Name of the texture.

    Returns:
        Tuple (width, height), or (0, 0) if not found.
    """
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


def update_hotspot(texture_name, hotspot_id, x, y, width, height, sync_to_disk=True):
    """Update an existing hotspot.

    Args:
        texture_name: Name of the texture.
        hotspot_id: ID of the hotspot to update.
        x: New X position.
        y: New Y position.
        width: New width.
        height: New height.
        sync_to_disk: If True, also write to JSON file. Set False during
                      drag operations for performance.

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
            return save_hotspots(data, sync_to_disk=sync_to_disk)

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
