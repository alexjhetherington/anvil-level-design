"""
Hotspot Mapping - JSON Storage

Handles reading/writing hotspot data.
Data is stored in scene properties for undo support.
Optionally synced to an external JSON file on blend save.

Data model: Each hotspottable texture has a list of bisecting lines.
Each line has: axis ("v" or "h"), pos (pixel position), start, end (extent).
A full line spans the entire image; a partial (anchored) line spans between
two perpendicular lines or image edges.

Rectangular hotspot cells are derived from the line arrangement using a
union-find algorithm on the micro-grid.
"""

import bpy
import json
import os

from ..core.logging import debug_log


# JSON schema version
HOTSPOTS_VERSION = "1.0"

# Orientation types for hotspots
ORIENTATION_TYPES = ('Any', 'Upwards', 'Floor', 'Ceiling')


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scene property load/save
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# File sync
# ---------------------------------------------------------------------------

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
        debug_log(f"[Hotspots] File not found, creating: {filepath}")
        data = load_hotspots()
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            debug_log(f"[Hotspots] Created: {filepath}")
            return True
        except IOError as e:
            print(f"Anvil Hotspots: Error creating {filepath}: {e}",
                  flush=True)
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
        print(f"Anvil Hotspots: Error syncing from {filepath}: {e}",
              flush=True)


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
        print(f"Anvil Hotspots: Error syncing to {filepath}: {e}",
              flush=True)


def scene_has_hotspots():
    """Check if the current scene has any hotspot data.

    Returns:
        True if there are any textures with hotspots defined.
    """
    data = load_hotspots()
    textures = data.get("textures", {})
    for tex_data in textures.values():
        if tex_data.get("lines") is not None:
            return True
        # Legacy formats
        if tex_data.get("tree") is not None:
            return True
        if tex_data.get("hotspots"):
            return True
    return False


def _create_empty_data():
    """Create empty hotspots data structure."""
    return {
        "version": HOTSPOTS_VERSION,
        "textures": {}
    }


# ---------------------------------------------------------------------------
# Texture management
# ---------------------------------------------------------------------------

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
    """Add a texture as hotspottable with a single full-image hotspot.

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
        "lines": [],
        "cell_orientations": {
            _cell_key(0, 0, width, height): "Any",
        },
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


# ---------------------------------------------------------------------------
# Cell key helpers
# ---------------------------------------------------------------------------

def _cell_key(x, y, w, h):
    """Create a string key for a cell from its bounds."""
    return f"{x}_{y}_{w}_{h}"


def _parse_cell_key(key):
    """Parse a cell key back into (x, y, w, h)."""
    parts = key.split("_")
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])


# ---------------------------------------------------------------------------
# Cell derivation
# ---------------------------------------------------------------------------

def derive_cells(lines, img_width, img_height):
    """Derive rectangular hotspot cells from a set of bisecting lines.

    Uses a micro-grid + union-find approach: lines create grid boundaries,
    and adjacent micro-cells without a separating line are merged.

    Args:
        lines: List of line dicts with axis, pos, start, end.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Sorted list of (x, y, w, h) tuples.
    """
    if not lines:
        return [(0, 0, img_width, img_height)]

    # Collect all unique grid coordinates
    x_set = {0, img_width}
    y_set = {0, img_height}
    for line in lines:
        if line["axis"] == "v":
            x_set.add(line["pos"])
            y_set.add(line["start"])
            y_set.add(line["end"])
        else:
            y_set.add(line["pos"])
            x_set.add(line["start"])
            x_set.add(line["end"])

    xs = sorted(x_set)
    ys = sorted(y_set)
    cols = len(xs) - 1
    rows = len(ys) - 1

    if cols <= 0 or rows <= 0:
        return [(0, 0, img_width, img_height)]

    # Index lookups
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}

    # Separation matrices
    # h_sep[r][c]: horizontal boundary between row r and r+1 at column c
    # v_sep[r][c]: vertical boundary between col c and c+1 at row r
    h_sep = [[False] * cols for _ in range(rows - 1)]
    v_sep = [[False] * (cols - 1) for _ in range(rows)]

    for line in lines:
        if line["axis"] == "h":
            pos = line["pos"]
            if pos not in y_idx or pos == 0 or pos == img_height:
                continue
            r = y_idx[pos] - 1
            start_c = x_idx.get(line["start"], 0)
            end_x = line["end"]
            for c in range(cols):
                if xs[c] >= line["start"] and xs[c + 1] <= end_x:
                    h_sep[r][c] = True
        else:
            pos = line["pos"]
            if pos not in x_idx or pos == 0 or pos == img_width:
                continue
            c = x_idx[pos] - 1
            start_r = y_idx.get(line["start"], 0)
            end_y = line["end"]
            for r in range(rows):
                if ys[r] >= line["start"] and ys[r + 1] <= end_y:
                    v_sep[r][c] = True

    # Union-find
    parent = list(range(rows * cols))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if c + 1 < cols and not v_sep[r][c]:
                union(idx, r * cols + c + 1)
            if r + 1 < rows and not h_sep[r][c]:
                union(idx, (r + 1) * cols + c)

    # Collect merged groups → bounding rectangles
    groups = {}
    for r in range(rows):
        for c in range(cols):
            root = find(r * cols + c)
            if root not in groups:
                groups[root] = []
            groups[root].append((r, c))

    cells = []
    for members in groups.values():
        min_r = min(r for r, c in members)
        max_r = max(r for r, c in members)
        min_c = min(c for r, c in members)
        max_c = max(c for r, c in members)
        x = xs[min_c]
        y = ys[min_r]
        w = xs[max_c + 1] - x
        h = ys[max_r + 1] - y
        cells.append((x, y, w, h))

    cells.sort(key=lambda c: (c[1], c[0]))
    return cells


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_texture_lines(texture_name, data=None):
    """Get the list of bisecting lines for a texture.

    Args:
        texture_name: Name of the texture.
        data: Pre-loaded data dict. If None, loads from storage.

    Returns:
        List of line dicts, or empty list if not found.
    """
    if data is None:
        data = load_hotspots()
    tex_data = data.get("textures", {}).get(texture_name)
    if tex_data is None:
        return []
    return tex_data.get("lines", [])


def get_texture_hotspots(texture_name, data=None):
    """Get all hotspots for a texture as a flat list.

    Derives cells from lines and attaches orientations.
    Also supports legacy formats.

    Args:
        texture_name: Name of the texture.
        data: Pre-loaded hotspot data dict. If None, loads from storage.

    Returns:
        List of hotspot dicts with x, y, width, height, orientation_type.
    """
    if data is None:
        data = load_hotspots()
    tex_data = data.get("textures", {}).get(texture_name, {})

    # Lines-based format (current)
    lines = tex_data.get("lines")
    if lines is not None:
        img_w = tex_data.get("image_width", 0)
        img_h = tex_data.get("image_height", 0)
        if img_w <= 0 or img_h <= 0:
            return []
        cells = derive_cells(lines, img_w, img_h)
        orientations = tex_data.get("cell_orientations", {})
        result = []
        for x, y, w, h in cells:
            key = _cell_key(x, y, w, h)
            orientation = orientations.get(key, "Any")
            result.append({
                "x": x, "y": y, "width": w, "height": h,
                "orientation_type": orientation,
            })
        return result

    # Legacy flat format
    return tex_data.get("hotspots", [])


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


def get_cells_with_orientations(texture_name, data=None):
    """Get derived cells with their orientations and keys.

    Args:
        texture_name: Name of the texture.
        data: Pre-loaded data dict. If None, loads from storage.

    Returns:
        List of (x, y, w, h, orientation_type, cell_key) tuples,
        or empty list.
    """
    if data is None:
        data = load_hotspots()
    tex_data = data.get("textures", {}).get(texture_name, {})

    lines = tex_data.get("lines")
    if lines is None:
        return []

    img_w = tex_data.get("image_width", 0)
    img_h = tex_data.get("image_height", 0)
    if img_w <= 0 or img_h <= 0:
        return []

    cells = derive_cells(lines, img_w, img_h)
    orientations = tex_data.get("cell_orientations", {})

    result = []
    for x, y, w, h in cells:
        key = _cell_key(x, y, w, h)
        orientation = orientations.get(key, "Any")
        result.append((x, y, w, h, orientation, key))
    return result


def find_cell_at_point(cells, px, py):
    """Find the cell containing a pixel coordinate.

    Args:
        cells: List of (x, y, w, h) tuples (or tuples with extra fields).
        px: X pixel coordinate.
        py: Y pixel coordinate.

    Returns:
        The matching tuple, or None.
    """
    for cell in cells:
        x, y, w, h = cell[0], cell[1], cell[2], cell[3]
        if x <= px < x + w and y <= py < y + h:
            return cell
    return None


def find_anchors(lines, px, py, axis, img_width, img_height):
    """Find anchor extent for a partial line at the cursor position.

    Looks for the two nearest perpendicular lines (or image edges)
    that bracket the cursor along the line's extent direction.

    Args:
        lines: List of line dicts.
        px: Cursor X in pixels.
        py: Cursor Y in pixels.
        axis: "v" or "h" - the axis of the line being added.
        img_width: Image width.
        img_height: Image height.

    Returns:
        Tuple (start, end) for the partial line's extent.
    """
    if axis == "v":
        # Vertical line: find horizontal boundaries active at x=px
        boundaries = [0, img_height]
        for line in lines:
            if line["axis"] == "h":
                if line["start"] <= px <= line["end"]:
                    boundaries.append(line["pos"])
        boundaries = sorted(set(boundaries))

        for i in range(len(boundaries) - 1):
            if boundaries[i] <= py < boundaries[i + 1]:
                return (boundaries[i], boundaries[i + 1])
        return (0, img_height)
    else:
        # Horizontal line: find vertical boundaries active at y=py
        boundaries = [0, img_width]
        for line in lines:
            if line["axis"] == "v":
                if line["start"] <= py <= line["end"]:
                    boundaries.append(line["pos"])
        boundaries = sorted(set(boundaries))

        for i in range(len(boundaries) - 1):
            if boundaries[i] <= px < boundaries[i + 1]:
                return (boundaries[i], boundaries[i + 1])
        return (0, img_width)


def get_line_move_range(lines, line_index, img_width, img_height):
    """Get the valid movement range for a line.

    The line can move between the nearest parallel lines whose extents
    overlap, ensuring at least 1px of cell width on each side.

    Args:
        lines: List of line dicts.
        line_index: Index of the line to move.
        img_width: Image width.
        img_height: Image height.

    Returns:
        Tuple (min_pos, max_pos) inclusive.
    """
    line = lines[line_index]
    axis = line["axis"]
    pos = line["pos"]
    start = line["start"]
    end = line["end"]

    if axis == "v":
        range_min = 1
        range_max = img_width - 1
    else:
        range_min = 1
        range_max = img_height - 1

    for i, other in enumerate(lines):
        if i == line_index or other["axis"] != axis:
            continue
        # Check extent overlap
        if other["start"] < end and start < other["end"]:
            if other["pos"] < pos:
                range_min = max(range_min, other["pos"] + 1)
            elif other["pos"] > pos:
                range_max = min(range_max, other["pos"] - 1)

    return (range_min, range_max)


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

def _rebuild_orientations(old_cells, old_orientations, new_cells):
    """Rebuild cell orientations after a line change.

    For each new cell, finds the old cell that contained its center
    and inherits that cell's orientation.

    Args:
        old_cells: List of (x, y, w, h) from before the change.
        old_orientations: Dict mapping cell keys to orientation strings.
        new_cells: List of (x, y, w, h) from after the change.

    Returns:
        New orientations dict.
    """
    new_orientations = {}
    for nx, ny, nw, nh in new_cells:
        cx = nx + nw / 2
        cy = ny + nh / 2
        # Find old cell containing this center
        inherited = "Any"
        for ox, oy, ow, oh in old_cells:
            if ox <= cx < ox + ow and oy <= cy < oy + oh:
                old_key = _cell_key(ox, oy, ow, oh)
                inherited = old_orientations.get(old_key, "Any")
                break
        new_orientations[_cell_key(nx, ny, nw, nh)] = inherited
    return new_orientations


def cycle_cell_orientation(texture_name, cell_key):
    """Cycle to the next orientation type for a cell.

    Args:
        texture_name: Name of the texture.
        cell_key: String key identifying the cell.

    Returns:
        The new orientation type string, or None if failed.
    """
    data = load_hotspots()
    tex_data = data.get("textures", {}).get(texture_name)
    if tex_data is None:
        return None

    orientations = tex_data.get("cell_orientations", {})
    current = orientations.get(cell_key, "Any")
    current_index = (ORIENTATION_TYPES.index(current)
                     if current in ORIENTATION_TYPES else 0)
    next_index = (current_index + 1) % len(ORIENTATION_TYPES)
    next_type = ORIENTATION_TYPES[next_index]

    orientations[cell_key] = next_type
    tex_data["cell_orientations"] = orientations
    save_hotspots(data)
    return next_type


# ---------------------------------------------------------------------------
# Line mutations
# ---------------------------------------------------------------------------

def add_line(texture_name, axis, pos, start, end):
    """Add a bisecting line to a texture.

    Rebuilds cell orientations so new cells inherit from the cell they
    were split from.

    Args:
        texture_name: Name of the texture.
        axis: "v" or "h".
        pos: Absolute pixel position.
        start: Extent start (pixels along perpendicular axis).
        end: Extent end.

    Returns:
        True if successful, False otherwise.
    """
    data = load_hotspots()
    tex_data = data.get("textures", {}).get(texture_name)
    if tex_data is None:
        return False

    lines = tex_data.get("lines", [])
    img_w = tex_data.get("image_width", 0)
    img_h = tex_data.get("image_height", 0)
    old_orientations = tex_data.get("cell_orientations", {})

    old_cells = derive_cells(lines, img_w, img_h)

    lines.append({"axis": axis, "pos": pos, "start": start, "end": end})
    tex_data["lines"] = lines

    new_cells = derive_cells(lines, img_w, img_h)
    tex_data["cell_orientations"] = _rebuild_orientations(
        old_cells, old_orientations, new_cells
    )

    return save_hotspots(data)


def remove_line(texture_name, line_index):
    """Remove a bisecting line, merging adjacent cells.

    The merged cell inherits the orientation of the first (top-left) cell.

    Args:
        texture_name: Name of the texture.
        line_index: Index into the lines list.

    Returns:
        True if successful, False otherwise.
    """
    data = load_hotspots()
    tex_data = data.get("textures", {}).get(texture_name)
    if tex_data is None:
        return False

    lines = tex_data.get("lines", [])
    if line_index < 0 or line_index >= len(lines):
        return False

    img_w = tex_data.get("image_width", 0)
    img_h = tex_data.get("image_height", 0)
    old_orientations = tex_data.get("cell_orientations", {})

    old_cells = derive_cells(lines, img_w, img_h)

    lines.pop(line_index)
    tex_data["lines"] = lines

    new_cells = derive_cells(lines, img_w, img_h)
    tex_data["cell_orientations"] = _rebuild_orientations(
        old_cells, old_orientations, new_cells
    )

    return save_hotspots(data)


def move_line(texture_name, line_index, new_pos):
    """Move a bisecting line to a new position.

    Rebuilds cell orientations.

    Args:
        texture_name: Name of the texture.
        line_index: Index into the lines list.
        new_pos: New absolute pixel position.

    Returns:
        True if successful, False otherwise.
    """
    data = load_hotspots()
    tex_data = data.get("textures", {}).get(texture_name)
    if tex_data is None:
        return False

    lines = tex_data.get("lines", [])
    if line_index < 0 or line_index >= len(lines):
        return False

    img_w = tex_data.get("image_width", 0)
    img_h = tex_data.get("image_height", 0)
    old_orientations = tex_data.get("cell_orientations", {})

    old_cells = derive_cells(lines, img_w, img_h)

    old_pos = lines[line_index]["pos"]
    moved_axis = lines[line_index]["axis"]
    lines[line_index]["pos"] = new_pos

    # Update perpendicular partial lines anchored to the moved line.
    # A partial line's start or end may reference the moved line's old pos.
    if old_pos != new_pos:
        for i, other in enumerate(lines):
            if i == line_index or other["axis"] == moved_axis:
                continue
            if other["start"] == old_pos:
                other["start"] = new_pos
            if other["end"] == old_pos:
                other["end"] = new_pos
            # Ensure start < end after update
            if other["start"] > other["end"]:
                other["start"], other["end"] = other["end"], other["start"]

    tex_data["lines"] = lines

    new_cells = derive_cells(lines, img_w, img_h)
    tex_data["cell_orientations"] = _rebuild_orientations(
        old_cells, old_orientations, new_cells
    )

    return save_hotspots(data)
