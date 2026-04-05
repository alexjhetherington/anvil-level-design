import bpy
import bmesh
import math
import random
from bpy.types import Operator

from ..core.logging import debug_log, DEBUG_KEEP_HOTSPOT_SEAMS
from ..core.materials import get_image_from_material
from ..core.uv_layers import get_render_active_uv_layer
from ..core.hotspot_queries import face_has_hotspot_material
from mathutils import Vector

from .topology import get_quad_islands

# Angle threshold for floor/ceiling classification (10 degrees from vertical)
FLOOR_CEILING_ANGLE_THRESHOLD = math.radians(10)

# Square aspect ratio tolerance (within 10% of 1.0)
SQUARE_ASPECT_TOLERANCE = 0.1

from ..handlers import cache_face_data, update_ui_from_selection
from ..hotspot_mapping.json_storage import (
    is_texture_hotspottable,
    get_texture_hotspots,
    get_texture_dimensions,
    load_hotspots,
)


def classify_face_type(face):
    """Classify a face as 'floor', 'ceiling', or 'wall' based on its normal.

    - Floor: normal pointing up (within 10 degrees of +Z)
    - Ceiling: normal pointing down (within 10 degrees of -Z)
    - Wall: anything else

    Args:
        face: BMFace with a normal vector

    Returns:
        'floor', 'ceiling', or 'wall'
    """
    normal = face.normal
    up = Vector((0, 0, 1))
    down = Vector((0, 0, -1))

    # Check angle to up
    angle_to_up = normal.angle(up)
    if angle_to_up <= FLOOR_CEILING_ANGLE_THRESHOLD:
        return 'floor'

    # Check angle to down
    angle_to_down = normal.angle(down)
    if angle_to_down <= FLOOR_CEILING_ANGLE_THRESHOLD:
        return 'ceiling'

    return 'wall'


def is_roughly_square(aspect_ratio):
    """Check if an aspect ratio is roughly square (within tolerance)."""
    return (1.0 - SQUARE_ASPECT_TOLERANCE) <= aspect_ratio <= (1.0 + SQUARE_ASPECT_TOLERANCE)


def compute_hotspot_rotation(island, uv_layer, island_aspect, hotspot):
    """Determine the UV rotation for applying a hotspot to an island.

    - Upwards on walls: compute_upward_rotation for texture-top-up
    - Upwards on non-walls: 0 degrees
    - Otherwise: pick rotation that best matches aspect ratio

    Args:
        island: List of BMFaces in the island
        uv_layer: UV layer
        island_aspect: Aspect ratio (width/height) of the island
        hotspot: Hotspot dict with width, height, orientation_type

    Returns:
        Rotation in degrees (0, 90, 180, or 270)
    """
    orientation = hotspot.get('orientation_type', 'Any')
    face_type = classify_face_type(island[0])
    hs_w = hotspot.get('width', 1)
    hs_h = hotspot.get('height', 1)
    hs_aspect = hs_w / hs_h if hs_h > 0 else 1.0

    if orientation == 'Upwards' and face_type == 'wall':
        return compute_upward_rotation(island, uv_layer)
    elif orientation == 'Upwards':
        return 0

    if is_roughly_square(island_aspect) and is_roughly_square(hs_aspect):
        return random.choice([0, 90, 180, 270])

    # Determine whether a 90-degree rotation better matches the aspect ratio
    if island_aspect > 0 and hs_aspect > 0:
        aspect_score_normal = abs(math.log(island_aspect / hs_aspect))
    else:
        aspect_score_normal = float('inf')

    rotated_aspect = 1.0 / island_aspect if island_aspect > 0.0001 else 1.0
    if rotated_aspect > 0 and hs_aspect > 0:
        aspect_score_rotated = abs(math.log(rotated_aspect / hs_aspect))
    else:
        aspect_score_rotated = float('inf')

    if aspect_score_rotated < aspect_score_normal:
        return random.choice([90, 270])
    else:
        return random.choice([0, 180])


def compute_upward_rotation(island, uv_layer):
    """Compute the UV rotation needed to make texture top point upward in world space.

    Examines all 4 possible rotations (0, 90, 180, 270) and returns the one
    where the texture's up direction is most aligned with world +Z.

    Args:
        island: List of faces in the island
        uv_layer: UV layer

    Returns:
        Rotation in degrees (0, 90, 180, or 270)
    """
    # Find the UV bounding box
    min_u = float('inf')
    max_u = float('-inf')
    min_v = float('inf')
    max_v = float('-inf')

    for face in island:
        for loop in face.loops:
            u, v = loop[uv_layer].uv.x, loop[uv_layer].uv.y
            min_u = min(min_u, u)
            max_u = max(max_u, u)
            min_v = min(min_v, v)
            max_v = max(max_v, v)

    uv_width = max_u - min_u
    uv_height = max_v - min_v

    if uv_width < 0.0001 or uv_height < 0.0001:
        return 0

    # Find representative vertices at each corner of the UV bounding box
    threshold = 0.05  # 5% from corner

    bottom_left_verts = []
    top_left_verts = []
    bottom_right_verts = []
    top_right_verts = []

    for face in island:
        for loop in face.loops:
            u, v = loop[uv_layer].uv.x, loop[uv_layer].uv.y
            norm_u = (u - min_u) / uv_width
            norm_v = (v - min_v) / uv_height

            if norm_u < threshold and norm_v < threshold:
                bottom_left_verts.append(loop.vert.co.copy())
            if norm_u < threshold and norm_v > 1 - threshold:
                top_left_verts.append(loop.vert.co.copy())
            if norm_u > 1 - threshold and norm_v < threshold:
                bottom_right_verts.append(loop.vert.co.copy())
            if norm_u > 1 - threshold and norm_v > 1 - threshold:
                top_right_verts.append(loop.vert.co.copy())

    def average_position(verts):
        if not verts:
            return None
        avg = Vector((0, 0, 0))
        for v in verts:
            avg += v
        return avg / len(verts)

    bl = average_position(bottom_left_verts)
    tl = average_position(top_left_verts)
    br = average_position(bottom_right_verts)
    tr = average_position(top_right_verts)

    if bl is None or tl is None or br is None or tr is None:
        debug_log("[compute_upward_rotation] Could not find all corners")
        return 0

    # Compute the 3D "up" direction for each rotation
    # rotation 0: V+ direction (bottom to top)
    # rotation 90: what was U- becomes V+ (CCW rotation)
    # rotation 180: V- direction (top to bottom)
    # rotation 270: what was U+ becomes V+

    up_0 = ((tl - bl) + (tr - br)) / 2  # Average of left and right vertical edges
    up_90 = ((bl - br) + (tl - tr)) / 2  # Average of bottom and top horizontal edges, negated
    up_180 = -up_0
    up_270 = -up_90

    ups = [(0, up_0), (90, up_90), (180, up_180), (270, up_270)]

    # Pick the rotation with the highest Z component (most upward)
    best_rotation = 0
    best_z = float('-inf')

    for rotation, up in ups:
        if up.length < 0.0001:
            continue
        up_normalized = up.normalized()
        if up_normalized.z > best_z:
            best_z = up_normalized.z
            best_rotation = rotation

    debug_log(f"[compute_upward_rotation] best_rotation={best_rotation}, best_z={best_z:.3f}")
    return best_rotation


def find_best_hotspot(island_aspect, hotspots, image_width, image_height, face_type, island, uv_layer,
                      island_world_area, pixels_per_meter, size_weight):
    """Find the best matching hotspot using weighted aspect ratio and size scoring.

    Filters hotspots by orientation type based on face type (floor/ceiling/wall).
    For Any/Floor/Ceiling, allows aspect flip and random rotation.
    For Upwards (walls only), no aspect flip and texture top must point up.

    Uses a weighted combination of aspect ratio match and size match:
    - aspect_score: log-ratio difference between island and hotspot aspect ratios
    - size_score: log-ratio difference between target pixel area and hotspot area
    - combined_score = (1 - size_weight) * aspect_score + size_weight * size_score

    Args:
        island_aspect: Aspect ratio (width/height) of the island's UVs
        hotspots: List of hotspot dicts from json_storage
        image_width: Texture width in pixels
        image_height: Texture height in pixels
        face_type: 'floor', 'ceiling', or 'wall'
        island: List of faces in the island (for upward rotation calculation)
        uv_layer: UV layer (for upward rotation calculation)
        island_world_area: World-space area of the island in Blender units squared
        pixels_per_meter: Target pixels per meter (used to calculate ideal hotspot size)
        size_weight: Balance between aspect ratio (0) and size (1) matching

    Returns:
        Tuple of (hotspot_dict, rotation_degrees) or (None, 0) if no valid hotspots
    """
    if not hotspots:
        return None, 0

    # Filter hotspots by orientation and face type
    valid_hotspots = []
    for hotspot in hotspots:
        orientation = hotspot.get('orientation_type', 'Any')

        if orientation == 'Any':
            valid_hotspots.append(hotspot)
        elif orientation == 'Floor' and face_type == 'floor':
            valid_hotspots.append(hotspot)
        elif orientation == 'Ceiling' and face_type == 'ceiling':
            valid_hotspots.append(hotspot)
        elif orientation == 'Upwards' and face_type == 'wall':
            valid_hotspots.append(hotspot)

    if not valid_hotspots:
        debug_log(f"[find_best_hotspot] No valid hotspots for face_type={face_type}")
        return None, 0

    # Calculate target pixel area based on world area and pixels per meter
    # target_area = world_area * (pixels_per_meter)^2
    target_pixel_area = island_world_area * (pixels_per_meter ** 2)

    # Pre-compute the upward rotation for 'Upwards' hotspots
    # This determines whether the aspect ratio will be flipped (90°/270°) or not (0°/180°)
    upward_rotation = compute_upward_rotation(island, uv_layer)
    upward_flips_aspect = upward_rotation in (90, 270)

    # Score all candidates: list of (score, hotspot, needs_90_rotation)
    candidates = []

    # Pre-compute rotated aspect ratio
    rotated_island_aspect = 1.0 / island_aspect if island_aspect > 0.0001 else 1.0

    for hotspot in valid_hotspots:
        hs_width = hotspot.get('width', 1)
        hs_height = hotspot.get('height', 1)

        if hs_width <= 0 or hs_height <= 0:
            continue

        hs_aspect = hs_width / hs_height
        hs_area = hs_width * hs_height
        orientation = hotspot.get('orientation_type', 'Any')

        # Calculate size score (same for both orientations)
        if target_pixel_area > 0 and hs_area > 0:
            size_score = abs(math.log(target_pixel_area / hs_area))
        else:
            size_score = 0.0

        # Calculate aspect scores for both orientations
        if island_aspect > 0 and hs_aspect > 0:
            aspect_score_normal = abs(math.log(island_aspect / hs_aspect))
        else:
            aspect_score_normal = float('inf')

        if rotated_island_aspect > 0 and hs_aspect > 0:
            aspect_score_rotated = abs(math.log(rotated_island_aspect / hs_aspect))
        else:
            aspect_score_rotated = float('inf')

        if orientation == 'Upwards':
            # For 'Upwards', only score the orientation that will actually be applied
            # based on the pre-computed upward rotation
            if upward_flips_aspect:
                combined = (1.0 - size_weight) * aspect_score_rotated + size_weight * size_score
                candidates.append((combined, hotspot, True))
            else:
                combined = (1.0 - size_weight) * aspect_score_normal + size_weight * size_score
                candidates.append((combined, hotspot, False))
        else:
            # For non-Upwards, check both orientations and add both as candidates
            combined_normal = (1.0 - size_weight) * aspect_score_normal + size_weight * size_score
            candidates.append((combined_normal, hotspot, False))

            combined_rotated = (1.0 - size_weight) * aspect_score_rotated + size_weight * size_score
            candidates.append((combined_rotated, hotspot, True))

    if not candidates:
        return None, 0

    # Find best score and collect all candidates with that score
    best_score = min(c[0] for c in candidates)
    best_candidates = [(h, r) for (s, h, r) in candidates if s == best_score]

    # Randomly choose among tied candidates
    best_hotspot, best_needs_90_rotation = random.choice(best_candidates)

    rotation = compute_hotspot_rotation(island, uv_layer, island_aspect, best_hotspot)

    debug_log(f"[find_best_hotspot] island_aspect={island_aspect:.3f}, face_type={face_type}, "
              f"orientation={best_hotspot.get('orientation_type', 'Any')}, rotation={rotation}, "
              f"score={best_score:.3f}, size_weight={size_weight:.2f}")
    return best_hotspot, rotation


def apply_hotspot_uvs(island, uv_layer, hotspot, image_width, image_height, rotation_degrees):
    """Apply hotspot UV coordinates to an island.

    Scales and positions the island's UVs to fit within the hotspot region.
    Applies rotation as specified (0, 90, 180, or 270 degrees).

    Args:
        island: List of faces in the island
        uv_layer: UV layer to modify
        hotspot: Hotspot dict with x, y, width, height in pixels
        image_width: Texture width in pixels
        image_height: Texture height in pixels
        rotation_degrees: Rotation to apply (0, 90, 180, or 270)
    """
    # Get hotspot bounds in UV space (0-1)
    hs_u_min = hotspot['x'] / image_width
    hs_v_min = 1.0 - (hotspot['y'] + hotspot['height']) / image_height  # Flip Y
    hs_u_max = (hotspot['x'] + hotspot['width']) / image_width
    hs_v_max = 1.0 - hotspot['y'] / image_height  # Flip Y

    hs_width = hs_u_max - hs_u_min
    hs_height = hs_v_max - hs_v_min

    # Get current UV bounds of island
    all_uvs = []
    for face in island:
        for loop in face.loops:
            all_uvs.append(loop[uv_layer].uv)

    min_u = min(uv.x for uv in all_uvs)
    max_u = max(uv.x for uv in all_uvs)
    min_v = min(uv.y for uv in all_uvs)
    max_v = max(uv.y for uv in all_uvs)

    island_width = max_u - min_u
    island_height = max_v - min_v

    if island_width < 0.0001 or island_height < 0.0001:
        return

    # Apply transformation to each UV
    for face in island:
        for loop in face.loops:
            uv = loop[uv_layer].uv

            # Normalize to 0-1 within island bounds
            norm_u = (uv.x - min_u) / island_width
            norm_v = (uv.y - min_v) / island_height

            # Apply rotation
            if rotation_degrees == 90:
                # Rotate 90 degrees CCW: (u, v) -> (v, 1-u)
                new_norm_u = norm_v
                new_norm_v = 1.0 - norm_u
                norm_u = new_norm_u
                norm_v = new_norm_v
            elif rotation_degrees == 180:
                # Rotate 180 degrees: (u, v) -> (1-u, 1-v)
                norm_u = 1.0 - norm_u
                norm_v = 1.0 - norm_v
            elif rotation_degrees == 270:
                # Rotate 270 degrees CCW (90 CW): (u, v) -> (1-v, u)
                new_norm_u = 1.0 - norm_v
                new_norm_v = norm_u
                norm_u = new_norm_u
                norm_v = new_norm_v
            # rotation_degrees == 0: no change

            # Scale to hotspot size and offset to hotspot position
            uv.x = hs_u_min + norm_u * hs_width
            uv.y = hs_v_min + norm_v * hs_height


def set_island_uvs_to_origin(island, uv_layer):
    """Set all UV coordinates in an island to (0, 0).

    Used when no valid hotspot can be applied to a face.

    Args:
        island: List of faces in the island
        uv_layer: UV layer to modify
    """
    for face in island:
        for loop in face.loops:
            loop[uv_layer].uv.x = 0.0
            loop[uv_layer].uv.y = 0.0


def get_uv_islands(bm, faces, uv_layer):
    """Get UV islands from a set of faces, categorized by geometry type and size.

    Flood-fills UV connectivity directly via BMesh (no operator calls).
    Two faces are in the same island if they share an edge and their UV
    coordinates match at that edge in both faces' loops.

    Args:
        bm: BMesh instance
        faces: List of faces to find islands within
        uv_layer: BMesh UV layer

    Returns:
        Tuple of:
            - multi_quad_islands: List of islands with multiple quad faces
            - single_quad_islands: List of islands with exactly one quad face
            - ngon_islands: List of islands containing any non-quad faces
    """
    if not faces:
        return [], [], []

    UV_EPSILON = 1e-5
    face_set = set(faces)
    visited = set()

    # Build edge -> face lookup restricted to our face set
    edge_faces = {}
    for face in faces:
        for edge in face.edges:
            edge_faces.setdefault(edge.index, []).append(face)

    all_islands = []

    for face in faces:
        if face in visited:
            continue

        island = []
        stack = [face]
        visited.add(face)

        while stack:
            current = stack.pop()
            island.append(current)

            for edge in current.edges:
                neighbors = edge_faces.get(edge.index)
                if not neighbors:
                    continue

                for neighbor in neighbors:
                    if neighbor in visited:
                        continue

                    # Check UV continuity across the shared edge
                    # Get UV coords at edge verts for both faces
                    v0, v1 = edge.verts
                    cur_uv0 = cur_uv1 = None
                    for loop in current.loops:
                        if loop.vert == v0:
                            cur_uv0 = loop[uv_layer].uv
                        elif loop.vert == v1:
                            cur_uv1 = loop[uv_layer].uv

                    nb_uv0 = nb_uv1 = None
                    for loop in neighbor.loops:
                        if loop.vert == v0:
                            nb_uv0 = loop[uv_layer].uv
                        elif loop.vert == v1:
                            nb_uv1 = loop[uv_layer].uv

                    if cur_uv0 is None or cur_uv1 is None or nb_uv0 is None or nb_uv1 is None:
                        continue

                    if (abs(cur_uv0.x - nb_uv0.x) < UV_EPSILON and abs(cur_uv0.y - nb_uv0.y) < UV_EPSILON
                            and abs(cur_uv1.x - nb_uv1.x) < UV_EPSILON and abs(cur_uv1.y - nb_uv1.y) < UV_EPSILON):
                        visited.add(neighbor)
                        stack.append(neighbor)

        all_islands.append(island)

    # Categorize islands
    multi_quad_islands = []
    single_quad_islands = []
    ngon_islands = []

    for island in all_islands:
        all_quads = all(len(f.loops) == 4 for f in island)
        if not all_quads:
            ngon_islands.append(island)
        elif len(island) == 1:
            single_quad_islands.append(island)
        else:
            multi_quad_islands.append(island)

    debug_log(f"[get_uv_islands] Found {len(multi_quad_islands)} multi-quad, {len(single_quad_islands)} single-quad, {len(ngon_islands)} ngon islands")
    return multi_quad_islands, single_quad_islands, ngon_islands


def make_single_quad_into_rectangle(bm, island, uv_layer):
    """Set UVs for a single quad face based on its 3D edge lengths.

    This always succeeds for a valid single-quad island.

    Args:
        bm: BMesh instance
        island: List containing exactly one quad face
        uv_layer: UV layer to operate on

    Returns:
        Dict with keys:
            'aspect_ratio': float - aspect ratio (width/height) of the result
    """
    face = island[0]
    loops = list(face.loops)

    # Calculate edge lengths
    edge1_len = (loops[0].vert.co - loops[1].vert.co).length
    edge2_len = (loops[1].vert.co - loops[2].vert.co).length

    # Set UVs directly
    loops[0][uv_layer].uv = Vector((0, 0))
    loops[1][uv_layer].uv = Vector((edge1_len, 0))
    loops[2][uv_layer].uv = Vector((edge1_len, edge2_len))
    loops[3][uv_layer].uv = Vector((0, edge2_len))

    aspect_ratio = edge1_len / edge2_len if edge2_len > 0.0001 else 1.0

    debug_log(f"[make_single_quad_into_rectangle] aspect={aspect_ratio:.3f}")

    return {
        'aspect_ratio': aspect_ratio,
    }


def _bfs_propagate_grid_uvs(faces, uv_layer, edge_faces):
    """Two-pass BFS grid UV generation with averaged row/column dimensions.

    Pass 1: BFS to assign integer grid coordinates (col, row) to each loop.
    Uses loops (not verts) so that seam-split vertices get independent grid
    coordinates on each side of the seam (e.g. a ring of quads cut by a seam).
    Pass 2: Measure 3D edge lengths per row/column, average them, then assign
    UVs from cumulative averaged dimensions.  This produces consistent results
    regardless of which face the BFS starts from.

    UVs are written in meters (not texture space).

    Args:
        faces: list of quad BMFaces forming the grid
        uv_layer: BMesh UV layer to write to
        edge_faces: dict of edge -> list of faces (adjacency within the grid)

    Returns:
        True if all loops were assigned grid positions, False otherwise.
    """
    first_face = faces[0]
    first_loops = list(first_face.loops)

    # --- Pass 1: BFS to assign grid coordinates per loop ---
    # Keyed by loop so the same BMVert can have different grid coords on
    # each side of a seam (different loops reference the same vert).
    loop_pos = {}
    loop_pos[first_loops[0]] = (0, 0)
    loop_pos[first_loops[1]] = (1, 0)
    loop_pos[first_loops[2]] = (1, 1)
    loop_pos[first_loops[3]] = (0, 1)

    # Helper: build vert -> loop lookup for a face
    def vert_to_loop(face):
        return {l.vert: l for l in face.loops}

    visited = {first_face}
    queue = [first_face]

    while queue:
        current = queue.pop(0)
        cur_vtl = vert_to_loop(current)

        for edge in current.edges:
            for neighbor in edge_faces.get(edge, []):
                if neighbor in visited:
                    continue

                sv0, sv1 = edge.verts
                cur_l0 = cur_vtl.get(sv0)
                cur_l1 = cur_vtl.get(sv1)
                if cur_l0 not in loop_pos or cur_l1 not in loop_pos:
                    continue

                p0 = loop_pos[cur_l0]
                p1 = loop_pos[cur_l1]
                shared_verts = {sv0, sv1}

                nb_vtl = vert_to_loop(neighbor)
                nb_loops = list(neighbor.loops)

                # Copy shared vert positions to neighbor's loops
                loop_pos[nb_vtl[sv0]] = p0
                loop_pos[nb_vtl[sv1]] = p1

                # Determine extension direction from the current face
                cur_loops_pos = [loop_pos[l] for l in current.loops]

                if p0[1] == p1[1]:
                    # Horizontal shared edge (same row)
                    shared_row = p0[1]
                    cur_other_row = next(r for (_, r) in cur_loops_pos if r != shared_row)
                    new_row = shared_row + (shared_row - cur_other_row)

                    for i, loop in enumerate(nb_loops):
                        if loop.vert not in shared_verts and loop not in loop_pos:
                            prev_v = nb_loops[(i - 1) % 4].vert
                            next_v = nb_loops[(i + 1) % 4].vert
                            adj_vert = prev_v if prev_v in shared_verts else next_v
                            loop_pos[loop] = (loop_pos[nb_vtl[adj_vert]][0], new_row)

                elif p0[0] == p1[0]:
                    # Vertical shared edge (same column)
                    shared_col = p0[0]
                    cur_other_col = next(c for (c, _) in cur_loops_pos if c != shared_col)
                    new_col = shared_col + (shared_col - cur_other_col)

                    for i, loop in enumerate(nb_loops):
                        if loop.vert not in shared_verts and loop not in loop_pos:
                            prev_v = nb_loops[(i - 1) % 4].vert
                            next_v = nb_loops[(i + 1) % 4].vert
                            adj_vert = prev_v if prev_v in shared_verts else next_v
                            loop_pos[loop] = (new_col, loop_pos[nb_vtl[adj_vert]][1])

                visited.add(neighbor)
                queue.append(neighbor)

    # Check all loops got grid positions (fails for non-grid topologies)
    for face in faces:
        for loop in face.loops:
            if loop not in loop_pos:
                return False

    # --- Pass 2: Measure, average, and assign UVs ---

    # Normalize grid coordinates to 0-based
    min_col = min(c for c, r in loop_pos.values())
    min_row = min(r for c, r in loop_pos.values())
    if min_col != 0 or min_row != 0:
        loop_pos = {l: (c - min_col, r - min_row) for l, (c, r) in loop_pos.items()}

    max_col = max(c for c, r in loop_pos.values())
    max_row = max(r for c, r in loop_pos.values())
    num_cols = max_col  # face columns (loop indices go 0..num_cols)
    num_rows = max_row

    # Collect width/height measurements from each face
    col_widths = [[] for _ in range(num_cols)]
    row_heights = [[] for _ in range(num_rows)]

    for face in faces:
        face_loops = list(face.loops)
        fc = min(loop_pos[l][0] for l in face_loops)
        fr = min(loop_pos[l][1] for l in face_loops)

        h_lengths = []
        v_lengths = []
        fl_vtl = vert_to_loop(face)
        for e in face.edges:
            ev0, ev1 = e.verts
            gp0 = loop_pos[fl_vtl[ev0]]
            gp1 = loop_pos[fl_vtl[ev1]]
            length = (ev0.co - ev1.co).length

            if gp0[1] == gp1[1]:
                h_lengths.append(length)  # horizontal edge -> column width
            elif gp0[0] == gp1[0]:
                v_lengths.append(length)  # vertical edge -> row height

        if h_lengths:
            col_widths[fc].append(sum(h_lengths) / len(h_lengths))
        if v_lengths:
            row_heights[fr].append(sum(v_lengths) / len(v_lengths))

    avg_col_w = [
        (sum(ws) / len(ws)) if ws else 1.0
        for ws in col_widths
    ]
    avg_row_h = [
        (sum(hs) / len(hs)) if hs else 1.0
        for hs in row_heights
    ]

    # Cumulative sums for vertex UV positions
    cum_x = [0.0] * (num_cols + 1)
    for c in range(num_cols):
        cum_x[c + 1] = cum_x[c] + avg_col_w[c]

    cum_y = [0.0] * (num_rows + 1)
    for r in range(num_rows):
        cum_y[r + 1] = cum_y[r] + avg_row_h[r]

    # Assign UVs
    for face in faces:
        for loop in face.loops:
            gc, gr = loop_pos[loop]
            loop[uv_layer].uv = Vector((cum_x[gc], cum_y[gr]))

    return True


def try_make_multi_quad_into_rectangle(bm, island, uv_layer):
    """Fit a multi-quad island's UVs into a rectangle by propagating UVs across shared edges.

    Starting from the first face (whose UVs are set from 3D edge lengths),
    BFS-walks to neighbors via shared edges and places each new face's UVs
    so the shared edge anchors it and the opposite edge uses 3D lengths.

    Args:
        bm: BMesh instance
        island: List of quad faces in the island (must have more than one face)
        uv_layer: UV layer to operate on

    Returns:
        Dict with keys:
            'success': bool - whether rectangle fitting succeeded
            'aspect_ratio': float - aspect ratio (width/height) of the result
        Or None if the island is invalid
    """
    if not island:
        return None

    debug_log(f"[try_make_multi_quad_into_rectangle] Processing island with {len(island)} faces")

    # Build edge -> faces lookup for this island only
    edge_faces = {}
    for face in island:
        for edge in face.edges:
            edge_faces.setdefault(edge, []).append(face)

    # Build a seam-filtered adjacency for BFS: exclude interior seam edges
    # so the grid walk doesn't wrap around (e.g. a ring of quads with a seam).
    bfs_edge_faces = {}
    for edge, faces_on_edge in edge_faces.items():
        if edge.seam and len(faces_on_edge) == 2:
            continue
        bfs_edge_faces[edge] = faces_on_edge

    # Run shared BFS propagation (using seam-filtered adjacency)
    if not _bfs_propagate_grid_uvs(island, uv_layer, bfs_edge_faces):
        debug_log(f"[try_make_multi_quad_into_rectangle] BFS grid failed (non-grid topology)")
        return {
            'success': False,
            'aspect_ratio': 0.0,
            'reason': 'not_rectangular',
        }

    # Calculate final aspect ratio from resulting UVs
    min_u = float('inf')
    max_u = float('-inf')
    min_v = float('inf')
    max_v = float('-inf')
    for face in island:
        for loop in face.loops:
            u, v = loop[uv_layer].uv
            if u < min_u: min_u = u
            if u > max_u: max_u = u
            if v < min_v: min_v = v
            if v > max_v: max_v = v

    width = max_u - min_u
    height = max_v - min_v

    if width < 0.0001 or height < 0.0001:
        return {
            'success': False,
            'aspect_ratio': 0.0,
            'reason': 'zero_size',
        }

    # Verify the faces form a rectangular grid. A rectangular grid of quads
    # has exactly 4 corner vertices (vertices touching exactly 1 face in the
    # island). Seam edges split a vertex in UV space, so a vertex on a seam
    # can count as multiple UV corners (e.g. a ring of quads with one seam
    # has 2 seam vertices each contributing 2 corners = 4 total).
    # For each vertex, collect which island faces touch it
    vert_faces = {}
    for face in island:
        for vert in face.verts:
            vert_faces.setdefault(vert, []).append(face)

    # Build set of seam edges that are interior to the island (both faces present)
    interior_seam_edges = set()
    for edge, faces_on_edge in edge_faces.items():
        if edge.seam and len(faces_on_edge) == 2:
            interior_seam_edges.add(edge)

    corner_count = 0
    for vert, faces_touching in vert_faces.items():
        # Group this vertex's faces by connectivity through non-seam edges.
        # Two faces are in the same group if they share an edge at this vertex
        # that is not a seam and is interior to the island.
        face_set = set(faces_touching)
        visited = set()
        for face in faces_touching:
            if face in visited:
                continue
            # BFS/flood from this face through non-seam edges at this vertex
            group = []
            queue = [face]
            visited.add(face)
            while queue:
                current = queue.pop()
                group.append(current)
                for edge in current.edges:
                    if vert not in edge.verts:
                        continue
                    if edge in interior_seam_edges:
                        continue
                    for neighbor in edge_faces.get(edge, []):
                        if neighbor in visited or neighbor not in face_set:
                            continue
                        visited.add(neighbor)
                        queue.append(neighbor)
            # A group with exactly 1 face means this is a UV corner
            if len(group) == 1:
                corner_count += 1

    if corner_count != 4:
        debug_log(f"[try_make_multi_quad_into_rectangle] Not rectangular: "
                  f"corner_count={corner_count}, interior_seams={len(interior_seam_edges)}")
        return {
            'success': False,
            'aspect_ratio': 0.0,
            'reason': 'not_rectangular',
        }

    aspect_ratio = width / height

    debug_log(f"[try_make_multi_quad_into_rectangle] Final aspect ratio: {aspect_ratio:.3f}")

    return {
        'success': True,
        'aspect_ratio': aspect_ratio,
    }


def apply_hotspots_to_mesh(bm, me, faces, allow_combined_faces, world_matrix, pixels_per_meter, size_weight, seam_angle, uv_layer=None, override_hotspot=None):
    import time
    t_total_start = time.perf_counter()

    # Store original user seams so we can restore them after hotspotting
    t0 = time.perf_counter()
    original_seams = set()
    for edge in bm.edges:
        if edge.seam:
            original_seams.add(edge.index)

    # Note: We no longer clear seams at the start - existing seams are respected
    # by group_quad_faces_by_angle_and_existing_seams() as group boundaries

    # Filter to faces with hotspottable materials, skipping fixed faces
    hotspottable_faces = []
    skipped_no_hotspot = 0

    fixed_layer = bm.faces.layers.int.get("anvil_fixed_hotspot")

    for face in faces:
        mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
        image = get_image_from_material(mat)

        if not image or not is_texture_hotspottable(image.name):
            skipped_no_hotspot += 1
            continue

        # Skip faces with the fixed hotspot flag (unless using override/choose)
        if override_hotspot is None and fixed_layer is not None and face[fixed_layer] != 0:
            continue

        hotspottable_faces.append(face)

    if not hotspottable_faces:
        return (0, skipped_no_hotspot, 0)

    # Get UV layer
    if uv_layer is None:
        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()
    debug_log(f"[Hotspot Perf] Setup & filter: {time.perf_counter() - t0:.4f}s ({len(hotspottable_faces)} hotspottable faces from {len(faces)} input)")

    # Phase 1: Group quad faces by angle and mark seams
    # This marks seams where face normals differ by > SEAM_ANGLE,
    # treats non-quads as blocking boundaries, and marks minimal cut graph seams
    t0 = time.perf_counter()
    debug_log(f"[Hotspot] Processing topology for {len(hotspottable_faces)} faces")
    quad_groups, non_quad_faces = get_quad_islands(bm, hotspottable_faces, seam_angle)
    debug_log(f"[Hotspot] Created {len(quad_groups)} quad groups, {len(non_quad_faces)} non-quad faces")
    debug_log(f"[Hotspot Perf] Phase 1 - Topology (get_quad_islands): {time.perf_counter() - t0:.4f}s ({len(quad_groups)} groups, {len(non_quad_faces)} non-quads)")

    # Phase 2: Select faces and unwrap with CONFORMAL
    t0 = time.perf_counter()
    for f in bm.faces:
        f.select = False
    for face in hotspottable_faces:
        face.select = True

    bpy.ops.uv.unwrap(method='CONFORMAL', margin=0.001)
    debug_log(f"[Hotspot Perf] Phase 2 - UV unwrap (CONFORMAL): {time.perf_counter() - t0:.4f}s")

    # Phase 3: Use quad groups from Phase 1 directly as islands.
    # (UV-connectivity detection would split groups at seam edges, since the
    # unwrap produces different UV coords on each side of a seam.)
    t0 = time.perf_counter()
    multi_quad_islands = [list(g) for g in quad_groups if len(g) > 1]
    single_quad_islands = [[f] for g in quad_groups if len(g) == 1 for f in g]
    ngon_islands = [[f] for f in non_quad_faces]
    debug_log(f"[Hotspot] Found {len(multi_quad_islands)} multi-quad, {len(single_quad_islands)} single-quad, {len(ngon_islands)} ngon islands")
    debug_log(f"[Hotspot Perf] Phase 3 - Island detection: {time.perf_counter() - t0:.4f}s ({len(multi_quad_islands)} multi-quad, {len(single_quad_islands)} single-quad, {len(ngon_islands)} ngon)")

    # If combined faces disabled (per-object setting), split all multi-quad islands into single faces
    # This makes the previous seam calculation redundant, so perhaps should be refactored
    if not allow_combined_faces:
        debug_log("[Hotspot] Combined faces disabled, splitting multi-quad islands")
        for island in multi_quad_islands:
            for face in island:
                single_quad_islands.append([face])
        multi_quad_islands = []

    # Phase 4a: Apply rectangle fitting to multi-quad islands (can fail)
    t0 = time.perf_counter()
    multi_quad_rectangled_islands = []
    multi_quad_not_rectangled_islands = []

    for i, island in enumerate(multi_quad_islands):
        result = try_make_multi_quad_into_rectangle(bm, island, uv_layer)

        if result is not None and result['success']:
            debug_log(f"[Hotspot] Multi-quad island {i}: SUCCESS - aspect={result['aspect_ratio']:.3f}")
            multi_quad_rectangled_islands.append((island, result))
        else:
            reason = result.get('reason', 'unknown') if result else 'invalid'
            debug_log(f"[Hotspot] Multi-quad island {i}: FAILED ({reason})")
            multi_quad_not_rectangled_islands.append(island)

    # Split failed multi-quad islands into individual quads
    for island in multi_quad_not_rectangled_islands:
        debug_log(f"[Hotspot] Splitting failed multi-quad into {len(island)} single quads")
        for face in island:
            single_quad_islands.append([face])
    debug_log(f"[Hotspot Perf] Phase 4a - Multi-quad rectangle fitting: {time.perf_counter() - t0:.4f}s ({len(multi_quad_rectangled_islands)} success, {len(multi_quad_not_rectangled_islands)} failed)")

    # Phase 4b: Apply rectangle fitting to single-quad islands (always succeeds)
    t0 = time.perf_counter()
    single_quad_rectangled_islands = []

    for i, island in enumerate(single_quad_islands):
        result = make_single_quad_into_rectangle(bm, island, uv_layer)
        debug_log(f"[Hotspot] Single-quad island {i}: aspect={result['aspect_ratio']:.3f}")
        single_quad_rectangled_islands.append((island, result))
    debug_log(f"[Hotspot Perf] Phase 4b - Single-quad rectangle fitting: {time.perf_counter() - t0:.4f}s ({len(single_quad_rectangled_islands)} islands)")

    # Log summary of categorized islands
    debug_log(f"[Hotspot] Island summary:")
    debug_log(f"  - multi_quad_rectangled_islands: {len(multi_quad_rectangled_islands)}")
    debug_log(f"  - single_quad_rectangled_islands: {len(single_quad_rectangled_islands)}")
    debug_log(f"  - ngon_islands: {len(ngon_islands)}")

    # Phase 5: Apply hotspot UVs to rectangled islands
    applied_count = 0
    no_match_count = 0

    # Calculate scale factor for world-space area conversion
    scale = world_matrix.to_scale()
    avg_scale = (abs(scale.x) + abs(scale.y) + abs(scale.z)) / 3
    area_scale_factor = avg_scale ** 2

    # Pre-load hotspot data once (avoids JSON parsing per-island)
    hotspot_data = load_hotspots()

    # Helper to apply hotspot to an island
    def apply_hotspot_to_island(island, aspect_ratio):
        nonlocal applied_count, no_match_count

        # Get texture from first face
        first_face = island[0]
        mat = me.materials[first_face.material_index] if first_face.material_index < len(me.materials) else None
        image = get_image_from_material(mat)

        if not image:
            debug_log(f"[Hotspot] No image found for island")
            return False

        texture_name = image.name
        hotspots = get_texture_hotspots(texture_name, hotspot_data)

        if not hotspots:
            debug_log(f"[Hotspot] No hotspots defined for texture: {texture_name}")
            return False

        image_width, image_height = get_texture_dimensions(texture_name, hotspot_data)
        if image_width <= 0 or image_height <= 0:
            # Fall back to actual image dimensions
            image_width = image.size[0]
            image_height = image.size[1]

        if image_width <= 0 or image_height <= 0:
            debug_log(f"[Hotspot] Invalid image dimensions for: {texture_name}")
            return False

        if override_hotspot is not None:
            # Use the caller-specified hotspot instead of finding best match
            rotation_degrees = compute_hotspot_rotation(
                island, uv_layer, aspect_ratio, override_hotspot
            )
            apply_hotspot_uvs(island, uv_layer, override_hotspot, image_width, image_height, rotation_degrees)
            applied_count += 1
            return True

        # Calculate island world-space area
        local_area = sum(f.calc_area() for f in island)
        island_world_area = local_area * area_scale_factor

        # Classify the face type based on the first face's normal
        face_type = classify_face_type(first_face)

        # Find best matching hotspot (considering orientation and size)
        best_hotspot, rotation_degrees = find_best_hotspot(
            aspect_ratio, hotspots, image_width, image_height,
            face_type, island, uv_layer,
            island_world_area, pixels_per_meter, size_weight
        )

        if best_hotspot is None:
            # No valid hotspot for this face type - set UVs to origin
            debug_log(f"[Hotspot] No valid hotspot for face_type={face_type}, setting UVs to origin")
            set_island_uvs_to_origin(island, uv_layer)
            no_match_count += 1
            return False

        # Apply the hotspot UVs
        apply_hotspot_uvs(island, uv_layer, best_hotspot, image_width, image_height, rotation_degrees)
        applied_count += 1

        debug_log(f"[Hotspot] Applied hotspot {best_hotspot.get('id')} (rotation={rotation_degrees})")
        return True

    # Apply to multi-quad rectangled islands
    t0 = time.perf_counter()
    for island, result in multi_quad_rectangled_islands:
        aspect_ratio = result.get('aspect_ratio', 1.0)
        apply_hotspot_to_island(island, aspect_ratio)

    # Apply to single-quad rectangled islands
    for island, result in single_quad_rectangled_islands:
        aspect_ratio = result.get('aspect_ratio', 1.0)
        apply_hotspot_to_island(island, aspect_ratio)
    debug_log(f"[Hotspot Perf] Phase 5 - Hotspot matching & UV apply (quads): {time.perf_counter() - t0:.4f}s ({applied_count} applied, {no_match_count} no match)")

    # Phase 6: Apply hotspot UVs to ngon islands (using CONFORMAL bounding box)
    t0 = time.perf_counter()
    ngon_applied_before = applied_count
    for island in ngon_islands:
        # Get UV bounding box aspect ratio from CONFORMAL unwrap
        all_uvs = []
        for face in island:
            for loop in face.loops:
                all_uvs.append(loop[uv_layer].uv)

        if not all_uvs:
            continue

        min_u = min(uv.x for uv in all_uvs)
        max_u = max(uv.x for uv in all_uvs)
        min_v = min(uv.y for uv in all_uvs)
        max_v = max(uv.y for uv in all_uvs)

        width = max_u - min_u
        height = max_v - min_v

        if width < 0.0001 or height < 0.0001:
            continue

        aspect_ratio = width / height
        apply_hotspot_to_island(island, aspect_ratio)

    debug_log(f"[Hotspot Perf] Phase 6 - Hotspot matching & UV apply (ngons): {time.perf_counter() - t0:.4f}s ({applied_count - ngon_applied_before} applied)")
    debug_log(f"[Hotspot] Applied hotspots to {applied_count} islands, {no_match_count} had no valid match")

    # Restore original user seams (clear scaffolding seams added during hotspotting)
    t0 = time.perf_counter()
    if not DEBUG_KEEP_HOTSPOT_SEAMS:
        for edge in bm.edges:
            edge.seam = False
        bm.edges.ensure_lookup_table()
        for edge_idx in original_seams:
            if edge_idx < len(bm.edges):
                bm.edges[edge_idx].seam = True
        debug_log(f"[Hotspot] Restored {len(original_seams)} user seams")
    else:
        debug_log("[Hotspot] DEBUG_KEEP_HOTSPOT_SEAMS: keeping all scaffolding seams")

    debug_log(f"[Hotspot Perf] Seam restore: {time.perf_counter() - t0:.4f}s")

    # Restore selection
    for face in hotspottable_faces:
        face.select = True
    bmesh.update_edit_mesh(me)

    debug_log(f"[Hotspot Perf] TOTAL: {time.perf_counter() - t_total_start:.4f}s")
    return (len(hotspottable_faces), skipped_no_hotspot, 0)


class LEVELDESIGN_OT_apply_specific_hotspot(Operator):
    """Apply a specific hotspot cell to selected faces"""
    bl_idname = "leveldesign.apply_specific_hotspot"
    bl_label = "Apply Specific Hotspot"
    bl_options = {'REGISTER', 'UNDO'}

    cell_key: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return (context.object and context.object.type == 'MESH'
                and context.mode == 'EDIT_MESH')

    def execute(self, context):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        # Ensure layers exist before collecting face references (creating layers invalidates refs)
        from ..core.face_id import get_face_id_layer, get_fixed_hotspot_layer, save_face_selection, restore_face_selection
        id_layer = get_face_id_layer(bm)
        fixed_layer = get_fixed_hotspot_layer(bm)

        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            self.report({'WARNING'}, "No faces selected")
            return {'CANCELLED'}

        # Parse cell key
        parts = self.cell_key.split('_')
        if len(parts) != 4:
            self.report({'ERROR'}, "Invalid cell key")
            return {'CANCELLED'}
        cell_x, cell_y, cell_w, cell_h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

        # Build the override hotspot dict with orientation from hotspot data
        hotspot_data = load_hotspots()
        orientation = 'Any'
        texture_name = None
        for face in selected_faces:
            mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
            image = get_image_from_material(mat)
            if image and is_texture_hotspottable(image.name):
                texture_name = image.name
                break

        if texture_name is None:
            self.report({'WARNING'}, "No hotspot faces in selection")
            return {'CANCELLED'}

        for hs in get_texture_hotspots(texture_name, hotspot_data):
            if hs['x'] == cell_x and hs['y'] == cell_y and hs['width'] == cell_w and hs['height'] == cell_h:
                orientation = hs.get('orientation_type', 'Any')
                break

        override = {
            'x': cell_x, 'y': cell_y,
            'width': cell_w, 'height': cell_h,
            'orientation_type': orientation,
        }

        # Save and restore selection since apply_hotspots_to_mesh modifies it
        selected_ids, active_id = save_face_selection(bm, id_layer)

        props = context.scene.level_design_props
        applied_count, skipped, _ = apply_hotspots_to_mesh(
            bm, me, selected_faces, obj.anvil_allow_combined_faces,
            obj.matrix_world, props.pixels_per_meter, obj.anvil_hotspot_size_weight,
            obj.anvil_hotspot_seam_angle,
            override_hotspot=override,
        )

        # Mark chosen faces as fixed
        if applied_count > 0:
            for face in selected_faces:
                face[fixed_layer] = 1

        restore_face_selection(bm, id_layer, selected_ids, active_id)
        bmesh.update_edit_mesh(me)
        cache_face_data(context)
        update_ui_from_selection(context)

        if applied_count > 0:
            self.report({'INFO'}, f"Applied hotspot to {applied_count} island(s)")
        else:
            self.report({'WARNING'}, "No hotspot faces in selection")
        return {'FINISHED'}


class LEVELDESIGN_OT_hotspot_palette(Operator):
    """Pick a specific hotspot cell by clicking on it in a visual atlas overlay"""
    bl_idname = "leveldesign.hotspot_palette"
    bl_label = "Hotspot Palette"

    @classmethod
    def poll(cls, context):
        if not (context.object and context.object.type == 'MESH' and context.mode == 'EDIT_MESH'):
            return False
        return True

    def invoke(self, context, event):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            self.report({'WARNING'}, "No faces selected")
            return {'CANCELLED'}

        # Get texture from first selected face
        face = selected_faces[0]
        mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
        image = get_image_from_material(mat)

        if not image or not is_texture_hotspottable(image.name):
            self.report({'WARNING'}, "Selected face has no hotspottable texture")
            return {'CANCELLED'}

        hotspot_data = load_hotspots()
        hotspots = get_texture_hotspots(image.name, hotspot_data)

        if not hotspots:
            self.report({'WARNING'}, "No hotspot cells defined for this texture")
            return {'CANCELLED'}

        self._hotspots = hotspots
        self._image = image
        self._img_w = image.size[0] if image.size[0] > 0 else 1
        self._img_h = image.size[1] if image.size[1] > 0 else 1
        self._hovered_cell = None
        self._gpu_texture = None
        self._draw_handler = None

        # Find a perspective 3D view, falling back to context.area
        self._area = context.area
        found = False
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D' and space.region_3d and not space.region_3d.is_orthographic_side_view:
                        self._area = area
                        found = True
                        break
                if found:
                    break

        # Compute overlay layout using the target area's region
        self._region = None
        for region in self._area.regions:
            if region.type == 'WINDOW':
                self._region = region
                break

        self._update_layout(context)

        # Register draw handler
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, (context,), 'WINDOW', 'POST_PIXEL'
        )

        context.window_manager.modal_handler_add(self)
        self._area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _update_layout(self, context):
        """Compute the screen-space rectangle for the atlas image."""
        region = self._region if self._region else context.region
        padding = 40

        avail_w = region.width - padding * 2
        avail_h = region.height - padding * 2

        img_aspect = self._img_w / self._img_h
        avail_aspect = avail_w / avail_h if avail_h > 0 else 1.0

        if img_aspect > avail_aspect:
            # Image is wider than available space
            self._draw_w = avail_w
            self._draw_h = avail_w / img_aspect
        else:
            # Image is taller than available space
            self._draw_h = avail_h
            self._draw_w = avail_h * img_aspect

        self._draw_x = (region.width - self._draw_w) / 2
        self._draw_y = (region.height - self._draw_h) / 2

    def _pixel_to_screen(self, px, py):
        """Convert image pixel coords (y-down) to screen coords (y-up)."""
        sx = self._draw_x + (px / self._img_w) * self._draw_w
        sy = self._draw_y + self._draw_h - (py / self._img_h) * self._draw_h
        return sx, sy

    def _screen_to_pixel(self, sx, sy):
        """Convert screen coords to image pixel coords (y-down)."""
        px = ((sx - self._draw_x) / self._draw_w) * self._img_w
        py = ((self._draw_y + self._draw_h - sy) / self._draw_h) * self._img_h
        return px, py

    def _hit_test(self, mx, my):
        """Return the hotspot dict under mouse, or None."""
        px, py = self._screen_to_pixel(mx, my)
        if px < 0 or py < 0 or px > self._img_w or py > self._img_h:
            return None
        for hs in self._hotspots:
            hx, hy, hw, hh = hs['x'], hs['y'], hs['width'], hs['height']
            if hx <= px <= hx + hw and hy <= py <= hy + hh:
                return hs
        return None

    def _ensure_gpu_texture(self):
        """Create GPU texture from image pixels."""
        if self._gpu_texture is not None:
            return
        import gpu
        image = self._image
        w, h = self._img_w, self._img_h
        pixels = image.pixels[:]
        buf = gpu.types.Buffer('FLOAT', w * h * 4, pixels)
        self._gpu_texture = gpu.types.GPUTexture((w, h), format='RGBA32F', data=buf)

    def _draw_callback(self, context):
        import gpu
        from gpu_extras.batch import batch_for_shader

        region = context.region

        # Draw darkened background
        gpu.state.blend_set('ALPHA')
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.uniform_float("color", (0.0, 0.0, 0.0, 0.7))
        bg_verts = [(0, 0), (region.width, 0), (region.width, region.height), (0, region.height)]
        bg_batch = batch_for_shader(shader, 'TRI_FAN', {"pos": bg_verts})
        bg_batch.draw(shader)

        # Draw the texture image
        self._ensure_gpu_texture()
        if self._gpu_texture is not None:
            shader_img = gpu.shader.from_builtin('IMAGE')
            shader_img.uniform_sampler("image", self._gpu_texture)

            x1, y1 = self._draw_x, self._draw_y
            x2, y2 = self._draw_x + self._draw_w, self._draw_y + self._draw_h

            img_verts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            img_uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
            img_batch = batch_for_shader(shader_img, 'TRI_FAN', {
                "pos": img_verts, "texCoord": img_uvs,
            })
            img_batch.draw(shader_img)

        # Draw cell outlines
        line_color = (1.0, 0.6, 0.0, 0.9)  # Orange
        hover_color = (1.0, 1.0, 1.0, 1.0)  # White
        hover_fill = (1.0, 1.0, 1.0, 0.15)  # Semi-transparent white

        for hs in self._hotspots:
            hx, hy, hw, hh = hs['x'], hs['y'], hs['width'], hs['height']
            sx1, sy1 = self._pixel_to_screen(hx, hy + hh)
            sx2, sy2 = self._pixel_to_screen(hx + hw, hy)

            is_hovered = (hs is self._hovered_cell)
            color = hover_color if is_hovered else line_color

            # Draw fill for hovered cell
            if is_hovered:
                shader_fill = gpu.shader.from_builtin('UNIFORM_COLOR')
                shader_fill.uniform_float("color", hover_fill)
                fill_verts = [(sx1, sy1), (sx2, sy1), (sx2, sy2), (sx1, sy2)]
                fill_batch = batch_for_shader(shader_fill, 'TRI_FAN', {"pos": fill_verts})
                fill_batch.draw(shader_fill)

            # Draw outline
            try:
                shader_line = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
                shader_line.uniform_float("viewportSize", (region.width, region.height))
                shader_line.uniform_float("lineWidth", 3.0 if is_hovered else 2.0)
                shader_line.uniform_float("color", color)
                outline_verts = [
                    (sx1, sy1), (sx2, sy1), (sx2, sy2), (sx1, sy2), (sx1, sy1)
                ]
                outline_batch = batch_for_shader(shader_line, 'LINE_STRIP', {"pos": outline_verts})
                outline_batch.draw(shader_line)
            except Exception:
                shader_line = gpu.shader.from_builtin('UNIFORM_COLOR')
                shader_line.uniform_float("color", color)
                outline_pairs = [
                    (sx1, sy1), (sx2, sy1),
                    (sx2, sy1), (sx2, sy2),
                    (sx2, sy2), (sx1, sy2),
                    (sx1, sy2), (sx1, sy1),
                ]
                outline_batch = batch_for_shader(shader_line, 'LINES', {"pos": outline_pairs})
                outline_batch.draw(shader_line)

            # Draw orientation icon
            import blf
            orientation = hs.get('orientation_type', 'Any')
            orientation_symbols = {
                'Any': '\u25CF',
                'Upwards': '\u2191',
                'Floor': '\u230A',
                'Ceiling': '\u2308',
            }
            symbol = orientation_symbols.get(orientation, '?')
            font_id = 0
            cx = (sx1 + sx2) / 2
            cy = (sy1 + sy2) / 2
            font_size = max(10, min(24, int((sy2 - sy1) * 0.3)))
            blf.size(font_id, font_size)
            blf.color(font_id, *color)
            tw, th = blf.dimensions(font_id, symbol)
            blf.position(font_id, cx - tw / 2, cy - th / 2, 0)
            blf.draw(font_id, symbol)

        gpu.state.blend_set('NONE')

    def _cleanup(self, context):
        if self._draw_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        self._gpu_texture = None
        self._area.tag_redraw()

    def _mouse_region_coords(self, event):
        """Convert absolute mouse coords to self._region-relative coords."""
        rgn = self._region
        return event.mouse_x - rgn.x, event.mouse_y - rgn.y

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            mx, my = self._mouse_region_coords(event)
            self._hovered_cell = self._hit_test(mx, my)
            self._area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            mx, my = self._mouse_region_coords(event)
            clicked = self._hit_test(mx, my)
            if clicked is not None:
                cell_key = f"{clicked['x']}_{clicked['y']}_{clicked['width']}_{clicked['height']}"
                self._cleanup(context)
                bpy.ops.leveldesign.apply_specific_hotspot(cell_key=cell_key)
                return {'FINISHED'}
            # Click outside image — cancel
            self._cleanup(context)
            return {'CANCELLED'}

        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            self._cleanup(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


class LEVELDESIGN_OT_apply_hotspot(Operator):
    """Apply hotspot mapping to selected faces based on aspect ratio"""
    bl_idname = "leveldesign.apply_hotspot"
    bl_label = "Apply Hotspot"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH' and context.mode in {'EDIT_MESH', 'OBJECT'}

    def execute(self, context):
        obj = context.object
        me = obj.data

        # Handle object mode: enter edit mode, operate on all faces, then return
        was_object_mode = context.mode == 'OBJECT'
        if was_object_mode:
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(me)

        # Store original selection to restore after operation (only relevant in edit mode)
        from ..core.face_id import get_face_id_layer, save_face_selection, restore_face_selection
        id_layer = get_face_id_layer(bm)
        selected_ids, active_id = save_face_selection(bm, id_layer)

        # In object mode, apply to all faces; in edit mode, use selection or all faces
        if was_object_mode:
            faces_to_process = list(bm.faces)
        else:
            selected_faces = [f for f in bm.faces if f.select]
            faces_to_process = selected_faces if selected_faces else list(bm.faces)

        props = context.scene.level_design_props
        allow_combined_faces = obj.anvil_allow_combined_faces
        size_weight = obj.anvil_hotspot_size_weight

        seam_angle = obj.anvil_hotspot_seam_angle
        applied_count, skipped_no_hotspot, skipped_not_quad = apply_hotspots_to_mesh(
            bm, me, faces_to_process, allow_combined_faces,
            obj.matrix_world, props.pixels_per_meter, size_weight, seam_angle
        )

        # Restore original face selection (only relevant if staying in edit mode)
        if not was_object_mode:
            restore_face_selection(bm, id_layer, selected_ids, active_id)

        bmesh.update_edit_mesh(me)

        # Update face cache
        cache_face_data(context)

        # Return to object mode if we started there
        if was_object_mode:
            bpy.ops.object.mode_set(mode='OBJECT')

        # Report results
        if applied_count > 0:
            msg = f"Applied hotspot to {applied_count} face(s)"
            if skipped_no_hotspot > 0 or skipped_not_quad > 0:
                msg += f" (skipped: {skipped_no_hotspot} no hotspot, {skipped_not_quad} non-quad)"
            self.report({'INFO'}, msg)
        else:
            self.report({'WARNING'}, "No faces could be hotspot mapped")

        return {'FINISHED'}


classes = (
    LEVELDESIGN_OT_apply_specific_hotspot,
    LEVELDESIGN_OT_hotspot_palette,
    LEVELDESIGN_OT_apply_hotspot,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
