import bpy
import bmesh
import math
import random
from bpy.types import Operator

from ..utils import is_level_design_workspace
from mathutils import Vector

from bpy_extras.view3d_utils import location_3d_to_region_2d

from ..utils import (
    align_2d_shape_to_square,
    derive_transform_from_uvs,
    face_aligned_project,
    get_texture_dimensions_from_material,
    get_selected_faces_or_report,
    get_image_from_material,
    get_render_active_uv_layer,
    debug_log,
    face_has_hotspot_material,
)
from .topology import get_quad_islands

# Angle threshold for seam detection in hotspot mapping (in radians)
SEAM_ANGLE = math.radians(30)

# Angle threshold for floor/ceiling classification (10 degrees from vertical)
FLOOR_CEILING_ANGLE_THRESHOLD = math.radians(10)

# Square aspect ratio tolerance (within 10% of 1.0)
SQUARE_ASPECT_TOLERANCE = 0.1

from ..handlers import cache_face_data, cache_single_face
from ..properties import set_updating_from_selection, sync_scale_tracking, apply_uv_to_face
from ..hotspot_mapping.json_storage import (
    is_texture_hotspottable,
    get_texture_hotspots,
    get_texture_dimensions,
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

    # Determine the rotation
    best_orientation = best_hotspot.get('orientation_type', 'Any')
    hs_aspect = best_hotspot.get('width', 1) / best_hotspot.get('height', 1)

    if best_orientation == 'Upwards':
        # Compute the specific rotation to make texture top point upward
        rotation = compute_upward_rotation(island, uv_layer)
    else:
        # Random rotation from valid options
        island_is_square = is_roughly_square(island_aspect)
        hotspot_is_square = is_roughly_square(hs_aspect)

        if island_is_square and hotspot_is_square:
            # All 4 rotations valid for square face + square hotspot
            rotation = random.choice([0, 90, 180, 270])
        elif best_needs_90_rotation:
            # Aspect flip needed: 90° or 270°
            rotation = random.choice([90, 270])
        else:
            # No aspect flip: 0° or 180°
            rotation = random.choice([0, 180])

    debug_log(f"[find_best_hotspot] island_aspect={island_aspect:.3f}, face_type={face_type}, "
              f"orientation={best_orientation}, rotation={rotation}, score={best_score:.3f}, "
              f"size_weight={size_weight:.2f}")
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


def get_uv_islands(bm, faces):
    """Get UV islands from a set of faces, categorized by geometry type and size.

    Uses select_linked with UV delimiter to find connected UV regions,
    then categorizes each island by face count and geometry type.

    Args:
        bm: BMesh instance
        faces: List of faces to find islands within

    Returns:
        Tuple of:
            - multi_quad_islands: List of islands with multiple quad faces
            - single_quad_islands: List of islands with exactly one quad face
            - ngon_islands: List of islands containing any non-quad faces
    """
    if not faces:
        return [], [], []

    # Deselect all faces first
    for f in bm.faces:
        f.select = False

    # Track which faces we still need to process
    remaining = set(faces)
    all_islands = []

    while remaining:
        # Select one unprocessed face
        start_face = next(iter(remaining))
        start_face.select = True

        # Expand selection to entire UV island
        bpy.ops.mesh.select_linked(delimit={'UV'})

        # Collect all selected faces that are in our original set
        island = []
        for f in bm.faces:
            if f.select and f in remaining:
                island.append(f)
                remaining.discard(f)
            f.select = False

        if island:
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

    # Select only this face
    for f in bm.faces:
        f.select = False
    face.select = True

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


def try_make_multi_quad_into_rectangle(bm, island, uv_layer):
    """Attempt to fit a multi-quad island's UVs into a rectangle using follow_active_quads.

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

    # Select only this island's faces
    for f in bm.faces:
        f.select = False
    for face in island:
        face.select = True

    # Set up the first quad with proper UVs based on its 3D proportions
    first_face = island[0]
    loops = list(first_face.loops)

    # Calculate edge lengths of first quad
    edge1_len = (loops[0].vert.co - loops[1].vert.co).length
    edge2_len = (loops[1].vert.co - loops[2].vert.co).length

    # Set UVs for first quad
    loops[0][uv_layer].uv = Vector((0, 0))
    loops[1][uv_layer].uv = Vector((edge1_len, 0))
    loops[2][uv_layer].uv = Vector((edge1_len, edge2_len))
    loops[3][uv_layer].uv = Vector((0, edge2_len))

    # Set as active face
    bm.faces.active = first_face

    # Run follow active quads
    bpy.ops.uv.follow_active_quads()

    # Calculate final aspect ratio from resulting UVs
    all_uvs = []
    for face in island:
        for loop in face.loops:
            all_uvs.append(loop[uv_layer].uv)

    min_u = min(uv.x for uv in all_uvs)
    max_u = max(uv.x for uv in all_uvs)
    min_v = min(uv.y for uv in all_uvs)
    max_v = max(uv.y for uv in all_uvs)

    width = max_u - min_u
    height = max_v - min_v

    if width < 0.0001 or height < 0.0001:
        return {
            'success': False,
            'aspect_ratio': 0.0,
            'reason': 'zero_size',
        }

    aspect_ratio = width / height

    debug_log(f"[try_make_multi_quad_into_rectangle] Final aspect ratio: {aspect_ratio:.3f}")

    return {
        'success': True,
        'aspect_ratio': aspect_ratio,
    }


def apply_hotspots_to_mesh(bm, me, faces, seam_mode, allow_combined_faces, world_matrix, pixels_per_meter, size_weight, seam_angle=None, uv_layer=None):
    # Store original seams if we need to restore them later
    original_seams = set()
    if seam_mode == 'MAINTAIN_USER':
        for edge in bm.edges:
            if edge.seam:
                original_seams.add(edge.index)

    # Note: We no longer clear seams at the start - existing seams are respected
    # by group_quad_faces_by_angle_and_existing_seams() as group boundaries

    # Filter to faces with hotspottable materials
    hotspottable_faces = []
    skipped_no_hotspot = 0

    for face in faces:
        mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
        image = get_image_from_material(mat)

        if not image or not is_texture_hotspottable(image.name):
            skipped_no_hotspot += 1
            continue

        hotspottable_faces.append(face)

    if not hotspottable_faces:
        return (0, skipped_no_hotspot, 0)

    # Get UV layer
    if uv_layer is None:
        from ..utils import get_render_active_uv_layer
        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

    # Phase 1: Group quad faces by angle and mark seams
    # This marks seams where face normals differ by > SEAM_ANGLE,
    # treats non-quads as blocking boundaries, and marks minimal cut graph seams
    debug_log(f"[Hotspot] Processing topology for {len(hotspottable_faces)} faces")
    quad_groups, non_quad_faces = get_quad_islands(bm, hotspottable_faces, SEAM_ANGLE)
    debug_log(f"[Hotspot] Created {len(quad_groups)} quad groups, {len(non_quad_faces)} non-quad faces")

    # Phase 2: Select faces and unwrap with CONFORMAL
    for f in bm.faces:
        f.select = False
    for face in hotspottable_faces:
        face.select = True

    bpy.ops.uv.unwrap(method='CONFORMAL', margin=0.001)

    # Phase 3: Detect UV islands and categorize by geometry type and size
    multi_quad_islands, single_quad_islands, ngon_islands = get_uv_islands(bm, hotspottable_faces)
    debug_log(f"[Hotspot] Found {len(multi_quad_islands)} multi-quad, {len(single_quad_islands)} single-quad, {len(ngon_islands)} ngon islands")

    # If combined faces disabled (per-object setting), split all multi-quad islands into single faces
    # This makes the previous seam calculation redundant, so perhaps should be refactored
    if not allow_combined_faces:
        debug_log("[Hotspot] Combined faces disabled, splitting multi-quad islands")
        for island in multi_quad_islands:
            for face in island:
                single_quad_islands.append([face])
        multi_quad_islands = []

    # Phase 4a: Apply rectangle fitting to multi-quad islands (can fail)
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

    # Phase 4b: Apply rectangle fitting to single-quad islands (always succeeds)
    single_quad_rectangled_islands = []

    for i, island in enumerate(single_quad_islands):
        result = make_single_quad_into_rectangle(bm, island, uv_layer)
        debug_log(f"[Hotspot] Single-quad island {i}: aspect={result['aspect_ratio']:.3f}")
        single_quad_rectangled_islands.append((island, result))

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
        hotspots = get_texture_hotspots(texture_name)

        if not hotspots:
            debug_log(f"[Hotspot] No hotspots defined for texture: {texture_name}")
            return False

        image_width, image_height = get_texture_dimensions(texture_name)
        if image_width <= 0 or image_height <= 0:
            # Fall back to actual image dimensions
            image_width = image.size[0]
            image_height = image.size[1]

        if image_width <= 0 or image_height <= 0:
            debug_log(f"[Hotspot] Invalid image dimensions for: {texture_name}")
            return False

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
    for island, result in multi_quad_rectangled_islands:
        aspect_ratio = result.get('aspect_ratio', 1.0)
        apply_hotspot_to_island(island, aspect_ratio)

    # Apply to single-quad rectangled islands
    for island, result in single_quad_rectangled_islands:
        aspect_ratio = result.get('aspect_ratio', 1.0)
        apply_hotspot_to_island(island, aspect_ratio)

    # Phase 6: Apply hotspot UVs to ngon islands (using CONFORMAL bounding box)
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

    debug_log(f"[Hotspot] Applied hotspots to {applied_count} islands, {no_match_count} had no valid match")

    # Handle seams based on seam_mode
    if seam_mode == 'MAINTAIN_USER':
        # Clear all seams first, then restore original user seams
        for edge in bm.edges:
            edge.seam = False
        bm.edges.ensure_lookup_table()
        for edge_idx in original_seams:
            if edge_idx < len(bm.edges):
                bm.edges[edge_idx].seam = True
        debug_log(f"[Hotspot] Restored {len(original_seams)} user seams")

    # Commented out because it's useful to see the islands to debug when combine faces is on but faces aren't combining
    #elif seam_mode == 'DISPLAY_ALL':
        # Keep existing seams and add seams around single quad islands
        #for island, result in single_quad_rectangled_islands:
        #    face = island[0]  # island is a list with one face for single quads
        #    for edge in face.edges:
        #        edge.seam = True
        #debug_log(f"[Hotspot] Added seams around {len(single_quad_rectangled_islands)} single quad islands")

    elif seam_mode == 'CLEAR_ALL':
        # Clear all seams on processed faces
        for face in hotspottable_faces:
            for edge in face.edges:
                edge.seam = False
        debug_log("[Hotspot] Cleared all seams")

    # Restore selection
    for face in hotspottable_faces:
        face.select = True
    bmesh.update_edit_mesh(me)

    return (len(hotspottable_faces), skipped_no_hotspot, 0)


class LEVELDESIGN_OT_face_aligned_project(Operator):
    """Project UVs aligned to each face independently"""
    bl_idname = "leveldesign.face_aligned_project"
    bl_label = "Face-Aligned Projection"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() and context.object and context.object.type == 'MESH' and context.mode in {'EDIT_MESH', 'OBJECT'}

    def execute(self, context):
        obj = context.object
        me = obj.data

        # Handle object mode: enter edit mode, select all faces, operate, then return
        was_object_mode = context.mode == 'OBJECT'
        if was_object_mode:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')

        bm = bmesh.from_edit_mesh(me)

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            if was_object_mode:
                bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'WARNING'}, "No faces selected")
            return {'CANCELLED'}

        props = context.scene.level_design_props
        ppm = props.pixels_per_meter
        scale = props.projection_scale

        projected_count = 0
        for face in selected_faces:
            # Skip faces with hotspottable materials
            if face_has_hotspot_material(face, me):
                continue

            mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
            face_aligned_project(face, uv_layer, mat, ppm, scale)
            projected_count += 1

        bmesh.update_edit_mesh(me)
        self.report({'INFO'}, f"Projected {projected_count} faces")

        # Update properties to reflect projection
        set_updating_from_selection(True)
        try:
            props.texture_scale_u = scale
            props.texture_scale_v = scale
            props.texture_rotation = 0.0
            # Offset will be derived from the resulting UVs
        finally:
            set_updating_from_selection(False)
            sync_scale_tracking(context)

        # Update face cache so depsgraph handler doesn't overwrite our changes
        cache_face_data(context)

        # Return to object mode if we started there
        if was_object_mode:
            bpy.ops.object.mode_set(mode='OBJECT')

        return {'FINISHED'}


class LEVELDESIGN_OT_align_uv(Operator):
    """Align texture to face edges"""
    bl_idname = "leveldesign.align_uv"
    bl_label = "Align UV"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        items=[
            ('LEFT', "Left", ""),
            ('RIGHT', "Right", ""),
            ('TOP', "Top", ""),
            ('BOTTOM', "Bottom", ""),
            ('CENTER', "Center", ""),
        ]
    )

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() and context.object and context.object.type == 'MESH' and context.mode in {'EDIT_MESH', 'OBJECT'}

    def execute(self, context):
        obj = context.object
        me = obj.data

        # Handle object mode: enter edit mode, select all faces, operate, then return
        was_object_mode = context.mode == 'OBJECT'
        if was_object_mode:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')

        bm = bmesh.from_edit_mesh(me)

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()
        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            if was_object_mode:
                bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'WARNING'}, "No faces selected")
            return {'CANCELLED'}

        for face in selected_faces:
            # Skip faces with hotspottable materials
            if face_has_hotspot_material(face, me):
                continue

            uvs = [loop[uv_layer].uv.copy() for loop in face.loops]

            if not uvs:
                continue

            # Get UV bounds
            min_u = min(uv.x for uv in uvs)
            max_u = max(uv.x for uv in uvs)
            min_v = min(uv.y for uv in uvs)
            max_v = max(uv.y for uv in uvs)

            center_u = (min_u + max_u) / 2
            center_v = (min_v + max_v) / 2

            # Calculate offset based on direction
            offset_u = 0
            offset_v = 0

            if self.direction == 'LEFT':
                offset_u = -min_u
            elif self.direction == 'RIGHT':
                offset_u = 1.0 - max_u
            elif self.direction == 'TOP':
                offset_v = 1.0 - max_v
            elif self.direction == 'BOTTOM':
                offset_v = -min_v
            elif self.direction == 'CENTER':
                offset_u = 0.5 - center_u
                offset_v = 0.5 - center_v

            # Apply offset
            for loop in face.loops:
                loop[uv_layer].uv.x += offset_u
                loop[uv_layer].uv.y += offset_v

        bmesh.update_edit_mesh(me)

        # Update face cache so depsgraph handler doesn't overwrite our changes
        cache_face_data(context)

        # Return to object mode if we started there
        if was_object_mode:
            bpy.ops.object.mode_set(mode='OBJECT')

        return {'FINISHED'}


class LEVELDESIGN_OT_fit_to_face(Operator):
    """Fit texture to fill the entire face (0-1 UV space)"""
    bl_idname = "leveldesign.fit_to_face"
    bl_label = "Fit to Face"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() and context.object and context.object.type == 'MESH' and context.mode in {'EDIT_MESH', 'OBJECT'}

    def execute(self, context):
        obj = context.object
        me = obj.data

        # Handle object mode: enter edit mode, select all faces, operate, then return
        was_object_mode = context.mode == 'OBJECT'
        if was_object_mode:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')

        bm = bmesh.from_edit_mesh(me)

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()
        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            if was_object_mode:
                bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'WARNING'}, "No faces selected")
            return {'CANCELLED'}

        for face in selected_faces:
            # Skip faces with hotspottable materials
            if face_has_hotspot_material(face, me):
                continue

            uvs = [loop[uv_layer].uv.copy() for loop in face.loops]

            if not uvs:
                continue

            # Get UV bounds
            min_u = min(uv.x for uv in uvs)
            max_u = max(uv.x for uv in uvs)
            min_v = min(uv.y for uv in uvs)
            max_v = max(uv.y for uv in uvs)

            width = max_u - min_u
            height = max_v - min_v

            if width < 0.0001 or height < 0.0001:
                continue

            # Scale and offset to fit 0-1
            for loop in face.loops:
                u = (loop[uv_layer].uv.x - min_u) / width
                v = (loop[uv_layer].uv.y - min_v) / height
                loop[uv_layer].uv = (u, v)

        bmesh.update_edit_mesh(me)

        # Update face cache so depsgraph handler doesn't overwrite our changes
        cache_face_data(context)

        # Return to object mode if we started there
        if was_object_mode:
            bpy.ops.object.mode_set(mode='OBJECT')

        return {'FINISHED'}


class LEVELDESIGN_OT_face_uv_mode(Operator):
    """Enter Face UV mode for streamlined texture edge alignment"""
    bl_idname = "leveldesign.face_uv_mode"
    bl_label = "Face UV Mode"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace() and context.object and context.object.type == 'MESH' and context.mode == 'EDIT_MESH'

    def invoke(self, context, event):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        selected_faces = [f for f in bm.faces if f.select]
        if len(selected_faces) != 1:
            self.report({'WARNING'}, "Select exactly one face")
            return {'CANCELLED'}

        # Skip faces with hotspottable materials
        if face_has_hotspot_material(selected_faces[0], me):
            self.report({'WARNING'}, "Cannot use Face UV Mode on hotspottable faces")
            return {'CANCELLED'}

        self.face_index = selected_faces[0].index
        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

        # Save initial transform for ESC revert
        props = context.scene.level_design_props
        self.saved_scale_u = props.texture_scale_u
        self.saved_scale_v = props.texture_scale_v
        self.saved_rotation = props.texture_rotation
        self.saved_offset_x = props.texture_offset_x
        self.saved_offset_y = props.texture_offset_y
        self.saved_edge_index = props.edge_index

        # Save initial UVs for revert
        face = selected_faces[0]
        self.saved_uvs = [(loop[uv_layer].uv.x, loop[uv_layer].uv.y) for loop in face.loops]

        # Default texture edge is BOTTOM
        self.texture_edge = 'BOTTOM'

        # Track last applied state to avoid redundant updates
        self.last_edge_index = -1
        self.last_texture_edge = None
        self.last_fit_mode = None

        # Fit mode state: None, 'vertical', or 'horizontal'
        self.fit_mode = None
        # Scales before fit mode was enabled (for reverting)
        self.pre_fit_scale_u = None
        self.pre_fit_scale_v = None

        # Initialize mouse position from invoke event
        self.mouse_x = event.mouse_region_x
        self.mouse_y = event.mouse_region_y

        context.window_manager.modal_handler_add(self)
        self._update_status_text(context)

        # Apply initial snap based on cursor position at invoke time
        self._apply_snap(context)

        return {'RUNNING_MODAL'}

    def _update_status_text(self, context):
        """Update status bar text based on current fit mode."""
        fit_indicator = ""
        if self.fit_mode == 'vertical':
            fit_indicator = " [V-Fit]"
        elif self.fit_mode == 'horizontal':
            fit_indicator = " [H-Fit]"
        context.workspace.status_text_set(
            f"W: Top    A: Left    S: Bottom    D: Right    Q: V-Fit    E: H-Fit    R: Reset Scale    LMB: Confirm    Esc: Cancel{fit_indicator}"
        )

    def _calculate_aspect_locked_fit_scale(self, face, uv_layer, fit_mode, current_scale_u, current_scale_v):
        """Calculate uniform scale for aspect-locked fit.

        The fit scale makes the texture fit the face in one dimension (vertical or horizontal),
        with both scale_u and scale_v set to the same value to maintain aspect ratio.

        Args:
            face: The BMesh face with aligned UVs
            uv_layer: UV layer
            fit_mode: 'vertical' or 'horizontal'
            current_scale_u: Current horizontal scale
            current_scale_v: Current vertical scale

        Returns:
            The uniform scale value that achieves the fit
        """
        uvs = [loop[uv_layer].uv for loop in face.loops]
        min_u = min(uv.x for uv in uvs)
        max_u = max(uv.x for uv in uvs)
        min_v = min(uv.y for uv in uvs)
        max_v = max(uv.y for uv in uvs)

        width_uv = max_u - min_u
        height_uv = max_v - min_v

        if fit_mode == 'vertical':
            if height_uv < 0.0001:
                return current_scale_v
            return height_uv * current_scale_v
        else:  # horizontal
            if width_uv < 0.0001:
                return current_scale_u
            return width_uv * current_scale_u

    def _get_closest_edge_index(self, context, face):
        """Find which face edge is closest to the mouse cursor in screen space."""
        region = context.region
        rv3d = context.region_data
        mouse_pos = Vector((self.mouse_x, self.mouse_y))

        obj = context.object
        world_matrix = obj.matrix_world

        loops = list(face.loops)
        num_edges = len(loops)

        closest_idx = 0
        closest_dist = float('inf')

        for i in range(num_edges):
            # Get edge midpoint in world space
            v1 = world_matrix @ loops[i].vert.co
            v2 = world_matrix @ loops[(i + 1) % num_edges].vert.co
            midpoint = (v1 + v2) / 2

            # Project to screen space
            screen_pos = location_3d_to_region_2d(region, rv3d, midpoint)
            if screen_pos is None:
                continue

            dist = (screen_pos - mouse_pos).length
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i

        return closest_idx

    def _apply_snap(self, context):
        """Apply snap to edge based on current texture_edge, closest face edge, and fit mode."""
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        bm.faces.ensure_lookup_table()
        if self.face_index >= len(bm.faces):
            return

        face = bm.faces[self.face_index]
        if not face.is_valid:
            return

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()
        props = context.scene.level_design_props

        # Get closest edge to mouse
        edge_index = self._get_closest_edge_index(context, face)

        # Skip if nothing changed
        if (edge_index == self.last_edge_index and
            self.texture_edge == self.last_texture_edge and
            self.fit_mode == self.last_fit_mode):
            return

        self.last_edge_index = edge_index
        self.last_texture_edge = self.texture_edge
        self.last_fit_mode = self.fit_mode

        # Map texture_edge to square_edge_index: 0=bottom, 1=right, 2=top, 3=left
        square_edge_map = {'BOTTOM': 0, 'RIGHT': 1, 'TOP': 2, 'LEFT': 3}
        square_edge_index = square_edge_map[self.texture_edge]

        # Update edge_index property
        props.edge_index = edge_index

        # Get current UV shape
        shape = [(loop[uv_layer].uv.x, loop[uv_layer].uv.y) for loop in face.loops]

        # Align the shape edge to the square edge
        aligned = align_2d_shape_to_square(shape, edge_index, square_edge_index)

        # Apply the transformed UVs back
        for i, loop in enumerate(face.loops):
            loop[uv_layer].uv.x = aligned[i][0]
            loop[uv_layer].uv.y = aligned[i][1]

        # Determine scale values
        if self.fit_mode:
            # Calculate fit scale from aligned UVs
            fit_scale = self._calculate_aspect_locked_fit_scale(
                face, uv_layer, self.fit_mode,
                props.texture_scale_u, props.texture_scale_v
            )
            scale_u = fit_scale
            scale_v = fit_scale
        else:
            # Use pre-fit scales if we have them (fit was disabled), otherwise current props
            if self.pre_fit_scale_u is not None:
                scale_u = self.pre_fit_scale_u
                scale_v = self.pre_fit_scale_v
            else:
                scale_u = props.texture_scale_u
                scale_v = props.texture_scale_v

        # Derive rotation and offset from the aligned UVs
        transform = derive_transform_from_uvs(face, uv_layer, props.pixels_per_meter, me)
        if transform:
            rotation = transform['rotation']
            offset_x = transform['offset_x']
            offset_y = transform['offset_y']

            # Re-apply UVs with final scale (this recalculates from scratch)
            mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
            apply_uv_to_face(face, uv_layer, scale_u, scale_v, rotation,
                             offset_x, offset_y, mat, props.pixels_per_meter, me)

            # Re-align to fix positioning after scale change
            shape = [(loop[uv_layer].uv.x, loop[uv_layer].uv.y) for loop in face.loops]
            aligned = align_2d_shape_to_square(shape, edge_index, square_edge_index)
            for i, loop in enumerate(face.loops):
                loop[uv_layer].uv.x = aligned[i][0]
                loop[uv_layer].uv.y = aligned[i][1]

            # Derive final transform after re-alignment
            final_transform = derive_transform_from_uvs(face, uv_layer, props.pixels_per_meter, me)
            if final_transform:
                set_updating_from_selection(True)
                try:
                    props.texture_scale_u = scale_u
                    props.texture_scale_v = scale_v
                    props.texture_rotation = final_transform['rotation']
                    props.texture_offset_x = final_transform['offset_x']
                    props.texture_offset_y = final_transform['offset_y']
                finally:
                    set_updating_from_selection(False)

        # Update cache
        cache_single_face(face, bm, props.pixels_per_meter, me)

        bmesh.update_edit_mesh(me)

    def _revert_transform(self, context):
        """Revert to saved transform on ESC."""
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        bm.faces.ensure_lookup_table()
        if self.face_index >= len(bm.faces):
            return

        face = bm.faces[self.face_index]
        if not face.is_valid:
            return

        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()
        props = context.scene.level_design_props

        # Restore saved UVs directly
        loops = list(face.loops)
        for i, loop in enumerate(loops):
            loop[uv_layer].uv.x = self.saved_uvs[i][0]
            loop[uv_layer].uv.y = self.saved_uvs[i][1]

        # Restore saved properties
        set_updating_from_selection(True)
        try:
            props.texture_scale_u = self.saved_scale_u
            props.texture_scale_v = self.saved_scale_v
            props.texture_rotation = self.saved_rotation
            props.texture_offset_x = self.saved_offset_x
            props.texture_offset_y = self.saved_offset_y
            props.edge_index = self.saved_edge_index
        finally:
            set_updating_from_selection(False)

        # Update cache with reverted state
        cache_single_face(face, bm, props.pixels_per_meter, me)

        bmesh.update_edit_mesh(me)

    def modal(self, context, event):
        # Store mouse position for edge detection
        self.mouse_x = event.mouse_region_x
        self.mouse_y = event.mouse_region_y

        props = context.scene.level_design_props

        # WASD to set texture edge
        if event.type == 'W' and event.value == 'PRESS':
            self.texture_edge = 'TOP'
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        if event.type == 'A' and event.value == 'PRESS':
            self.texture_edge = 'LEFT'
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        if event.type == 'S' and event.value == 'PRESS':
            self.texture_edge = 'BOTTOM'
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        if event.type == 'D' and event.value == 'PRESS':
            self.texture_edge = 'RIGHT'
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        # Q for vertical fit toggle
        if event.type == 'Q' and event.value == 'PRESS':
            if self.fit_mode == 'vertical':
                # Disable fit mode
                self.fit_mode = None
            else:
                # Enable vertical fit (save scales if first time enabling any fit)
                if self.fit_mode is None:
                    self.pre_fit_scale_u = props.texture_scale_u
                    self.pre_fit_scale_v = props.texture_scale_v
                self.fit_mode = 'vertical'
            self._update_status_text(context)
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        # E for horizontal fit toggle
        if event.type == 'E' and event.value == 'PRESS':
            if self.fit_mode == 'horizontal':
                # Disable fit mode
                self.fit_mode = None
            else:
                # Enable horizontal fit (save scales if first time enabling any fit)
                if self.fit_mode is None:
                    self.pre_fit_scale_u = props.texture_scale_u
                    self.pre_fit_scale_v = props.texture_scale_v
                self.fit_mode = 'horizontal'
            self._update_status_text(context)
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        # R for reset scale to 1
        if event.type == 'R' and event.value == 'PRESS':
            # Disable any fit mode
            self.fit_mode = None
            # Set pre_fit scales to 1 so disabling fit returns to 1
            self.pre_fit_scale_u = 1.0
            self.pre_fit_scale_v = 1.0
            # Force re-apply by clearing last state
            self.last_fit_mode = 'force_update'
            self._update_status_text(context)
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        # Mouse move - update snap based on new cursor position
        if event.type == 'MOUSEMOVE':
            self._apply_snap(context)
            return {'RUNNING_MODAL'}

        # Left click - confirm and exit
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # Clear pre-fit scales on confirm (they're now the real scales)
            self.pre_fit_scale_u = None
            self.pre_fit_scale_v = None
            context.workspace.status_text_set(None)
            return {'FINISHED'}

        # Escape - revert and exit
        if event.type == 'ESC':
            self._revert_transform(context)
            context.workspace.status_text_set(None)
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
        from ..utils import get_face_id_layer, save_face_selection, restore_face_selection
        id_layer = get_face_id_layer(bm)
        selected_ids, active_id = save_face_selection(bm, id_layer)

        # In object mode, apply to all faces; in edit mode, use selection or all faces
        if was_object_mode:
            faces_to_process = list(bm.faces)
        else:
            selected_faces = [f for f in bm.faces if f.select]
            faces_to_process = selected_faces if selected_faces else list(bm.faces)

        # Get seam mode and hotspot settings from per-object properties
        props = context.scene.level_design_props
        seam_mode = obj.anvil_hotspot_seam_mode
        allow_combined_faces = obj.anvil_allow_combined_faces
        size_weight = obj.anvil_hotspot_size_weight

        applied_count, skipped_no_hotspot, skipped_not_quad = apply_hotspots_to_mesh(
            bm, me, faces_to_process, seam_mode, allow_combined_faces,
            obj.matrix_world, props.pixels_per_meter, size_weight
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
    LEVELDESIGN_OT_face_aligned_project,
    LEVELDESIGN_OT_align_uv,
    LEVELDESIGN_OT_fit_to_face,
    LEVELDESIGN_OT_face_uv_mode,
    LEVELDESIGN_OT_apply_hotspot,
)

addon_keymaps = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Register keymap for Face UV Mode
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Mesh', space_type='EMPTY')
        kmi = km.keymap_items.new(
            "leveldesign.face_uv_mode",
            'T', 'PRESS',
            head=True
        )
        addon_keymaps.append((km, kmi))

        # UV shortcut keymaps (unbound by default)
        kmi = km.keymap_items.new(
            "leveldesign.face_aligned_project",
            'NONE', 'PRESS',
            head=True
        )
        addon_keymaps.append((km, kmi))

        kmi = km.keymap_items.new(
            "leveldesign.align_uv",
            'NONE', 'PRESS',
            head=True
        )
        kmi.properties.direction = 'CENTER'
        addon_keymaps.append((km, kmi))

        kmi = km.keymap_items.new(
            "leveldesign.fit_to_face",
            'NONE', 'PRESS',
            head=True
        )
        addon_keymaps.append((km, kmi))


def unregister():
    # Remove keymaps
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
