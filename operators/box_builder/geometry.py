"""
Box Builder - Geometry

Creates box meshes with correct outward normals, material assignment, and UV mapping.
"""

import bmesh
from mathutils import Vector

from ..texture_apply import set_uv_from_other_face
from ...properties import apply_uv_to_face
from ...handlers import cache_single_face, get_active_image
from ...utils import find_material_with_image, create_material_with_image, debug_log


def execute_box_builder(first_vertex, second_vertex, depth, local_x, local_y, local_z,
                        obj, ppm, reverse_plane_normal):
    """
    Create a box mesh from the modal draw parameters.

    Args:
        first_vertex: First corner of the rectangle (world space)
        second_vertex: Opposite corner of the rectangle (world space)
        depth: Depth of the box (can be negative or zero)
        local_x: Rectangle's local X axis
        local_y: Rectangle's local Y axis
        local_z: Rectangle's local Z axis (depth direction)
        obj: The active mesh object
        ppm: Pixels per meter setting
        reverse_plane_normal: If True, zero-depth plane faces -lz instead of +lz

    Returns:
        tuple: (success: bool, message: str)
    """
    if obj is None or obj.type != 'MESH':
        return (False, "No active mesh object")

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # Transform to object local space
    world_to_local = obj.matrix_world.inverted()
    local_first = world_to_local @ first_vertex
    local_second = world_to_local @ second_vertex

    rot = world_to_local.to_3x3()
    lx = (rot @ local_x).normalized()
    ly = (rot @ local_y).normalized()
    lz = (rot @ local_z).normalized()

    scale_factor = (rot @ local_z).length
    local_depth = depth * scale_factor

    # Compute rectangle dimensions along local axes
    diff = local_second - local_first
    dx = diff.dot(lx)
    dy = diff.dot(ly)

    # Track axis flips for winding correction
    flip_count = 0
    if dx < 0:
        dx = -dx
        lx = -lx
        flip_count += 1
    if dy < 0:
        dy = -dy
        ly = -ly
        flip_count += 1

    # Find active selected face for material/UV source (before creating new geometry)
    source_face = None
    uv_layer = bm.loops.layers.uv.active
    if bm.faces.active is not None and bm.faces.active.is_valid and bm.faces.active.select:
        source_face = bm.faces.active

    is_zero_depth = abs(local_depth) < 1e-5

    if is_zero_depth:
        new_faces = _create_plane(bm, local_first, dx, dy, local_depth,
                                  lx, ly, lz, flip_count, reverse_plane_normal)
    else:
        new_faces = _create_box(bm, local_first, dx, dy, local_depth,
                                lx, ly, lz, flip_count)

    if not new_faces:
        bmesh.update_edit_mesh(me)
        return (False, "Failed to create box geometry")

    # Normals must be computed before UV application (UV functions use face.normal)
    bm.normal_update()

    # Apply material and UVs
    _apply_material_and_uvs(bm, new_faces, source_face, uv_layer, ppm, me, obj)

    bmesh.update_edit_mesh(me)

    if is_zero_depth:
        return (True, "Plane created")
    return (True, "Box created")


def _create_box(bm, origin, dx, dy, depth, lx, ly, lz, flip_count):
    """Create a 6-faced box with outward normals guaranteed by construction.

    Uses a canonical face winding table for the positive-depth right-handed case,
    then corrects for axis flips and negative depth.

    Returns:
        list: List of newly created BMFaces
    """
    # Build 8 vertices
    #   Front face (depth=0):  0=BL, 1=BR, 2=TR, 3=TL
    #   Back face (depth=d):   4=BL, 5=BR, 6=TR, 7=TL
    v = []
    v.append(bm.verts.new(origin))                                      # 0: front BL
    v.append(bm.verts.new(origin + lx * dx))                            # 1: front BR
    v.append(bm.verts.new(origin + lx * dx + ly * dy))                  # 2: front TR
    v.append(bm.verts.new(origin + ly * dy))                            # 3: front TL
    v.append(bm.verts.new(origin + lz * depth))                         # 4: back BL
    v.append(bm.verts.new(origin + lx * dx + lz * depth))               # 5: back BR
    v.append(bm.verts.new(origin + lx * dx + ly * dy + lz * depth))     # 6: back TR
    v.append(bm.verts.new(origin + ly * dy + lz * depth))               # 7: back TL

    # Canonical face winding table (outward normals for positive depth, no flips)
    # Each face is ordered so the cross product of consecutive edges points outward
    face_windings = [
        [v[0], v[3], v[2], v[1]],  # Front  (-lz)
        [v[4], v[5], v[6], v[7]],  # Back   (+lz)
        [v[0], v[4], v[7], v[3]],  # Left   (-lx)
        [v[1], v[2], v[6], v[5]],  # Right  (+lx)
        [v[0], v[1], v[5], v[4]],  # Bottom (-ly)
        [v[3], v[7], v[6], v[2]],  # Top    (+ly)
    ]

    # Odd number of axis flips reverses all windings
    reverse = (flip_count % 2 == 1)
    # Negative depth also reverses
    if depth < 0:
        reverse = not reverse

    if reverse:
        face_windings = [list(reversed(fw)) for fw in face_windings]

    faces = []
    for winding in face_windings:
        try:
            f = bm.faces.new(winding)
            faces.append(f)
        except ValueError:
            debug_log(f"[BoxBuilder] Failed to create face with winding {[vt.co[:] for vt in winding]}")

    return faces


def _create_plane(bm, origin, dx, dy, depth, lx, ly, lz, flip_count, reverse_plane_normal):
    """Create a single-plane (4-vert) face for zero-depth case.

    The plane faces toward the camera. The caller determines whether +lz or -lz
    points toward the camera and sets reverse_plane_normal accordingly.

    Returns:
        list: List containing the single new BMFace
    """
    v0 = bm.verts.new(origin)
    v1 = bm.verts.new(origin + lx * dx)
    v2 = bm.verts.new(origin + lx * dx + ly * dy)
    v3 = bm.verts.new(origin + ly * dy)

    # Default winding [v0, v1, v2, v3] produces normal in +lz direction.
    # If reverse_plane_normal is True, lz points away from camera so we reverse
    # to get -lz (toward camera). Otherwise +lz already faces the camera.
    if reverse_plane_normal:
        winding = [v0, v3, v2, v1]  # Normal in -lz
    else:
        winding = [v0, v1, v2, v3]  # Normal in +lz

    # Odd number of axis flips reverses winding
    if flip_count % 2 == 1:
        winding = list(reversed(winding))

    try:
        f = bm.faces.new(winding)
        return [f]
    except ValueError:
        debug_log(f"[BoxBuilder] Failed to create plane face")
        return []


def _apply_material_and_uvs(bm, new_faces, source_face, uv_layer, ppm, me, obj):
    """Apply material and UVs to newly created faces.

    If a selected active face exists: copy its material and use set_uv_from_other_face
    (alt-click style projection).

    Otherwise: use the active image to find/create a material and apply default UVs.
    """
    if source_face is not None and source_face.is_valid and uv_layer is not None:
        # Alt-click style: copy material and UV from source face
        mat_idx = source_face.material_index
        obj_matrix = obj.matrix_world

        for face in new_faces:
            if not face.is_valid:
                continue
            face.material_index = mat_idx
            set_uv_from_other_face(source_face, face, uv_layer, ppm, me, obj_matrix)
    else:
        # Default: use active image
        image = get_active_image()
        if image is None:
            return

        # Find or create material for this image
        mat = find_material_with_image(image)
        if mat is None:
            mat = create_material_with_image(image)

        # Ensure material is on the object
        mat_idx = None
        for i, slot_mat in enumerate(me.materials):
            if slot_mat == mat:
                mat_idx = i
                break
        if mat_idx is None:
            me.materials.append(mat)
            mat_idx = len(me.materials) - 1

        # Ensure UV layer exists
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.new("UVMap")

        for face in new_faces:
            if not face.is_valid:
                continue
            face.material_index = mat_idx
            apply_uv_to_face(face, uv_layer, 1.0, 1.0, 0, 0, 0, mat, ppm, me)
            cache_single_face(face, uv_layer, ppm, me)
