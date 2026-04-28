"""
Box Builder - Geometry

Creates box meshes with correct outward normals, material assignment, and UV mapping.
"""

import bpy
import bmesh
from mathutils import Vector

from ..texture_apply import set_uv_from_other_face
from ...handlers import cache_single_face, get_active_image, get_previous_image
from ...core.logging import debug_log
from ...core.materials import (
    find_material_with_image,
    create_material_with_image,
    ensure_material_slot,
)
from ...core.face_id import get_face_id_layer
from ...core.uv_projection import face_aligned_project
from ...core.uv_layers import get_render_active_uv_layer


def _faces_coplanar_antiparallel(face_a, face_b):
    dot = face_a.normal.dot(face_b.normal)
    if dot > -0.99:
        return False
    dist = abs((face_b.verts[0].co - face_a.verts[0].co).dot(face_a.normal))
    return dist < 0.001


def _project_face_2d(face, axis_u, axis_v):
    return [(v.co.dot(axis_u), v.co.dot(axis_v)) for v in face.verts]


def _polygons_overlap_2d(poly_a, poly_b):
    for poly in [poly_a, poly_b]:
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge_x = poly[j][0] - poly[i][0]
            edge_y = poly[j][1] - poly[i][1]
            axis = (-edge_y, edge_x)

            min_a = min(p[0] * axis[0] + p[1] * axis[1] for p in poly_a)
            max_a = max(p[0] * axis[0] + p[1] * axis[1] for p in poly_a)
            min_b = min(p[0] * axis[0] + p[1] * axis[1] for p in poly_b)
            max_b = max(p[0] * axis[0] + p[1] * axis[1] for p in poly_b)

            if max_a <= min_b + 1e-6 or max_b <= min_a + 1e-6:
                return False
    return True


def _faces_overlap(face_a, face_b):
    normal = face_a.normal
    if abs(normal.z) < 0.9:
        up = Vector((0, 0, 1))
    else:
        up = Vector((1, 0, 0))
    axis_u = normal.cross(up).normalized()
    axis_v = normal.cross(axis_u).normalized()

    poly_a = _project_face_2d(face_a, axis_u, axis_v)
    poly_b = _project_face_2d(face_b, axis_u, axis_v)
    return _polygons_overlap_2d(poly_a, poly_b)


def _face_overlaps_antiparallel_coplanar_existing(face, existing_faces):
    for existing_face in existing_faces:
        if _faces_coplanar_antiparallel(face, existing_face):
            if _faces_overlap(face, existing_face):
                return True
    return False


def _remove_antiparallel_coplanar_faces(bm, new_faces, existing_faces):
    bm.faces.index_update()
    bm.faces.ensure_lookup_table()
    faces_to_remove = [
        face for face in new_faces
        if face.is_valid and _face_overlaps_antiparallel_coplanar_existing(face, existing_faces)
    ]
    if not faces_to_remove:
        return new_faces

    edges_to_remove = []
    for edge in bm.edges:
        linked_removed_faces = [
            face for face in edge.link_faces
            if face in faces_to_remove
        ]
        if len(linked_removed_faces) >= 2 and len(linked_removed_faces) == len(edge.link_faces):
            edges_to_remove.append(edge)

    kept_faces = [face for face in new_faces if face not in faces_to_remove]
    for face in faces_to_remove:
        debug_log(f"[BoxBuilder] Removing anti-parallel coplanar face {face.index}")
    bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES_ONLY')
    edges_to_remove = [edge for edge in edges_to_remove if edge.is_valid]
    if edges_to_remove:
        debug_log(f"[BoxBuilder] Removing {len(edges_to_remove)} edges between removed faces")
        bmesh.ops.delete(bm, geom=edges_to_remove, context='EDGES')
    bm.faces.index_update()
    bm.faces.ensure_lookup_table()
    bm.normal_update()

    return [face for face in kept_faces if face.is_valid]


def execute_box_builder(first_vertex, second_vertex, depth, local_x, local_y, local_z,
                        obj, ppm, view_forward, keep_anti_parallel_coplanar_faces):
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
        view_forward: Camera forward direction (world space), used for plane normal orientation
        keep_anti_parallel_coplanar_faces: Keep new faces that overlap
            existing coplanar faces with opposite normals

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
    # Detect left-handed axis system (e.g. TOP, BACK, RIGHT ortho views).
    # Only affects boxes; planes handle view-facing via view_forward.
    left_handed = lx.cross(ly).dot(lz) < 0
    if dx < 0:
        dx = -dx
        lx = -lx
        flip_count += 1
    if dy < 0:
        dy = -dy
        ly = -ly
        flip_count += 1

    # Transform view_forward to object local space for plane normal orientation
    local_view_forward = (rot @ view_forward).normalized()

    # Find active selected face for material/UV source (before creating new geometry)
    # Ensure custom layers exist before creating geometry (creating layers invalidates BMesh refs)
    source_face = None
    uv_layer = get_render_active_uv_layer(bm, me)
    if uv_layer is None:
        uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        uv_layer = bm.loops.layers.uv.new("UVMap")
    get_face_id_layer(bm)
    if (bm.faces.active is not None and bm.faces.active.is_valid
            and not bm.faces.active.hide and bm.faces.active.select):
        source_face = bm.faces.active

    existing_faces = [
        face for face in bm.faces
        if face.is_valid and not face.hide
    ]

    is_zero_depth = abs(local_depth) < 1e-5
    is_zero_dx = dx < 1e-5
    is_zero_dy = dy < 1e-5
    is_plane = is_zero_depth or is_zero_dx or is_zero_dy

    if is_plane:
        if is_zero_dx and not is_zero_depth:
            new_faces = _create_plane(bm, local_first, ly, dy, lz, local_depth, local_view_forward)
        elif is_zero_dy and not is_zero_depth:
            new_faces = _create_plane(bm, local_first, lx, dx, lz, local_depth, local_view_forward)
        else:
            new_faces = _create_plane(bm, local_first, lx, dx, ly, dy, local_view_forward)
    else:
        box_flip_count = flip_count + (1 if left_handed else 0)
        new_faces = _create_box(bm, local_first, dx, dy, local_depth,
                                lx, ly, lz, box_flip_count)

    if not new_faces:
        bmesh.update_edit_mesh(me)
        return (False, "Failed to create box geometry")

    # Normals must be computed before UV application (UV functions use face.normal)
    bm.normal_update()

    if not is_plane and not keep_anti_parallel_coplanar_faces:
        new_faces = _remove_antiparallel_coplanar_faces(bm, new_faces, existing_faces)

    # Apply material and UVs
    _apply_material_and_uvs(bm, new_faces, source_face, uv_layer, ppm, me, obj)

    # Diagnostic: check for zero-area UVs after box creation
    if uv_layer is not None:
        for face in new_faces:
            if not face.is_valid:
                continue
            uvs = [loop[uv_layer].uv.copy() for loop in face.loops]
            uv_area = 0.0
            for i in range(1, len(uvs) - 1):
                ea = uvs[i] - uvs[0]
                eb = uvs[i + 1] - uvs[0]
                uv_area += abs(ea.x * eb.y - ea.y * eb.x)
            if uv_area < 1e-8:
                debug_log(f"[BoxBuilder] WARNING: face {face.index} has zero-area UVs after creation "
                          f"(source_face={'index ' + str(source_face.index) if source_face else 'None'}, "
                          f"mat_idx={face.material_index})")

    # Add newly created box geometry to the existing selection and collect
    # indexed vertex signatures for weld tracking.
    new_face_vert_positions = []
    bm.faces.index_update()
    bm.faces.ensure_lookup_table()
    for f in new_faces:
        if f.is_valid:
            f.select = True
            new_face_vert_positions.append(
                (f.index, frozenset(tuple(v.co) for v in f.verts))
            )
    bm.select_flush(True)

    bmesh.update_edit_mesh(me)

    if is_plane:
        return (True, "Plane created", new_face_vert_positions)
    return (True, "Box created", new_face_vert_positions)


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


def _create_plane(bm, origin, axis1, dim1, axis2, dim2, local_view_forward):
    """Create a single quad plane facing toward the camera.

    The plane spans axis1*dim1 and axis2*dim2 from origin.
    Normal is oriented to face the camera (opposite local_view_forward).

    Args:
        bm: BMesh instance
        origin: Plane corner position (object-local space)
        axis1: First spanning axis (normalized, object-local)
        dim1: Extent along axis1 (signed)
        axis2: Second spanning axis (normalized, object-local)
        dim2: Extent along axis2 (signed)
        local_view_forward: Camera forward direction (object-local space)

    Returns:
        list: List containing the single new BMFace
    """
    v0 = bm.verts.new(origin)
    v1 = bm.verts.new(origin + axis1 * dim1)
    v2 = bm.verts.new(origin + axis1 * dim1 + axis2 * dim2)
    v3 = bm.verts.new(origin + axis2 * dim2)

    # Default winding [v0, v1, v2, v3] produces normal along (axis1*dim1) x (axis2*dim2).
    # We want the normal to face the camera (opposite to local_view_forward).
    geometric_normal = (axis1 * dim1).cross(axis2 * dim2)

    if geometric_normal.dot(local_view_forward) > 0:
        winding = [v0, v3, v2, v1]  # Reverse to face camera
    else:
        winding = [v0, v1, v2, v3]  # Already faces camera

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
    if (source_face is not None and source_face.is_valid
            and not source_face.hide and uv_layer is not None):
        # Alt-click style: copy material and UV from source face
        mat_idx = source_face.material_index
        obj_matrix = obj.matrix_world

        for face in new_faces:
            if not face.is_valid:
                continue
            face.material_index = mat_idx
            result = set_uv_from_other_face(source_face, face, uv_layer, ppm, me, obj_matrix)
            if not result:
                debug_log(f"[BoxBuilder] set_uv_from_other_face FAILED for face {face.index} "
                          f"(source face {source_face.index}, source area={source_face.calc_area():.6f})")
    else:
        # Default: use active image, falling back to previous image
        image = get_active_image()
        if image is None:
            image = get_previous_image()
        if image is None:
            for face in new_faces:
                if not face.is_valid:
                    continue
                mat = (
                    me.materials[face.material_index]
                    if face.material_index < len(me.materials)
                    else None
                )
                face_aligned_project(face, uv_layer, mat, ppm)
                cache_single_face(face, bm, ppm, me)
            return

        # Find or create material for this image
        mat = find_material_with_image(image)
        if mat is None:
            mat = create_material_with_image(image)

        mat_idx = ensure_material_slot(me, mat)

        for face in new_faces:
            if not face.is_valid:
                continue
            face.material_index = mat_idx
            face_aligned_project(face, uv_layer, mat, ppm)
            cache_single_face(face, bm, ppm, me)


def execute_box_builder_object_mode(first_vertex, second_vertex, depth,
                                    local_x, local_y, local_z,
                                    ppm, view_forward):
    """Create a new object with box geometry in object mode.

    Object origin is placed at first_vertex; geometry is built relative to it.

    Args:
        first_vertex: First corner of the rectangle (world space)
        second_vertex: Opposite corner of the rectangle (world space)
        depth: Depth of the box (can be negative or zero)
        local_x: Rectangle's local X axis
        local_y: Rectangle's local Y axis
        local_z: Rectangle's local Z axis (depth direction)
        ppm: Pixels per meter setting
        view_forward: Camera forward direction (world space), used for plane normal orientation

    Returns:
        tuple: (success: bool, message: str)
    """
    # Compute dimensions relative to first_vertex (will be object origin)
    diff = second_vertex - first_vertex
    dx = diff.dot(local_x)
    dy = diff.dot(local_y)

    flip_count = 0
    lx = local_x.copy()
    ly = local_y.copy()
    lz = local_z.copy()

    # Detect left-handed axis system (e.g. TOP, BACK, RIGHT ortho views).
    # Only affects boxes; planes handle view-facing via view_forward.
    left_handed = lx.cross(ly).dot(lz) < 0
    if dx < 0:
        dx = -dx
        lx = -lx
        flip_count += 1
    if dy < 0:
        dy = -dy
        ly = -ly
        flip_count += 1

    is_zero_depth = abs(depth) < 1e-5
    is_zero_dx = dx < 1e-5
    is_zero_dy = dy < 1e-5
    is_plane = is_zero_depth or is_zero_dx or is_zero_dy

    # Build geometry in a new bmesh (origin at 0,0,0)
    bm = bmesh.new()
    origin = Vector((0, 0, 0))

    if is_plane:
        if is_zero_dx and not is_zero_depth:
            new_faces = _create_plane(bm, origin, ly, dy, lz, depth, view_forward)
        elif is_zero_dy and not is_zero_depth:
            new_faces = _create_plane(bm, origin, lx, dx, lz, depth, view_forward)
        else:
            new_faces = _create_plane(bm, origin, lx, dx, ly, dy, view_forward)
    else:
        box_flip_count = flip_count + (1 if left_handed else 0)
        new_faces = _create_box(bm, origin, dx, dy, depth,
                                lx, ly, lz, box_flip_count)

    if not new_faces:
        bm.free()
        return (False, "Failed to create box geometry")

    bm.normal_update()

    # Create new mesh data and object
    me = bpy.data.meshes.new("Box")
    obj = bpy.data.objects.new("Box", me)
    obj.location = first_vertex

    # Link to active collection
    collection = bpy.context.collection
    collection.objects.link(obj)

    # Write initial geometry to mesh
    bm.to_mesh(me)
    bm.free()

    # Deselect all, then set new object as active and selected
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Ensure a UV map exists (matching Blender's default Add Cube behaviour)
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")

    # Apply material and UVs via edit mode (apply_uv_to_face requires edit mesh)
    image = get_active_image()
    if image is None:
        image = get_previous_image()
    if image is not None:
        mat = find_material_with_image(image)
        if mat is None:
            mat = create_material_with_image(image)

        mat_idx = ensure_material_slot(me, mat)

        bpy.ops.object.mode_set(mode='EDIT')

        bm_edit = bmesh.from_edit_mesh(me)
        bm_edit.faces.ensure_lookup_table()

        uv_layer = get_render_active_uv_layer(bm_edit, me)
        if uv_layer is None:
            uv_layer = bm_edit.loops.layers.uv.new("UVMap")

        for face in bm_edit.faces:
            if not face.is_valid:
                continue
            face.material_index = mat_idx
            face_aligned_project(face, uv_layer, mat, ppm)
            cache_single_face(face, bm_edit, ppm, me)

        bmesh.update_edit_mesh(me)
        bpy.ops.object.mode_set(mode='OBJECT')

    if is_plane:
        return (True, "Plane object created")
    return (True, "Box object created")
