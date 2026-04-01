import bpy
import bmesh
import math
from bpy.types import Operator
from bpy_extras import view3d_utils

from ..utils import is_level_design_workspace, debug_log
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..utils import (
    find_material_with_image,
    create_material_with_image,
    get_texture_dimensions_from_material,
    get_face_local_axes,
    derive_transform_from_uvs,
    normalize_offset,
    get_image_from_material,
    face_has_hotspot_material,
    any_connected_face_has_hotspot,
    get_all_hotspot_faces,
    get_face_id_layer,
    save_face_selection,
    restore_face_selection,
)
from ..properties import apply_uv_to_face
from ..handlers import (
    cache_single_face, get_active_image, set_active_image, redraw_ui_panels,
    update_ui_from_selection, update_active_image_from_face,
)
from .backface_select.paint_base import ModalPaintBase
from .backface_select.raycast import (
    raycast_bvh_skip_backfaces,
    raycast_bvh_skip_backfaces_polys,
)


def set_uv_from_source_params(target_face, uv_layer, ppm, me, obj_matrix,
                              scale_u, scale_v, source_rotation,
                              source_normal, source_u, source_v,
                              ref_point_co, ref_point_uv,
                              bm=None):
    """Apply UV to target_face using explicit source transform parameters.

    Instead of reading from a live source BMFace, this accepts the source's
    derived transform, axes, and a reference point (position + UV) to compute
    the correct rotation and offset for the target face.

    Args:
        target_face: BMesh face to apply UVs to
        uv_layer: BMesh UV layer
        ppm: Pixels per meter setting
        me: Mesh data for target face
        obj_matrix: Target object world matrix (for parallel plane reference)
        scale_u, scale_v: Source texture scale factors
        source_rotation: Source rotation in degrees
        source_normal: Source face normal (Vector, already in target space)
        source_u, source_v: Source face local axes (Vectors, already in target space)
        ref_point_co: 3D position of a reference point (already in target space)
        ref_point_uv: UV coordinates of that reference point (Vector2D)
        bm: BMesh instance (optional, for cache_single_face per-layer caching)
    """
    target_normal = target_face.normal.normalized()
    target_axes = get_face_local_axes(target_face)
    if not target_axes:
        return False

    target_u, target_v = target_axes

    # Compute reference axis for rotation calculation
    # For intersecting planes: use the intersection line
    # For parallel planes: use "most upward" direction on the plane
    intersection = source_normal.cross(target_normal)
    if intersection.length < 0.0001:
        # Parallel planes - compute "most upward" vector on plane as reference
        local_up = (obj_matrix.inverted().to_3x3() @ Vector((0, 0, 1))).normalized()
        reference = local_up - source_normal.dot(local_up) * source_normal
        if reference.length < 0.0001:
            # Plane is horizontal, use texture V direction as reference
            rot_rad = math.radians(source_rotation)
            reference = source_u * math.sin(rot_rad) + source_v * math.cos(rot_rad)
        reference = reference.normalized()
    else:
        reference = intersection.normalized()

    # Compute angle from reference to each face's U direction
    angle_ref_to_source_u = math.atan2(
        reference.cross(source_u).dot(source_normal),
        reference.dot(source_u)
    )
    angle_ref_to_target_u = math.atan2(
        reference.cross(target_u).dot(target_normal),
        reference.dot(target_u)
    )

    # Source's texture U direction relative to reference
    texture_angle_from_ref = angle_ref_to_source_u - math.radians(source_rotation)

    # For anti-parallel faces (opposite normals), mirror the rotation
    if source_normal.dot(target_normal) < -0.9999:
        texture_angle_from_ref = -texture_angle_from_ref + math.radians(180)

    # Target rotation needed to achieve the same texture angle from reference
    target_rotation = math.degrees(angle_ref_to_target_u - texture_angle_from_ref)

    # Check for opposite winding (u→v rotation direction differs between faces)
    source_handedness = source_u.cross(source_v).dot(source_normal)
    target_handedness = target_u.cross(target_v).dot(target_normal)
    if source_handedness * target_handedness < 0:
        target_rotation += 180

    # Compute offset: project the reference point onto the target face and
    # figure out what offset is needed to match its known UV.
    target_plane_point = list(target_face.loops)[0].vert.co
    dist_to_plane = (ref_point_co - target_plane_point).dot(target_normal)
    projected_point = ref_point_co - dist_to_plane * target_normal

    target_mat = me.materials[target_face.material_index] if target_face.material_index < len(me.materials) else None
    tex_meters_u, tex_meters_v = get_texture_dimensions_from_material(target_mat, ppm)

    rot_rad = math.radians(target_rotation)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    proj_x = target_u * cos_r - target_v * sin_r
    proj_y = target_u * sin_r + target_v * cos_r

    delta = projected_point - target_plane_point
    x = delta.dot(proj_x)
    y = delta.dot(proj_y)

    u = x / (scale_u * tex_meters_u)
    v = y / (scale_v * tex_meters_v)

    target_offset_x = normalize_offset(ref_point_uv.x - u)
    target_offset_y = normalize_offset(ref_point_uv.y - v)

    # Apply to target face
    debug_log(f"[ApplyImage] set_uv_from_source_params: face {target_face.index} | "
              f"scale=({scale_u:.4f}, {scale_v:.4f}) rotation={target_rotation:.2f} offset=({target_offset_x:.4f}, {target_offset_y:.4f})")
    apply_uv_to_face(target_face, uv_layer, scale_u, scale_v, target_rotation,
                     target_offset_x, target_offset_y, target_mat, ppm, me)
    if bm is not None:
        cache_single_face(target_face, bm, ppm, me)

    return True


def set_uv_from_other_face(source_face, target_face, uv_layer, ppm, me, obj_matrix, bm=None,
                            source_uv_layer=None, source_me=None, source_to_target=None):
    """Copy UV settings from source face to target face with proper rotation/offset handling.

    Extracts transform parameters from the source face and delegates to
    set_uv_from_source_params().

    Args:
        source_face: BMesh face to copy UV settings from
        target_face: BMesh face to apply UV settings to
        uv_layer: BMesh UV layer (target face's layer)
        ppm: Pixels per meter setting
        me: Mesh data for target face
        obj_matrix: Target object world matrix (for parallel plane reference calculation)
        bm: BMesh instance (optional, for cache_single_face per-layer caching)
        source_uv_layer: UV layer for source face (cross-object only; defaults to uv_layer)
        source_me: Mesh data for source face (cross-object only; defaults to me)
        source_to_target: 4x4 matrix from source object space to target object space
    """
    # ---- Cross-object: resolve source data references ----
    src_uv = source_uv_layer if source_uv_layer is not None else uv_layer
    src_me = source_me if source_me is not None else me

    source_transform = derive_transform_from_uvs(source_face, src_uv, ppm, src_me)
    if not source_transform:
        return False

    scale_u = source_transform['scale_u']
    scale_v = source_transform['scale_v']

    # Zero scale means source has collapsed/zero-area UVs — can't derive settings
    if abs(scale_u) < 1e-8 or abs(scale_v) < 1e-8:
        return False
    source_rotation = source_transform['rotation']

    source_normal = source_face.normal.normalized()
    source_axes = get_face_local_axes(source_face)
    if not source_axes:
        return False
    source_u, source_v = source_axes

    # Choose the best reference point for offset calculation:
    # prefer a shared vertex (exact match), fall back to projecting source's first vert
    source_verts = set(source_face.verts)
    target_verts = set(target_face.verts)
    shared_verts = source_verts & target_verts

    if len(shared_verts) >= 1:
        shared_vert = list(shared_verts)[0]
        ref_point_co = shared_vert.co.copy()
        # Find UV of shared vert in source
        ref_point_uv = None
        for loop in source_face.loops:
            if loop.vert == shared_vert:
                ref_point_uv = loop[src_uv].uv.copy()
                break
        if ref_point_uv is None:
            ref_point_uv = list(source_face.loops)[0][src_uv].uv.copy()
            ref_point_co = list(source_face.loops)[0].vert.co.copy()
    else:
        source_loop_0 = list(source_face.loops)[0]
        ref_point_co = source_loop_0.vert.co.copy()
        ref_point_uv = source_loop_0[src_uv].uv.copy()

    # ---- Cross-object: transform source geometry to target's local space ----
    if source_to_target is not None:
        rot = source_to_target.to_3x3()
        source_normal = (rot @ source_normal).normalized()
        source_u = (rot @ source_u).normalized()
        source_v = (rot @ source_v).normalized()
        ref_point_co = source_to_target @ ref_point_co

    return set_uv_from_source_params(
        target_face, uv_layer, ppm, me, obj_matrix,
        scale_u, scale_v, source_rotation,
        source_normal, source_u, source_v,
        ref_point_co, ref_point_uv,
        bm=bm,
    )


def _transfer_affine_across_edge(M_source, source_axes, target_axes,
                                  shared_edge_verts, source_face, target_face,
                                  source_normal, target_normal):
    """Transfer an affine matrix from source face to target face via a shared edge.

    Uses the shared edge as a bridge: decomposes the source matrix into
    edge-parallel and edge-perpendicular components, then re-expresses those
    in the target face's local 2D coordinate system.

    Args:
        M_source: 2x2 Matrix mapping source local 2D → UV.
        source_axes: (local_x, local_y) of source face (3D Vectors).
        target_axes: (local_x, local_y) of target face (3D Vectors).
        shared_edge_verts: Tuple of 2 BMVert on the shared edge.
        source_face: Source BMFace (for centroid calculation).
        target_face: Target BMFace (for centroid calculation).
        source_normal: Source face normal (3D Vector).
        target_normal: Target face normal (3D Vector).

    Returns:
        2x2 Matrix mapping target local 2D → UV, or None on failure.
    """
    from mathutils import Matrix

    src_x, src_y = source_axes
    tgt_x, tgt_y = target_axes

    # Edge direction in 3D
    e_3d = (shared_edge_verts[1].co - shared_edge_verts[0].co).normalized()

    # Edge direction in each face's local 2D
    e_s = Vector((e_3d.dot(src_x), e_3d.dot(src_y)))
    e_t = Vector((e_3d.dot(tgt_x), e_3d.dot(tgt_y)))

    if e_s.length < 1e-8 or e_t.length < 1e-8:
        return None

    # Perpendicular to edge in each face's plane, using normal x edge.
    # This gives a consistent direction: for coplanar faces both perpendiculars
    # point the same way in 3D; for corners they naturally "unfold".
    p_s_3d = source_normal.cross(e_3d)
    if p_s_3d.length < 1e-8:
        return None
    p_s_3d.normalize()
    p_s = Vector((p_s_3d.dot(src_x), p_s_3d.dot(src_y)))

    p_t_3d = target_normal.cross(e_3d)
    if p_t_3d.length < 1e-8:
        return None
    p_t_3d.normalize()
    p_t = Vector((p_t_3d.dot(tgt_x), p_t_3d.dot(tgt_y)))

    # Build basis matrices (columns = edge dir, perp dir)
    B_source = Matrix(((e_s.x, p_s.x), (e_s.y, p_s.y)))
    B_target = Matrix(((e_t.x, p_t.x), (e_t.y, p_t.y)))

    try:
        B_target_inv = B_target.inverted()
    except ValueError:
        return None

    # M_target = M_source @ B_source @ inv(B_target)
    return M_source @ B_source @ B_target_inv


def set_uv_from_other_face_affine(source_face, target_face, uv_layer, ppm, me,
                                   shared_edge, bm,
                                   source_uv_layer=None, source_me=None):
    """Transfer UV from source face to target face using affine transform.

    Extracts the affine transform from the source tessellation triangle
    touching the shared edge and transfers it to the target face via an
    edge-based basis change.

    Args:
        source_face: BMesh face to copy UV from.
        target_face: BMesh face to apply UV to.
        uv_layer: BMesh UV layer (target).
        ppm: Pixels per meter setting.
        me: Mesh data for target face.
        shared_edge: BMEdge shared between source and target.
        bm: BMesh instance (for tessellation and cache).
        source_uv_layer: UV layer for source face (defaults to uv_layer).
        source_me: Mesh data for source face (defaults to me).

    Returns:
        True on success, False on failure.
    """
    from ..utils import extract_affine_from_face, get_face_local_axes
    from ..properties import apply_affine_to_face

    src_uv = source_uv_layer if source_uv_layer is not None else uv_layer

    shared_edge_verts = (shared_edge.verts[0], shared_edge.verts[1])

    # Get tessellation from bmesh
    loop_triangles = bm.calc_loop_triangles()

    # Extract affine from source face using triangle on shared edge
    affine_result = extract_affine_from_face(
        source_face, src_uv, shared_edge_verts, loop_triangles
    )
    if affine_result is None:
        return False

    M_source, t_source, source_axes = affine_result

    source_normal = source_face.normal.normalized()
    target_normal = target_face.normal.normalized()
    target_axes = get_face_local_axes(target_face)
    if not target_axes:
        return False

    # Transfer affine across the shared edge
    M_target = _transfer_affine_across_edge(
        M_source, source_axes, target_axes,
        shared_edge_verts, source_face, target_face,
        source_normal, target_normal,
    )
    if M_target is None:
        return False

    # Compute offset by anchoring to a shared vertex
    anchor_vert = shared_edge_verts[0]
    anchor_uv = None
    for loop in source_face.loops:
        if loop.vert == anchor_vert:
            anchor_uv = loop[src_uv].uv.copy()
            break
    if anchor_uv is None:
        return False

    # Shared vert's position in target face's local 2D
    tgt_x, tgt_y = target_axes
    target_first_vert = list(target_face.loops)[0].vert.co
    delta = anchor_vert.co - target_first_vert
    anchor_local_2d = Vector((delta.dot(tgt_x), delta.dot(tgt_y)))

    t_target = anchor_uv - M_target @ anchor_local_2d

    debug_log(f"[AffineUV] face {target_face.index} | M=[{M_target[0][0]:.4f},{M_target[0][1]:.4f};"
              f"{M_target[1][0]:.4f},{M_target[1][1]:.4f}] t=({t_target.x:.4f},{t_target.y:.4f})")

    apply_affine_to_face(target_face, uv_layer, M_target, t_target, me)
    cache_single_face(target_face, bm, ppm, me)

    return True


def _dispatch_set_uv_from_other_face(source_face, target_face, uv_layer, ppm, me, obj_matrix,
                           bm=None, source_uv_layer=None, source_me=None,
                           source_to_target=None):
    """Choose and apply the best UV transfer method for the given faces.

    Affine transformation is in theory a superset of the functionality that
    scalar transformation (using our scale / offset / rotation abstraction)
    offers.  It has not been well tested, so we use it only in cases we
    *know* that scalar transformation will fail.

    Signature matches set_uv_from_other_face for use as a drop-in callback.
    """
    from ..utils import needs_affine_transfer

    src_uv = source_uv_layer if source_uv_layer is not None else uv_layer

    # Check whether affine is needed and whether we have a shared edge
    use_affine = False
    shared_edge = None
    if source_to_target is None and needs_affine_transfer(source_face, src_uv):
        source_edges = set(source_face.edges)
        target_edges = set(target_face.edges)
        shared_edges = source_edges & target_edges
        if shared_edges:
            shared_edge = next(iter(shared_edges))
            use_affine = True

    if use_affine:
        debug_log("Level Design Tools: Using affine UV transfer")
        success = set_uv_from_other_face_affine(
            source_face, target_face, uv_layer, ppm, me,
            shared_edge, bm,
            source_uv_layer=source_uv_layer, source_me=source_me,
        )
        if success:
            return True
        debug_log("Level Design Tools: Affine UV transfer failed, falling back to scalar")

    debug_log("Level Design Tools: Using scalar UV transfer")
    return set_uv_from_other_face(
        source_face, target_face, uv_layer, ppm, me, obj_matrix,
        bm=bm, source_uv_layer=source_uv_layer, source_me=source_me,
        source_to_target=source_to_target,
    )


def _get_top_edge_index(face, obj_matrix):
    """Return the loop index of the edge highest in world space.

    The "top" edge is the one whose two vertices have the highest average
    world-space Z coordinate.
    """
    loops = list(face.loops)
    n = len(loops)
    best_idx = 0
    best_z = -float('inf')
    for i in range(n):
        a = obj_matrix @ loops[i].vert.co
        b = obj_matrix @ loops[(i + 1) % n].vert.co
        avg_z = (a.z + b.z) * 0.5
        if avg_z > best_z:
            best_z = avg_z
            best_idx = i
    return best_idx


def _get_bottom_edge_index(face, obj_matrix):
    """Return the loop index of the edge lowest in world space."""
    loops = list(face.loops)
    n = len(loops)
    best_idx = 0
    best_z = float('inf')
    for i in range(n):
        a = obj_matrix @ loops[i].vert.co
        b = obj_matrix @ loops[(i + 1) % n].vert.co
        avg_z = (a.z + b.z) * 0.5
        if avg_z < best_z:
            best_z = avg_z
            best_idx = i
    return best_idx


def stretch_uv_from_other_face(source_face, target_face, uv_layer, ppm, me, obj_matrix, bm=None,
                                source_uv_layer=None, source_me=None, source_to_target=None):
    """Apply UV from source face stretched to fit target face.

    Copies the source face's UV region directly onto the target face.
    For quad-to-quad, source UVs are copied with a vertex offset so the
    visually top edge matches between source and target.
    For non-quad faces, target vertices are projected into the source UV
    bounding rectangle proportionally.
    """
    src_uv = source_uv_layer if source_uv_layer is not None else uv_layer

    source_loops = list(source_face.loops)
    target_loops = list(target_face.loops)
    source_uvs = [loop[src_uv].uv.copy() for loop in source_loops]

    # Determine object matrices for world-space comparisons
    source_obj_matrix = obj_matrix
    if source_to_target is not None:
        # source_to_target = target_world_inv @ source_world
        # so source_world = target_world @ source_to_target
        source_obj_matrix = obj_matrix @ source_to_target

    n_src = len(source_loops)
    n_tgt = len(target_loops)

    if n_src == 4 and n_tgt == 4:
        # Quad-to-quad: copy UVs directly with a rotation offset to match top edges
        src_top = _get_top_edge_index(source_face, source_obj_matrix)
        tgt_top = _get_top_edge_index(target_face, obj_matrix)
        offset = tgt_top - src_top

        for i in range(4):
            src_idx = (i - offset) % 4
            target_loops[i][uv_layer].uv = source_uvs[src_idx].copy()

        debug_log(f"[StretchApply] quad-to-quad face {target_face.index} | "
                  f"src_top={src_top} tgt_top={tgt_top} offset={offset}")
    else:
        # Non-quad: map target vertices into source UV bounding rectangle
        min_u = min(uv.x for uv in source_uvs)
        max_u = max(uv.x for uv in source_uvs)
        min_v = min(uv.y for uv in source_uvs)
        max_v = max(uv.y for uv in source_uvs)
        uv_width = max_u - min_u
        uv_height = max_v - min_v
        if uv_width < 1e-8 or uv_height < 1e-8:
            return False

        # Derive local axes from the bottom edge of the target face
        bot_idx = _get_bottom_edge_index(target_face, obj_matrix)
        n = len(target_loops)
        edge_vec = target_loops[(bot_idx + 1) % n].vert.co - target_loops[bot_idx].vert.co
        if edge_vec.length < 1e-8:
            return False
        local_x = edge_vec.normalized()
        normal = target_face.normal.normalized()
        local_y = normal.cross(local_x).normalized()

        # Project all target vertices onto local axes
        ref = target_loops[0].vert.co
        xs = []
        ys = []
        for loop in target_loops:
            delta = loop.vert.co - ref
            xs.append(delta.dot(local_x))
            ys.append(delta.dot(local_y))

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range < 1e-8 or y_range < 1e-8:
            return False

        # Map normalized position into source UV bounding rectangle
        for i, loop in enumerate(target_loops):
            t_x = (xs[i] - x_min) / x_range
            t_y = (ys[i] - y_min) / y_range
            loop[uv_layer].uv.x = min_u + t_x * uv_width
            loop[uv_layer].uv.y = min_v + t_y * uv_height

        debug_log(f"[StretchApply] non-quad face {target_face.index} | "
                  f"src_uv_box=({min_u:.4f},{min_v:.4f})-({max_u:.4f},{max_v:.4f})")

    if me.is_editmode:
        bmesh.update_edit_mesh(me)
    if bm is not None:
        cache_single_face(target_face, bm, ppm, me)

    return True


# ---- Shared helpers for apply/stretch-apply paint operators ----

def _invoke_apply_setup(op, context, event):
    """Common invoke setup for apply paint operators.

    Validates preconditions, stores paint session state on `op`, and builds
    cross-object BVH trees. Returns None on success or a Blender result set
    (e.g. {'PASS_THROUGH'}) on failure.
    """
    obj = context.object
    if not obj or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
        debug_log("[ApplyImage] PASS_THROUGH: no mesh object or not in EDIT_MESH mode")
        return {'PASS_THROUGH'}

    if not context.tool_settings.mesh_select_mode[2]:
        debug_log("[ApplyImage] PASS_THROUGH: not in face select mode")
        return {'PASS_THROUGH'}

    image = get_active_image()
    if not image:
        debug_log("[ApplyImage] PASS_THROUGH: no active image in file browser")
        return {'PASS_THROUGH'}

    bm_check = bmesh.from_edit_mesh(obj.data)
    bm_check.faces.ensure_lookup_table()
    selected_count = sum(1 for f in bm_check.faces if f.select)
    if selected_count != 1:
        debug_log(f"[ApplyImage] PASS_THROUGH: need exactly 1 face selected, got {selected_count}")
        return {'PASS_THROUGH'}

    source_face = bm_check.faces.active
    if source_face is None or not source_face.select:
        debug_log(f"[ApplyImage] PASS_THROUGH: active face is None ({source_face is None}) or not selected")
        return {'PASS_THROUGH'}

    op._source_face_index = source_face.index
    op._image = image
    op._obj_matrix = obj.matrix_world.copy()

    props = context.scene.level_design_props
    op._ppm = props.pixels_per_meter
    op._auto_hotspot = obj.anvil_auto_hotspot
    op._allow_combined_faces = obj.anvil_allow_combined_faces
    op._size_weight = obj.anvil_hotspot_size_weight
    op._painted_face_indices = set()
    op._faces_previously_hotspottable = set()

    mat = find_material_with_image(image)
    if mat is None:
        mat = create_material_with_image(image)
    op._mat = mat
    if mat.name not in obj.data.materials:
        obj.data.materials.append(mat)
    op._mat_index = obj.data.materials.find(mat.name)

    op._other_objects_info = []
    op._other_bmeshes = {}
    op._paint_visited_other = set()
    for other_obj in context.view_layer.objects:
        if other_obj == obj or other_obj.type != 'MESH' or not other_obj.visible_get():
            continue
        other_me = other_obj.data
        if other_me is None or len(other_me.polygons) == 0:
            continue
        bvh = BVHTree.FromPolygons(
            [v.co for v in other_me.vertices],
            [p.vertices for p in other_me.polygons]
        )
        op._other_objects_info.append({
            'obj': other_obj,
            'bvh': bvh,
            'polygons': other_me.polygons,
            'materials': other_me.materials,
        })

    debug_log(f"[ApplyImage] invoke OK: source_face={op._source_face_index}, image={image.name}, mat={mat.name}")
    return None


def _paint_sample_impl(op, context, mouse_2d, region, rv3d, uv_func):
    """Shared paint_sample implementation for apply paint operators."""
    obj = op._paint_obj
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()

    origin_local, dir_local = op._paint_ray_local(region, rv3d, mouse_2d)

    location, normal, face_index, distance = raycast_bvh_skip_backfaces(
        op._paint_bvh, origin_local, dir_local,
        bm, me.materials, max_iterations=64
    )

    # ---- Cross-object raycast: find closest hit among other visible meshes ----
    hit_other_obj = None
    hit_other_face_index = None
    if op._other_objects_info:
        best_distance = distance if face_index is not None else float('inf')

        coord = (mouse_2d.x, mouse_2d.y)
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        for info in op._other_objects_info:
            other_obj = info['obj']
            matrix_inv = other_obj.matrix_world.inverted()
            origin_other = matrix_inv @ ray_origin
            dir_other = (matrix_inv.to_3x3() @ view_vector).normalized()

            loc, norm, fidx, dist = raycast_bvh_skip_backfaces_polys(
                info['bvh'], origin_other, dir_other,
                info['polygons'], info['materials'], max_iterations=64
            )

            if fidx is not None and dist < best_distance:
                best_distance = dist
                hit_other_obj = other_obj
                hit_other_face_index = fidx

    # ---- Apply to other object if it was the closest hit ----
    if hit_other_obj is not None:
        visit_key = (id(hit_other_obj), hit_other_face_index)
        if visit_key not in op._paint_visited_other:
            op._paint_visited_other.add(visit_key)

            other_me = hit_other_obj.data
            obj_id = id(hit_other_obj)

            if obj_id not in op._other_bmeshes:
                other_bm = bmesh.new()
                other_bm.from_mesh(other_me)
                op._other_bmeshes[obj_id] = {
                    'bm': other_bm,
                    'obj': hit_other_obj,
                }
            other_data = op._other_bmeshes[obj_id]
            other_bm = other_data['bm']
            other_bm.faces.ensure_lookup_table()

            if op._mat.name not in other_me.materials:
                other_me.materials.append(op._mat)
            other_mat_index = other_me.materials.find(op._mat.name)

            target_face = other_bm.faces[hit_other_face_index]
            source_face = bm.faces[op._source_face_index]

            target_face.material_index = other_mat_index

            from ..utils import get_render_active_uv_layer
            source_uv = get_render_active_uv_layer(bm, me)
            if source_uv is None:
                source_uv = bm.loops.layers.uv.verify()
            target_uv = other_bm.loops.layers.uv.verify()

            source_to_target = hit_other_obj.matrix_world.inverted() @ op._paint_obj.matrix_world

            uv_func(
                source_face, target_face, target_uv,
                op._ppm, other_me, hit_other_obj.matrix_world,
                bm=bm, source_uv_layer=source_uv, source_me=me,
                source_to_target=source_to_target,
            )

            debug_log(f"[ApplyImage] cross-object hit: {hit_other_obj.name} face {hit_other_face_index}")
        return

    if face_index is None:
        debug_log(f"[ApplyImage] raycast miss at mouse ({mouse_2d.x:.0f}, {mouse_2d.y:.0f}) - no face hit")
        return

    debug_log(f"[ApplyImage] raycast hit face {face_index} at distance {distance:.4f}, mouse ({mouse_2d.x:.0f}, {mouse_2d.y:.0f})")
    if face_index in op._paint_visited:
        return
    op._paint_visited.add(face_index)

    target_face = bm.faces[face_index]
    source_face = bm.faces[op._source_face_index]

    from ..utils import get_render_active_uv_layer
    uv_layer = get_render_active_uv_layer(bm, me)
    if uv_layer is None:
        uv_layer = bm.loops.layers.uv.verify()

    if face_has_hotspot_material(target_face, me):
        op._faces_previously_hotspottable.add(face_index)

    target_face.material_index = op._mat_index

    uv_func(source_face, target_face, uv_layer, op._ppm, me, op._obj_matrix, bm=bm)

    op._painted_face_indices.add(face_index)


def _flush_other_bmeshes(op):
    """Write back and free all cross-object bmeshes."""
    for data in op._other_bmeshes.values():
        data['bm'].to_mesh(data['obj'].data)
        data['bm'].free()
    op._other_bmeshes.clear()


def _discard_other_bmeshes(op):
    """Free all cross-object bmeshes without writing back."""
    for data in op._other_bmeshes.values():
        data['bm'].free()
    op._other_bmeshes.clear()


def _invoke_uv_transform_setup(op, context, event):
    """Common invoke setup for UV-transform-only paint operators.

    Like _invoke_apply_setup but does not require a file browser image and
    does not touch materials.  Only the UV transform (offset, rotation, scale)
    from the selected source face is used.

    Returns None on success or a Blender result set on failure.
    """
    obj = context.object
    if not obj or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
        debug_log("[UVTransform] PASS_THROUGH: no mesh object or not in EDIT_MESH mode")
        return {'PASS_THROUGH'}

    if not context.tool_settings.mesh_select_mode[2]:
        debug_log("[UVTransform] PASS_THROUGH: not in face select mode")
        return {'PASS_THROUGH'}

    bm_check = bmesh.from_edit_mesh(obj.data)
    bm_check.faces.ensure_lookup_table()
    selected_count = sum(1 for f in bm_check.faces if f.select)
    if selected_count != 1:
        debug_log(f"[UVTransform] PASS_THROUGH: need exactly 1 face selected, got {selected_count}")
        return {'PASS_THROUGH'}

    source_face = bm_check.faces.active
    if source_face is None or not source_face.select:
        debug_log(f"[UVTransform] PASS_THROUGH: active face is None ({source_face is None}) or not selected")
        return {'PASS_THROUGH'}

    op._source_face_index = source_face.index
    op._obj_matrix = obj.matrix_world.copy()

    props = context.scene.level_design_props
    op._ppm = props.pixels_per_meter
    op._painted_face_indices = set()

    op._other_objects_info = []
    op._other_bmeshes = {}
    op._paint_visited_other = set()
    for other_obj in context.view_layer.objects:
        if other_obj == obj or other_obj.type != 'MESH' or not other_obj.visible_get():
            continue
        other_me = other_obj.data
        if other_me is None or len(other_me.polygons) == 0:
            continue
        bvh = BVHTree.FromPolygons(
            [v.co for v in other_me.vertices],
            [p.vertices for p in other_me.polygons]
        )
        op._other_objects_info.append({
            'obj': other_obj,
            'bvh': bvh,
            'polygons': other_me.polygons,
            'materials': other_me.materials,
        })

    debug_log(f"[UVTransform] invoke OK: source_face={op._source_face_index}")
    return None


def _paint_sample_uv_transform_impl(op, context, mouse_2d, region, rv3d):
    """Paint sample implementation for UV-transform-only operators.

    Like _paint_sample_impl but does not change materials on target faces.
    """
    obj = op._paint_obj
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()

    origin_local, dir_local = op._paint_ray_local(region, rv3d, mouse_2d)

    location, normal, face_index, distance = raycast_bvh_skip_backfaces(
        op._paint_bvh, origin_local, dir_local,
        bm, me.materials, max_iterations=64
    )

    # ---- Cross-object raycast: find closest hit among other visible meshes ----
    hit_other_obj = None
    hit_other_face_index = None
    if op._other_objects_info:
        best_distance = distance if face_index is not None else float('inf')

        coord = (mouse_2d.x, mouse_2d.y)
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        for info in op._other_objects_info:
            other_obj = info['obj']
            matrix_inv = other_obj.matrix_world.inverted()
            origin_other = matrix_inv @ ray_origin
            dir_other = (matrix_inv.to_3x3() @ view_vector).normalized()

            loc, norm, fidx, dist = raycast_bvh_skip_backfaces_polys(
                info['bvh'], origin_other, dir_other,
                info['polygons'], info['materials'], max_iterations=64
            )

            if fidx is not None and dist < best_distance:
                best_distance = dist
                hit_other_obj = other_obj
                hit_other_face_index = fidx

    # ---- Apply to other object if it was the closest hit ----
    if hit_other_obj is not None:
        visit_key = (id(hit_other_obj), hit_other_face_index)
        if visit_key not in op._paint_visited_other:
            op._paint_visited_other.add(visit_key)

            other_me = hit_other_obj.data
            obj_id = id(hit_other_obj)

            if obj_id not in op._other_bmeshes:
                other_bm = bmesh.new()
                other_bm.from_mesh(other_me)
                op._other_bmeshes[obj_id] = {
                    'bm': other_bm,
                    'obj': hit_other_obj,
                }
            other_data = op._other_bmeshes[obj_id]
            other_bm = other_data['bm']
            other_bm.faces.ensure_lookup_table()

            target_face = other_bm.faces[hit_other_face_index]
            source_face = bm.faces[op._source_face_index]

            # No material change — only UV transform

            from ..utils import get_render_active_uv_layer
            source_uv = get_render_active_uv_layer(bm, me)
            if source_uv is None:
                source_uv = bm.loops.layers.uv.verify()
            target_uv = other_bm.loops.layers.uv.verify()

            source_to_target = hit_other_obj.matrix_world.inverted() @ op._paint_obj.matrix_world

            _dispatch_set_uv_from_other_face(
                source_face, target_face, target_uv,
                op._ppm, other_me, hit_other_obj.matrix_world,
                bm=bm, source_uv_layer=source_uv, source_me=me,
                source_to_target=source_to_target,
            )

            debug_log(f"[UVTransform] cross-object hit: {hit_other_obj.name} face {hit_other_face_index}")
        return

    if face_index is None:
        debug_log(f"[UVTransform] raycast miss at mouse ({mouse_2d.x:.0f}, {mouse_2d.y:.0f}) - no face hit")
        return

    debug_log(f"[UVTransform] raycast hit face {face_index} at distance {distance:.4f}")
    if face_index in op._paint_visited:
        return
    op._paint_visited.add(face_index)

    target_face = bm.faces[face_index]
    source_face = bm.faces[op._source_face_index]

    from ..utils import get_render_active_uv_layer
    uv_layer = get_render_active_uv_layer(bm, me)
    if uv_layer is None:
        uv_layer = bm.loops.layers.uv.verify()

    # No material change — only UV transform

    _dispatch_set_uv_from_other_face(source_face, target_face, uv_layer, op._ppm, me, op._obj_matrix, bm=bm)

    op._painted_face_indices.add(face_index)


def _pick_source_from_cursor(op, context, event):
    """Raycast to find source face under cursor. Shared by pick operators.

    Returns (hit_obj, hit_face_index, image) on success,
    or sets op.report and returns None on failure.
    """
    edit_obj = context.object
    if not edit_obj or edit_obj.type != 'MESH' or context.mode != 'EDIT_MESH':
        return None

    bm_edit = bmesh.from_edit_mesh(edit_obj.data)
    # Ensure layers exist before collecting face references (creating layers invalidates refs)
    from ..utils import get_face_id_layer, get_render_active_uv_layer
    uv_layer = get_render_active_uv_layer(bm_edit, edit_obj.data)
    if uv_layer is None:
        bm_edit.loops.layers.uv.verify()
    get_face_id_layer(bm_edit)
    bm_edit.faces.ensure_lookup_table()
    selected_faces = [f for f in bm_edit.faces if f.select]
    if not selected_faces:
        return None

    region = context.region
    rv3d = context.region_data
    coord = (event.mouse_region_x, event.mouse_region_y)

    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

    closest_distance = float('inf')
    hit_obj = None
    hit_mat_index = None
    hit_face_index = None

    for obj in context.view_layer.objects:
        if obj.type != 'MESH' or not obj.visible_get():
            continue
        me = obj.data
        if me is None:
            continue

        matrix_inv = obj.matrix_world.inverted()
        ray_origin_local = matrix_inv @ ray_origin
        ray_direction_local = (matrix_inv.to_3x3() @ view_vector).normalized()

        if obj.mode == 'EDIT' and obj == context.object:
            bm = bmesh.from_edit_mesh(me)
            bm.faces.ensure_lookup_table()
            bvh = BVHTree.FromBMesh(bm)

            location, normal, face_index, distance = raycast_bvh_skip_backfaces(
                bvh, ray_origin_local, ray_direction_local,
                bm, me.materials, max_iterations=64
            )

            if face_index is not None and distance < closest_distance:
                closest_distance = distance
                hit_obj = obj
                hit_mat_index = bm.faces[face_index].material_index
                hit_face_index = face_index
        else:
            depsgraph = context.evaluated_depsgraph_get()
            obj_eval = obj.evaluated_get(depsgraph)
            me_eval = obj_eval.to_mesh()
            if me_eval is None or len(me_eval.polygons) == 0:
                if me_eval:
                    obj_eval.to_mesh_clear()
                continue

            bvh = BVHTree.FromPolygons(
                [v.co for v in me_eval.vertices],
                [p.vertices for p in me_eval.polygons]
            )

            location, normal, face_index, distance = raycast_bvh_skip_backfaces_polys(
                bvh, ray_origin_local, ray_direction_local,
                me_eval.polygons, me_eval.materials, max_iterations=64
            )

            if face_index is not None and distance < closest_distance:
                closest_distance = distance
                hit_obj = obj
                hit_mat_index = me_eval.polygons[face_index].material_index
                hit_face_index = face_index

            obj_eval.to_mesh_clear()

    if hit_obj is None:
        op.report({'WARNING'}, "No face under cursor")
        return None

    hit_me = hit_obj.data
    hit_mat = hit_me.materials[hit_mat_index] if hit_mat_index < len(hit_me.materials) else None
    if not hit_mat:
        op.report({'WARNING'}, "Face has no material")
        return None

    image = get_image_from_material(hit_mat)
    if not image:
        op.report({'WARNING'}, "Material has no image texture")
        return None

    return hit_obj, hit_face_index, image, selected_faces, bm_edit


# ---- Apply Image to Face (Alt+Left Click) ----

class apply_image_to_face(ModalPaintBase, Operator):
    """Apply selected File Browser image to hovered face (drag to paint)"""
    bl_idname = "leveldesign.apply_image_to_face"
    bl_label = "Apply Image to Face"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        result = _invoke_apply_setup(self, context, event)
        if result is not None:
            return result
        return self._invoke_paint(context, event)

    def modal(self, context, event):
        return self._modal_paint(context, event)

    def paint_begin(self, context, event):
        return True

    def paint_cancel(self, context):
        _discard_other_bmeshes(self)

    def paint_sample(self, context, mouse_2d, region, rv3d):
        _paint_sample_impl(self, context, mouse_2d, region, rv3d, _dispatch_set_uv_from_other_face)

    def paint_finish(self, context):
        _flush_other_bmeshes(self)

        if not self._painted_face_indices:
            return

        from ..hotspot_mapping.json_storage import is_texture_hotspottable
        from .uv_tools import apply_hotspots_to_mesh

        obj = self._paint_obj
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()

        from ..utils import get_render_active_uv_layer
        uv_layer = get_render_active_uv_layer(bm, me)
        if uv_layer is None:
            uv_layer = bm.loops.layers.uv.verify()

        # Ensure layers exist before collecting face references (creating layers invalidates refs)
        get_face_id_layer(bm)

        new_is_hotspottable = is_texture_hotspottable(self._image.name)

        if self._auto_hotspot and new_is_hotspottable:
            all_hotspot_faces = get_all_hotspot_faces(bm, me)

            if all_hotspot_faces:
                id_layer = get_face_id_layer(bm)
                selected_ids, active_id = save_face_selection(bm, id_layer)

                apply_hotspots_to_mesh(
                    bm, me, all_hotspot_faces,
                    self._allow_combined_faces, self._obj_matrix,
                    self._ppm, self._size_weight
                )

                restore_face_selection(bm, id_layer, selected_ids, active_id)

                for face in all_hotspot_faces:
                    if face.is_valid:
                        cache_single_face(face, bm, self._ppm, me)
        elif (self._auto_hotspot and not new_is_hotspottable
              and self._faces_previously_hotspottable):
            has_connected = False
            for fi in self._painted_face_indices:
                if fi < len(bm.faces) and any_connected_face_has_hotspot(bm.faces[fi], me):
                    has_connected = True
                    break

            if has_connected:
                all_hotspot_faces = get_all_hotspot_faces(bm, me)

                if all_hotspot_faces:
                    id_layer = get_face_id_layer(bm)
                    selected_ids, active_id = save_face_selection(bm, id_layer)

                    apply_hotspots_to_mesh(
                        bm, me, all_hotspot_faces,
                        self._allow_combined_faces, self._obj_matrix,
                        self._ppm, self._size_weight
                    )

                    restore_face_selection(bm, id_layer, selected_ids, active_id)

                    for face in all_hotspot_faces:
                        if face.is_valid:
                            cache_single_face(face, bm, self._ppm, me)

        update_ui_from_selection(context)
        update_active_image_from_face(context)
        redraw_ui_panels(context)


# ---- Stretch Apply Image to Face (Shift+Alt+Left Click) ----

class stretch_apply_image_to_face(ModalPaintBase, Operator):
    """Stretch-apply selected File Browser image to hovered face (drag to paint)"""
    bl_idname = "leveldesign.stretch_apply_image_to_face"
    bl_label = "Stretch Apply Image to Face"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        result = _invoke_apply_setup(self, context, event)
        if result is not None:
            return result
        return self._invoke_paint(context, event)

    def modal(self, context, event):
        return self._modal_paint(context, event)

    def paint_begin(self, context, event):
        return True

    def paint_cancel(self, context):
        _discard_other_bmeshes(self)

    def paint_sample(self, context, mouse_2d, region, rv3d):
        _paint_sample_impl(self, context, mouse_2d, region, rv3d, stretch_uv_from_other_face)

    def paint_finish(self, context):
        _flush_other_bmeshes(self)
        if not self._painted_face_indices:
            return
        update_ui_from_selection(context)
        update_active_image_from_face(context)
        redraw_ui_panels(context)


# ---- Pick Image from Face (Alt+Right Click) ----

class pick_image_from_face(Operator):
    """Pick texture from hovered face and apply to selected faces"""
    bl_idname = "leveldesign.pick_image_from_face"
    bl_label = "Pick and Apply Texture"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        from ..hotspot_mapping.json_storage import is_texture_hotspottable
        from .uv_tools import apply_hotspots_to_mesh

        pick_result = _pick_source_from_cursor(self, context, event)
        if pick_result is None:
            return {'PASS_THROUGH'}
        hit_obj, hit_face_index, image, selected_faces, bm_edit = pick_result

        edit_obj = context.object
        me = edit_obj.data
        from ..utils import get_render_active_uv_layer
        uv_layer = get_render_active_uv_layer(bm_edit, me)
        if uv_layer is None:
            uv_layer = bm_edit.loops.layers.uv.verify()
        props = context.scene.level_design_props
        ppm = props.pixels_per_meter

        faces_with_previous_hotspot = [f for f in selected_faces if face_has_hotspot_material(f, me)]
        any_previous_was_hotspottable = len(faces_with_previous_hotspot) > 0

        any_connected_has_hotspot = False
        for f in selected_faces:
            if any_connected_face_has_hotspot(f, me):
                any_connected_has_hotspot = True
                break

        mat = find_material_with_image(image)
        if mat is None:
            mat = create_material_with_image(image)
        if mat.name not in me.materials:
            me.materials.append(mat)
        mat_index = me.materials.find(mat.name)

        for face in selected_faces:
            face.material_index = mat_index

        is_same_object = (hit_obj == edit_obj)
        _source_bm = None
        if is_same_object:
            _source_face = bm_edit.faces[hit_face_index]
            _source_uv = uv_layer
            _source_me = me
            _source_to_target = None
        else:
            _source_bm = bmesh.new()
            _source_bm.from_mesh(hit_obj.data)
            _source_bm.faces.ensure_lookup_table()
            _source_face = _source_bm.faces[hit_face_index]
            _source_uv = _source_bm.loops.layers.uv.verify()
            _source_me = hit_obj.data
            _source_to_target = edit_obj.matrix_world.inverted() @ hit_obj.matrix_world

        new_is_hotspottable = is_texture_hotspottable(image.name)

        if edit_obj.anvil_auto_hotspot and new_is_hotspottable:
            all_hotspot_faces = get_all_hotspot_faces(bm_edit, me)

            if all_hotspot_faces:
                id_layer = get_face_id_layer(bm_edit)
                selected_ids, active_id = save_face_selection(bm_edit, id_layer)

                allow_combined_faces = edit_obj.anvil_allow_combined_faces
                size_weight = edit_obj.anvil_hotspot_size_weight

                apply_hotspots_to_mesh(
                    bm_edit, me, all_hotspot_faces, allow_combined_faces,
                    edit_obj.matrix_world, ppm, size_weight
                )

                restore_face_selection(bm_edit, id_layer, selected_ids, active_id)

                for face in all_hotspot_faces:
                    if face.is_valid:
                        cache_single_face(face, bm_edit, ppm, me)
        else:
            if (edit_obj.anvil_auto_hotspot and not new_is_hotspottable
                    and any_previous_was_hotspottable and any_connected_has_hotspot):
                all_hotspot_faces = get_all_hotspot_faces(bm_edit, me)

                if all_hotspot_faces:
                    id_layer = get_face_id_layer(bm_edit)
                    selected_ids, active_id = save_face_selection(bm_edit, id_layer)

                    allow_combined_faces = edit_obj.anvil_allow_combined_faces
                    size_weight = edit_obj.anvil_hotspot_size_weight

                    apply_hotspots_to_mesh(
                        bm_edit, me, all_hotspot_faces, allow_combined_faces,
                        edit_obj.matrix_world, ppm, size_weight
                    )

                    restore_face_selection(bm_edit, id_layer, selected_ids, active_id)

                    for face in all_hotspot_faces:
                        if face.is_valid:
                            cache_single_face(face, bm_edit, ppm, me)

            for target_face in selected_faces:
                set_uv_from_other_face(
                    _source_face, target_face, uv_layer,
                    ppm, me, edit_obj.matrix_world,
                    bm=bm_edit,
                    source_uv_layer=_source_uv,
                    source_me=_source_me,
                    source_to_target=_source_to_target,
                )

        if _source_bm is not None:
            _source_bm.free()

        bmesh.update_edit_mesh(me)

        update_ui_from_selection(context)
        update_active_image_from_face(context)
        redraw_ui_panels(context)
        self.report({'INFO'}, f"Applied: {image.name}")

        return {'FINISHED'}


# ---- Stretch Pick Image from Face (Shift+Alt+Right Click) ----

class stretch_pick_image_from_face(Operator):
    """Stretch-pick texture from hovered face and apply to selected faces"""
    bl_idname = "leveldesign.stretch_pick_image_from_face"
    bl_label = "Stretch Pick and Apply Texture"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        pick_result = _pick_source_from_cursor(self, context, event)
        if pick_result is None:
            return {'PASS_THROUGH'}
        hit_obj, hit_face_index, image, selected_faces, bm_edit = pick_result

        edit_obj = context.object
        me = edit_obj.data
        from ..utils import get_render_active_uv_layer
        uv_layer = get_render_active_uv_layer(bm_edit, me)
        if uv_layer is None:
            uv_layer = bm_edit.loops.layers.uv.verify()
        props = context.scene.level_design_props
        ppm = props.pixels_per_meter

        mat = find_material_with_image(image)
        if mat is None:
            mat = create_material_with_image(image)
        if mat.name not in me.materials:
            me.materials.append(mat)
        mat_index = me.materials.find(mat.name)

        for face in selected_faces:
            face.material_index = mat_index

        is_same_object = (hit_obj == edit_obj)
        _source_bm = None
        if is_same_object:
            _source_face = bm_edit.faces[hit_face_index]
            _source_uv = uv_layer
            _source_me = me
            _source_to_target = None
        else:
            _source_bm = bmesh.new()
            _source_bm.from_mesh(hit_obj.data)
            _source_bm.faces.ensure_lookup_table()
            _source_face = _source_bm.faces[hit_face_index]
            _source_uv = _source_bm.loops.layers.uv.verify()
            _source_me = hit_obj.data
            _source_to_target = edit_obj.matrix_world.inverted() @ hit_obj.matrix_world

        for target_face in selected_faces:
            stretch_uv_from_other_face(
                _source_face, target_face, uv_layer,
                ppm, me, edit_obj.matrix_world,
                bm=bm_edit,
                source_uv_layer=_source_uv,
                source_me=_source_me,
                source_to_target=_source_to_target,
            )

        if _source_bm is not None:
            _source_bm.free()

        bmesh.update_edit_mesh(me)

        update_ui_from_selection(context)
        update_active_image_from_face(context)
        redraw_ui_panels(context)
        self.report({'INFO'}, f"Stretch applied: {image.name}")

        return {'FINISHED'}


# ---- Apply UV Transform to Face (Ctrl+Alt+Left Click) ----

class apply_uv_transform_to_face(ModalPaintBase, Operator):
    """Apply UV transform from selected face to hovered face without changing material (drag to paint)"""
    bl_idname = "leveldesign.apply_uv_transform_to_face"
    bl_label = "Apply UV Transform to Face"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        result = _invoke_uv_transform_setup(self, context, event)
        if result is not None:
            return result
        return self._invoke_paint(context, event)

    def modal(self, context, event):
        return self._modal_paint(context, event)

    def paint_begin(self, context, event):
        return True

    def paint_cancel(self, context):
        _discard_other_bmeshes(self)

    def paint_sample(self, context, mouse_2d, region, rv3d):
        _paint_sample_uv_transform_impl(self, context, mouse_2d, region, rv3d)

    def paint_finish(self, context):
        _flush_other_bmeshes(self)
        if not self._painted_face_indices:
            return
        update_ui_from_selection(context)
        update_active_image_from_face(context)
        redraw_ui_panels(context)


# ---- Pick UV Transform from Face (Ctrl+Alt+Right Click) ----

class pick_uv_transform_from_face(Operator):
    """Pick UV transform from hovered face and apply to selected faces without changing material"""
    bl_idname = "leveldesign.pick_uv_transform_from_face"
    bl_label = "Pick and Apply UV Transform"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_level_design_workspace()

    def invoke(self, context, event):
        pick_result = _pick_source_from_cursor(self, context, event)
        if pick_result is None:
            return {'PASS_THROUGH'}
        hit_obj, hit_face_index, image, selected_faces, bm_edit = pick_result

        edit_obj = context.object
        me = edit_obj.data
        from ..utils import get_render_active_uv_layer
        uv_layer = get_render_active_uv_layer(bm_edit, me)
        if uv_layer is None:
            uv_layer = bm_edit.loops.layers.uv.verify()
        props = context.scene.level_design_props
        ppm = props.pixels_per_meter

        # No material change — only UV transform

        is_same_object = (hit_obj == edit_obj)
        _source_bm = None
        if is_same_object:
            _source_face = bm_edit.faces[hit_face_index]
            _source_uv = uv_layer
            _source_me = me
            _source_to_target = None
        else:
            _source_bm = bmesh.new()
            _source_bm.from_mesh(hit_obj.data)
            _source_bm.faces.ensure_lookup_table()
            _source_face = _source_bm.faces[hit_face_index]
            _source_uv = _source_bm.loops.layers.uv.verify()
            _source_me = hit_obj.data
            _source_to_target = edit_obj.matrix_world.inverted() @ hit_obj.matrix_world

        for target_face in selected_faces:
            set_uv_from_other_face(
                _source_face, target_face, uv_layer,
                ppm, me, edit_obj.matrix_world,
                bm=bm_edit,
                source_uv_layer=_source_uv,
                source_me=_source_me,
                source_to_target=_source_to_target,
            )

        if _source_bm is not None:
            _source_bm.free()

        bmesh.update_edit_mesh(me)

        update_ui_from_selection(context)
        update_active_image_from_face(context)
        redraw_ui_panels(context)
        self.report({'INFO'}, "Applied UV transform")

        return {'FINISHED'}


classes = (
    apply_image_to_face,
    pick_image_from_face,
    stretch_apply_image_to_face,
    stretch_pick_image_from_face,
    apply_uv_transform_to_face,
    pick_uv_transform_from_face,
)

addon_keymaps = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    km = kc.keymaps.new(name='Mesh', space_type='EMPTY')

    # Alt+Left Click to apply image to face
    kmi = km.keymap_items.new(
        apply_image_to_face.bl_idname,
        'LEFTMOUSE', 'PRESS',
        alt=True,
        head=True
    )
    addon_keymaps.append((km, kmi))

    # Alt+Right Click to pick image from face (edit mode)
    kmi = km.keymap_items.new(
        pick_image_from_face.bl_idname,
        'RIGHTMOUSE', 'PRESS',
        alt=True,
        head=True
    )
    addon_keymaps.append((km, kmi))

    # Shift+Alt+Left Click to stretch-apply image to face
    kmi = km.keymap_items.new(
        stretch_apply_image_to_face.bl_idname,
        'LEFTMOUSE', 'PRESS',
        shift=True,
        alt=True,
        head=True
    )
    addon_keymaps.append((km, kmi))

    # Shift+Alt+Right Click to stretch-pick image from face
    kmi = km.keymap_items.new(
        stretch_pick_image_from_face.bl_idname,
        'RIGHTMOUSE', 'PRESS',
        shift=True,
        alt=True,
        head=True
    )
    addon_keymaps.append((km, kmi))

    # Ctrl+Alt+Left Click to apply UV transform to face
    kmi = km.keymap_items.new(
        apply_uv_transform_to_face.bl_idname,
        'LEFTMOUSE', 'PRESS',
        ctrl=True,
        alt=True,
        head=True
    )
    addon_keymaps.append((km, kmi))

    # Ctrl+Alt+Right Click to pick UV transform from face
    kmi = km.keymap_items.new(
        pick_uv_transform_from_face.bl_idname,
        'RIGHTMOUSE', 'PRESS',
        ctrl=True,
        alt=True,
        head=True
    )
    addon_keymaps.append((km, kmi))

    # Register Alt+Right Click in 3D View for object mode as well
    km = kc.keymaps.new(name='3D View', space_type='VIEW_3D')
    kmi = km.keymap_items.new(
        pick_image_from_face.bl_idname,
        'RIGHTMOUSE', 'PRESS',
        alt=True,
        head=True
    )
    addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
