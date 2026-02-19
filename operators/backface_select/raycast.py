from mathutils import Vector


def is_face_backfacing(face_normal_local, ray_direction_local):
    """Check if a face is back-facing relative to the ray direction.

    A positive dot product means the face normal points away from the viewer
    (same direction as the ray), making it a back face.
    """
    return face_normal_local.dot(ray_direction_local) > 0


def has_backface_culling_enabled(material_index, materials):
    """Check if the material at the given index has backface culling enabled.

    Returns False if there is no material (unassigned faces are always visible
    and should block selection).
    """
    if material_index >= len(materials):
        return False
    mat = materials[material_index]
    if mat is None:
        return False
    return mat.use_backface_culling


def raycast_bvh_skip_backfaces(bvh, ray_origin_local, ray_direction_local,
                                bm, materials, max_iterations):
    """Raycast through a BVHTree, skipping back-facing faces with culling enabled.

    Iteratively casts rays, advancing past any hit face that is both back-facing
    and has backface culling enabled on its material. Caller must call
    bm.faces.ensure_lookup_table() before calling this function.

    Returns (location, normal, face_index, distance) or
            (None, None, None, None) if no visible face is hit.
    """
    origin = ray_origin_local.copy()
    total_distance = 0.0
    epsilon = 0.0001

    for _ in range(max_iterations):
        location, normal, face_index, distance = bvh.ray_cast(origin, ray_direction_local)

        if face_index is None:
            return (None, None, None, None)

        face = bm.faces[face_index]

        if (is_face_backfacing(face.normal, ray_direction_local)
                and has_backface_culling_enabled(face.material_index, materials)):
            # Advance origin past this hit
            total_distance += distance + epsilon
            origin = origin + ray_direction_local * (distance + epsilon)
            continue

        # Visible face hit
        return (location, normal, face_index, total_distance + distance)

    return (None, None, None, None)


def raycast_bvh_skip_backfaces_polys(bvh, ray_origin_local, ray_direction_local,
                                      polygons, materials, max_iterations):
    """Raycast through a BVHTree built from polygons, skipping culled backfaces.

    Same as raycast_bvh_skip_backfaces but uses Mesh.polygons instead of
    bmesh faces. For use with evaluated meshes (BVHTree.FromPolygons).

    Returns (location, normal, face_index, distance) or
            (None, None, None, None) if no visible face is hit.
    """
    origin = ray_origin_local.copy()
    total_distance = 0.0
    epsilon = 0.0001

    for _ in range(max_iterations):
        location, normal, face_index, distance = bvh.ray_cast(origin, ray_direction_local)

        if face_index is None:
            return (None, None, None, None)

        poly = polygons[face_index]

        if (is_face_backfacing(Vector(poly.normal), ray_direction_local)
                and has_backface_culling_enabled(poly.material_index, materials)):
            total_distance += distance + epsilon
            origin = origin + ray_direction_local * (distance + epsilon)
            continue

        return (location, normal, face_index, total_distance + distance)

    return (None, None, None, None)


def raycast_scene_skip_backfaces(depsgraph, scene, origin, direction,
                                  max_iterations):
    """Raycast through the scene, skipping back-facing faces with culling enabled.

    Wraps scene.ray_cast(), advancing past back-facing culled hits.
    Checks material culling on the evaluated mesh.

    Returns (hit, location, normal, face_index, object, matrix) matching
    the format of scene.ray_cast().
    """
    current_origin = origin.copy()
    epsilon = 0.0001

    for _ in range(max_iterations):
        hit, location, normal, face_index, obj, matrix = scene.ray_cast(
            depsgraph, current_origin, direction
        )

        if not hit:
            return (False, None, None, None, None, None)

        # Check if hit face is back-facing with culling
        obj_eval = obj.evaluated_get(depsgraph)
        me_eval = obj_eval.to_mesh()

        if me_eval is not None and face_index < len(me_eval.polygons):
            poly = me_eval.polygons[face_index]
            face_normal_world = (matrix.to_3x3() @ Vector(poly.normal)).normalized()
            mat_index = poly.material_index
            materials = me_eval.materials

            backfacing = is_face_backfacing(face_normal_world, direction)
            culling = has_backface_culling_enabled(mat_index, materials)

            obj_eval.to_mesh_clear()

            if backfacing and culling:
                current_origin = location + direction * epsilon
                continue
        else:
            if me_eval is not None:
                obj_eval.to_mesh_clear()

        return (hit, location, normal, face_index, obj, matrix)

    return (False, None, None, None, None, None)
