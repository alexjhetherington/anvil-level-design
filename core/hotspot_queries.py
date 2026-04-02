from .materials import get_image_from_material


def face_has_hotspot_material(face, me):
    """Check if a face has a material with a hotspottable texture.

    Args:
        face: BMesh face to check
        me: Mesh data (for accessing materials)

    Returns:
        True if the face's material has a hotspottable texture, False otherwise.
    """
    from ..hotspot_mapping.json_storage import is_texture_hotspottable

    mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
    image = get_image_from_material(mat)
    if image and is_texture_hotspottable(image.name):
        return True
    return False


def object_has_hotspot_material(obj):
    """Check if any material on an object uses a hotspottable texture.

    Args:
        obj: Blender object to check

    Returns:
        True if any material on the object is hotspottable, False otherwise.
    """
    from ..hotspot_mapping.json_storage import is_texture_hotspottable

    if not obj or obj.type != 'MESH' or not obj.data.materials:
        return False
    for mat in obj.data.materials:
        image = get_image_from_material(mat)
        if image and is_texture_hotspottable(image.name):
            return True
    return False


def get_connected_faces(face):
    """Get all faces connected to a face via shared edges.

    Args:
        face: BMesh face to get connected faces for

    Returns:
        Set of BMesh faces connected to the input face (excluding the input face itself).
    """
    connected = set()
    for edge in face.edges:
        for linked_face in edge.link_faces:
            if linked_face != face and linked_face.is_valid:
                connected.add(linked_face)
    return connected


def any_connected_face_has_hotspot(face, me):
    """Check if any face connected to the given face has a hotspot material.

    Args:
        face: BMesh face to check neighbors of
        me: Mesh data (for accessing materials)

    Returns:
        True if any connected face has a hotspottable texture, False otherwise.
    """
    for connected_face in get_connected_faces(face):
        if face_has_hotspot_material(connected_face, me):
            return True
    return False


def get_all_hotspot_faces(bm, me):
    """Get all faces in the bmesh that have hotspot materials.

    Args:
        bm: BMesh to search
        me: Mesh data (for accessing materials)

    Returns:
        List of BMesh faces that have hotspottable textures.
    """
    return [f for f in bm.faces if f.is_valid and face_has_hotspot_material(f, me)]
