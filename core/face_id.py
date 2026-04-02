import bpy
import bmesh


# ---- Managed Face ID System ----
# BMesh int layer that gives each face a stable, unique ID.
# ID 0 = unassigned (new face).  Positive integers are assigned IDs.
# Unlike Blender's face.index, these survive topology changes.

FACE_ID_LAYER_NAME = "anvil_face_id"
FIXED_HOTSPOT_LAYER_NAME = "anvil_fixed_hotspot"
_next_face_id = 1


def get_face_id_layer(bm):
    """Get or create the managed face ID layer on a BMesh.

    Returns the BMesh int layer for face IDs.
    """
    layer = bm.faces.layers.int.get(FACE_ID_LAYER_NAME)
    if layer is None:
        layer = bm.faces.layers.int.new(FACE_ID_LAYER_NAME)
    return layer


def get_fixed_hotspot_layer(bm):
    """Get or create the fixed-hotspot flag layer on a BMesh.

    Returns the BMesh int layer (0 = not fixed, 1 = fixed).
    """
    layer = bm.faces.layers.int.get(FIXED_HOTSPOT_LAYER_NAME)
    if layer is None:
        layer = bm.faces.layers.int.new(FIXED_HOTSPOT_LAYER_NAME)
    return layer


def ensure_face_ids(bm, id_layer):
    """Assign managed IDs to all faces that don't have one.

    Faces with ID 0 (unassigned) get a unique positive integer.

    Args:
        bm: BMesh instance
        id_layer: The face ID int layer from get_face_id_layer()
    """
    global _next_face_id

    # Find current max to ensure we don't reuse IDs
    current_max = 0
    unassigned = []
    for face in bm.faces:
        fid = face[id_layer]
        if fid == 0:
            unassigned.append(face)
        elif fid > current_max:
            current_max = fid

    _next_face_id = max(_next_face_id, current_max + 1)

    for face in unassigned:
        face[id_layer] = _next_face_id
        _next_face_id += 1


def assign_face_id(face, id_layer):
    """Assign a managed ID to a single face if it doesn't have one.

    Returns the assigned (or existing) face ID.
    """
    global _next_face_id
    fid = face[id_layer]
    if fid != 0:
        return fid

    face[id_layer] = _next_face_id
    _next_face_id += 1
    return face[id_layer]


def find_face_by_id(bm, face_id, id_layer):
    """Find a face by its managed ID. Returns None if not found."""
    for face in bm.faces:
        if face[id_layer] == face_id:
            return face
    return None


def reindex_face_ids(bm, id_layer):
    """Assign fresh unique sequential IDs to ALL faces.

    Called after processing (UV projection, caching) to guarantee every face
    has a unique ID.  Since we only compare between consecutive depsgraph
    updates, IDs don't need long-term stability.

    Args:
        bm: BMesh instance
        id_layer: The face ID int layer from get_face_id_layer()
    """
    global _next_face_id
    _next_face_id = 1
    for face in bm.faces:
        face[id_layer] = _next_face_id
        _next_face_id += 1


def save_face_selection(bm, id_layer):
    """Save face selection state using managed IDs.

    Returns (selected_ids, active_id) where active_id may be None.
    """
    selected_ids = {f[id_layer] for f in bm.faces if f.select}
    active_id = bm.faces.active[id_layer] if bm.faces.active else None
    return selected_ids, active_id


def restore_face_selection(bm, id_layer, selected_ids, active_id):
    """Restore face selection state from managed IDs."""
    bm.faces.ensure_lookup_table()
    for face in bm.faces:
        face.select = face[id_layer] in selected_ids
    if active_id is not None:
        for face in bm.faces:
            if face[id_layer] == active_id:
                bm.faces.active = face
                break


def get_selected_faces_or_report(operator, bm):
    """Get selected faces from bmesh, or report warning and return None.

    Args:
        operator: The operator instance (for self.report)
        bm: BMesh instance

    Returns:
        List of selected faces, or None if no faces selected (warning reported)
    """
    selected_faces = [f for f in bm.faces if f.select]
    if not selected_faces:
        operator.report({'WARNING'}, "No faces selected")
        return None
    return selected_faces


def get_selected_face_count(context):
    """Get number of selected faces"""
    if context.mode != 'EDIT_MESH' or not context.object or context.object.type != 'MESH':
        return 0

    me = context.object.data
    bm = bmesh.from_edit_mesh(me)
    return sum(1 for f in bm.faces if f.select)
