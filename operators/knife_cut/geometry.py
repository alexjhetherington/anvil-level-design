"""
Knife Cut Tool - Geometry Operations

Delegates to Blender's built-in knife_project operator for reliable mesh cutting.
A temporary edge-only mesh is created from the snapped cut points, then projected
onto the active edit mesh from the current view direction.
"""

import bpy

from ...core.logging import debug_log


def execute_knife_cut(context, obj, cut_points):
    """Execute knife cut by projecting cut path edges via knife_project.

    Creates a temporary edge-only mesh from the cut points, selects it,
    and uses bpy.ops.mesh.knife_project to cut the active edit mesh.

    Args:
        context: Blender context
        obj: Active mesh object (must be in edit mode)
        cut_points: List of (world_position, face_normal) tuples

    Returns:
        tuple: (success: bool, message: str)
    """
    if len(cut_points) < 2:
        return (False, "Need at least 2 points")

    # Build temporary edge mesh from cut points (world space)
    verts = [tuple(p[0]) for p in cut_points]
    edges = [(i, i + 1) for i in range(len(verts) - 1)]

    temp_mesh = bpy.data.meshes.new("_anvil_knife_temp")
    temp_mesh.from_pydata(verts, edges, [])
    temp_mesh.update()

    temp_obj = bpy.data.objects.new("_anvil_knife_temp", temp_mesh)
    context.collection.objects.link(temp_obj)

    # Ensure the view layer picks up the new object
    context.view_layer.update()

    # Select the temp object alongside the active edit mesh
    temp_obj.select_set(True)

    success = False
    message = "Knife cut failed"

    try:
        result = bpy.ops.mesh.knife_project(cut_through=False)
        if result == {'FINISHED'}:
            success = True
            n = len(cut_points) - 1
            message = f"Knife cut: {n} segment{'s' if n != 1 else ''}"
            debug_log(f"[KnifeCut] knife_project succeeded with {len(cut_points)} points")
        else:
            message = "Knife project did not complete"
            debug_log(f"[KnifeCut] knife_project returned {result}")
    except Exception as e:
        message = f"Knife cut failed: {e}"
        debug_log(f"[KnifeCut] knife_project exception: {e}")
    finally:
        # Remove temporary object and mesh
        try:
            temp_obj.select_set(False)
        except ReferenceError:
            pass
        try:
            bpy.data.objects.remove(temp_obj, do_unlink=True)
        except (ReferenceError, RuntimeError):
            pass
        try:
            if temp_mesh.users == 0:
                bpy.data.meshes.remove(temp_mesh)
        except (ReferenceError, RuntimeError):
            pass

    return (success, message)
