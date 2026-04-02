import os

import bpy
import bmesh

from ..core.materials import create_material_with_image, find_material_with_image
from ..core.uv_projection import face_aligned_project
from ..core.uv_projection import apply_uv_to_face

TEXTURE_PATH = os.path.join(os.path.dirname(__file__), "dev_orange_wall.png")


def _get_context_override():
    """Build a temp_override context dict for operators in timer callbacks."""
    window = bpy.context.window or bpy.context.window_manager.windows[0]
    return {"window": window}


def create_vertical_plane(name):
    """Create a 1x1 vertical plane in the XZ plane (facing +Y).

    Vertices: (0,0,0), (1,0,0), (1,0,1), (0,0,1)

    The face is textured with dev_orange_wall.png using the addon's
    material/UV pipeline.

    Returns the new object, linked to the active scene.
    """
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    v0 = bm.verts.new((0, 0, 0))
    v1 = bm.verts.new((1, 0, 0))
    v2 = bm.verts.new((1, 0, 1))
    v3 = bm.verts.new((0, 0, 1))
    bm.faces.new((v0, v1, v2, v3))

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    _apply_material(obj, 1.0, 1.0)

    return obj


def create_textured_cube(name, scale_u, scale_v, face_aligned=False):
    """Create a 1x1x1 cube with all faces textured at the given UV scale.

    The cube spans (0,0,0) to (1,1,1). Returns the object in object mode.

    Args:
        name: Object name
        scale_u: Horizontal UV scale (ignored when face_aligned is True)
        scale_v: Vertical UV scale (ignored when face_aligned is True)
        face_aligned: If True, use face-aligned world-axis projection
                      instead of per-face local projection
    """
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)

    # bmesh.ops.create_cube centers at origin; shift to (0,0,0)-(1,1,1)
    for v in bm.verts:
        v.co.x += 0.5
        v.co.y += 0.5
        v.co.z += 0.5

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    if face_aligned:
        _apply_material_face_aligned(obj)
    else:
        _apply_material(obj, scale_u, scale_v)

    return obj


def _apply_material_face_aligned(obj):
    """Load dev_orange_wall.png and apply it with face-aligned projection."""
    image = bpy.data.images.load(TEXTURE_PATH, check_existing=True)

    mat = find_material_with_image(image)
    if not mat:
        mat = create_material_with_image(image)

    obj.data.materials.append(mat)
    mat_index = obj.data.materials.find(mat.name)

    ppm = bpy.context.scene.level_design_props.pixels_per_meter

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()

    for face in bm.faces:
        face.material_index = mat_index
        face_aligned_project(face, uv_layer, mat, ppm)

    bmesh.update_edit_mesh(obj.data)
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')


def add_uv_layer(obj, layer_name, scale_u, scale_v):
    """Add a new UV layer and project all faces with apply_uv_to_face."""
    mat = obj.data.materials[0]
    ppm = bpy.context.scene.level_design_props.pixels_per_meter
    was_edit = (obj.mode == 'EDIT')

    ctx = _get_context_override()
    if not was_edit:
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.new(layer_name)

    for face in bm.faces:
        apply_uv_to_face(face, uv_layer, scale_u, scale_v, 0.0, 0.0, 0.0,
                         mat, ppm, obj.data)

    bmesh.update_edit_mesh(obj.data)
    if not was_edit:
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')


def add_uv_layer_face_aligned(obj, layer_name, scale):
    """Add a new UV layer and project all faces with face_aligned_project."""
    mat = obj.data.materials[0]
    ppm = bpy.context.scene.level_design_props.pixels_per_meter
    was_edit = (obj.mode == 'EDIT')

    ctx = _get_context_override()
    if not was_edit:
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.new(layer_name)

    for face in bm.faces:
        face_aligned_project(face, uv_layer, mat, ppm, scale=scale)

    bmesh.update_edit_mesh(obj.data)
    if not was_edit:
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.mode_set(mode='OBJECT')


def _apply_material(obj, scale_u=1.0, scale_v=1.0):
    """Load dev_orange_wall.png and apply it as a material with UVs to all faces."""
    image = bpy.data.images.load(TEXTURE_PATH, check_existing=True)

    mat = find_material_with_image(image)
    if not mat:
        mat = create_material_with_image(image)

    obj.data.materials.append(mat)
    mat_index = obj.data.materials.find(mat.name)

    ppm = bpy.context.scene.level_design_props.pixels_per_meter

    ctx = _get_context_override()
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()

    for face in bm.faces:
        face.material_index = mat_index
        apply_uv_to_face(face, uv_layer, scale_u, scale_v, 0.0, 0.0, 0.0,
                         mat, ppm, obj.data)

    bmesh.update_edit_mesh(obj.data)
    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')
