import bpy
import bmesh
import math
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty, EnumProperty
from mathutils import Vector


# Flag to prevent recursive updates
_updating_from_selection = False
# Flag to prevent linked scale from causing infinite recursion
_updating_linked_scale = False
# Flag to prevent offset normalization from causing infinite recursion
_normalizing_offset = False
# Flag to prevent rotation normalization from causing infinite recursion
_normalizing_rotation = False
# Track last scale values to detect which one changed
_last_scale_u = 1.0
_last_scale_v = 1.0


def set_updating_from_selection(value):
    global _updating_from_selection
    _updating_from_selection = value


def get_updating_from_selection():
    return _updating_from_selection


def sync_scale_tracking(context):
    """Update scale tracking to match current property values. Call after selection changes."""
    global _last_scale_u, _last_scale_v
    props = context.scene.level_design_props
    _last_scale_u = props.texture_scale_u
    _last_scale_v = props.texture_scale_v


def update_uv_lock(self, context):
    """Called when UV lock is toggled"""
    from .handlers import cache_face_data
    cache_face_data(context)


def apply_uv_to_face(face, uv_layer, scale_u, scale_v, rotation_deg, offset_x, offset_y,
                     mat, ppm, me):
    """Apply UV coordinates to a single face from scratch with given parameters.

    This is the low-level function that does the actual UV projection.
    Order: rotate projection axes → project → scale → offset

    Rotation is applied by rotating the projection axes in face space, which
    avoids distortion with non-square textures. Scale is applied in texture space,
    meaning scale_u always affects the texture's horizontal axis and scale_v
    always affects the texture's vertical axis, regardless of rotation.

    Args:
        face: BMesh face to apply UVs to
        uv_layer: BMesh UV layer
        scale_u, scale_v: Texture scale factors
        rotation_deg: Rotation in degrees
        offset_x, offset_y: UV offset
        mat: Material to get texture dimensions from (can be None)
        ppm: Pixels per meter setting
        me: Mesh data (for bmesh.update_edit_mesh)
    """
    from .utils import get_face_local_axes, get_texture_dimensions_from_material

    # Guard against invalid face data during modal operators (e.g., loop cut)
    try:
        loops = list(face.loops)
    except (ReferenceError, RuntimeError, OSError):
        return

    if len(loops) < 2:
        return

    face_axes = get_face_local_axes(face)
    if not face_axes:
        return
    face_local_x, face_local_y = face_axes

    # Get texture dimensions from material
    tex_meters_u, tex_meters_v = get_texture_dimensions_from_material(mat, ppm)

    # Rotate the projection axes in face space
    # This determines which face direction maps to texture U vs V
    rotation_rad = math.radians(rotation_deg)
    cos_rot = math.cos(rotation_rad)
    sin_rot = math.sin(rotation_rad)

    # Rotated projection axes (3D vectors in face plane)
    proj_x = face_local_x * cos_rot - face_local_y * sin_rot
    proj_y = face_local_x * sin_rot + face_local_y * cos_rot

    first_vert = loops[0].vert.co

    for loop in loops:
        delta = loop.vert.co - first_vert

        # Project onto rotated axes
        x = delta.dot(proj_x)
        y = delta.dot(proj_y)

        # Convert to UV with scale (texture-based since x is along texture U direction)
        u = x / (scale_u * tex_meters_u)
        v = y / (scale_v * tex_meters_v)

        # Offset
        loop[uv_layer].uv.x = u + offset_x
        loop[uv_layer].uv.y = v + offset_y

    bmesh.update_edit_mesh(me)


def apply_scale_to_selected_faces(context):
    """Apply scale from panel to selected faces, preserving each face's rotation and offset."""
    if get_updating_from_selection() or context.mode != 'EDIT_MESH':
        return

    obj = context.object
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    uv_layer = bm.loops.layers.uv.verify()

    props = context.scene.level_design_props
    new_scale_u = props.texture_scale_u
    new_scale_v = props.texture_scale_v
    ppm = props.pixels_per_meter

    selected_faces = [f for f in bm.faces if f.select]
    if not selected_faces:
        return

    from .handlers import cache_single_face
    from .utils import derive_transform_from_uvs

    for face in selected_faces:
        # Get current transform from this face's UVs
        current = derive_transform_from_uvs(face, uv_layer, ppm, me)
        if current is None:
            continue

        mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
        # Apply new scale, but keep this face's rotation and offset
        apply_uv_to_face(face, uv_layer, new_scale_u, new_scale_v,
                         current['rotation'], current['offset_x'], current['offset_y'],
                         mat, ppm, me)
        cache_single_face(face, uv_layer, ppm, me)


def apply_rotation_to_selected_faces(context):
    """Apply rotation from panel to selected faces, preserving each face's scale and offset."""
    if get_updating_from_selection() or context.mode != 'EDIT_MESH':
        return

    obj = context.object
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    uv_layer = bm.loops.layers.uv.verify()

    props = context.scene.level_design_props
    new_rotation = props.texture_rotation
    ppm = props.pixels_per_meter

    selected_faces = [f for f in bm.faces if f.select]
    if not selected_faces:
        return

    from .handlers import cache_single_face
    from .utils import derive_transform_from_uvs

    for face in selected_faces:
        # Get current transform from this face's UVs
        current = derive_transform_from_uvs(face, uv_layer, ppm, me)
        if current is None:
            continue

        mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
        # Apply new rotation, but keep this face's scale and offset
        apply_uv_to_face(face, uv_layer, current['scale_u'], current['scale_v'],
                         new_rotation, current['offset_x'], current['offset_y'],
                         mat, ppm, me)
        cache_single_face(face, uv_layer, ppm, me)


def apply_offset_to_selected_faces(context):
    """Apply offset from panel to selected faces, preserving each face's scale and rotation."""
    if get_updating_from_selection() or context.mode != 'EDIT_MESH':
        return

    obj = context.object
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    uv_layer = bm.loops.layers.uv.verify()

    props = context.scene.level_design_props
    new_offset_x = props.texture_offset_x
    new_offset_y = props.texture_offset_y
    ppm = props.pixels_per_meter

    selected_faces = [f for f in bm.faces if f.select]
    if not selected_faces:
        return

    from .handlers import cache_single_face
    from .utils import derive_transform_from_uvs

    for face in selected_faces:
        # Get current transform from this face's UVs
        current = derive_transform_from_uvs(face, uv_layer, ppm, me)
        if current is None:
            continue

        mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
        # Apply new offset, but keep this face's scale and rotation
        apply_uv_to_face(face, uv_layer, current['scale_u'], current['scale_v'],
                         current['rotation'], new_offset_x, new_offset_y,
                         mat, ppm, me)
        cache_single_face(face, uv_layer, ppm, me)


def update_texture_scale(self, context):
    """Called when scale_u or scale_v changes"""
    global _updating_linked_scale, _last_scale_u, _last_scale_v

    if _updating_linked_scale or get_updating_from_selection():
        return

    props = context.scene.level_design_props

    # Handle linked scale - detect which value changed and sync the other
    if props.texture_scale_linked and props.texture_scale_u != props.texture_scale_v:
        _updating_linked_scale = True
        try:
            u_changed = props.texture_scale_u != _last_scale_u
            v_changed = props.texture_scale_v != _last_scale_v

            if u_changed:
                props.texture_scale_v = props.texture_scale_u
            elif v_changed:
                props.texture_scale_u = props.texture_scale_v
        finally:
            _updating_linked_scale = False

    # Update tracking
    _last_scale_u = props.texture_scale_u
    _last_scale_v = props.texture_scale_v

    apply_scale_to_selected_faces(context)


def update_texture_rotation(self, context):
    """Called when rotation changes"""
    global _normalizing_rotation

    if _normalizing_rotation or get_updating_from_selection():
        return

    props = context.scene.level_design_props

    # Normalize rotation to 0-360 range
    if props.texture_rotation < 0 or props.texture_rotation >= 360:
        _normalizing_rotation = True
        try:
            props.texture_rotation = props.texture_rotation % 360.0
        finally:
            _normalizing_rotation = False

    apply_rotation_to_selected_faces(context)


def update_texture_offset(self, context):
    """Called when offset_x or offset_y changes"""
    global _normalizing_offset

    if _normalizing_offset or get_updating_from_selection():
        return

    props = context.scene.level_design_props

    # Normalize offset values if they're outside 0-1 range
    needs_normalize_x = props.texture_offset_x < 0 or props.texture_offset_x >= 1
    needs_normalize_y = props.texture_offset_y < 0 or props.texture_offset_y >= 1
    if needs_normalize_x or needs_normalize_y:
        _normalizing_offset = True
        try:
            if needs_normalize_x:
                props.texture_offset_x = props.texture_offset_x % 1.0
            if needs_normalize_y:
                props.texture_offset_y = props.texture_offset_y % 1.0
        finally:
            _normalizing_offset = False

    apply_offset_to_selected_faces(context)


def update_projection_scale(self, context):
    """Called when projection_scale changes - automatically runs face-aligned projection"""
    if get_updating_from_selection() or context.mode != 'EDIT_MESH':
        return

    bpy.ops.leveldesign.face_aligned_project()


class LevelDesignProperties(bpy.types.PropertyGroup):
    """Combined properties for Level Design Tools"""

    # === UV Tools Properties ===
    uv_lock: BoolProperty(
        name="UV Lock",
        description="Lock UVs to geometry when transforming",
        default=False,
        update=update_uv_lock,
    )

    pixels_per_meter: IntProperty(
        name="Pixels per Meter",
        description="Number of texture pixels that represent 1 meter at scale 1",
        default=128,
        min=1,
        max=4096,
    )

    texture_scale_u: FloatProperty(
        name="Scale U",
        description="Horizontal texture scale",
        default=1.0,
        min=0.001,
        max=100.0,
        update=update_texture_scale,
    )

    texture_scale_v: FloatProperty(
        name="Scale V",
        description="Vertical texture scale",
        default=1.0,
        min=0.001,
        max=100.0,
        update=update_texture_scale,
    )

    texture_scale_linked: BoolProperty(
        name="Link Scale",
        description="Link U and V scale values together",
        default=True,
    )

    texture_rotation: FloatProperty(
        name="Rotation",
        description="Texture rotation in degrees",
        default=0.0,
        update=update_texture_rotation,
    )

    texture_offset_x: FloatProperty(
        name="Offset X",
        description="Horizontal texture offset in texture tiles",
        default=0.0,
        update=update_texture_offset,
    )

    texture_offset_y: FloatProperty(
        name="Offset Y",
        description="Vertical texture offset in texture tiles",
        default=0.0,
        update=update_texture_offset,
    )

    edge_index: IntProperty(
        name="Edge Index",
        description="Current edge index for snapping",
        default=0,
        min=0,
    )

    projection_scale: FloatProperty(
        name="Scale",
        description="Scale to use for face-aligned projection",
        default=1.0,
        min=0.001,
        max=100.0,
        update=update_projection_scale,
    )

    # === Export Properties (last used settings) ===
    last_export_filepath: StringProperty(
        name="Last Export Path",
        description="Path of the last GLB export",
        default="",
        subtype='FILE_PATH',
    )

    last_export_scale: FloatProperty(
        name="Last Export Scale",
        description="Scale used in last export",
        default=1.0,
        min=0.001,
        max=1000.0,
    )

    last_export_format: EnumProperty(
        name="Last Export Format",
        items=[
            ('GLB', "GLB (.glb)", ""),
            ('GLTF_SEPARATE', "GLTF + Bin + Textures", ""),
            ('GLTF_EMBEDDED', "GLTF Embedded (.gltf)", ""),
        ],
        default='GLB',
    )

    last_export_textures: BoolProperty(
        name="Last Export Textures",
        description="Whether textures were included in last export",
        default=True,
    )

    last_export_normals: BoolProperty(
        name="Last Export Normals",
        description="Whether normals were included in last export",
        default=True,
    )

    last_export_apply_modifiers: BoolProperty(
        name="Last Export Apply Modifiers",
        description="Whether modifiers were applied in last export",
        default=True,
    )


def register():
    bpy.utils.register_class(LevelDesignProperties)
    bpy.types.Scene.level_design_props = PointerProperty(type=LevelDesignProperties)


def unregister():
    del bpy.types.Scene.level_design_props
    bpy.utils.unregister_class(LevelDesignProperties)
