import bpy
import bmesh
import math
import time
from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty, EnumProperty, CollectionProperty

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

# Undo push tracking - only push if enough time has passed (new edit vs dragging)
_UNDO_PUSH_THRESHOLD = 0.3  # seconds
_last_scale_update_time = 0.0
_last_rotation_update_time = 0.0
_last_offset_update_time = 0.0


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


def update_uv_map_lock(self, context):
    """Called when a per-UV-map lock is toggled"""
    from .handlers import cache_face_data
    cache_face_data(context)


class AnvilUVMapSettings(bpy.types.PropertyGroup):
    """Per-UV-map settings (lock state)"""
    locked: BoolProperty(
        name="Locked",
        description="Lock this UV map to geometry (sticker mode)",
        default=False,
        update=update_uv_map_lock,
    )


def apply_scale_to_selected_faces(context):
    """Apply scale from panel to selected faces, preserving each face's rotation and offset."""
    if get_updating_from_selection() or context.mode != 'EDIT_MESH':
        return

    obj = context.object
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    from .core.uv_layers import get_render_active_uv_layer
    uv_layer = get_render_active_uv_layer(bm, me)
    if uv_layer is None:
        return

    props = context.scene.level_design_props
    new_scale_u = props.texture_scale_u
    new_scale_v = props.texture_scale_v
    ppm = props.pixels_per_meter

    selected_faces = [f for f in bm.faces if f.select]
    if not selected_faces:
        return

    from .handlers import cache_single_face
    from .core.uv_projection import derive_transform_from_uvs, apply_uv_to_face
    from .core.hotspot_queries import face_has_hotspot_material

    for face in selected_faces:
        # Skip faces with hotspottable materials
        if face_has_hotspot_material(face, me):
            continue

        # Get current transform from this face's UVs
        current = derive_transform_from_uvs(face, uv_layer, ppm, me)
        if current is None:
            continue

        mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
        # Apply new scale, but keep this face's rotation and offset
        apply_uv_to_face(face, uv_layer, new_scale_u, new_scale_v,
                         current['rotation'], current['offset_x'], current['offset_y'],
                         mat, ppm, me)
        cache_single_face(face, bm, ppm, me)


def apply_rotation_to_selected_faces(context):
    """Apply rotation from panel to selected faces, preserving each face's scale and offset."""
    if get_updating_from_selection() or context.mode != 'EDIT_MESH':
        return

    obj = context.object
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    from .core.uv_layers import get_render_active_uv_layer
    uv_layer = get_render_active_uv_layer(bm, me)
    if uv_layer is None:
        return

    props = context.scene.level_design_props
    new_rotation = props.texture_rotation
    ppm = props.pixels_per_meter

    selected_faces = [f for f in bm.faces if f.select]
    if not selected_faces:
        return

    from .handlers import cache_single_face
    from .core.uv_projection import derive_transform_from_uvs, apply_uv_to_face
    from .core.hotspot_queries import face_has_hotspot_material

    for face in selected_faces:
        # Skip faces with hotspottable materials
        if face_has_hotspot_material(face, me):
            continue

        # Get current transform from this face's UVs
        current = derive_transform_from_uvs(face, uv_layer, ppm, me)
        if current is None:
            continue

        mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
        # Apply new rotation, but keep this face's scale and offset
        apply_uv_to_face(face, uv_layer, current['scale_u'], current['scale_v'],
                         new_rotation, current['offset_x'], current['offset_y'],
                         mat, ppm, me)
        cache_single_face(face, bm, ppm, me)


def apply_offset_to_selected_faces(context):
    """Apply offset from panel to selected faces, preserving each face's scale and rotation."""
    if get_updating_from_selection() or context.mode != 'EDIT_MESH':
        return

    obj = context.object
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    from .core.uv_layers import get_render_active_uv_layer
    uv_layer = get_render_active_uv_layer(bm, me)
    if uv_layer is None:
        return

    props = context.scene.level_design_props
    new_offset_x = props.texture_offset_x
    new_offset_y = props.texture_offset_y
    ppm = props.pixels_per_meter

    selected_faces = [f for f in bm.faces if f.select]
    if not selected_faces:
        return

    from .handlers import cache_single_face
    from .core.uv_projection import derive_transform_from_uvs, apply_uv_to_face
    from .core.hotspot_queries import face_has_hotspot_material

    for face in selected_faces:
        # Skip faces with hotspottable materials
        if face_has_hotspot_material(face, me):
            continue

        # Get current transform from this face's UVs
        current = derive_transform_from_uvs(face, uv_layer, ppm, me)
        if current is None:
            continue

        mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
        # Apply new offset, but keep this face's scale and rotation
        apply_uv_to_face(face, uv_layer, current['scale_u'], current['scale_v'],
                         current['rotation'], new_offset_x, new_offset_y,
                         mat, ppm, me)
        cache_single_face(face, bm, ppm, me)


def update_texture_scale(self, context):
    """Called when scale_u or scale_v changes"""
    global _updating_linked_scale, _last_scale_u, _last_scale_v, _last_scale_update_time

    if _updating_linked_scale or get_updating_from_selection():
        return

    from .handlers import mark_multi_face_set_scale
    mark_multi_face_set_scale()

    # Push undo if this is a new edit session (not mid-drag)
    current_time = time.time()
    if current_time - _last_scale_update_time > _UNDO_PUSH_THRESHOLD:
        bpy.ops.ed.undo_push(message="Change Texture Scale")
    _last_scale_update_time = current_time

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
    global _normalizing_rotation, _last_rotation_update_time

    if _normalizing_rotation or get_updating_from_selection():
        return

    from .handlers import mark_multi_face_set_rotation
    mark_multi_face_set_rotation()

    # Push undo if this is a new edit session (not mid-drag)
    current_time = time.time()
    if current_time - _last_rotation_update_time > _UNDO_PUSH_THRESHOLD:
        bpy.ops.ed.undo_push(message="Change Texture Rotation")
    _last_rotation_update_time = current_time

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
    global _normalizing_offset, _last_offset_update_time

    if _normalizing_offset or get_updating_from_selection():
        return

    from .handlers import mark_multi_face_set_offset
    mark_multi_face_set_offset()

    # Push undo if this is a new edit session (not mid-drag)
    current_time = time.time()
    if current_time - _last_offset_update_time > _UNDO_PUSH_THRESHOLD:
        bpy.ops.ed.undo_push(message="Change Texture Offset")
    _last_offset_update_time = current_time

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
    if get_updating_from_selection():
        return

    # Only run in edit mode or object mode (operator handles mode switching)
    if context.mode not in {'EDIT_MESH', 'OBJECT'}:
        return

    bpy.ops.leveldesign.face_aligned_project()


class LevelDesignProperties(bpy.types.PropertyGroup):
    """Combined properties for Level Design Tools"""

    # Dummy prop so the UV Lock button renders even with no object selected
    uv_lock_placeholder: BoolProperty(name="UV Lock", default=False)

    debug_logging: BoolProperty(
        name="Debug Logging",
        description="Enable debug logging to the console",
        default=False,
    )

    # === UV Tools Properties ===
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

    # auto_hotspot moved to per-object properties
    # (see register() below)

    # Face orientation overlay state saved before forcing it on in vertex paint/sculpt.
    # -1 = not saved, 0 = was off, 1 = was on
    saved_face_orientation: IntProperty(
        name="Saved Face Orientation",
        default=-1,
    )

    # === Default Material Settings ===
    default_interpolation: EnumProperty(
        name="Default Interpolation",
        description="Interpolation mode for new materials",
        items=[
            ('Closest', "Closest", "No interpolation (pixelated)"),
            ('Linear', "Linear", "Linear interpolation (smooth)"),
        ],
        default='Linear',
    )

    default_texture_as_alpha: BoolProperty(
        name="Texture as Alpha",
        description="Connect texture alpha to material alpha for new materials",
        default=False,
    )

    default_vertex_colors: BoolProperty(
        name="Vertex Colors",
        description="Multiply texture by vertex colors for new materials",
        default=False,
    )

    default_roughness: FloatProperty(
        name="Roughness",
        description="Roughness value for new materials",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    default_metallic: FloatProperty(
        name="Metallic",
        description="Metallic value for new materials",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    # === Default Experimental Material Settings ===
    default_emission_strength: FloatProperty(
        name="Emission Strength",
        description="Emission strength for new materials",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    default_emission_color: FloatVectorProperty(
        name="Emission Color",
        description="Emission color for new materials",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
    )

    default_specular: FloatProperty(
        name="Specular",
        description="Specular (IOR Level) value for new materials",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    # === UI Collapse State ===
    show_experimental_settings: BoolProperty(
        name="Experimental Settings",
        description="Show experimental material settings",
        default=False,
    )

    show_default_experimental_settings: BoolProperty(
        name="Experimental Settings",
        description="Show default experimental material settings",
        default=False,
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

    anvil_grid_scale: FloatProperty(
        name="Anvil Grid Scale",
        description="Logical grid scale used by Anvil",
        min=0.0001,
        max=100000.0,
    )

    # === Grid Overlay ===
    show_grid_overlay: BoolProperty(
        name="Show Grid Overlay",
        description="Display a grid overlay on all visible geometry",
        default=False,
    )

    # === Fixed Hotspot Overlay ===
    show_fixed_hotspot_overlay: BoolProperty(
        name="Show Fixed Hotspot Overlay",
        description="Display a white overlay on faces marked as fixed hotspot",
        default=False,
    )

    # === Context Weld State ===
    weld_mode: EnumProperty(
        name="Weld Mode",
        description="Next weld action to perform",
        items=[
            ('NONE', "None", "No weld action pending"),
            ('BRIDGE', "Bridge Edge Loops", "Bridge two edge loops"),
            ('CORRIDOR', "Corridor", "Create face and extrude corridor"),
            ('INVERT', "Invert", "Flip normals on the mesh"),
            ('FOLDED_PLANE', "Folded Plane", "Fill faces on cuboid side planes"),
        ],
        default='NONE',
    )

    weld_depth: FloatProperty(
        name="Weld Depth",
        description="Depth for corridor extrusion",
        default=0.0,
    )

    weld_direction: FloatVectorProperty(
        name="Weld Direction",
        description="Extrusion direction for corridor (from cube cut)",
        size=3,
        default=(0.0, 0.0, 0.0),
    )

    weld_back_plane_offset: FloatProperty(
        name="Weld Back Plane Offset",
        description="Projection of the cube cut back plane onto the extrusion direction",
        default=0.0,
    )


def register():
    bpy.utils.register_class(AnvilUVMapSettings)
    bpy.utils.register_class(LevelDesignProperties)
    bpy.types.Scene.level_design_props = PointerProperty(type=LevelDesignProperties)

    # Per-object, per-UV-map settings collection
    bpy.types.Object.anvil_uv_map_settings = CollectionProperty(type=AnvilUVMapSettings)

    # Per-object allow combined faces property for hotspotting
    bpy.types.Object.anvil_allow_combined_faces = BoolProperty(
        name="Combine Faces",
        description="Allow multi-face islands during hotspot mapping. When disabled, each face is mapped independently",
        default=True,
    )

    # Per-object balance between aspect ratio and size matching (0 = pure aspect, 1 = pure size)
    bpy.types.Object.anvil_hotspot_size_weight = FloatProperty(
        name="Size Weight",
        description="Balance between aspect ratio matching (0) and size matching (1)",
        default=0.1,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    # Per-object seam angle threshold for hotspot topology grouping (stored in radians, displayed as degrees)
    bpy.types.Object.anvil_hotspot_seam_angle = FloatProperty(
        name="Seam Angle",
        description="Maximum angle between face normals before adding a seam. Lower values create more groups, higher values combine more faces",
        default=math.radians(33),
        min=0.0,
        max=math.pi,
        subtype='ANGLE',
    )

    # Per-object auto hotspot toggle
    bpy.types.Object.anvil_auto_hotspot = BoolProperty(
        name="Auto Hotspot",
        description="Automatically apply hotspot mapping after geometry changes",
        default=False,
    )



def unregister():
    del bpy.types.Object.anvil_auto_hotspot
    del bpy.types.Object.anvil_hotspot_seam_angle
    del bpy.types.Object.anvil_hotspot_size_weight
    del bpy.types.Object.anvil_allow_combined_faces
    del bpy.types.Object.anvil_uv_map_settings
    del bpy.types.Scene.level_design_props
    bpy.utils.unregister_class(LevelDesignProperties)
    bpy.utils.unregister_class(AnvilUVMapSettings)
