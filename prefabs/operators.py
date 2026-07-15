"""Prefab operator classes and registration."""

import math
import os
import random

import bpy
from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty
from bpy.types import Operator
from mathutils import Matrix, Vector

from ..core.logging import debug_log
from ..core.workspace_check import is_level_design_workspace
from ..operators.modal_draw import snapping, utils
from ..operators.modal_draw.base_operator import ModalDrawBase
from ..operators.modal_draw.default_grid_pivot import DefaultGridPivotMixin
from ..operators.modal_draw.prefab_ghost import build_prefab_albedo_ghost
from ..operators.weld import (
    set_repeat_prefab_on_object,
    set_repeat_prefab_override,
)
from . import browser
from .assets import (
    append_prefab_object,
    clear_prefab_asset,
    create_object_override,
    find_existing_linked_object,
    find_loaded_library,
    focus_selected_in_3d_views,
    iter_scene_prefab_assets,
    link_prefab_object,
    make_all_free_objects_assets,
    normalize_path,
    refresh_library_objects,
    reload_library,
    scan_library_prefab_assets,
    select_prefab_asset,
)
from .previews import (
    capture_library_previews,
    cleanup_preview_cache,
    invalidate_preview_cache,
)


_PREFAB_ROTATE_LEFT_ID = "leveldesign.prefab_rotate_left"
_PREFAB_ROTATE_RIGHT_ID = "leveldesign.prefab_rotate_right"
_PREFAB_ROTATION_DRAG_RADIANS_PER_PIXEL = math.radians(0.5)
_PREFAB_QUARTER_TURN = math.radians(90.0)
_PREFAB_UP_FALLBACK = Vector((0.0, 0.0, 1.0))
_PREFAB_NORMAL_EPSILON = 0.000001
_PREFAB_IDENTITY_SCALE = (1.0, 1.0, 1.0)
_PREFAB_ZERO_ROTATION = (0.0, 0.0, 0.0)


def _poll_level_design(context):
    return is_level_design_workspace() and getattr(context, "scene", None) is not None


def _poll_prefab_scene_mode(context):
    scene = getattr(context, "scene", None)
    return (_poll_level_design(context)
            and getattr(scene, "anvil_prefab_mode", 'SCENE') == 'SCENE')


def _poll_prefab_library_mode(context):
    scene = getattr(context, "scene", None)
    return (_poll_level_design(context)
            and getattr(scene, "anvil_prefab_mode", 'SCENE') == 'LIBRARY')


def _normalize_rotation_angle(rotation):
    return rotation % (math.pi * 2.0)


def _prefab_safe_up_vector(up_vector):
    if up_vector is None:
        return _PREFAB_UP_FALLBACK.copy()
    up = up_vector.copy()
    if up.length < _PREFAB_NORMAL_EPSILON:
        return _PREFAB_UP_FALLBACK.copy()
    up.normalize()
    return up


def _prefab_placement_up(inherit_normal, face_normal):
    if inherit_normal:
        return _prefab_safe_up_vector(face_normal)
    return _PREFAB_UP_FALLBACK.copy()


def _prefab_up_alignment_matrix(up_vector):
    local_z = _prefab_safe_up_vector(up_vector)
    world_up = _PREFAB_UP_FALLBACK

    if abs(local_z.dot(world_up)) > 0.999:
        local_x = Vector((1.0, 0.0, 0.0))
    else:
        local_x = world_up.cross(local_z)
        local_x.normalize()

    local_y = local_z.cross(local_x)
    if local_y.length < _PREFAB_NORMAL_EPSILON:
        local_y = Vector((0.0, 1.0, 0.0))
    else:
        local_y.normalize()

    return Matrix((
        (local_x.x, local_y.x, local_z.x, 0.0),
        (local_x.y, local_y.y, local_z.y, 0.0),
        (local_x.z, local_y.z, local_z.z, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))


def _prefab_placement_matrix(
        base_matrix,
        location,
        rotation,
        up_vector,
        random_scale,
        random_rotation):
    matrix = (
        _prefab_up_alignment_matrix(up_vector)
        @ Matrix.Rotation(rotation, 4, 'Z')
        @ _prefab_local_random_matrix(random_scale, random_rotation)
        @ base_matrix
    )
    matrix.translation = location
    return matrix


def _prefab_random_axis_value(min_value, max_value):
    lower = min(min_value, max_value)
    upper = max(min_value, max_value)
    if lower == upper:
        return lower
    return random.uniform(lower, upper)


def _prefab_random_vector(min_values, max_values):
    return Vector((
        _prefab_random_axis_value(min_values[0], max_values[0]),
        _prefab_random_axis_value(min_values[1], max_values[1]),
        _prefab_random_axis_value(min_values[2], max_values[2]),
    ))


def _prefab_random_scale_is_linked(props):
    return (
        bool(props.prefab_random_scale_min_linked)
        and bool(props.prefab_random_scale_max_linked)
    )


def _sample_prefab_random_scale(props):
    if _prefab_random_scale_is_linked(props):
        value = _prefab_random_axis_value(
            props.prefab_random_scale_min[0],
            props.prefab_random_scale_max[0],
        )
        return Vector((value, value, value))

    return _prefab_random_vector(
        props.prefab_random_scale_min,
        props.prefab_random_scale_max,
    )


def _prefab_random_settings_key(props):
    return (
        bool(props.prefab_random_scale_enabled),
        tuple(props.prefab_random_scale_min),
        bool(props.prefab_random_scale_min_linked),
        tuple(props.prefab_random_scale_max),
        bool(props.prefab_random_scale_max_linked),
        bool(props.prefab_random_rotation_enabled),
        tuple(props.prefab_random_rotation_min),
        tuple(props.prefab_random_rotation_max),
    )


def _sample_prefab_random_transform(props):
    scale = Vector(_PREFAB_IDENTITY_SCALE)
    rotation = Vector(_PREFAB_ZERO_ROTATION)

    if props.prefab_random_scale_enabled:
        scale = _sample_prefab_random_scale(props)

    if props.prefab_random_rotation_enabled:
        rotation = _prefab_random_vector(
            props.prefab_random_rotation_min,
            props.prefab_random_rotation_max,
        )

    return scale, rotation


def _prefab_scale_matrix(scale):
    return Matrix((
        (scale[0], 0.0, 0.0, 0.0),
        (0.0, scale[1], 0.0, 0.0),
        (0.0, 0.0, scale[2], 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))


def _prefab_euler_rotation_matrix(rotation):
    return (
        Matrix.Rotation(rotation[2], 4, 'Z')
        @ Matrix.Rotation(rotation[1], 4, 'Y')
        @ Matrix.Rotation(rotation[0], 4, 'X')
    )


def _prefab_local_random_matrix(scale, rotation):
    return _prefab_euler_rotation_matrix(rotation) @ _prefab_scale_matrix(scale)


def _prefab_rotation_snap_increment(tool_settings):
    if not getattr(tool_settings, "use_snap", False):
        return 0.0
    if hasattr(tool_settings, "use_snap_rotate") and not tool_settings.use_snap_rotate:
        return 0.0

    for attr_name in (
            "snap_angle_increment_3d",
            "snap_angle_increment",
            "snap_angle_increment_2d"):
        snap_value = getattr(tool_settings, attr_name, 0.0)
        if snap_value > 0.0:
            return snap_value

    return math.radians(15.0)


def _snap_prefab_rotation(tool_settings, rotation):
    snap_increment = _prefab_rotation_snap_increment(tool_settings)
    if snap_increment <= 0.0:
        return rotation
    return round(rotation / snap_increment) * snap_increment


def _event_matches_keymap_item_with_ctrl(event, keymap_item, event_ctrl):
    if not getattr(keymap_item, "active", True):
        return False
    if event.value != 'PRESS':
        return False
    if keymap_item.value not in {'PRESS', 'ANY'}:
        return False
    if event.type != keymap_item.type:
        return False
    if getattr(keymap_item, "any", False):
        return True

    if event_ctrl != getattr(keymap_item, "ctrl", False):
        return False

    for attr_name in ('shift', 'alt', 'oskey'):
        if getattr(event, attr_name, False) != getattr(keymap_item, attr_name, False):
            return False
    return True


def _event_matches_prefab_rotation_keymap_item(event, keymap_item):
    event_ctrl = getattr(event, "ctrl", False)
    if _event_matches_keymap_item_with_ctrl(event, keymap_item, event_ctrl):
        return True
    if not event_ctrl or getattr(keymap_item, "ctrl", False):
        return False
    return _event_matches_keymap_item_with_ctrl(event, keymap_item, False)


def _prefab_rotation_key_direction(window_manager, event):
    keyconfigs = (window_manager.keyconfigs.user, window_manager.keyconfigs.addon)
    for keyconfig in keyconfigs:
        if keyconfig is None:
            continue

        found_prefab_rotation_key = False
        for km_name, _space_type in KEYMAPS_TO_REGISTER:
            keymap = keyconfig.keymaps.get(km_name)
            if keymap is None:
                continue

            for keymap_item in keymap.keymap_items:
                if keymap_item.idname not in {_PREFAB_ROTATE_LEFT_ID, _PREFAB_ROTATE_RIGHT_ID}:
                    continue

                found_prefab_rotation_key = True
                if _event_matches_prefab_rotation_keymap_item(event, keymap_item):
                    if keymap_item.idname == _PREFAB_ROTATE_LEFT_ID:
                        return 1
                    return -1

        if found_prefab_rotation_key:
            return 0

    return 0


def _resolve_prefab_linked_asset(scene, library_index, object_name, asset_type):
    if not (0 <= library_index < len(scene.anvil_prefab_libraries)):
        return None, "", False, "Invalid library index"

    lib_entry = scene.anvil_prefab_libraries[library_index]
    abs_path = normalize_path(lib_entry.filepath)
    if not os.path.isfile(abs_path):
        return None, abs_path, False, f"Library not found: {abs_path}"
    if asset_type != 'OBJECT':
        return None, abs_path, False, f"Invalid asset type: {asset_type}"
    if not object_name:
        return None, abs_path, False, "No prefab name supplied"

    loaded_lib = find_loaded_library(abs_path)
    linked_asset = None
    reused_linked_asset = False
    if loaded_lib is not None:
        linked_asset = find_existing_linked_object(abs_path, object_name)
        reused_linked_asset = linked_asset is not None

    if linked_asset is None:
        linked_asset = link_prefab_object(abs_path, object_name)
        if linked_asset is None:
            return None, abs_path, False, f"Object '{object_name}' not found in {abs_path}"

    return linked_asset, abs_path, reused_linked_asset, ""


def _generated_prefab_name_index(name, base_name, suffix):
    if suffix and name.endswith(suffix):
        stem = name[:-len(suffix)]
    else:
        stem = name

    if stem == base_name:
        return 0

    numeric_prefix = base_name + "."
    if not stem.startswith(numeric_prefix):
        return None

    numeric_part = stem[len(numeric_prefix):]
    if len(numeric_part) != 3 or not numeric_part.isdigit():
        return None

    return int(numeric_part)


def _next_prefab_object_name(base_name, suffix, placed_object):
    used_indices = set()
    for obj in bpy.data.objects:
        if obj == placed_object or obj.library is not None:
            continue
        index = _generated_prefab_name_index(obj.name, base_name, suffix)
        if index is not None:
            used_indices.add(index)

    index = 0
    while index in used_indices:
        index += 1

    if index == 0:
        return base_name + suffix
    return f"{base_name}.{index:03d}{suffix}"


def _find_repeat_prefab_object(object_name):
    if not object_name:
        return None, "No repeat prefab object supplied"

    obj = bpy.data.objects.get(object_name)
    if obj is None:
        return None, f"Repeat prefab object '{object_name}' not found"
    return obj, ""


def _instantiate_repeated_prefab_object(
        scene,
        collection,
        view_layer,
        source_object,
        object_name,
        source_object_name,
        placement_matrix,
        name_suffix):
    placed_object = source_object.copy()
    target_collection = collection if collection is not None else scene.collection
    target_collection.objects.link(placed_object)

    base_name = object_name if object_name else source_object_name
    placed_object.name = _next_prefab_object_name(
        base_name,
        name_suffix,
        placed_object,
    )
    placed_object.matrix_basis = placement_matrix.copy()

    for obj in view_layer.objects:
        obj.select_set(False)
    placed_object.select_set(True)
    view_layer.objects.active = placed_object

    debug_log(
        f"[Prefabs] Repeated local object '{source_object.name}' "
        f"as '{placed_object.name}'"
    )
    return placed_object, "Prefab placed"


def _instantiate_prefab_object(
        scene,
        collection,
        view_layer,
        library_index,
        source_object_name,
        object_name,
        asset_type,
        placement_matrix,
        name_suffix,
        make_fully_local):
    linked_asset, abs_path, reused_linked_asset, error = _resolve_prefab_linked_asset(
        scene, library_index, source_object_name, asset_type
    )
    if linked_asset is None:
        return None, error

    placed_object = None
    if make_fully_local:
        debug_log(
            "[Prefabs] Appending fully local object "
            f"name='{source_object_name}' library='{abs_path}'"
        )
        placed_object = append_prefab_object(abs_path, source_object_name)
        if placed_object is None:
            return None, f"Object '{source_object_name}' not found in {abs_path}"

    if placed_object is None:
        debug_log(
            "[Prefabs] Creating object override for localization "
            f"name='{source_object_name}' library='{abs_path}' "
            f"linked_name='{linked_asset.name}' users={linked_asset.users}"
        )
        override, override_error = create_object_override(linked_asset)
        if override is None and not override_error and reused_linked_asset:
            debug_log(
                "[Prefabs] Re-linking stale object asset after override_create returned None "
                f"name='{source_object_name}' linked_name='{linked_asset.name}' "
                f"library='{getattr(linked_asset.library, 'filepath', None)}' "
                f"users={linked_asset.users}"
            )
            try:
                bpy.data.objects.remove(linked_asset, do_unlink=True)
            except RuntimeError as exc:
                debug_log(
                    "[Prefabs] Could not remove stale linked object asset "
                    f"name='{source_object_name}' linked_name='{linked_asset.name}': {exc}"
                )
            linked_asset = link_prefab_object(abs_path, source_object_name)
            if linked_asset is None:
                return None, f"Object '{source_object_name}' not found in {abs_path}"
            override, override_error = create_object_override(linked_asset)

        if override_error:
            message = (
                f"Could not prepare object prefab '{source_object_name}' "
                f"from {abs_path}: {override_error}"
            )
            print(f"Anvil Level Design: {message}", flush=True)
            debug_log(
                "[Prefabs] override_create raised "
                f"name='{source_object_name}' linked_name='{linked_asset.name}' "
                f"library='{getattr(linked_asset.library, 'filepath', None)}' "
                f"override_library={linked_asset.override_library is not None}"
            )
            return None, message
        if override is None:
            message = (
                f"Could not prepare object prefab '{source_object_name}' from {abs_path}. "
                "Blender returned no override object."
            )
            print(f"Anvil Level Design: {message}", flush=True)
            debug_log(
                "[Prefabs] override_create returned None "
                f"name='{source_object_name}' linked_name='{linked_asset.name}' "
                f"library='{getattr(linked_asset.library, 'filepath', None)}' "
                f"override_library={linked_asset.override_library is not None}"
            )
            return None, message

        placed_object = override.make_local(
            clear_liboverride=True,
            clear_asset_data=True,
        )
        if placed_object is None:
            message = f"Could not make object prefab '{source_object_name}' local"
            print(f"Anvil Level Design: {message}", flush=True)
            return None, message

    target_collection = collection if collection is not None else scene.collection
    try:
        target_collection.objects.link(placed_object)
    except RuntimeError:
        pass

    base_name = object_name if object_name else source_object_name
    placed_object.name = _next_prefab_object_name(
        base_name,
        name_suffix,
        placed_object,
    )
    placed_object.matrix_basis = placement_matrix.copy()

    for obj in view_layer.objects:
        obj.select_set(False)
    placed_object.select_set(True)
    view_layer.objects.active = placed_object

    debug_log(
        f"[Prefabs] Instantiated object '{source_object_name}' "
        f"as '{placed_object.name}' from {abs_path}"
    )
    return placed_object, "Prefab placed"


class LEVELDESIGN_OT_prefab_rotate_left(Operator):
    """Rotate prefab placement left by 90 degrees"""
    bl_idname = _PREFAB_ROTATE_LEFT_ID
    bl_label = "Prefab Rotate Left"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return _poll_prefab_scene_mode(context)

    def execute(self, context):
        return {'PASS_THROUGH'}


class LEVELDESIGN_OT_prefab_rotate_right(Operator):
    """Rotate prefab placement right by 90 degrees"""
    bl_idname = _PREFAB_ROTATE_RIGHT_ID
    bl_label = "Prefab Rotate Right"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return _poll_prefab_scene_mode(context)

    def execute(self, context):
        return {'PASS_THROUGH'}


class LEVELDESIGN_OT_prefab_add_library(Operator):
    """Add a .blend file as a prefab library"""
    bl_idname = "leveldesign.prefab_add_library"
    bl_label = "Add Library"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.blend", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _poll_prefab_scene_mode(context)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene
        abs_path = normalize_path(self.filepath)
        if not abs_path or not os.path.isfile(abs_path):
            self.report({'ERROR'}, "Pick a .blend file")
            return {'CANCELLED'}

        for existing in scene.anvil_prefab_libraries:
            if normalize_path(existing.filepath) == abs_path:
                self.report({'WARNING'}, f"Library already added: {abs_path}")
                return {'CANCELLED'}

        lib_entry = scene.anvil_prefab_libraries.add()
        lib_entry.filepath = abs_path
        if not refresh_library_objects(lib_entry):
            scene.anvil_prefab_libraries.remove(len(scene.anvil_prefab_libraries) - 1)
            self.report({'ERROR'}, f"Failed to read: {abs_path}")
            return {'CANCELLED'}

        invalidate_preview_cache(abs_path)
        scene.anvil_prefab_active_library_index = len(scene.anvil_prefab_libraries) - 1
        return {'FINISHED'}


class LEVELDESIGN_OT_prefab_remove_library(Operator):
    """Remove the selected prefab library"""
    bl_idname = "leveldesign.prefab_remove_library"
    bl_label = "Remove Library"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not _poll_prefab_scene_mode(context):
            return False
        scene = context.scene
        idx = scene.anvil_prefab_active_library_index
        return 0 <= idx < len(scene.anvil_prefab_libraries)

    def execute(self, context):
        scene = context.scene
        idx = scene.anvil_prefab_active_library_index
        invalidate_preview_cache(scene.anvil_prefab_libraries[idx].filepath)
        scene.anvil_prefab_libraries.remove(idx)
        new_count = len(scene.anvil_prefab_libraries)
        if new_count == 0:
            scene.anvil_prefab_active_library_index = 0
        elif idx >= new_count:
            scene.anvil_prefab_active_library_index = new_count - 1
        return {'FINISHED'}


class LEVELDESIGN_OT_prefab_refresh_libraries(Operator):
    """Reload all prefab libraries and rescan their object lists"""
    bl_idname = "leveldesign.prefab_refresh_libraries"
    bl_label = "Refresh Libraries"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _poll_prefab_scene_mode(context)

    def execute(self, context):
        scene = context.scene
        missing = []
        failed_reload = []
        for lib_entry in scene.anvil_prefab_libraries:
            loaded = find_loaded_library(lib_entry.filepath)
            if loaded is not None:
                if not reload_library(loaded):
                    failed_reload.append(loaded.name)
            invalidate_preview_cache(lib_entry.filepath)
            if not refresh_library_objects(lib_entry):
                missing.append(lib_entry.filepath)

        msgs = []
        if missing:
            msgs.append(f"Missing libraries: {', '.join(missing)}")
        if failed_reload:
            msgs.append(f"Failed to reload: {', '.join(failed_reload)}")
        if msgs:
            self.report({'WARNING'}, " | ".join(msgs))
        else:
            self.report({'INFO'}, "Prefab libraries refreshed")
        browser.update_prefab_browser_availability(scene)
        return {'FINISHED'}


class LEVELDESIGN_OT_prefab_reset_random_transform(Operator):
    """Reset prefab placement randomization values"""
    bl_idname = "leveldesign.prefab_reset_random_transform"
    bl_label = "Reset Prefab Randomization"
    bl_options = {'REGISTER', 'UNDO'}

    target: StringProperty(options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _poll_prefab_scene_mode(context)

    def execute(self, context):
        if self.target not in {'SIZE', 'ROTATION', 'ALL'}:
            self.report({'ERROR'}, f"Invalid prefab randomization target: {self.target}")
            return {'CANCELLED'}

        props = context.scene.level_design_props
        if self.target in {'SIZE', 'ALL'}:
            props.prefab_random_scale_enabled = False
            props.prefab_random_scale_min = _PREFAB_IDENTITY_SCALE
            props.prefab_random_scale_min_linked = True
            props.prefab_random_scale_max = _PREFAB_IDENTITY_SCALE
            props.prefab_random_scale_max_linked = True

        if self.target in {'ROTATION', 'ALL'}:
            props.prefab_random_rotation_enabled = False
            props.prefab_random_rotation_min = _PREFAB_ZERO_ROTATION
            props.prefab_random_rotation_max = _PREFAB_ZERO_ROTATION

        self.report({'INFO'}, "Prefab randomization reset")
        return {'FINISHED'}


class LEVELDESIGN_OT_prefab_instantiate(DefaultGridPivotMixin, ModalDrawBase, Operator):
    """Enter prefab placement mode or instantiate a prefab directly"""
    bl_idname = "leveldesign.prefab_instantiate"
    bl_label = "Instantiate Prefab"
    bl_options = {'REGISTER', 'UNDO'}

    library_index: IntProperty(
        name="Library Index",
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    source_object_name: StringProperty(
        name="Source Object",
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    repeat_source_object_name: StringProperty(
        name="Repeat Source Object",
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    asset_type: StringProperty(
        name="Asset Type",
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    object_name: StringProperty(
        name="Object Name",
        description="Name for the placed prefab object before its optional suffix",
    )
    name_suffix: StringProperty(
        name="Suffix",
        description="Suffix appended after Blender numbering, e.g. Tree.001-col",
        default="",
    )
    make_fully_local: BoolProperty(
        name="Make Local",
        description="Make the prefab hierarchy and all of its data local to this file",
        default=False,
    )
    action_pivot: FloatVectorProperty(
        size=3,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    action_normal: FloatVectorProperty(
        size=3,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    action_rotation: FloatProperty(
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    action_random_scale: FloatVectorProperty(
        size=3,
        default=_PREFAB_IDENTITY_SCALE,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    action_random_rotation: FloatVectorProperty(
        size=3,
        subtype='EULER',
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    placement_rotation: FloatProperty(
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    use_placement: BoolProperty(
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    @classmethod
    def poll(cls, context):
        return _poll_prefab_scene_mode(context)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = layout.column(align=True)
        settings.prop(self, "object_name")
        settings.prop(self, "name_suffix")
        settings.prop(self, "make_fully_local")

    def _sample_random_transform_from_scene(self, scene):
        props = scene.level_design_props
        self._random_transform_settings_key = _prefab_random_settings_key(props)
        self._prefab_random_scale, self._prefab_random_rotation = _sample_prefab_random_transform(
            props,
        )

    def _sync_random_transform_from_scene(self, scene):
        props = scene.level_design_props
        settings_key = _prefab_random_settings_key(props)
        if settings_key == getattr(self, "_random_transform_settings_key", None):
            return False

        self._random_transform_settings_key = settings_key
        self._prefab_random_scale, self._prefab_random_rotation = _sample_prefab_random_transform(
            props,
        )
        return True

    def invoke(self, context, event):
        asset_type = self.asset_type or 'OBJECT'
        if self.repeat_source_object_name:
            placement_source, error = _find_repeat_prefab_object(
                self.repeat_source_object_name,
            )
        else:
            placement_source, _abs_path, _reused, error = _resolve_prefab_linked_asset(
                context.scene,
                self.library_index,
                self.source_object_name,
                asset_type,
            )
        if placement_source is None:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}

        self.asset_type = asset_type
        self._placement_rotation = _normalize_rotation_angle(self.placement_rotation)
        self._sample_random_transform_from_scene(context.scene)
        self._rotation_drag_active = False
        self._rotation_drag_start_mouse_x = 0
        self._rotation_drag_start_rotation = self._placement_rotation
        self._inherit_normal = context.scene.level_design_props.prefab_inherit_normal
        self._prefab_ghost = build_prefab_albedo_ghost(placement_source)
        self._ghost_base_matrix = placement_source.matrix_basis.copy()
        set_repeat_prefab_override(
            context.scene,
            self.library_index,
            self.source_object_name,
            asset_type,
            self._placement_rotation,
        )
        return super().invoke(context, event)

    def _is_valid_mode(self, context):
        return context.mode in ('EDIT_MESH', 'OBJECT')

    def _update_first_vertex_preview(self, context, event):
        if self._preview is None:
            return
        self._preview.set_prefab_ghost(self._prefab_ghost)

        if event.shift:
            if not self._rotation_drag_active:
                if not self._begin_rotation_drag(context, event):
                    super()._update_first_vertex_preview(context, event)
                    self._update_prefab_ghost_transform_from_preview(context.scene)
                    return
            self._update_rotation_from_mouse_drag(context, event)
            self._update_prefab_ghost_transform_from_preview(context.scene)
            self._update_header(context)
            return

        if self._rotation_drag_active:
            self._rotation_drag_active = False

        super()._update_first_vertex_preview(context, event)
        self._update_prefab_ghost_transform_from_preview(context.scene)

    def _update_prefab_ghost_transform_from_preview(self, scene):
        if self._preview is None:
            return
        if getattr(self, "_prefab_ghost", None) is None:
            return
        self._sync_random_transform_from_scene(scene)
        self._preview.set_prefab_ghost(self._prefab_ghost)
        snap_point = self._preview._snap_point
        if snap_point is None:
            self._preview.update_prefab_ghost_matrix(None)
            return
        matrix = _prefab_placement_matrix(
            self._ghost_base_matrix,
            snap_point,
            self._placement_rotation,
            _prefab_placement_up(self._inherit_normal, self._preview._face_plane_normal),
            self._prefab_random_scale,
            self._prefab_random_rotation,
        )
        self._preview.update_prefab_ghost_matrix(matrix)

    def _begin_rotation_drag(self, context, event):
        if self._preview is None:
            return False
        if self._preview._snap_point is None:
            super()._update_first_vertex_preview(context, event)
            self._update_prefab_ghost_transform_from_preview(context.scene)
        if self._preview._snap_point is None:
            return False

        self._rotation_drag_active = True
        self._rotation_drag_start_mouse_x = event.mouse_region_x
        self._rotation_drag_start_rotation = self._placement_rotation
        return True

    def _update_rotation_from_mouse_drag(self, context, event):
        delta_x = event.mouse_region_x - self._rotation_drag_start_mouse_x
        rotation = (
            self._rotation_drag_start_rotation
            + delta_x * _PREFAB_ROTATION_DRAG_RADIANS_PER_PIXEL
        )
        rotation = _snap_prefab_rotation(context.tool_settings, rotation)
        self._placement_rotation = _normalize_rotation_angle(rotation)

    def _rotate_prefab_by_quarter_turn(self, context, direction):
        self._placement_rotation = _normalize_rotation_angle(
            self._placement_rotation + _PREFAB_QUARTER_TURN * direction
        )
        self._rotation_drag_start_rotation = self._placement_rotation
        self._update_prefab_ghost_transform_from_preview(context.scene)
        target = getattr(self, "_active_view_target", None)
        if target is not None and target.is_live():
            with context.temp_override(**target.override_kwargs()):
                self._update_header(context)
        else:
            self._update_header(context)
        utils.tag_redraw_all_3d_views()

    def _toggle_inherit_normal(self, context):
        self._inherit_normal = not self._inherit_normal
        context.scene.level_design_props.prefab_inherit_normal = self._inherit_normal
        self._update_prefab_ghost_transform_from_preview(context.scene)
        target = getattr(self, "_active_view_target", None)
        if target is not None and target.is_live():
            with context.temp_override(**target.override_kwargs()):
                self._update_header(context)
        else:
            self._update_header(context)
        utils.tag_redraw_all_3d_views()

    def _reroll_random_transform(self, context):
        self._sample_random_transform_from_scene(context.scene)
        self._update_prefab_ghost_transform_from_preview(context.scene)
        target = getattr(self, "_active_view_target", None)
        if target is not None and target.is_live():
            with context.temp_override(**target.override_kwargs()):
                self._update_header(context)
        else:
            self._update_header(context)
        utils.tag_redraw_all_3d_views()

    def _handle_rotation_modifier_event(self, context, event):
        if self._state != self.STATE_FIRST_VERTEX:
            return None
        if event.type not in {'LEFT_SHIFT', 'RIGHT_SHIFT'}:
            return None

        target = getattr(self, "_active_view_target", None)
        if target is None or not target.is_live():
            return None

        if event.value == 'PRESS':
            fake_event = self._synthetic_event_for_last_mouse(event.ctrl, True, event.alt)
            if fake_event is not None:
                with context.temp_override(**target.override_kwargs()):
                    self._is_2d_view = utils.is_2d_view(context)
                    self._begin_rotation_drag(context, fake_event)
                    self._update_header(context)
                    utils.tag_redraw_all_3d_views()
            return {'PASS_THROUGH'}

        if event.value == 'RELEASE':
            self._rotation_drag_active = False
            with context.temp_override(**target.override_kwargs()):
                self._update_header(context)
                utils.tag_redraw_all_3d_views()
            return {'PASS_THROUGH'}

        return None

    def modal(self, context, event):
        if (event.value == 'PRESS'
                and getattr(self, "_state", None) == self.STATE_FIRST_VERTEX):
            if event.type == 'W':
                self._toggle_inherit_normal(context)
                return {'RUNNING_MODAL'}

            if event.type == 'R':
                self._reroll_random_transform(context)
                return {'RUNNING_MODAL'}

            rotation_direction = _prefab_rotation_key_direction(
                context.window_manager,
                event,
            )
            if rotation_direction != 0:
                self._rotate_prefab_by_quarter_turn(context, rotation_direction)
                return {'RUNNING_MODAL'}

        modifier_result = self._handle_rotation_modifier_event(context, event)
        if modifier_result is not None:
            return modifier_result

        return super().modal(context, event)

    def _confirm_first_vertex(self, context, event):
        if (event.shift or self._rotation_drag_active) and self._preview is not None:
            snapped = self._preview._snap_point
            face_normal = self._preview._face_plane_normal
            if snapped is not None:
                snapped = snapped.copy()

        else:
            snapped = None
            face_normal = None

        if snapped is None and self._is_2d_view:
            snapped, face_normal = self._calculate_first_vertex_snap_2d(context, event)
        elif snapped is None and self._axis_lock_normal is not None:
            snapped, face_normal, _obj, _was_clamped = snapping.calculate_first_vertex_snap_3d_on_plane(
                context,
                event,
                self._axis_lock_plane_point,
                self._axis_lock_normal,
            )
        elif snapped is None:
            snapped, face_normal, _obj, _was_clamped = self._calculate_first_vertex_snap_3d(
                context, event
            )

        if snapped is None:
            return {'RUNNING_MODAL'}

        self._update_prefab_ghost_transform_from_preview(context.scene)
        pivot = snapped
        local_z = _prefab_placement_up(self._inherit_normal, face_normal)
        result = self._run_action(
            context,
            pivot,
            pivot,
            0.0,
            Vector((1, 0, 0)),
            Vector((0, 1, 0)),
            local_z,
        )
        success, message = result[0], result[1]

        if not getattr(self, "_action_reported", False):
            if success:
                self.report({'INFO'}, message)
            else:
                self.report({'ERROR'}, message)

        self._cleanup(context)
        if success:
            return {'FINISHED'}
        return {'CANCELLED'}

    def _execute_action(self, context, pivot, second_vertex, depth,
                        local_x, local_y, local_z):
        return self._instantiate_at_pivot(
            context,
            pivot,
            self._placement_rotation,
            local_z,
            self._prefab_random_scale,
            self._prefab_random_rotation,
        )

    def _capture_action_properties(self, context, pivot, second_vertex,
                                   depth, local_x, local_y, local_z):
        self._sync_random_transform_from_scene(context.scene)
        self.action_pivot = pivot
        self.action_normal = _prefab_safe_up_vector(local_z)
        self.action_rotation = self._placement_rotation
        self.action_random_scale = self._prefab_random_scale
        self.action_random_rotation = self._prefab_random_rotation
        self.use_placement = True

    def _instantiate_at_pivot(
            self,
            context,
            pivot,
            rotation,
            up_vector,
            random_scale,
            random_rotation):
        if context.mode == 'EDIT_MESH':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError as exc:
                return (False, f"Could not leave edit mode: {exc}")

        asset_type = self.asset_type or 'OBJECT'
        placement_matrix = _prefab_placement_matrix(
            self._ghost_base_matrix,
            pivot,
            rotation,
            up_vector,
            random_scale,
            random_rotation,
        )
        if self.repeat_source_object_name:
            repeat_source, error = _find_repeat_prefab_object(
                self.repeat_source_object_name,
            )
            if repeat_source is None:
                return (False, error)
            override, message = _instantiate_repeated_prefab_object(
                context.scene,
                context.collection,
                context.view_layer,
                repeat_source,
                self.object_name,
                self.source_object_name,
                placement_matrix,
                self.name_suffix,
            )
        else:
            override, message = _instantiate_prefab_object(
                context.scene,
                context.collection,
                context.view_layer,
                self.library_index,
                self.source_object_name,
                self.object_name,
                asset_type,
                placement_matrix,
                self.name_suffix,
                self.make_fully_local,
            )
        if override is None:
            return (False, message)

        set_repeat_prefab_on_object(
            override,
            context.scene,
            self.library_index,
            self.source_object_name,
            asset_type,
            rotation,
        )
        return (True, message)

    def execute(self, context):
        if self.use_placement:
            pivot = Vector(self.action_pivot)
            up_vector = _prefab_safe_up_vector(Vector(self.action_normal))
            rotation = _normalize_rotation_angle(self.action_rotation)
            random_scale = Vector(self.action_random_scale)
            random_rotation = Vector(self.action_random_rotation)
        else:
            pivot = Vector((0.0, 0.0, 0.0))
            up_vector = _PREFAB_UP_FALLBACK.copy()
            rotation = _normalize_rotation_angle(self.placement_rotation)
            random_scale, random_rotation = _sample_prefab_random_transform(
                context.scene.level_design_props,
            )

        asset_type = self.asset_type or 'OBJECT'
        if self.repeat_source_object_name:
            placement_source, error = _find_repeat_prefab_object(
                self.repeat_source_object_name,
            )
        else:
            placement_source, _abs_path, _reused, error = _resolve_prefab_linked_asset(
                context.scene,
                self.library_index,
                self.source_object_name,
                asset_type,
            )
        if placement_source is None:
            self._last_action_result = (False, error)
            self.report({'ERROR'}, error)
            self._action_reported = True
            return {'CANCELLED'}

        self.asset_type = asset_type
        self._ghost_base_matrix = placement_source.matrix_basis.copy()

        result = self._instantiate_at_pivot(
            context,
            pivot,
            rotation,
            up_vector,
            random_scale,
            random_rotation,
        )
        self._last_action_result = result
        success, message = result[0], result[1]
        if browser.prefab_browser_modal.is_popup_window(context.window):
            browser.prefab_browser_modal.restore_preferences(
                context.preferences,
                context.window_manager.windows,
                True,
                True,
            )
            try:
                bpy.ops.wm.window_close()
            except RuntimeError as exc:
                debug_log(f"[Prefabs] Could not close prefab browser window after add: {exc}")
        if success:
            self.report({'INFO'}, message)
            self._action_reported = True
            return {'FINISHED'}

        self.report({'ERROR'}, message)
        self._action_reported = True
        return {'CANCELLED'}

    def _get_tool_name(self):
        return "Prefab Placement"

    def _cleanup(self, context):
        context.workspace.status_text_set(None)
        super()._cleanup(context)

    def _update_header(self, context):
        rotation_degrees = math.degrees(self._placement_rotation) % 360.0
        normal_indicator = " [Inherit Normal]" if self._inherit_normal else " [World Up]"
        lock_indicator = " [Grid Locked]" if self._axis_lock_normal is not None else ""
        props = context.scene.level_design_props
        random_labels = []
        if props.prefab_random_scale_enabled:
            random_labels.append("Random Size")
        if props.prefab_random_rotation_enabled:
            random_labels.append("Random Rotation")
        random_indicator = ""
        if random_labels:
            random_indicator = f" [{' + '.join(random_labels)}]"
        if self._rotation_drag_active:
            rotation_hint = f"Shift-drag: Rotate {rotation_degrees:.1f} deg"
        else:
            rotation_hint = f"Shift-drag: Rotate ({rotation_degrees:.1f} deg)"
        context.workspace.status_text_set(
            f"W: Inherit Normal    R: Reroll Random    Q: Rotate Left    E: Rotate Right    {rotation_hint}    Ctrl: Lock Grid    LMB: Place    Esc: Cancel{normal_indicator}{lock_indicator}{random_indicator}"
        )


class LEVELDESIGN_OT_set_prefab_mode(Operator):
    """Set this scene's prefab mode (Scene vs Library)"""
    bl_idname = "leveldesign.set_prefab_mode"
    bl_label = "Set Prefab Mode"
    bl_options = {'REGISTER', 'UNDO'}

    mode: StringProperty()

    @classmethod
    def poll(cls, context):
        return _poll_level_design(context)

    def execute(self, context):
        if self.mode not in {'SCENE', 'LIBRARY'}:
            self.report({'ERROR'}, f"Invalid mode: {self.mode}")
            return {'CANCELLED'}
        context.scene.anvil_prefab_mode = self.mode
        return {'FINISHED'}


class LEVELDESIGN_OT_prefab_make_free_objects_assets(Operator):
    """Mark all free objects as prefab assets"""
    bl_idname = "leveldesign.prefab_make_free_objects_assets"
    bl_label = "Make All Free Objects Assets"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _poll_prefab_library_mode(context)

    def execute(self, context):
        scene = context.scene
        marked_count = make_all_free_objects_assets(scene)
        capture_library_previews(scene)
        invalidate_preview_cache(bpy.data.filepath)
        self.report(
            {'INFO'},
            f"Marked {marked_count} object(s) as prefab assets and generated prefab previews",
        )
        return {'FINISHED'}


class LEVELDESIGN_OT_prefab_top_level_modifier_warning(Operator):
    """Explain why top-level prefab modifiers will not stay linked"""
    bl_idname = "leveldesign.prefab_top_level_modifier_warning"
    bl_label = "Top-Level Modifier Warning"
    bl_description = (
        "This prefab has a modifier on its top-level object. When placed, that object and its "
        "modifiers become local and will not receive library updates. Put updateable modifiers "
        "on a child object instead"
    )
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return _poll_prefab_library_mode(context)

    def execute(self, context):
        self.report({'WARNING'}, self.bl_description)
        return {'CANCELLED'}


class LEVELDESIGN_OT_prefab_clear_asset(Operator):
    """Remove this object from prefab assets"""
    bl_idname = "leveldesign.prefab_clear_asset"
    bl_label = "Remove Prefab Asset"
    bl_options = {'REGISTER', 'UNDO'}

    asset_type: StringProperty()
    asset_name: StringProperty()

    @classmethod
    def poll(cls, context):
        return _poll_prefab_library_mode(context)

    def execute(self, context):
        scene = context.scene
        if clear_prefab_asset(scene, self.asset_type, self.asset_name):
            invalidate_preview_cache(bpy.data.filepath)
            self.report({'INFO'}, f"Removed prefab asset: {self.asset_name}")
            return {'FINISHED'}
        self.report({'WARNING'}, f"Prefab asset not found: {self.asset_name}")
        return {'CANCELLED'}


class LEVELDESIGN_OT_prefab_select_asset(Operator):
    """Select this prefab asset in the current scene"""
    bl_idname = "leveldesign.prefab_select_asset"
    bl_label = "Select Prefab Asset"
    bl_options = {'REGISTER'}

    asset_type: StringProperty()
    asset_name: StringProperty()

    @classmethod
    def poll(cls, context):
        return _poll_prefab_library_mode(context)

    def execute(self, context):
        view_layer = context.view_layer
        if select_prefab_asset(view_layer, self.asset_type, self.asset_name):
            focus_selected_in_3d_views(context)
            return {'FINISHED'}
        self.report({'WARNING'}, f"Prefab asset not found: {self.asset_name}")
        return {'CANCELLED'}


class LEVELDESIGN_OT_prefab_generate_previews(Operator):
    """Generate viewport previews for existing prefab assets"""
    bl_idname = "leveldesign.prefab_generate_previews"
    bl_label = "Generate Previews"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _poll_prefab_library_mode(context)

    def execute(self, context):
        capture_library_previews(context.scene)
        invalidate_preview_cache(bpy.data.filepath)
        self.report({'INFO'}, "Generated prefab previews")
        return {'FINISHED'}


classes = (
    LEVELDESIGN_OT_prefab_rotate_left,
    LEVELDESIGN_OT_prefab_rotate_right,
    LEVELDESIGN_OT_prefab_add_library,
    LEVELDESIGN_OT_prefab_remove_library,
    LEVELDESIGN_OT_prefab_refresh_libraries,
    LEVELDESIGN_OT_prefab_reset_random_transform,
    browser.LEVELDESIGN_OT_prefab_browser,
    browser.LEVELDESIGN_OT_prefab_browser_set_library_filter,
    browser.LEVELDESIGN_OT_prefab_browser_interaction,
    browser.LEVELDESIGN_OT_prefab_browser_close,
    browser.LEVELDESIGN_OT_prefab_browser_fix_layout,
    LEVELDESIGN_OT_prefab_instantiate,
    LEVELDESIGN_OT_set_prefab_mode,
    LEVELDESIGN_OT_prefab_make_free_objects_assets,
    LEVELDESIGN_OT_prefab_top_level_modifier_warning,
    LEVELDESIGN_OT_prefab_clear_asset,
    LEVELDESIGN_OT_prefab_select_asset,
    LEVELDESIGN_OT_prefab_generate_previews,
)

_addon_keymaps = []

KEYMAPS_TO_REGISTER = [
    ("Object Mode", 'EMPTY'),
    ("Mesh", 'EMPTY'),
]


def _register_prefab_browser_keymap(kc, km_name, space_type):
    km = kc.keymaps.new(name=km_name, space_type=space_type)
    kmi = km.keymap_items.new(
        "leveldesign.prefab_browser",
        'THREE',
        'PRESS',
        shift=True,
        head=True,
    )
    _addon_keymaps.append((km, kmi))


def _register_prefab_rotation_keymap(kc, km_name, space_type, operator_id, key):
    km = kc.keymaps.new(name=km_name, space_type=space_type)
    kmi = km.keymap_items.new(
        operator_id,
        key,
        'PRESS',
    )
    _addon_keymaps.append((km, kmi))


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.anvil_prefab_browser_search = StringProperty(
        name="Search",
        description="Filter prefabs by name or library",
        options={'TEXTEDIT_UPDATE'},
        update=browser.prefab_browser_search_update,
    )
    bpy.types.WindowManager.anvil_prefab_browser_library_filter = IntProperty(
        name="Library Filter",
        description="Selected prefab library filter, or all libraries",
        default=-1,
        min=-1,
        update=browser.prefab_browser_library_filter_update,
    )
    bpy.types.WindowManager.anvil_prefab_browser_preview_scale = FloatProperty(
        name="Preview Scale",
        description="Prefab browser preview thumbnail size",
        min=0.0,
        max=100.0,
        default=50.0,
        subtype='PERCENTAGE',
        update=browser.prefab_browser_preview_scale_update,
    )
    bpy.types.WindowManager.anvil_prefab_browser_scroll_offset = IntProperty(
        name="Prefab Browser Scroll Offset",
        description="Remembered prefab browser scroll position",
        default=0,
        min=0,
        options={'HIDDEN'},
    )

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        for km_name, space_type in KEYMAPS_TO_REGISTER:
            _register_prefab_browser_keymap(kc, km_name, space_type)
            _register_prefab_rotation_keymap(
                kc,
                km_name,
                space_type,
                _PREFAB_ROTATE_LEFT_ID,
                'Q',
            )
            _register_prefab_rotation_keymap(
                kc,
                km_name,
                space_type,
                _PREFAB_ROTATE_RIGHT_ID,
                'E',
            )


def unregister():
    browser.prefab_browser_modal.restore_preferences(
        bpy.context.preferences,
        bpy.context.window_manager.windows,
        False,
        False,
    )
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    _addon_keymaps.clear()

    if hasattr(bpy.types.WindowManager, "anvil_prefab_browser_scroll_offset"):
        del bpy.types.WindowManager.anvil_prefab_browser_scroll_offset
    if hasattr(bpy.types.WindowManager, "anvil_prefab_browser_preview_scale"):
        del bpy.types.WindowManager.anvil_prefab_browser_preview_scale
    if hasattr(bpy.types.WindowManager, "anvil_prefab_browser_library_filter"):
        del bpy.types.WindowManager.anvil_prefab_browser_library_filter
    if hasattr(bpy.types.WindowManager, "anvil_prefab_browser_search"):
        del bpy.types.WindowManager.anvil_prefab_browser_search
    cleanup_preview_cache()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
