"""
Visible Select

Toolbar tool that raycasts past backface-culled faces, allowing selection of
visible geometry behind them without X-ray mode.
"""

import bpy
from . import operator
from ...core.workspace_check import is_level_design_workspace


_CURSOR_TOOL_ID = "builtin.cursor"
_TOOLBAR_ENTRY_ATTR = "_level_design_toolbar_entry"
_APPLY_IMAGE_TO_FACE_ID = "leveldesign.apply_image_to_face"
_STRETCH_APPLY_IMAGE_TO_FACE_ID = "leveldesign.stretch_apply_image_to_face"
_APPLY_UV_TRANSFORM_TO_FACE_ID = "leveldesign.apply_uv_transform_to_face"
_PICK_IMAGE_FROM_FACE_ID = "leveldesign.pick_image_from_face"
_STRETCH_PICK_IMAGE_FROM_FACE_ID = "leveldesign.stretch_pick_image_from_face"
_PICK_UV_TRANSFORM_FROM_FACE_ID = "leveldesign.pick_uv_transform_from_face"


class LEVELDESIGN_TOOL_visible_select_edit_mesh(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'

    bl_idname = "leveldesign.visible_select_edit_mesh"
    bl_label = "Visible Select (Edit Mode)"
    bl_description = "Select visible edit mesh elements through culled surfaces"
    bl_icon = "ops.generic.select"
    bl_options = {'KEYMAP_FALLBACK'}
    bl_widget = None
    bl_keymap = (
        (
            operator.LEVELDESIGN_OT_visible_select.bl_idname,
            {"type": 'LEFTMOUSE', "value": 'PRESS'},
            {"properties": [("extend", False), ("loop", False)]},
        ),
        (
            operator.LEVELDESIGN_OT_visible_select.bl_idname,
            {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True},
            {"properties": [("extend", True), ("loop", False)]},
        ),
        (
            operator.LEVELDESIGN_OT_visible_select.bl_idname,
            {"type": 'LEFTMOUSE', "value": 'PRESS', "alt": True},
            {"properties": [("extend", False), ("loop", True)]},
        ),
        (
            operator.LEVELDESIGN_OT_visible_select.bl_idname,
            {"type": 'LEFTMOUSE', "value": 'PRESS',
             "shift": True, "alt": True},
            {"properties": [("extend", True), ("loop", True)]},
        ),
        (
            operator.LEVELDESIGN_OT_visible_shortest_path_pick.bl_idname,
            {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True},
            {"properties": [("use_fill", False)]},
        ),
        (
            operator.LEVELDESIGN_OT_visible_shortest_path_pick.bl_idname,
            {"type": 'LEFTMOUSE', "value": 'PRESS',
             "ctrl": True, "shift": True},
            {"properties": [("use_fill", True)]},
        ),
        # Keep the texture paint/pick/apply shortcuts active while this tool is
        # selected. Tool keymap items later in this tuple get the first chance,
        # so the apply operators can consume Alt-click when exactly one face is
        # selected and PASS_THROUGH back to visible loop selection otherwise.
        (
            _PICK_IMAGE_FROM_FACE_ID,
            {"type": 'RIGHTMOUSE', "value": 'PRESS', "alt": True},
            {"properties": []},
        ),
        (
            _STRETCH_PICK_IMAGE_FROM_FACE_ID,
            {"type": 'RIGHTMOUSE', "value": 'PRESS',
             "shift": True, "alt": True},
            {"properties": []},
        ),
        (
            _PICK_UV_TRANSFORM_FROM_FACE_ID,
            {"type": 'RIGHTMOUSE', "value": 'PRESS', "ctrl": True, "alt": True},
            {"properties": []},
        ),
        (
            _APPLY_IMAGE_TO_FACE_ID,
            {"type": 'LEFTMOUSE', "value": 'PRESS', "alt": True},
            {"properties": []},
        ),
        (
            _STRETCH_APPLY_IMAGE_TO_FACE_ID,
            {"type": 'LEFTMOUSE', "value": 'PRESS',
             "shift": True, "alt": True},
            {"properties": []},
        ),
        (
            _APPLY_UV_TRANSFORM_TO_FACE_ID,
            {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True, "alt": True},
            {"properties": []},
        ),
    )


class LEVELDESIGN_TOOL_visible_select_object(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'OBJECT'

    bl_idname = "leveldesign.visible_select_object"
    bl_label = "Visible Select (Object Mode)"
    bl_description = "Select visible objects through culled surfaces"
    bl_icon = "ops.generic.select"
    bl_options = {'KEYMAP_FALLBACK'}
    bl_widget = None
    bl_keymap = (
        (
            operator.LEVELDESIGN_OT_visible_object_select.bl_idname,
            {"type": 'LEFTMOUSE', "value": 'PRESS'},
            {"properties": [("extend", False)]},
        ),
        (
            operator.LEVELDESIGN_OT_visible_object_select.bl_idname,
            {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True},
            {"properties": [("extend", True)]},
        ),
    )


def _remove_tool_def_from_toolbar(tools, tool_def):
    from bl_ui.space_toolsystem_common import ToolDef

    for index, item in enumerate(tools):
        if item is tool_def:
            del tools[index]
            return True

        if isinstance(item, tuple) and not isinstance(item, ToolDef):
            if tool_def not in item:
                continue

            remaining_items = tuple(
                sub_item for sub_item in item if sub_item is not tool_def
            )
            if remaining_items:
                tools[index] = remaining_items
            else:
                del tools[index]
            return True

    return False


def _insert_toolbar_item_before_cursor(tools, toolbar_item):
    for index, item in enumerate(tools):
        if getattr(item, "idname", None) == _CURSOR_TOOL_ID:
            tools[index:index] = (toolbar_item,)
            return True

    tools.append(toolbar_item)
    return False


def _make_level_design_toolbar_entry(tool_def):
    def level_design_toolbar_entry(context):
        if context is None or is_level_design_workspace():
            return (tool_def,)
        return ()

    return level_design_toolbar_entry


def _toolbar_tools_for_tool(tool_cls):
    from bl_ui.space_toolsystem_common import ToolSelectPanelHelper

    toolbar_cls = ToolSelectPanelHelper._tool_class_from_space_type(
        tool_cls.bl_space_type
    )
    if toolbar_cls is None:
        return None

    return toolbar_cls._tools[tool_cls.bl_context_mode]


def _place_tool_before_cursor(tool_cls):
    # register_tool() inserts after builtin.select inside Blender's select tuple.
    # Lift the tool out so it draws as its own toolbar button before Cursor.
    tools = _toolbar_tools_for_tool(tool_cls)
    if tools is None:
        return

    tool_def = tool_cls._bl_tool

    if not _remove_tool_def_from_toolbar(tools, tool_def):
        return

    toolbar_entry = _make_level_design_toolbar_entry(tool_def)
    setattr(tool_cls, _TOOLBAR_ENTRY_ATTR, toolbar_entry)
    _insert_toolbar_item_before_cursor(tools, toolbar_entry)


def _restore_tool_def_for_unregister(tool_cls):
    toolbar_entry = getattr(tool_cls, _TOOLBAR_ENTRY_ATTR, None)
    if toolbar_entry is None:
        return

    tools = _toolbar_tools_for_tool(tool_cls)
    if tools is None:
        return

    for index, item in enumerate(tools):
        if item is toolbar_entry:
            tools[index] = tool_cls._bl_tool
            delattr(tool_cls, _TOOLBAR_ENTRY_ATTR)
            return

    delattr(tool_cls, _TOOLBAR_ENTRY_ATTR)


def register():
    operator.register()

    bpy.utils.register_tool(
        LEVELDESIGN_TOOL_visible_select_object,
        after={"builtin.select"},
        group=False,
    )
    _place_tool_before_cursor(LEVELDESIGN_TOOL_visible_select_object)
    bpy.utils.register_tool(
        LEVELDESIGN_TOOL_visible_select_edit_mesh,
        after={"builtin.select"},
        group=False,
    )
    _place_tool_before_cursor(LEVELDESIGN_TOOL_visible_select_edit_mesh)


def unregister():
    _restore_tool_def_for_unregister(LEVELDESIGN_TOOL_visible_select_edit_mesh)
    _restore_tool_def_for_unregister(LEVELDESIGN_TOOL_visible_select_object)
    bpy.utils.unregister_tool(LEVELDESIGN_TOOL_visible_select_edit_mesh)
    bpy.utils.unregister_tool(LEVELDESIGN_TOOL_visible_select_object)

    operator.unregister()
