"""
Hotspot Mapping - Gizmo System

GPU drawing and interactive editing for line-based hotspots in Image Editor.
Draws cell outlines, bisecting lines, orientation icons, and a preview line
under the cursor. Handles click-to-split (full line by default, ctrl for
anchored partial line), line dragging, and line deletion.
"""

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader

from . import json_storage
from .properties import SNAP_SIZES
from ..utils import debug_log, is_hotspot_mapping_workspace


# ---------------------------------------------------------------------------
# Snap size adjustment operators
# ---------------------------------------------------------------------------

class HOTSPOT_OT_snap_size_up(bpy.types.Operator):
    """Increase hotspot snap size"""
    bl_idname = "hotspot.snap_size_up"
    bl_label = "Increase Snap Size"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return is_hotspot_mapping_workspace()

    def execute(self, context):
        props = context.scene.hotspot_mapping_props
        current = props.snap_size
        for size in SNAP_SIZES:
            if size > current:
                props.snap_size = size
                self.report({'INFO'}, f"Hotspot snap: {size}px")
                break
        else:
            self.report({'INFO'}, f"Hotspot snap: {current}px (max)")

        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()
        return {'FINISHED'}


class HOTSPOT_OT_snap_size_down(bpy.types.Operator):
    """Decrease hotspot snap size"""
    bl_idname = "hotspot.snap_size_down"
    bl_label = "Decrease Snap Size"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return is_hotspot_mapping_workspace()

    def execute(self, context):
        props = context.scene.hotspot_mapping_props
        current = props.snap_size
        for size in reversed(SNAP_SIZES):
            if size < current:
                props.snap_size = size
                self.report({'INFO'}, f"Hotspot snap: {size}px")
                break
        else:
            self.report({'INFO'}, f"Hotspot snap: {current}px (min)")

        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()
        return {'FINISHED'}


class HOTSPOT_OT_cycle_orientation(bpy.types.Operator):
    """Cycle hotspot orientation type"""
    bl_idname = "hotspot.cycle_orientation"
    bl_label = "Cycle Orientation"
    bl_options = {'REGISTER', 'UNDO'}

    cell_key: bpy.props.StringProperty(
        name="Cell Key",
        description="Key of the cell to cycle orientation",
        options={'SKIP_SAVE', 'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        if context.area.type != 'IMAGE_EDITOR':
            return False
        space = context.space_data
        return space.image is not None

    def execute(self, context):
        space = context.space_data
        image = space.image

        new_type = json_storage.cycle_cell_orientation(
            image.name, self.cell_key
        )
        if new_type:
            self.report({'INFO'}, f"Orientation: {new_type}")
            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.tag_redraw()
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to cycle orientation")
            return {'CANCELLED'}


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

class HOTSPOT_TOOL_edit(bpy.types.WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'VIEW'

    bl_idname = "hotspot.edit_tool"
    bl_label = "Hotspot Edit"
    bl_description = (
        "Click to add bisecting line (Ctrl for partial). "
        "Drag lines to move. X/Del to delete."
    )
    bl_icon = "ops.transform.resize"

    bl_keymap = (
        ("hotspot.click_select",
         {"type": 'LEFTMOUSE', "value": 'PRESS', "any": True},
         None),
        ("hotspot.update_cursor",
         {"type": 'MOUSEMOVE', "value": 'ANY', "any": True},
         None),
        ("hotspot.delete_line", {"type": 'X', "value": 'PRESS'},
         None),
        ("hotspot.delete_line", {"type": 'DEL', "value": 'PRESS'},
         None),
    )


# ---------------------------------------------------------------------------
# Visual styling
# ---------------------------------------------------------------------------

COLOR_CELL = (1.0, 0.5, 0.0, 0.8)               # Orange - cell outlines
COLOR_LINE = (1.0, 0.5, 0.0, 0.8)                # Orange - bisecting lines
COLOR_LINE_HOVER = (1.0, 0.8, 0.0, 0.9)          # Yellow - hovered line
COLOR_LINE_ACTIVE = (0.0, 1.0, 0.5, 0.9)         # Green - selected line
COLOR_PREVIEW = (0.4, 0.7, 1.0, 0.7)             # Blue - preview line
COLOR_ICON = (1.0, 1.0, 1.0, 0.9)                # White - orientation icons
LINE_WIDTH = 2.0
SPLIT_LINE_WIDTH = 2.0
PREVIEW_LINE_WIDTH = 2.0
LINE_HIT_THRESHOLD = 8  # pixels (screen space)

# Icon sizing
ICON_FRACTION_MAX = 1 / 20
ICON_FRACTION_MIN = 1 / 40
ZOOM_HIDE_FACTOR = 0.8
ZOOM_FULL_FACTOR = 5.0


# ---------------------------------------------------------------------------
# Module state for preview and hover
# ---------------------------------------------------------------------------

_draw_handler = None
_last_tool_idname = None

# Cached by update_cursor for the draw handler
_preview_axis = None        # "v" or "h"
_preview_pos = None         # absolute pixel position
_preview_extent = None      # (start, end) pixel range
_hovered_line_idx = -1      # index of hovered line, -1 if none
_cursor_ctrl = False        # whether ctrl is held (for preview extent)


def _check_tool_change():
    """Timer callback to check if active tool changed and redraw if needed."""
    global _last_tool_idname

    try:
        context = bpy.context
        if not context or not context.workspace:
            return 0.25

        if not is_hotspot_mapping_workspace():
            return 0.25

        tool = context.workspace.tools.from_space_image_mode('VIEW')
        current_idname = tool.idname if tool else None

        if current_idname != _last_tool_idname:
            _last_tool_idname = current_idname
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        area.tag_redraw()
    except Exception:
        pass

    return 0.25


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _pixel_to_view(x, y, img_w, img_h):
    """Convert pixel coords (y from top) to view/UV coords (y from bottom)."""
    return (x / img_w, (img_h - y) / img_h)


def _pixel_to_region(px, py, img_w, img_h, view2d):
    """Convert pixel coords to screen region coords."""
    vx, vy = _pixel_to_view(px, py, img_w, img_h)
    return view2d.view_to_region(vx, vy, clip=False)


def _region_to_pixel(mx, my, img_w, img_h, view2d):
    """Convert screen region coords to pixel coords (y from top)."""
    vx, vy = view2d.region_to_view(mx, my)
    return (vx * img_w, img_h - (vy * img_h))


# ---------------------------------------------------------------------------
# Hit detection
# ---------------------------------------------------------------------------

def _find_hit_line(context, event, image):
    """Find the bisecting line under the mouse cursor.

    Returns:
        The line's index, or -1 if none.
    """
    lines = json_storage.get_texture_lines(image.name)
    if not lines:
        return -1

    img_w = image.size[0] if image.size[0] > 0 else 1
    img_h = image.size[1] if image.size[1] > 0 else 1

    region = context.region
    view2d = region.view2d
    mx, my = event.mouse_region_x, event.mouse_region_y

    best_idx = -1
    best_dist = LINE_HIT_THRESHOLD

    for i, line in enumerate(lines):
        axis = line["axis"]
        pos = line["pos"]
        start = line["start"]
        end = line["end"]

        if axis == "v":
            sx, _ = _pixel_to_region(pos, 0, img_w, img_h, view2d)
            _, sy1 = _pixel_to_region(0, end, img_w, img_h, view2d)
            _, sy2 = _pixel_to_region(0, start, img_w, img_h, view2d)
            dist = abs(mx - sx)
            in_extent = sy1 <= my <= sy2
        else:
            _, sy = _pixel_to_region(0, pos, img_w, img_h, view2d)
            sx1, _ = _pixel_to_region(start, 0, img_w, img_h, view2d)
            sx2, _ = _pixel_to_region(end, 0, img_w, img_h, view2d)
            dist = abs(my - sy)
            in_extent = sx1 <= mx <= sx2

        if in_extent and dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def _find_hit_icon(context, event, image):
    """Find the cell whose orientation icon is under the cursor.

    Returns:
        The cell's key string, or None.
    """
    cells = json_storage.get_cells_with_orientations(image.name)
    if not cells:
        return None

    img_w = image.size[0] if image.size[0] > 0 else 1
    img_h = image.size[1] if image.size[1] > 0 else 1

    region = context.region
    view2d = region.view2d

    # Zoom calculation for icon sizing
    _, origin_y = view2d.view_to_region(0, 0, clip=False)
    _, unit_y = view2d.view_to_region(0, 1, clip=False)
    pixels_per_uv = abs(unit_y - origin_y)

    viewport_height = region.height
    zoom_hide = viewport_height * ZOOM_HIDE_FACTOR
    if pixels_per_uv < zoom_hide:
        return None

    zoom_full = viewport_height * ZOOM_FULL_FACTOR
    icon_size_max = viewport_height * ICON_FRACTION_MAX
    icon_size_min = viewport_height * ICON_FRACTION_MIN

    if pixels_per_uv >= zoom_full:
        icon_size = icon_size_max
    else:
        t = (pixels_per_uv - zoom_hide) / (zoom_full - zoom_hide)
        icon_size = icon_size_min + t * (icon_size_max - icon_size_min)

    mx, my = event.mouse_region_x, event.mouse_region_y
    hs = icon_size / 2 + 2

    for cx, cy, cw, ch, _orient, key in cells:
        icon_px = cx + cw / 2
        icon_py = cy + ch / 2
        rcx, rcy = _pixel_to_region(icon_px, icon_py, img_w, img_h, view2d)
        if abs(mx - rcx) <= hs and abs(my - rcy) <= hs:
            return key

    return None


def _compute_preview(context, event, image, use_anchored):
    """Compute the preview bisecting line for the cursor position.

    Args:
        context: Blender context.
        event: Mouse event.
        image: Blender image.
        use_anchored: If True (ctrl held), use anchored partial extent.

    Returns:
        Tuple (axis, pos, extent_start, extent_end) or None.
    """
    img_w = image.size[0] if image.size[0] > 0 else 1
    img_h = image.size[1] if image.size[1] > 0 else 1

    region = context.region
    view2d = region.view2d
    mx, my = event.mouse_region_x, event.mouse_region_y

    px, py = _region_to_pixel(mx, my, img_w, img_h, view2d)

    # Must be inside image bounds
    if px < 0 or px >= img_w or py < 0 or py >= img_h:
        return None

    # Suppress preview near icon centers
    cells = json_storage.get_cells_with_orientations(image.name)
    for cx, cy, cw, ch, _orient, _key in cells:
        icon_cx = cx + cw / 2
        icon_cy = cy + ch / 2
        dead_zone = min(cw, ch) / 4
        if abs(px - icon_cx) < dead_zone and abs(py - icon_cy) < dead_zone:
            return None

    # Determine axis from closest edge (loop-cut semantics:
    # nearest vertical edge → horizontal cut, and vice versa)
    # Find the cell the cursor is in for edge distance calculation
    cell = json_storage.find_cell_at_point(cells, px, py)
    if cell is not None:
        cx, cy, cw, ch = cell[0], cell[1], cell[2], cell[3]
    else:
        cx, cy, cw, ch = 0, 0, img_w, img_h

    dist_left = px - cx
    dist_right = (cx + cw) - px
    dist_top = py - cy
    dist_bottom = (cy + ch) - py

    min_horiz = min(dist_left, dist_right)
    min_vert = min(dist_top, dist_bottom)

    if min_horiz <= min_vert:
        axis = "h"
        raw_pos = py
    else:
        axis = "v"
        raw_pos = px

    # Snap
    props = context.scene.hotspot_mapping_props
    if props.snap_enabled:
        snap = props.snap_size
        snapped = round(raw_pos / snap) * snap
    else:
        snapped = int(raw_pos)

    # Compute extent
    lines = json_storage.get_texture_lines(image.name)
    if use_anchored:
        ext_start, ext_end = json_storage.find_anchors(
            lines, px, py, axis, img_w, img_h
        )
    else:
        if axis == "v":
            ext_start, ext_end = 0, img_h
        else:
            ext_start, ext_end = 0, img_w

    # Clamp within extent (at least 1px from edges)
    if axis == "v":
        snapped = max(1, min(snapped, img_w - 1))
    else:
        snapped = max(1, min(snapped, img_h - 1))

    # Don't show preview if extent is too small to split
    if ext_end - ext_start < 2:
        return None

    return (axis, snapped, ext_start, ext_end)


# ---------------------------------------------------------------------------
# Cursor update operator
# ---------------------------------------------------------------------------

class HOTSPOT_OT_update_cursor(bpy.types.Operator):
    """Update cursor and preview line based on mouse position"""
    bl_idname = "hotspot.update_cursor"
    bl_label = "Update Hotspot Cursor"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        global _preview_axis, _preview_pos, _preview_extent
        global _hovered_line_idx, _cursor_ctrl

        space = context.space_data
        if not space or space.type != 'IMAGE_EDITOR':
            return {'PASS_THROUGH'}

        image = space.image
        if not image or not json_storage.is_texture_hotspottable(image.name):
            _preview_axis = None
            _preview_pos = None
            _preview_extent = None
            _hovered_line_idx = -1
            context.window.cursor_set('DEFAULT')
            return {'PASS_THROUGH'}

        _cursor_ctrl = event.ctrl

        # Check if hovering a line
        hit_line = _find_hit_line(context, event, image)
        _hovered_line_idx = hit_line

        if hit_line >= 0:
            lines = json_storage.get_texture_lines(image.name)
            if hit_line < len(lines):
                if lines[hit_line]["axis"] == "v":
                    context.window.cursor_set('MOVE_X')
                else:
                    context.window.cursor_set('MOVE_Y')
            else:
                context.window.cursor_set('DEFAULT')
        else:
            hit_icon = _find_hit_icon(context, event, image)
            if hit_icon is not None:
                context.window.cursor_set('DEFAULT')
            else:
                context.window.cursor_set('CROSSHAIR')

        # Compute preview line (not shown when hovering a line)
        if hit_line < 0:
            preview = _compute_preview(context, event, image, event.ctrl)
            if preview is not None:
                _preview_axis = preview[0]
                _preview_pos = preview[1]
                _preview_extent = (preview[2], preview[3])
            else:
                _preview_axis = None
                _preview_pos = None
                _preview_extent = None
        else:
            _preview_axis = None
            _preview_pos = None
            _preview_extent = None

        context.area.tag_redraw()
        return {'PASS_THROUGH'}


# ---------------------------------------------------------------------------
# Click handler
# ---------------------------------------------------------------------------

class HOTSPOT_OT_click_select(bpy.types.Operator):
    """Click to add line, drag existing line, or cycle orientation"""
    bl_idname = "hotspot.click_select"
    bl_label = "Click Select Hotspot"
    bl_options = {'REGISTER', 'UNDO'}

    use_anchored: bpy.props.BoolProperty(
        name="Use Anchored",
        description="Add a partial line anchored between perpendicular lines",
        options={'SKIP_SAVE', 'HIDDEN'},
    )

    def invoke(self, context, event):
        space = context.space_data
        if not space or space.type != 'IMAGE_EDITOR':
            return {'PASS_THROUGH'}

        image = space.image
        if not image or not json_storage.is_texture_hotspottable(image.name):
            return {'PASS_THROUGH'}

        props = context.scene.hotspot_mapping_props

        # Priority 1: Click on orientation icon
        hit_icon = _find_hit_icon(context, event, image)
        if hit_icon is not None:
            debug_log(f"[Hotspots Click] ICON click on cell {hit_icon}")
            bpy.ops.hotspot.cycle_orientation(cell_key=hit_icon)
            return {'FINISHED'}

        # Priority 2: Click on line → select and start drag
        hit_line = _find_hit_line(context, event, image)
        if hit_line >= 0:
            debug_log(f"[Hotspots Click] LINE click on #{hit_line}")
            props.active_line_idx = hit_line
            context.area.tag_redraw()
            bpy.ops.hotspot.drag_line('INVOKE_DEFAULT',
                                      line_index=hit_line)
            return {'FINISHED'}

        # Priority 3: Click to add a new line
        use_anchored = self.use_anchored or event.ctrl
        preview = _compute_preview(context, event, image, use_anchored)
        if preview is not None:
            axis, pos, ext_start, ext_end = preview
            debug_log(f"[Hotspots Click] ADD line axis={axis} pos={pos} "
                      f"extent=[{ext_start},{ext_end}] "
                      f"anchored={use_anchored}")
            if json_storage.add_line(image.name, axis, pos,
                                     ext_start, ext_end):
                props.active_line_idx = -1
                for area in context.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        area.tag_redraw()
                return {'FINISHED'}

        return {'PASS_THROUGH'}


# ---------------------------------------------------------------------------
# Drag line operator
# ---------------------------------------------------------------------------

class HOTSPOT_OT_drag_line(bpy.types.Operator):
    """Drag a bisecting line to move it"""
    bl_idname = "hotspot.drag_line"
    bl_label = "Drag Line"
    bl_options = {'REGISTER', 'UNDO'}

    line_index: bpy.props.IntProperty(
        name="Line Index",
        options={'SKIP_SAVE', 'HIDDEN'},
    )

    _drag_start_pos = None
    _line_axis = None
    _move_range = None
    _texture_name = None
    _img_w = None
    _img_h = None

    @classmethod
    def poll(cls, context):
        if context.area.type != 'IMAGE_EDITOR':
            return False
        space = context.space_data
        return space.image is not None

    def invoke(self, context, event):
        space = context.space_data
        image = space.image
        self._texture_name = image.name

        self._img_w = image.size[0] if image.size[0] > 0 else 1
        self._img_h = image.size[1] if image.size[1] > 0 else 1

        lines = json_storage.get_texture_lines(self._texture_name)
        if self.line_index < 0 or self.line_index >= len(lines):
            return {'CANCELLED'}

        line = lines[self.line_index]
        self._line_axis = line["axis"]
        self._drag_start_pos = line["pos"]
        self._move_range = json_storage.get_line_move_range(
            lines, self.line_index, self._img_w, self._img_h
        )

        if self._line_axis == "v":
            context.window.cursor_modal_set('MOVE_X')
        else:
            context.window.cursor_modal_set('MOVE_Y')

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            self._update_drag(context, event)

        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            context.window.cursor_modal_restore()
            return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.window.cursor_modal_restore()
            json_storage.move_line(self._texture_name,
                                   self.line_index,
                                   self._drag_start_pos)
            context.area.tag_redraw()
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _update_drag(self, context, event):
        region = context.region
        view2d = region.view2d

        mx, my = event.mouse_region_x, event.mouse_region_y
        vx, vy = view2d.region_to_view(mx, my)

        if self._line_axis == "v":
            raw_pos = vx * self._img_w
        else:
            raw_pos = self._img_h - (vy * self._img_h)

        props = context.scene.hotspot_mapping_props
        if props.snap_enabled:
            snap = props.snap_size
            new_pos = round(raw_pos / snap) * snap
        else:
            new_pos = int(raw_pos)

        if self._move_range:
            range_min, range_max = self._move_range
            new_pos = max(range_min, min(new_pos, range_max))

        json_storage.move_line(self._texture_name, self.line_index,
                               new_pos)
        context.area.tag_redraw()


# ---------------------------------------------------------------------------
# Delete line operator
# ---------------------------------------------------------------------------

class HOTSPOT_OT_delete_line(bpy.types.Operator):
    """Delete the selected or hovered bisecting line"""
    bl_idname = "hotspot.delete_line"
    bl_label = "Delete Line"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not is_hotspot_mapping_workspace():
            return False
        if context.area.type != 'IMAGE_EDITOR':
            return False
        space = context.space_data
        if not space.image:
            return False
        props = context.scene.hotspot_mapping_props
        return props.active_line_idx >= 0 or _hovered_line_idx >= 0

    def execute(self, context):
        space = context.space_data
        image = space.image
        props = context.scene.hotspot_mapping_props

        # Prefer selected, fall back to hovered
        idx = props.active_line_idx
        if idx < 0:
            idx = _hovered_line_idx
        if idx < 0:
            self.report({'WARNING'}, "No line selected")
            return {'CANCELLED'}

        if json_storage.remove_line(image.name, idx):
            self.report({'INFO'}, "Removed line")
            props.active_line_idx = -1
            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.tag_redraw()
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to remove line")
            return {'CANCELLED'}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_hotspots():
    """Draw handler callback for hotspot visualization."""
    context = bpy.context

    if context.area is None or context.area.type != 'IMAGE_EDITOR':
        return

    if not is_hotspot_mapping_workspace():
        return

    tool = context.workspace.tools.from_space_image_mode('VIEW')
    if not tool or tool.idname != "hotspot.edit_tool":
        return

    space = context.space_data
    image = space.image
    if not image:
        return

    texture_name = image.name
    if not json_storage.is_texture_hotspottable(texture_name):
        return

    img_w = image.size[0] if image.size[0] > 0 else 1
    img_h = image.size[1] if image.size[1] > 0 else 1

    region = context.region
    view2d = region.view2d

    # Zoom calculation for icon sizing
    _, origin_y = view2d.view_to_region(0, 0, clip=False)
    _, unit_y = view2d.view_to_region(0, 1, clip=False)
    pixels_per_uv = abs(unit_y - origin_y)

    viewport_height = region.height
    zoom_hide = viewport_height * ZOOM_HIDE_FACTOR
    zoom_full = viewport_height * ZOOM_FULL_FACTOR
    icon_size_max = viewport_height * ICON_FRACTION_MAX
    icon_size_min = viewport_height * ICON_FRACTION_MIN

    if pixels_per_uv >= zoom_full:
        zoom_icon_size = icon_size_max
    elif pixels_per_uv <= zoom_hide:
        zoom_icon_size = icon_size_min
    else:
        t = (pixels_per_uv - zoom_hide) / (zoom_full - zoom_hide)
        zoom_icon_size = icon_size_min + t * (icon_size_max - icon_size_min)

    props = context.scene.hotspot_mapping_props
    active_line = props.active_line_idx

    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(LINE_WIDTH)

    try:
        # Pass 1: Draw cell outlines
        cells = json_storage.get_cells_with_orientations(texture_name)
        cell_regions = []
        for cx, cy, cw, ch, orientation, _key in cells:
            rx1, ry1 = _pixel_to_region(cx, cy + ch, img_w, img_h, view2d)
            rx2, ry2 = _pixel_to_region(cx + cw, cy, img_w, img_h, view2d)
            cell_regions.append((rx1, ry1, rx2, ry2, orientation))
            _draw_rectangle(rx1, ry1, rx2, ry2, COLOR_CELL)

        # Pass 2: Draw bisecting lines with highlighting
        lines = json_storage.get_texture_lines(texture_name)
        for i, line in enumerate(lines):
            axis = line["axis"]
            pos = line["pos"]
            start = line["start"]
            end = line["end"]

            if axis == "v":
                p1 = _pixel_to_region(pos, end, img_w, img_h, view2d)
                p2 = _pixel_to_region(pos, start, img_w, img_h, view2d)
            else:
                p1 = _pixel_to_region(start, pos, img_w, img_h, view2d)
                p2 = _pixel_to_region(end, pos, img_w, img_h, view2d)

            if i == active_line:
                color = COLOR_LINE_ACTIVE
            elif i == _hovered_line_idx:
                color = COLOR_LINE_HOVER
            else:
                color = COLOR_LINE

            _draw_line(p1, p2, color, SPLIT_LINE_WIDTH)

        # Pass 3: Preview bisecting line
        if (_preview_axis is not None and _preview_pos is not None
                and _preview_extent is not None):
            ext_s, ext_e = _preview_extent
            if _preview_axis == "v":
                p1 = _pixel_to_region(_preview_pos, ext_e, img_w, img_h,
                                      view2d)
                p2 = _pixel_to_region(_preview_pos, ext_s, img_w, img_h,
                                      view2d)
            else:
                p1 = _pixel_to_region(ext_s, _preview_pos, img_w, img_h,
                                      view2d)
                p2 = _pixel_to_region(ext_e, _preview_pos, img_w, img_h,
                                      view2d)
            _draw_line(p1, p2, COLOR_PREVIEW, PREVIEW_LINE_WIDTH)

        # Pass 4: Orientation icons
        if pixels_per_uv >= zoom_hide:
            for rx1, ry1, rx2, ry2, orientation in cell_regions:
                icon_cx = (rx1 + rx2) / 2
                icon_cy = (ry1 + ry2) / 2
                _draw_orientation_icon(icon_cx, icon_cy, orientation,
                                       COLOR_ICON, zoom_icon_size)

    finally:
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)


def _draw_rectangle(x1, y1, x2, y2, color):
    """Draw a rectangle outline."""
    points = [
        (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1),
    ]

    try:
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        region = bpy.context.region
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", LINE_WIDTH)
        shader.uniform_float("color", color)

        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": points})
        batch.draw(shader)
    except Exception:
        try:
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            shader.uniform_float("color", color)
            line_pairs = [
                (x1, y1), (x2, y1),
                (x2, y1), (x2, y2),
                (x2, y2), (x1, y2),
                (x1, y2), (x1, y1),
            ]
            batch = batch_for_shader(shader, 'LINES', {"pos": line_pairs})
            batch.draw(shader)
        except Exception:
            pass


def _draw_line(p1, p2, color, width):
    """Draw a single line segment."""
    try:
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        region = bpy.context.region
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", width)
        shader.uniform_float("color", color)

        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": [p1, p2]})
        batch.draw(shader)
    except Exception:
        try:
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            shader.uniform_float("color", color)
            batch = batch_for_shader(shader, 'LINES', {"pos": [p1, p2]})
            batch.draw(shader)
        except Exception:
            pass


def _draw_orientation_icon(cx, cy, orientation_type, color, icon_size):
    """Draw orientation symbol at center position (cx, cy)."""
    orientation_symbols = {
        'Any': '●',
        'Upwards': '↑',
        'Floor': '⌊',
        'Ceiling': '⌈',
    }
    symbol = orientation_symbols.get(orientation_type, '*')

    font_id = 0
    font_size = int(icon_size + 4)

    blf.size(font_id, font_size)
    blf.color(font_id, color[0], color[1], color[2], color[3])

    text_width, text_height = blf.dimensions(font_id, symbol)

    text_x = cx - text_width / 2
    text_y = cy - text_height / 2

    blf.position(font_id, text_x, text_y, 0)
    blf.draw(font_id, symbol)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_keymaps = []


def register():
    global _draw_handler, _keymaps

    bpy.utils.register_class(HOTSPOT_OT_snap_size_up)
    bpy.utils.register_class(HOTSPOT_OT_snap_size_down)
    bpy.utils.register_class(HOTSPOT_OT_cycle_orientation)
    bpy.utils.register_class(HOTSPOT_OT_update_cursor)
    bpy.utils.register_class(HOTSPOT_OT_drag_line)
    bpy.utils.register_class(HOTSPOT_OT_click_select)
    bpy.utils.register_class(HOTSPOT_OT_delete_line)

    _draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
        draw_hotspots, (), 'WINDOW', 'POST_PIXEL'
    )
    debug_log(f"[Hotspots] Draw handler registered: {_draw_handler}")

    bpy.utils.register_tool(HOTSPOT_TOOL_edit, after={"builtin.sample"},
                            separator=True)
    debug_log("[Hotspots] Tool registered: hotspot.edit_tool")

    bpy.app.timers.register(_check_tool_change, persistent=True)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Image Generic', space_type='IMAGE_EDITOR')

        kmi = km.keymap_items.new(
            'hotspot.snap_size_up',
            'RIGHT_BRACKET', 'PRESS',
            head=True
        )
        _keymaps.append((km, kmi))

        kmi = km.keymap_items.new(
            'hotspot.snap_size_down',
            'LEFT_BRACKET', 'PRESS',
            head=True
        )
        _keymaps.append((km, kmi))

        debug_log("[Hotspots] Snap keymaps registered")


def unregister():
    global _draw_handler, _keymaps

    if bpy.app.timers.is_registered(_check_tool_change):
        bpy.app.timers.unregister(_check_tool_change)

    for km, kmi in _keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    _keymaps.clear()

    bpy.utils.unregister_tool(HOTSPOT_TOOL_edit)

    if _draw_handler is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_draw_handler,
                                                       'WINDOW')
        _draw_handler = None

    bpy.utils.unregister_class(HOTSPOT_OT_delete_line)
    bpy.utils.unregister_class(HOTSPOT_OT_click_select)
    bpy.utils.unregister_class(HOTSPOT_OT_drag_line)
    bpy.utils.unregister_class(HOTSPOT_OT_update_cursor)
    bpy.utils.unregister_class(HOTSPOT_OT_cycle_orientation)
    bpy.utils.unregister_class(HOTSPOT_OT_snap_size_down)
    bpy.utils.unregister_class(HOTSPOT_OT_snap_size_up)
