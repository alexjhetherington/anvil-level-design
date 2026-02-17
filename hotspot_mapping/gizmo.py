"""
Hotspot Mapping - Gizmo System

GPU drawing and interactive editing for hotspot rectangles in Image Editor.
Uses a Blender Tool for proper mouse input handling.
"""

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader

from . import json_storage
from .properties import SNAP_SIZES
from ..utils import debug_log, is_hotspot_mapping_workspace


# Snap size adjustment operators
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
        # Find next larger size in list
        for size in SNAP_SIZES:
            if size > current:
                props.snap_size = size
                self.report({'INFO'}, f"Hotspot snap: {size}px")
                break
        else:
            self.report({'INFO'}, f"Hotspot snap: {current}px (max)")

        # Refresh UI panels
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
        # Find next smaller size in list
        for size in reversed(SNAP_SIZES):
            if size < current:
                props.snap_size = size
                self.report({'INFO'}, f"Hotspot snap: {size}px")
                break
        else:
            self.report({'INFO'}, f"Hotspot snap: {current}px (min)")

        # Refresh UI panels
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()
        return {'FINISHED'}


class HOTSPOT_OT_cycle_orientation(bpy.types.Operator):
    """Cycle hotspot orientation type"""
    bl_idname = "hotspot.cycle_orientation"
    bl_label = "Cycle Orientation"
    bl_options = {'REGISTER', 'UNDO'}

    hotspot_id: bpy.props.StringProperty(
        name="Hotspot ID",
        description="ID of the hotspot to cycle orientation",
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

        if not self.hotspot_id:
            self.report({'WARNING'}, "No hotspot specified")
            return {'CANCELLED'}

        new_type = json_storage.cycle_hotspot_orientation(image.name, self.hotspot_id)
        if new_type:
            self.report({'INFO'}, f"Orientation: {new_type}")
            # Force redraw
            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.tag_redraw()
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to cycle orientation")
            return {'CANCELLED'}


# Tool definition for Image Editor
class HOTSPOT_TOOL_edit(bpy.types.WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'VIEW'

    bl_idname = "hotspot.edit_tool"
    bl_label = "Hotspot Edit"
    bl_description = "Click and drag to move/resize hotspot regions"
    bl_icon = "ops.transform.resize"

    bl_keymap = (
        ("hotspot.click_select", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        ("hotspot.update_cursor", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    )


# Visual styling
COLOR_HOTSPOT = (1.0, 0.5, 0.0, 0.8)           # Orange - normal hotspot
COLOR_HOTSPOT_ACTIVE = (0.0, 1.0, 0.5, 0.9)    # Green - selected hotspot
COLOR_HOTSPOT_HOVER = (1.0, 0.8, 0.0, 0.9)     # Yellow - hovered hotspot
COLOR_HANDLE = (1.0, 1.0, 1.0, 0.9)            # White - resize handles
COLOR_ICON = (1.0, 1.0, 1.0, 0.9)              # White - orientation icons
LINE_WIDTH = 2.0
HANDLE_SIZE = 8  # pixels

# Icon sizing - relative to viewport and image size
# Thresholds are based on how much the image fills the viewport ("fit" zoom)
ICON_FRACTION_MAX = 1 / 20   # max icon size as fraction of viewport height
ICON_FRACTION_MIN = 1 / 40   # min icon size as fraction of viewport height
ZOOM_HIDE_FACTOR = 0.8       # hide icons when zoomed out past this fraction of fit zoom
ZOOM_FULL_FACTOR = 5.0       # icons reach max size at this multiple of fit zoom


# Global draw handler reference
_draw_handler = None
_last_tool_idname = None


def _check_tool_change():
    """Timer callback to check if active tool changed and redraw if needed."""
    global _last_tool_idname

    try:
        context = bpy.context
        if not context or not context.workspace:
            return 0.25

        tool = context.workspace.tools.from_space_image_mode('VIEW')
        current_idname = tool.idname if tool else None

        if current_idname != _last_tool_idname:
            _last_tool_idname = current_idname
            # Tool changed - redraw all Image Editors
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        area.tag_redraw()
    except Exception:
        pass

    return 0.25  # Check every 0.5 seconds


def _get_image_editor_image(context):
    """Get the image from the active Image Editor space."""
    space = context.space_data
    if space and space.type == 'IMAGE_EDITOR':
        return space.image
    return None


def _pixel_to_view(x, y, img_width, img_height):
    """Convert pixel coordinates to view (UV) coordinates.

    Blender's Image Editor view2d uses UV coordinates (0-1 range).
    We store hotspots in pixel coords with (0,0) at top-left.
    View coords have (0,0) at bottom-left.

    Args:
        x: X position in pixels (from left)
        y: Y position in pixels (from top, as stored in JSON)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple (view_x, view_y) in UV space (0-1 range)
    """
    # Convert to UV space (0-1) and flip Y
    view_x = x / img_width
    view_y = (img_height - y) / img_height
    return (view_x, view_y)


def _get_hit_zone_for_hotspot(context, event, hotspot, image):
    """Determine which part of a hotspot was clicked.

    Standalone function used by both click_select and interactive_edit.

    Returns: 'center', 'top', 'bottom', 'left', 'right',
             'top_left', 'top_right', 'bottom_left', 'bottom_right', or None
    """
    region = context.region
    view2d = region.view2d

    img_width = image.size[0] if image.size[0] > 0 else 1
    img_height = image.size[1] if image.size[1] > 0 else 1

    # Get hotspot bounds in view (UV) coords
    px = hotspot.get("x", 0)
    py = hotspot.get("y", 0)
    pw = hotspot.get("width", 0)
    ph = hotspot.get("height", 0)

    x1, y2 = _pixel_to_view(px, py, img_width, img_height)
    x2, y1 = _pixel_to_view(px + pw, py + ph, img_width, img_height)

    # Convert to region coords
    rx1, ry1 = view2d.view_to_region(x1, y1, clip=False)
    rx2, ry2 = view2d.view_to_region(x2, y2, clip=False)

    mx, my = event.mouse_region_x, event.mouse_region_y
    hs = HANDLE_SIZE

    # Check corners first (higher priority)
    if abs(mx - rx1) < hs and abs(my - ry1) < hs:
        return 'bottom_left'
    if abs(mx - rx2) < hs and abs(my - ry1) < hs:
        return 'bottom_right'
    if abs(mx - rx2) < hs and abs(my - ry2) < hs:
        return 'top_right'
    if abs(mx - rx1) < hs and abs(my - ry2) < hs:
        return 'top_left'

    # Check edges - detect anywhere along the edge, not just at midpoint
    on_left_edge = abs(mx - rx1) < hs and ry1 + hs < my < ry2 - hs
    on_right_edge = abs(mx - rx2) < hs and ry1 + hs < my < ry2 - hs
    on_bottom_edge = abs(my - ry1) < hs and rx1 + hs < mx < rx2 - hs
    on_top_edge = abs(my - ry2) < hs and rx1 + hs < mx < rx2 - hs

    if on_bottom_edge:
        return 'bottom'
    if on_top_edge:
        return 'top'
    if on_left_edge:
        return 'left'
    if on_right_edge:
        return 'right'

    # Check if inside rectangle (center move)
    if rx1 <= mx <= rx2 and ry1 <= my <= ry2:
        return 'center'

    return None


# Cursor mapping for different drag modes
CURSOR_FOR_MODE = {
    'center': 'HAND',
    'left': 'MOVE_X',
    'right': 'MOVE_X',
    'top': 'MOVE_Y',
    'bottom': 'MOVE_Y',
    'top_left': 'SCROLL_XY',
    'top_right': 'SCROLL_XY',
    'bottom_left': 'SCROLL_XY',
    'bottom_right': 'SCROLL_XY',
}


class HOTSPOT_OT_update_cursor(bpy.types.Operator):
    """Update cursor based on what's under the mouse"""
    bl_idname = "hotspot.update_cursor"
    bl_label = "Update Hotspot Cursor"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        space = context.space_data
        if not space or space.type != 'IMAGE_EDITOR':
            return {'PASS_THROUGH'}

        image = space.image
        if not image:
            context.window.cursor_set('DEFAULT')
            return {'PASS_THROUGH'}

        if not json_storage.is_texture_hotspottable(image.name):
            context.window.cursor_set('DEFAULT')
            return {'PASS_THROUGH'}

        props = context.scene.hotspot_mapping_props
        hotspots = json_storage.get_texture_hotspots(image.name)

        # Check if hovering over any orientation icon first
        for hotspot in hotspots:
            if _is_click_on_icon(context, event, hotspot, image):
                context.window.cursor_set('DEFAULT')
                return {'PASS_THROUGH'}

        hit_zone = None

        # Check active hotspot first (priority for selected hotspot)
        if props.active_hotspot_id:
            active_hotspot = json_storage.get_hotspot_by_id(image.name, props.active_hotspot_id)
            if active_hotspot:
                hit_zone = _get_hit_zone_for_hotspot(context, event, active_hotspot, image)

        # If not on active hotspot, check others
        if not hit_zone:
            for hotspot in hotspots:
                hit_zone = _get_hit_zone_for_hotspot(context, event, hotspot, image)
                if hit_zone:
                    break

        if hit_zone:
            cursor = CURSOR_FOR_MODE.get(hit_zone, 'DEFAULT')
            context.window.cursor_set(cursor)
        else:
            context.window.cursor_set('DEFAULT')

        return {'PASS_THROUGH'}


def _view_to_pixel(view_x, view_y, img_width, img_height):
    """Convert view (UV) coordinates to pixel coordinates.

    Args:
        view_x: X in UV space (0-1)
        view_y: Y in UV space (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple (pixel_x, pixel_y) where Y is from top (JSON format)
    """
    # Convert from UV space and flip Y back
    pixel_x = view_x * img_width
    pixel_y = img_height - (view_y * img_height)
    return (pixel_x, pixel_y)


def draw_hotspots():
    """Draw handler callback for hotspot rectangles."""
    context = bpy.context

    # Only draw in Image Editor
    if context.area is None or context.area.type != 'IMAGE_EDITOR':
        return

    # Only draw in the Hotspot Mapping workspace
    if not is_hotspot_mapping_workspace():
        return

    # Only draw when hotspot edit tool is active
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

    # Get image dimensions
    img_width = image.size[0] if image.size[0] > 0 else 1
    img_height = image.size[1] if image.size[1] > 0 else 1

    # Get hotspots
    hotspots = json_storage.get_texture_hotspots(texture_name)
    if not hotspots:
        return

    # Get active hotspot ID
    props = context.scene.hotspot_mapping_props
    active_id = props.active_hotspot_id

    # Get view2d for coordinate conversion
    region = context.region
    view2d = region.view2d

    # Calculate zoom level: screen pixels per UV unit (vertical)
    # Measured vertically so it directly relates to image height vs viewport height
    origin_x, origin_y = view2d.view_to_region(0, 0, clip=False)
    _, unit_y = view2d.view_to_region(0, 1, clip=False)
    pixels_per_uv = abs(unit_y - origin_y)

    # "Fit" zoom = when image height fills the viewport
    viewport_height = region.height
    fit_zoom = viewport_height

    # Zoom thresholds relative to fit zoom
    zoom_hide = fit_zoom * ZOOM_HIDE_FACTOR
    zoom_full = fit_zoom * ZOOM_FULL_FACTOR

    # Icon size limits based on viewport size
    icon_size_max = viewport_height * ICON_FRACTION_MAX
    icon_size_min = viewport_height * ICON_FRACTION_MIN

    # Interpolate icon size based on zoom
    if pixels_per_uv >= zoom_full:
        zoom_icon_size = icon_size_max
    elif pixels_per_uv <= zoom_hide:
        zoom_icon_size = icon_size_min
    else:
        t = (pixels_per_uv - zoom_hide) / (zoom_full - zoom_hide)
        zoom_icon_size = icon_size_min + t * (icon_size_max - icon_size_min)

    # Set up GPU state
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(LINE_WIDTH)

    try:
        # First pass: draw all rectangles and handles
        hotspot_regions = []
        for hotspot in hotspots:
            hotspot_id = hotspot.get("id", "")
            is_active = (hotspot_id == active_id)

            # Get hotspot bounds in pixel coords (y from top)
            px = hotspot.get("x", 0)
            py = hotspot.get("y", 0)
            pw = hotspot.get("width", 0)
            ph = hotspot.get("height", 0)

            # Convert corners to view (UV) coords
            # Top-left corner in JSON → top-left in UV (but Y flipped)
            x1, y2 = _pixel_to_view(px, py, img_width, img_height)
            # Bottom-right corner in JSON
            x2, y1 = _pixel_to_view(px + pw, py + ph, img_width, img_height)

            # Convert view coords to region (screen) coords
            rx1, ry1 = view2d.view_to_region(x1, y1, clip=False)
            rx2, ry2 = view2d.view_to_region(x2, y2, clip=False)

            # Store for icon pass
            orientation = hotspot.get("orientation_type", "Any")
            hotspot_regions.append((rx1, ry1, rx2, ry2, orientation))

            # Choose color based on state
            color = COLOR_HOTSPOT_ACTIVE if is_active else COLOR_HOTSPOT

            # Draw rectangle
            _draw_rectangle(rx1, ry1, rx2, ry2, color)

            # Draw handles for active hotspot
            if is_active:
                _draw_handles(rx1, ry1, rx2, ry2)

        # Second pass: draw all icons on top
        if pixels_per_uv >= zoom_hide:
            for rx1, ry1, rx2, ry2, orientation in hotspot_regions:
                icon_cx = (rx1 + rx2) / 2
                icon_cy = (ry1 + ry2) / 2
                _draw_orientation_icon(icon_cx, icon_cy, orientation, COLOR_ICON, zoom_icon_size)

    finally:
        # Restore GPU state
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)


def _draw_rectangle(x1, y1, x2, y2, color):
    """Draw a rectangle outline."""
    # Build line strip for rectangle (5 points to close the loop)
    points = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
        (x1, y1),  # Close the loop
    ]

    try:
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        region = bpy.context.region
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", LINE_WIDTH)
        shader.uniform_float("color", color)

        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": points})
        batch.draw(shader)
    except Exception as e:
        # Fallback to simple shader
        try:
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            shader.uniform_float("color", color)

            # Draw as individual lines
            lines = [
                (x1, y1), (x2, y1),
                (x2, y1), (x2, y2),
                (x2, y2), (x1, y2),
                (x1, y2), (x1, y1),
            ]
            batch = batch_for_shader(shader, 'LINES', {"pos": lines})
            batch.draw(shader)
        except Exception as e2:
            pass


def _draw_handles(x1, y1, x2, y2):
    """Draw resize handles at corners and edge midpoints."""
    hs = HANDLE_SIZE // 2

    # Calculate handle positions
    handles = [
        # Corners
        (x1, y1),  # bottom-left
        (x2, y1),  # bottom-right
        (x2, y2),  # top-right
        (x1, y2),  # top-left
        # Edge midpoints
        ((x1 + x2) / 2, y1),  # bottom
        ((x1 + x2) / 2, y2),  # top
        (x1, (y1 + y2) / 2),  # left
        (x2, (y1 + y2) / 2),  # right
    ]

    # Draw each handle as a small square
    for hx, hy in handles:
        _draw_handle_square(hx - hs, hy - hs, hx + hs, hy + hs)


def _draw_handle_square(x1, y1, x2, y2):
    """Draw a filled square for a handle."""
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.uniform_float("color", COLOR_HANDLE)

        # Draw as two triangles (filled quad)
        vertices = [
            (x1, y1), (x2, y1), (x2, y2),
            (x1, y1), (x2, y2), (x1, y2),
        ]
        batch = batch_for_shader(shader, 'TRIS', {"pos": vertices})
        batch.draw(shader)
    except Exception:
        pass


def _draw_orientation_icon(cx, cy, orientation_type, color, icon_size):
    """Draw orientation symbol at center position (cx, cy).

    Args:
        cx, cy: Center position of icon in region coords
        orientation_type: One of 'Any', 'Upwards', 'Floor', 'Ceiling'
        color: RGBA tuple for icon color
        icon_size: Size of the icon in pixels
    """
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

    # Get text dimensions to center it
    text_width, text_height = blf.dimensions(font_id, symbol)

    # Position text centered on cx, cy
    text_x = cx - text_width / 2
    text_y = cy - text_height / 2

    blf.position(font_id, text_x, text_y, 0)
    blf.draw(font_id, symbol)


def _get_icon_region(rx1, ry1, rx2, ry2, pixels_per_uv, viewport_height):
    """Get the icon click region for a hotspot.

    Args:
        rx1, ry1, rx2, ry2: Hotspot bounds in region coords
        pixels_per_uv: Zoom level (screen pixels per UV unit)
        viewport_height: Region height in pixels

    Returns:
        Tuple (icon_x1, icon_y1, icon_x2, icon_y2) for icon bounds,
        or None if zoomed out too far to show icons
    """
    # Thresholds relative to fit zoom (image height fills viewport)
    fit_zoom = viewport_height
    zoom_hide = fit_zoom * ZOOM_HIDE_FACTOR
    zoom_full = fit_zoom * ZOOM_FULL_FACTOR

    # No icon if zoomed out too far
    if pixels_per_uv < zoom_hide:
        return None

    # Icon size limits based on viewport
    icon_size_max = viewport_height * ICON_FRACTION_MAX
    icon_size_min = viewport_height * ICON_FRACTION_MIN

    # Calculate icon size based on zoom with linear interpolation
    if pixels_per_uv >= zoom_full:
        icon_size = icon_size_max
    else:
        t = (pixels_per_uv - zoom_hide) / (zoom_full - zoom_hide)
        icon_size = icon_size_min + t * (icon_size_max - icon_size_min)


    # Icon is centered on hotspot
    icon_cx = (rx1 + rx2) / 2
    icon_cy = (ry1 + ry2) / 2
    hs = icon_size / 2 + 2  # Slight padding for click area
    return (icon_cx - hs, icon_cy - hs, icon_cx + hs, icon_cy + hs)


def _is_click_on_icon(context, event, hotspot, image):
    """Check if click is on the orientation icon of a hotspot.

    Args:
        context: Blender context
        event: Mouse event
        hotspot: Hotspot dict
        image: Image object

    Returns:
        True if click is on icon, False otherwise
    """
    region = context.region
    view2d = region.view2d

    # Calculate zoom level: screen pixels per UV unit (vertical)
    origin_x, origin_y = view2d.view_to_region(0, 0, clip=False)
    _, unit_y = view2d.view_to_region(0, 1, clip=False)
    pixels_per_uv = abs(unit_y - origin_y)

    img_width = image.size[0] if image.size[0] > 0 else 1
    img_height = image.size[1] if image.size[1] > 0 else 1

    # Get hotspot bounds in view (UV) coords
    px = hotspot.get("x", 0)
    py = hotspot.get("y", 0)
    pw = hotspot.get("width", 0)
    ph = hotspot.get("height", 0)

    x1, y2 = _pixel_to_view(px, py, img_width, img_height)
    x2, y1 = _pixel_to_view(px + pw, py + ph, img_width, img_height)

    # Convert to region coords
    rx1, ry1 = view2d.view_to_region(x1, y1, clip=False)
    rx2, ry2 = view2d.view_to_region(x2, y2, clip=False)

    # Get icon bounds (returns None if zoomed out too far)
    icon_bounds = _get_icon_region(rx1, ry1, rx2, ry2, pixels_per_uv, region.height)
    if icon_bounds is None:
        return False

    icon_x1, icon_y1, icon_x2, icon_y2 = icon_bounds
    mx, my = event.mouse_region_x, event.mouse_region_y

    return icon_x1 <= mx <= icon_x2 and icon_y1 <= my <= icon_y2


class HOTSPOT_OT_interactive_edit(bpy.types.Operator):
    """Interactively edit hotspot position and size"""
    bl_idname = "hotspot.interactive_edit"
    bl_label = "Edit Hotspot"
    bl_options = {'REGISTER', 'UNDO'}

    # Hit zone passed from click_select to avoid redundant detection
    hit_zone: bpy.props.StringProperty(
        name="Hit Zone",
        description="Which part of the hotspot was clicked",
        options={'SKIP_SAVE', 'HIDDEN'},
    )

    # Internal state
    _dragging = False
    _drag_mode = None  # 'move', 'resize_tl', 'resize_tr', 'resize_bl', 'resize_br', etc.
    _drag_start_mouse = None
    _drag_start_hotspot = None  # Copy of hotspot data at drag start

    @classmethod
    def poll(cls, context):
        if context.area.type != 'IMAGE_EDITOR':
            return False
        space = context.space_data
        if not space.image:
            return False
        props = context.scene.hotspot_mapping_props
        if not props.active_hotspot_id:
            return False
        return json_storage.is_texture_hotspottable(space.image.name)

    def invoke(self, context, event):
        self._dragging = False
        self._drag_mode = None
        self._drag_start_mouse = None
        self._drag_start_hotspot = None

        # Get current hotspot data
        space = context.space_data
        image = space.image
        props = context.scene.hotspot_mapping_props

        hotspot = json_storage.get_hotspot_by_id(
            image.name, props.active_hotspot_id
        )
        if not hotspot:
            self.report({'WARNING'}, "No active hotspot")
            return {'CANCELLED'}

        # Use hit zone passed from click_select, or detect if called directly
        zone = self.hit_zone if self.hit_zone else _get_hit_zone_for_hotspot(context, event, hotspot, image)
        if zone is None:
            return {'CANCELLED'}

        # Start dragging
        self._dragging = True
        self._drag_mode = zone
        self._drag_start_mouse = (event.mouse_region_x, event.mouse_region_y)
        self._drag_start_hotspot = hotspot.copy()

        # Set cursor based on drag mode
        cursor = CURSOR_FOR_MODE.get(zone, 'DEFAULT')
        context.window.cursor_modal_set(cursor)

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            self._update_drag(context, event)

        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self._dragging = False
            context.window.cursor_modal_restore()
            # Save final position
            self._save_hotspot(context)
            return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Cancel - restore original position
            context.window.cursor_modal_restore()
            self._restore_hotspot(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _update_drag(self, context, event):
        """Update hotspot position/size based on mouse movement."""
        if not self._dragging or not self._drag_start_hotspot:
            return

        space = context.space_data
        image = space.image
        props = context.scene.hotspot_mapping_props
        region = context.region
        view2d = region.view2d

        img_width = image.size[0] if image.size[0] > 0 else 1
        img_height = image.size[1] if image.size[1] > 0 else 1

        # Calculate mouse delta in view coords
        mx1, my1 = self._drag_start_mouse
        mx2, my2 = event.mouse_region_x, event.mouse_region_y

        # Convert region coords to view (UV) coords
        vx1, vy1 = view2d.region_to_view(mx1, my1)
        vx2, vy2 = view2d.region_to_view(mx2, my2)

        # Delta in UV space, then convert to pixels
        dx = (vx2 - vx1) * img_width
        dy = (vy2 - vy1) * img_height  # Note: view Y increases upward

        # Get original hotspot values
        orig_x = self._drag_start_hotspot.get("x", 0)
        orig_y = self._drag_start_hotspot.get("y", 0)
        orig_w = self._drag_start_hotspot.get("width", 0)
        orig_h = self._drag_start_hotspot.get("height", 0)

        # Calculate new values based on drag mode
        new_x, new_y, new_w, new_h = orig_x, orig_y, orig_w, orig_h

        mode = self._drag_mode

        if mode == 'center':
            # Move entire rectangle
            new_x = orig_x + int(dx)
            new_y = orig_y - int(dy)  # Flip Y for JSON format
        elif mode == 'left':
            # Resize from left edge
            new_x = orig_x + int(dx)
            new_w = orig_w - int(dx)
        elif mode == 'right':
            # Resize from right edge
            new_w = orig_w + int(dx)
        elif mode == 'top':
            # Resize from top edge (in view coords, top is higher Y)
            new_y = orig_y - int(dy)  # Flip Y
            new_h = orig_h + int(dy)
        elif mode == 'bottom':
            # Resize from bottom edge
            new_h = orig_h - int(dy)
        elif mode == 'top_left':
            new_x = orig_x + int(dx)
            new_w = orig_w - int(dx)
            new_y = orig_y - int(dy)
            new_h = orig_h + int(dy)
        elif mode == 'top_right':
            new_w = orig_w + int(dx)
            new_y = orig_y - int(dy)
            new_h = orig_h + int(dy)
        elif mode == 'bottom_left':
            new_x = orig_x + int(dx)
            new_w = orig_w - int(dx)
            new_h = orig_h - int(dy)
        elif mode == 'bottom_right':
            new_w = orig_w + int(dx)
            new_h = orig_h - int(dy)

        # Minimum size is snap size when snapping enabled, otherwise 8
        min_size = props.snap_size if props.snap_enabled else 8

        # Apply pixel snapping if enabled (absolute snap, not relative)
        if props.snap_enabled:
            snap = props.snap_size

            def snap_to_grid(val):
                return round(val / snap) * snap

            if mode == 'center':
                # Snap position only, preserve size
                new_x = snap_to_grid(new_x)
                new_y = snap_to_grid(new_y)
            elif mode == 'left':
                # Snap left edge, keep right edge fixed
                right_edge = orig_x + orig_w
                new_x = snap_to_grid(new_x)
                new_w = right_edge - new_x
            elif mode == 'right':
                # Snap right edge, keep left edge fixed
                right_edge = snap_to_grid(new_x + new_w)
                new_w = right_edge - new_x
            elif mode == 'top':
                # Snap top edge, keep bottom edge fixed
                bottom_edge = orig_y + orig_h
                new_y = snap_to_grid(new_y)
                new_h = bottom_edge - new_y
            elif mode == 'bottom':
                # Snap bottom edge, keep top edge fixed
                bottom_edge = snap_to_grid(new_y + new_h)
                new_h = bottom_edge - new_y
            elif mode == 'top_left':
                right_edge = orig_x + orig_w
                bottom_edge = orig_y + orig_h
                new_x = snap_to_grid(new_x)
                new_y = snap_to_grid(new_y)
                new_w = right_edge - new_x
                new_h = bottom_edge - new_y
            elif mode == 'top_right':
                bottom_edge = orig_y + orig_h
                right_edge = snap_to_grid(new_x + new_w)
                new_y = snap_to_grid(new_y)
                new_w = right_edge - new_x
                new_h = bottom_edge - new_y
            elif mode == 'bottom_left':
                right_edge = orig_x + orig_w
                new_x = snap_to_grid(new_x)
                bottom_edge = snap_to_grid(new_y + new_h)
                new_w = right_edge - new_x
                new_h = bottom_edge - new_y
            elif mode == 'bottom_right':
                right_edge = snap_to_grid(new_x + new_w)
                bottom_edge = snap_to_grid(new_y + new_h)
                new_w = right_edge - new_x
                new_h = bottom_edge - new_y

        # Enforce minimum size (only for resize modes)
        if mode != 'center':
            if new_w < min_size:
                if mode in ('left', 'top_left', 'bottom_left'):
                    new_x = orig_x + orig_w - min_size
                new_w = min_size
            if new_h < min_size:
                if mode in ('top', 'top_left', 'top_right'):
                    new_y = orig_y + orig_h - min_size
                new_h = min_size

        # Clamp to image bounds
        if mode == 'center':
            # For moving, clamp position but never change size
            new_x = max(0, min(new_x, img_width - new_w))
            new_y = max(0, min(new_y, img_height - new_h))
        else:
            # For left edge drags: keep right edge fixed when clamping
            if mode in ('left', 'top_left', 'bottom_left'):
                right_edge = orig_x + orig_w
                if new_x < 0:
                    new_x = 0
                    new_w = right_edge  # Recalculate width to keep right edge fixed
                new_x = min(new_x, img_width - min_size)

            # For top edge drags: keep bottom edge fixed when clamping
            if mode in ('top', 'top_left', 'top_right'):
                bottom_edge = orig_y + orig_h
                if new_y < 0:
                    new_y = 0
                    new_h = bottom_edge  # Recalculate height to keep bottom edge fixed
                new_y = min(new_y, img_height - min_size)

            # For right/bottom edge drags: just clamp the size
            new_w = max(min_size, min(new_w, img_width - new_x))
            new_h = max(min_size, min(new_h, img_height - new_y))

        # Ensure integer values
        new_x = int(new_x)
        new_y = int(new_y)
        new_w = int(new_w)
        new_h = int(new_h)

        # Update hotspot in scene property only (no disk write during drag)
        json_storage.update_hotspot(
            image.name, props.active_hotspot_id,
            new_x, new_y, new_w, new_h,
            sync_to_disk=False
        )

        # Force redraw
        context.area.tag_redraw()

    def _save_hotspot(self, context):
        """Sync hotspot data to disk after drag completes."""
        json_storage.sync_to_file()

    def _restore_hotspot(self, context):
        """Restore hotspot to original position on cancel."""
        if not self._drag_start_hotspot:
            return

        space = context.space_data
        image = space.image
        props = context.scene.hotspot_mapping_props

        orig = self._drag_start_hotspot
        # Restore and sync to disk
        json_storage.update_hotspot(
            image.name, props.active_hotspot_id,
            orig.get("x", 0), orig.get("y", 0),
            orig.get("width", 0), orig.get("height", 0)
        )
        context.area.tag_redraw()


class HOTSPOT_OT_click_select(bpy.types.Operator):
    """Click to select a hotspot in the Image Editor"""
    bl_idname = "hotspot.click_select"
    bl_label = "Click Select Hotspot"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        space = context.space_data
        if not space or space.type != 'IMAGE_EDITOR':
            return {'PASS_THROUGH'}

        image = space.image
        if not image:
            return {'PASS_THROUGH'}

        if not json_storage.is_texture_hotspottable(image.name):
            return {'PASS_THROUGH'}

        hotspots = json_storage.get_texture_hotspots(image.name)
        props = context.scene.hotspot_mapping_props

        # First check if click is on any hotspot's orientation icon
        for hotspot in hotspots:
            if _is_click_on_icon(context, event, hotspot, image):
                hotspot_id = hotspot.get("id", "")
                debug_log(f"[Hotspots Click] ICON click on {hotspot_id}")
                # Cycle orientation and redraw
                bpy.ops.hotspot.cycle_orientation(hotspot_id=hotspot_id)
                return {'FINISHED'}

        clicked_hotspot = None
        hit_zone = None

        # Check active hotspot first (priority for selected hotspot)
        if props.active_hotspot_id:
            active_hotspot = json_storage.get_hotspot_by_id(image.name, props.active_hotspot_id)
            if active_hotspot:
                zone = _get_hit_zone_for_hotspot(context, event, active_hotspot, image)
                if zone:
                    clicked_hotspot = active_hotspot
                    hit_zone = zone
                    debug_log(f"[Hotspots Click] HIT on active {active_hotspot.get('id')} zone={zone}")

        # If not on active hotspot, check others
        if not hit_zone:
            for hotspot in hotspots:
                zone = _get_hit_zone_for_hotspot(context, event, hotspot, image)
                if zone:
                    clicked_hotspot = hotspot
                    hit_zone = zone
                    debug_log(f"[Hotspots Click] HIT on {hotspot.get('id')} zone={zone}")
                    break

        if clicked_hotspot and hit_zone:
            # Select this hotspot
            props.active_hotspot_id = clicked_hotspot.get("id", "")
            context.area.tag_redraw()

            # Start dragging immediately, passing the hit zone
            bpy.ops.hotspot.interactive_edit('INVOKE_DEFAULT', hit_zone=hit_zone)
            return {'FINISHED'}
        else:
            # Clicked outside any hotspot - pass through to normal Blender behavior
            return {'PASS_THROUGH'}


_keymaps = []


def register():
    global _draw_handler, _keymaps

    bpy.utils.register_class(HOTSPOT_OT_snap_size_up)
    bpy.utils.register_class(HOTSPOT_OT_snap_size_down)
    bpy.utils.register_class(HOTSPOT_OT_cycle_orientation)
    bpy.utils.register_class(HOTSPOT_OT_update_cursor)
    bpy.utils.register_class(HOTSPOT_OT_interactive_edit)
    bpy.utils.register_class(HOTSPOT_OT_click_select)

    # Register draw handler
    _draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
        draw_hotspots, (), 'WINDOW', 'POST_PIXEL'
    )
    debug_log(f"[Hotspots] Draw handler registered: {_draw_handler}")

    # Register the Hotspot Edit tool in Image Editor toolbar
    bpy.utils.register_tool(HOTSPOT_TOOL_edit, after={"builtin.sample"}, separator=True)
    debug_log("[Hotspots] Tool registered: hotspot.edit_tool")

    # Register timer to check for tool changes
    bpy.app.timers.register(_check_tool_change, persistent=True)

    # Register keymaps for snap size adjustment ([ and ])
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

    # Unregister tool change timer
    if bpy.app.timers.is_registered(_check_tool_change):
        bpy.app.timers.unregister(_check_tool_change)

    # Remove keymaps
    for km, kmi in _keymaps:
        try:
            km.keymap_items.remove(kmi)
        except ReferenceError:
            pass
    _keymaps.clear()

    # Unregister tool
    bpy.utils.unregister_tool(HOTSPOT_TOOL_edit)

    # Remove draw handler
    if _draw_handler is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None

    bpy.utils.unregister_class(HOTSPOT_OT_click_select)
    bpy.utils.unregister_class(HOTSPOT_OT_interactive_edit)
    bpy.utils.unregister_class(HOTSPOT_OT_update_cursor)
    bpy.utils.unregister_class(HOTSPOT_OT_cycle_orientation)
    bpy.utils.unregister_class(HOTSPOT_OT_snap_size_down)
    bpy.utils.unregister_class(HOTSPOT_OT_snap_size_up)
