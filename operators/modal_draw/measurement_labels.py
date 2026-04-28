"""
Modal Draw - Measurement Labels

Screen-space text labels for shape-preview measurements.
"""

import bpy
import blf
from bpy_extras.view3d_utils import location_3d_to_region_2d

from ...core.workspace_check import is_level_design_workspace


FONT_ID = 0
FONT_SIZE = 11
TEXT_MARGIN = 24
MIN_LENGTH = 1e-10


def segments_from_loop(points):
    """Return measurement segments for a closed point loop."""
    segments = []
    if len(points) < 2:
        return segments

    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        segments.append((start, end, None))

    return segments


def segments_from_indexed_vertices(vertices, edges):
    """Return measurement segments from indexed vertex edge data."""
    segments = []
    for start_index, end_index in edges:
        if start_index < 0 or end_index < 0:
            continue
        if start_index >= len(vertices) or end_index >= len(vertices):
            continue
        segments.append((vertices[start_index], vertices[end_index], None))

    return segments


def draw_measurement_segments(region, rv3d, space, unit_settings, theme_3d, segments):
    """Draw labels for the supplied world-space measurement segments."""
    if not _should_draw_measurements(space):
        return
    if region is None or rv3d is None:
        return

    text_color = _edge_length_text_color(theme_3d)

    blf.size(FONT_ID, FONT_SIZE)

    occupied = set()
    for segment in segments:
        start, end, label = _segment_parts(segment)
        if (end - start).length < MIN_LENGTH:
            continue

        text = label
        if text is None:
            text = _format_length(unit_settings, start, end)
        if not text:
            continue

        midpoint = (start + end) * 0.5
        screen_pos = location_3d_to_region_2d(region, rv3d, midpoint)
        if screen_pos is None:
            continue
        if not _is_inside_label_margin(region, screen_pos):
            continue

        # Blender's native measurement text uses the projected midpoint as the
        # text origin, so the glyphs sit just off the edge instead of centered
        # over it.
        x = round(screen_pos.x)
        y = round(screen_pos.y)

        occupied_key = (round(x / 2), round(y / 2), text)
        if occupied_key in occupied:
            continue
        occupied.add(occupied_key)

        _draw_label(text, x, y, text_color)


def _should_draw_measurements(space):
    if not is_level_design_workspace():
        return False

    if space is None or space.type != 'VIEW_3D':
        return False
    overlay = space.overlay
    if not overlay.show_overlays:
        return False
    return overlay.show_extra_edge_length


def _segment_parts(segment):
    if len(segment) >= 3:
        return (segment[0], segment[1], segment[2])
    return (segment[0], segment[1], None)


def _format_length(unit_settings, start, end):
    length = (end - start).length

    if unit_settings is not None and unit_settings.system != 'NONE':
        scaled_length = length * unit_settings.scale_length
        try:
            return bpy.utils.units.to_string(
                unit_settings.system,
                'LENGTH',
                scaled_length,
                precision=3,
                split_unit=unit_settings.use_separate,
                compatible_unit=False,
            )
        except Exception:
            pass

    return f"{length:.3f}"


def _edge_length_text_color(theme_3d):
    try:
        color = theme_3d.extra_edge_len
        return (color[0], color[1], color[2], 1.0)
    except Exception:
        return (0.0, 0.0, 0.0, 1.0)


def _is_inside_label_margin(region, screen_pos):
    return (
        screen_pos.x >= -TEXT_MARGIN and
        screen_pos.y >= -TEXT_MARGIN and
        screen_pos.x <= region.width + TEXT_MARGIN and
        screen_pos.y <= region.height + TEXT_MARGIN
    )


def _draw_label(text, x, y, text_color):
    blf.color(FONT_ID, text_color[0], text_color[1], text_color[2], text_color[3])
    blf.position(FONT_ID, x, y, 0)
    blf.draw(FONT_ID, text)
