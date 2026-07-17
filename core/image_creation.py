"""Create simple image files, including fitted and centred text."""

import os
import tempfile

import blf
import bpy
import imbuf


class ImageFileCreationError(RuntimeError):
    pass


class ImageTextRenderError(RuntimeError):
    pass


class CreatedImageLoadError(RuntimeError):
    pass


def _normal_path(filepath):
    if not filepath:
        return ""
    return os.path.normcase(os.path.abspath(bpy.path.abspath(filepath)))


def _loaded_image_for_filepath(filepath):
    target_path = _normal_path(filepath)
    for image in bpy.data.images:
        try:
            image_path = image.filepath_from_user()
        except RuntimeError:
            image_path = image.filepath
        if _normal_path(image_path) == target_path:
            return image
    return None


def _measure_text_center_correction(
        font_id,
        text,
        font_size,
        text_width,
        text_height):
    measurement_scale = min(1.0, 512.0 / max(text_width, text_height))
    measurement_font_size = max(1.0, font_size * measurement_scale)
    blf.size(font_id, measurement_font_size)
    measurement_width, measurement_height = blf.dimensions(font_id, text)

    padding = max(8, int(measurement_height * 0.25) + 2)
    buffer_width = int(measurement_width) + padding * 2 + 2
    buffer_height = int(measurement_height * 2.0) + padding * 2 + 2
    draw_x = float(padding)
    draw_y = float(padding) + measurement_height
    measurement_buffer = imbuf.new((buffer_width, buffer_height))
    measurement_filepath = ""
    measurement_image = None

    try:
        with blf.bind_imbuf(font_id, measurement_buffer):
            blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
            blf.position(font_id, draw_x, draw_y, 0.0)
            blf.draw_buffer(font_id, text)

        temporary_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        measurement_filepath = temporary_file.name
        temporary_file.close()
        imbuf.write(measurement_buffer, filepath=measurement_filepath)
        measurement_image = bpy.data.images.load(
            measurement_filepath,
            check_existing=False,
        )

        pixels = measurement_image.pixels[:]
        min_x = buffer_width
        min_y = buffer_height
        max_x = -1
        max_y = -1
        for pixel_index in range(buffer_width * buffer_height):
            if pixels[pixel_index * 4 + 3] <= 0.001:
                continue
            pixel_x = pixel_index % buffer_width
            pixel_y = pixel_index // buffer_width
            min_x = min(min_x, pixel_x)
            min_y = min(min_y, pixel_y)
            max_x = max(max_x, pixel_x)
            max_y = max(max_y, pixel_y)

        if max_x < min_x or max_y < min_y:
            raise RuntimeError("Could not measure rendered text")

        ink_center_x = (min_x + max_x + 1.0) * 0.5
        ink_center_y = (min_y + max_y + 1.0) * 0.5
        layout_center_x = draw_x + measurement_width * 0.5
        layout_center_y = draw_y + measurement_height * 0.5
        correction_scale = font_size / measurement_font_size
        descender_threshold = max(1.0, measurement_height * 0.1)
        has_descender = min_y < draw_y - descender_threshold
        return (
            (layout_center_x - ink_center_x) * correction_scale,
            (layout_center_y - ink_center_y) * correction_scale,
            has_descender,
        )
    finally:
        if measurement_image is not None:
            bpy.data.images.remove(measurement_image)
        measurement_buffer.free()
        if measurement_filepath and os.path.exists(measurement_filepath):
            os.remove(measurement_filepath)


def _draw_centered_text_on_image(filepath, text, width, height, text_color):
    image_buffer = imbuf.load(filepath)
    if image_buffer is None:
        raise RuntimeError("Could not load image buffer")

    font_id = 0
    available_width = max(1.0, width * 0.84)
    available_height = max(1.0, height * 0.75)
    font_size = float(max(1, min(width, height)))

    try:
        blf.size(font_id, font_size)
        text_width, text_height = blf.dimensions(font_id, text)
        if text_width <= 0.0 or text_height <= 0.0:
            raise RuntimeError("Could not measure text")

        fit_scale = min(
            available_width / text_width,
            available_height / text_height,
            1.0,
        )
        font_size = max(1.0, font_size * fit_scale)
        blf.size(font_id, font_size)
        text_width, text_height = blf.dimensions(font_id, text)
        center_correction_x, center_correction_y, has_descender = (
            _measure_text_center_correction(
                font_id,
                text,
                font_size,
                text_width,
                text_height,
            )
        )
        vertical_center_height = text_height
        if has_descender:
            blf.size(font_id, font_size)
            # Ignore descenders, but include cap height when the label uses capitals.
            vertical_center_reference = (
                "H" if any(character.isupper() for character in text) else "x"
            )
            reference_width, reference_height = blf.dimensions(
                font_id,
                vertical_center_reference,
            )
            _reference_correction_x, center_correction_y, _has_descender = (
                _measure_text_center_correction(
                    font_id,
                    vertical_center_reference,
                    font_size,
                    reference_width,
                    reference_height,
                )
            )
            vertical_center_height = reference_height
        blf.size(font_id, font_size)

        with blf.bind_imbuf(font_id, image_buffer):
            blf.color(font_id, *text_color)
            blf.position(
                font_id,
                (width - text_width) * 0.5 + center_correction_x,
                (height - vertical_center_height) * 0.5 + center_correction_y,
                0.0,
            )
            blf.draw_buffer(font_id, text)

        imbuf.write(image_buffer, filepath=filepath)
    finally:
        image_buffer.free()


def _save_solid_color_png(filepath, width, height, color, existing_image):
    if existing_image is not None and existing_image.source == 'GENERATED':
        existing_image.scale(width, height)
        existing_image.generated_color = color
        existing_image.filepath_raw = filepath
        existing_image.file_format = 'PNG'
        existing_image.save()
        return

    filename = os.path.basename(filepath)
    writer = bpy.data.images.new(
        f"ANVIL_Write_{filename}",
        width=width,
        height=height,
        alpha=True,
    )
    try:
        writer.generated_color = color
        writer.filepath_raw = filepath
        writer.file_format = 'PNG'
        writer.save()
    finally:
        bpy.data.images.remove(writer)


def create_image(filepath, width, height, color, text, text_color):
    """Create a PNG and return its loaded Blender image datablock."""
    existing_image = _loaded_image_for_filepath(filepath)
    try:
        _save_solid_color_png(
            filepath,
            width,
            height,
            color,
            existing_image,
        )
    except (OSError, RuntimeError) as exc:
        raise ImageFileCreationError(str(exc)) from exc

    if text:
        try:
            _draw_centered_text_on_image(
                filepath,
                text,
                width,
                height,
                text_color,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ImageTextRenderError(str(exc)) from exc

    try:
        if existing_image is not None:
            if existing_image.source == 'GENERATED':
                if text:
                    existing_image.source = 'FILE'
                    existing_image.reload()
            else:
                existing_image.reload()
            return existing_image
        return bpy.data.images.load(filepath, check_existing=False)
    except RuntimeError as exc:
        raise CreatedImageLoadError(str(exc)) from exc
