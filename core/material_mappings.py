"""Analysis and application logic for Fix Material Mappings."""

from collections import namedtuple

import bpy

from .material_shader import infer_primary_shader_image, shader_images
from .materials import (
    MATERIAL_SCHEMA_VERSION,
    clear_material_mapping,
    get_primary_image_from_material,
    material_name_for_image,
    set_material_primary_image,
)


MaterialMappingAnalysis = namedtuple(
    "MaterialMappingAnalysis",
    ("material", "shader_images", "suggested_image", "is_already_mapped"),
)


def material_is_mapping_eligible(material):
    return (
        material is not None
        and material.library is None
        and material.name != "ANVIL_Unassigned"
    )


def eligible_materials():
    return [
        material for material in bpy.data.materials
        if material_is_mapping_eligible(material)
    ]


def analyze_material_mapping(material):
    images = tuple(shader_images(material))
    current_image = get_primary_image_from_material(material)
    suggested_image = current_image or infer_primary_shader_image(material)
    return MaterialMappingAnalysis(
        material,
        images,
        suggested_image,
        (
            current_image is not None
            and getattr(material, "anvil_material_schema_version", 0)
            == MATERIAL_SCHEMA_VERSION
        ),
    )


def analyze_material_mappings(materials):
    return [analyze_material_mapping(material) for material in materials]


def mapping_choice_conflicts(choices):
    by_image = {}
    for material, image in choices:
        if image is None:
            continue
        by_image.setdefault(image.as_pointer(), []).append(material)
    return {
        pointer: materials
        for pointer, materials in by_image.items()
        if len(materials) > 1
    }


def apply_material_mapping_choices(choices, rename_materials, name_pattern):
    """Apply mappings without changing shaders, slots, or material users."""
    conflicts = mapping_choice_conflicts(choices)
    if conflicts:
        names = []
        for materials in conflicts.values():
            names.append(", ".join(material.name for material in materials))
        raise ValueError(
            "Each image can map to only one material: " + "; ".join(names)
        )

    local_choices = []
    for material, image in choices:
        if not material_is_mapping_eligible(material):
            continue
        local_choices.append((material, image))

    # Clear the reviewed set first so a mapping can move between materials
    # without ever reassigning object material slots.
    for material, _image in local_choices:
        clear_material_mapping(material)

    mapped_count = 0
    renamed_count = 0
    for material, image in local_choices:
        if image is None:
            continue
        set_material_primary_image(material, image)
        mapped_count += 1
        if rename_materials:
            requested_name = material_name_for_image(image, name_pattern)
            if requested_name and material.name != requested_name:
                material.name = requested_name
                renamed_count += 1

    return mapped_count, renamed_count
