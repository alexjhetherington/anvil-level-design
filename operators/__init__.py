from . import texture_apply
from . import hotspot_apply
from . import face_uv_mode
from . import grid_snapping_mode
from . import grid_tools
from . import walk_navigation
from . import ortho_navigation
from . import glb_export
from . import material_tools
from . import pixels_per_meter
from . import modal_draw
from . import cube_cut
from . import box_builder
from . import backface_select
from . import select_linked
from . import uv_select_invalid
from . import weld
from . import vertex_paint_color_picker
from . import overlap_check
from . import grid_overlay
from . import fixed_hotspot_overlay
from . import uv_transform_modal


def register():
    texture_apply.register()
    hotspot_apply.register()
    face_uv_mode.register()
    grid_snapping_mode.register()
    grid_tools.register()
    walk_navigation.register()
    ortho_navigation.register()
    glb_export.register()
    material_tools.register()
    pixels_per_meter.register()
    modal_draw.register()
    cube_cut.register()
    box_builder.register()
    backface_select.register()
    select_linked.register()
    uv_select_invalid.register()
    weld.register()
    vertex_paint_color_picker.register()
    overlap_check.register()
    grid_overlay.register()
    fixed_hotspot_overlay.register()
    uv_transform_modal.register()


def unregister():
    uv_transform_modal.unregister()
    fixed_hotspot_overlay.unregister()
    grid_overlay.unregister()
    overlap_check.unregister()
    vertex_paint_color_picker.unregister()
    weld.unregister()
    uv_select_invalid.unregister()
    select_linked.unregister()
    backface_select.unregister()
    box_builder.unregister()
    cube_cut.unregister()
    modal_draw.unregister()
    pixels_per_meter.unregister()
    material_tools.unregister()
    glb_export.unregister()
    ortho_navigation.unregister()
    walk_navigation.unregister()
    grid_tools.unregister()
    grid_snapping_mode.unregister()
    face_uv_mode.unregister()
    hotspot_apply.unregister()
    texture_apply.unregister()
