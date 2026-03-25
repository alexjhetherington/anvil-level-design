from . import texture_apply
from . import uv_tools
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


def register():
    texture_apply.register()
    uv_tools.register()
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


def unregister():
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
    uv_tools.unregister()
    texture_apply.unregister()
