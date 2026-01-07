from . import texture_apply
from . import uv_tools
from . import grid_tools
from . import walk_navigation
from . import ortho_navigation
from . import glb_export
from . import material_tools
from . import pixels_per_meter


def register():
    texture_apply.register()
    uv_tools.register()
    grid_tools.register()
    walk_navigation.register()
    ortho_navigation.register()
    glb_export.register()
    material_tools.register()
    pixels_per_meter.register()


def unregister():
    material_tools.unregister()
    glb_export.unregister()
    ortho_navigation.unregister()
    walk_navigation.unregister()
    grid_tools.unregister()
    uv_tools.unregister()
    texture_apply.unregister()
