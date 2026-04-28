from . import uv_panel
from . import prefab_panel


def register():
    # Sidebar tab order is determined by which category each panel is first
    # registered into: Materials -> Prefabs -> Settings -> Export.
    uv_panel.register_materials()
    prefab_panel.register()
    uv_panel.register_settings_and_export()


def unregister():
    prefab_panel.unregister()
    uv_panel.unregister()
