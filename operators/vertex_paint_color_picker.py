import bpy


_addon_keymaps = []


def register():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Vertex Paint', space_type='EMPTY')
        kmi = km.keymap_items.new(
            "wm.call_panel",
            type='C',
            value='PRESS',
            ctrl=False,
            shift=False,
            alt=False
        )
        kmi.properties.name = 'VIEW3D_PT_paint_vertex_context_menu'
        _addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()
