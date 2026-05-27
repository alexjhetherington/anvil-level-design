"""
Backface-Aware Select

Custom click-select operator that raycasts through backface-culled faces,
allowing selection of visible geometry behind them without X-ray mode.
"""

import bpy
from . import operator


_addon_keymaps = []


def register():
    operator.register()

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    km = kc.keymaps.new(name='Mesh', space_type='EMPTY')

    # Plain click
    kmi = km.keymap_items.new(
        operator.LEVELDESIGN_OT_backface_select.bl_idname,
        'LEFTMOUSE', 'CLICK',
        head=True
    )
    kmi.properties.extend = False
    kmi.properties.loop = False
    _addon_keymaps.append((km, kmi))

    # Shift+click (extend selection)
    kmi = km.keymap_items.new(
        operator.LEVELDESIGN_OT_backface_select.bl_idname,
        'LEFTMOUSE', 'CLICK',
        shift=True,
        head=True
    )
    kmi.properties.extend = True
    kmi.properties.loop = False
    _addon_keymaps.append((km, kmi))

    # Alt+click (loop select)
    kmi = km.keymap_items.new(
        operator.LEVELDESIGN_OT_backface_select.bl_idname,
        'LEFTMOUSE', 'CLICK',
        alt=True,
        head=True
    )
    kmi.properties.extend = False
    kmi.properties.loop = True
    _addon_keymaps.append((km, kmi))

    # Shift+Alt+click (extend loop select)
    kmi = km.keymap_items.new(
        operator.LEVELDESIGN_OT_backface_select.bl_idname,
        'LEFTMOUSE', 'CLICK',
        shift=True,
        alt=True,
        head=True
    )
    kmi.properties.extend = True
    kmi.properties.loop = True
    _addon_keymaps.append((km, kmi))

    # Ctrl+click (shortest path)
    kmi = km.keymap_items.new(
        operator.LEVELDESIGN_OT_backface_shortest_path_pick.bl_idname,
        'LEFTMOUSE', 'PRESS',
        ctrl=True,
        head=True
    )
    kmi.properties.use_fill = False
    _addon_keymaps.append((km, kmi))

    # Ctrl+Shift+click (filled shortest path)
    kmi = km.keymap_items.new(
        operator.LEVELDESIGN_OT_backface_shortest_path_pick.bl_idname,
        'LEFTMOUSE', 'PRESS',
        ctrl=True,
        shift=True,
        head=True
    )
    kmi.properties.use_fill = True
    _addon_keymaps.append((km, kmi))

    # Object mode keymaps
    km_obj = kc.keymaps.new(name='Object Mode', space_type='EMPTY')

    # Plain click
    kmi = km_obj.keymap_items.new(
        operator.LEVELDESIGN_OT_backface_object_select.bl_idname,
        'LEFTMOUSE', 'CLICK',
        head=True
    )
    kmi.properties.extend = False
    _addon_keymaps.append((km_obj, kmi))

    # Shift+click (extend selection)
    kmi = km_obj.keymap_items.new(
        operator.LEVELDESIGN_OT_backface_object_select.bl_idname,
        'LEFTMOUSE', 'CLICK',
        shift=True,
        head=True
    )
    kmi.properties.extend = True
    _addon_keymaps.append((km_obj, kmi))


def unregister():
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    operator.unregister()
