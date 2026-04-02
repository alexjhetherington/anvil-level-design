def sync_uv_map_settings(obj):
    """Sync per-UV-map settings collection to match the object's UV maps.

    Adds entries for new UV maps and removes entries for deleted ones.
    Preserves lock state for existing maps.

    Args:
        obj: Blender object with mesh data
    """
    me = obj.data
    settings = obj.anvil_uv_map_settings

    # Build set of current UV map names
    uv_names = {uv.name for uv in me.uv_layers}

    # Remove settings for UV maps that no longer exist
    for i in range(len(settings) - 1, -1, -1):
        if settings[i].name not in uv_names:
            settings.remove(i)

    # Add settings for new UV maps
    existing_names = {s.name for s in settings}
    for uv in me.uv_layers:
        if uv.name not in existing_names:
            item = settings.add()
            item.name = uv.name


def get_render_active_uv_layer(bm, me):
    """Get the bmesh UV layer corresponding to the active render UV map.

    Args:
        bm: BMesh instance
        me: Mesh data

    Returns:
        BMesh UV layer for the active render UV map, or None if no UV layers exist.
    """
    if not me.uv_layers:
        return None

    # Find the active render UV map
    for uv_map in me.uv_layers:
        if uv_map.active_render:
            layer = bm.loops.layers.uv.get(uv_map.name)
            if layer is not None:
                return layer

    # Fallback to first UV layer
    if bm.loops.layers.uv:
        return bm.loops.layers.uv[0]
    return None


def get_unlocked_uv_layers(bm, obj, me):
    """Get list of bmesh UV layers that are unlocked.

    Args:
        bm: BMesh instance
        obj: Blender object (for accessing anvil_uv_map_settings)
        me: Mesh data

    Returns:
        List of unlocked BMesh UV layers. If no settings exist, all layers are unlocked.
    """
    sync_uv_map_settings(obj)
    settings = obj.anvil_uv_map_settings
    locked_names = {s.name for s in settings if s.locked}

    layers = []
    for uv_map in me.uv_layers:
        if uv_map.name not in locked_names:
            layer = bm.loops.layers.uv.get(uv_map.name)
            if layer is not None:
                layers.append(layer)
    return layers


def get_all_uv_layers(bm, me):
    """Get list of all bmesh UV layers.

    Args:
        bm: BMesh instance
        me: Mesh data

    Returns:
        List of all BMesh UV layers.
    """
    layers = []
    for uv_map in me.uv_layers:
        layer = bm.loops.layers.uv.get(uv_map.name)
        if layer is not None:
            layers.append(layer)
    return layers


def get_locked_uv_layers(bm, obj, me):
    """Get list of bmesh UV layers that are locked.

    Args:
        bm: BMesh instance
        obj: Blender object (for accessing anvil_uv_map_settings)
        me: Mesh data

    Returns:
        List of locked BMesh UV layers.
    """
    sync_uv_map_settings(obj)
    settings = obj.anvil_uv_map_settings
    locked_names = {s.name for s in settings if s.locked}

    layers = []
    for uv_map in me.uv_layers:
        if uv_map.name in locked_names:
            layer = bm.loops.layers.uv.get(uv_map.name)
            if layer is not None:
                layers.append(layer)
    return layers
