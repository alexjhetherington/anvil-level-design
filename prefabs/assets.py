"""Prefab asset, library, and selection helpers."""

import os

import bpy


def normalize_path(path):
    if not path:
        return ""
    return os.path.normpath(bpy.path.abspath(path))


def scan_library_prefab_assets(filepath):
    """Open the .blend file and return object asset entries."""
    abs_path = normalize_path(filepath)
    if not os.path.isfile(abs_path):
        return None

    with bpy.data.libraries.load(abs_path, link=True, assets_only=True) as (data_from, _data_to):
        # Don't link anything; we only want asset names.
        assets = []
        for name in data_from.objects:
            assets.append(('OBJECT', name))
        return assets


def refresh_library_objects(lib_entry):
    """Re-populate lib_entry.objects by scanning the .blend file."""
    assets = scan_library_prefab_assets(lib_entry.filepath)
    lib_entry.objects.clear()
    if assets is None:
        return False
    for asset_type, name in sorted(assets, key=lambda item: (item[0], item[1])):
        item = lib_entry.objects.add()
        item.name = name
        item.asset_type = asset_type
    return True


def find_loaded_library(filepath):
    """Return the bpy.data.libraries entry matching filepath, if any."""
    target = normalize_path(filepath)
    for lib in bpy.data.libraries:
        if normalize_path(lib.filepath) == target:
            return lib
    return None


def reload_library(library_db):
    """Reload a Library datablock from disk.

    wm.lib_reload accepts `library=` (the Library datablock name) only when
    the file-browser-style `directory` + `filename` properties also point to
    the .blend on disk; otherwise it errors with "Not a library". A bare
    `library=` or context.id override is not enough.
    """
    abs_path = normalize_path(library_db.filepath)
    if not os.path.isfile(abs_path):
        print(f"Anvil Level Design: Library file not found on disk: {abs_path}", flush=True)
        return False
    try:
        result = bpy.ops.wm.lib_reload(
            'EXEC_DEFAULT',
            library=library_db.name,
            directory=os.path.dirname(abs_path) + os.sep,
            filename=os.path.basename(abs_path),
        )
        if 'FINISHED' in result:
            return True
        print(f"Anvil Level Design: lib_reload returned {result} for {abs_path}", flush=True)
    except RuntimeError as exc:
        print(f"Anvil Level Design: lib_reload failed for {abs_path}: {exc}", flush=True)
    return False


def find_existing_linked_object(abs_path, obj_name):
    for obj in bpy.data.objects:
        if (obj.library is not None
                and normalize_path(obj.library.filepath) == abs_path
                and obj.name == obj_name):
            return obj
    return None


def link_prefab_object(abs_path, obj_name):
    with bpy.data.libraries.load(abs_path, link=True, assets_only=True) as (data_from, data_to):
        if obj_name not in data_from.objects:
            return None
        data_to.objects = [obj_name]
    if not data_to.objects:
        return None
    return data_to.objects[0]


def append_prefab_object(abs_path, obj_name):
    """Append one fresh, fully local prefab without reusing earlier appended IDs."""
    with bpy.data.libraries.load(
            abs_path,
            link=False,
            recursive=True,
            reuse_local_id=False,
            assets_only=True,
            clear_asset_data=True) as (data_from, data_to):
        if obj_name not in data_from.objects:
            return None
        data_to.objects = [obj_name]
    if not data_to.objects:
        return None
    return data_to.objects[0]


def create_object_override(linked_asset):
    try:
        return linked_asset.override_create(remap_local_usages=True), ""
    except RuntimeError as exc:
        return None, str(exc)


def iter_scene_prefab_assets(scene):
    assets = []
    for obj in scene.collection.all_objects:
        if obj.asset_data is not None:
            assets.append(('OBJECT', obj.name))
    return sorted(assets, key=lambda item: (item[0], item[1]))


def _has_asset_object_ancestor(obj):
    parent = obj.parent
    while parent is not None:
        if parent.asset_data is not None:
            return True
        parent = parent.parent
    return False


def _object_children(obj):
    return [child for child in bpy.data.objects if child.parent == obj]


def _object_has_asset_descendant(obj):
    for child in _object_children(obj):
        if child.asset_data is not None:
            return True
        if _object_has_asset_descendant(child):
            return True
    return False


def make_all_free_objects_assets(scene):
    marked_count = 0
    for obj in scene.collection.all_objects:
        if obj.asset_data is not None:
            continue
        if obj.parent is not None:
            continue
        if _has_asset_object_ancestor(obj):
            continue
        if _object_has_asset_descendant(obj):
            continue
        obj.asset_mark()
        marked_count += 1
    return marked_count


def clear_prefab_asset(scene, asset_type, asset_name):
    if asset_type == 'OBJECT':
        obj = next((o for o in scene.collection.all_objects if o.name == asset_name), None)
        if obj is None or obj.asset_data is None:
            return False
        obj.asset_clear()
        return True
    return False


def select_prefab_asset(view_layer, asset_type, asset_name):
    for obj in view_layer.objects:
        obj.select_set(False)

    if asset_type == 'OBJECT':
        obj = next((o for o in view_layer.objects if o.name == asset_name), None)
        if obj is None:
            return False
        obj.select_set(True)
        view_layer.objects.active = obj
        return True
    return False


def focus_selected_in_3d_views(context):
    window = context.window
    if window is None:
        return
    for area in window.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if region is None:
            continue
        space = area.spaces.active
        with context.temp_override(window=window, area=area, region=region, space_data=space):
            try:
                bpy.ops.view3d.view_selected(use_all_regions=False)
            except RuntimeError:
                pass
