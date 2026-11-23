"""Prefab asset, library, and selection helpers."""

import json
import os

import bpy


_PREFAB_ASSET_REFERENCE_PROP = "_anvil_prefab_asset_reference"
_LEGACY_PREFAB_PROPERTIES = (
    "_aw_prefab_library_index",
    "_aw_prefab_object_name",
    "_aw_prefab_asset_type",
    "_aw_prefab_rotation",
    "_aw_prefab_base_matrix",
)
_prefab_dependency_reference_cache = {}


def normalize_path(path):
    if not path:
        return ""
    return os.path.normpath(bpy.path.abspath(path))


def _prefab_reference_filepath(filepath, blend_filepath):
    """Return a portable library path when the owning blend file is saved."""
    if filepath.startswith("//") and blend_filepath:
        abs_path = os.path.normpath(
            os.path.join(os.path.dirname(blend_filepath), filepath[2:]),
        )
    else:
        abs_path = normalize_path(filepath)

    if not abs_path or not blend_filepath:
        return abs_path

    try:
        relative_path = os.path.relpath(abs_path, os.path.dirname(blend_filepath))
    except ValueError:
        return abs_path
    return "//" + relative_path.replace(os.sep, "/")


def _prefab_asset_reference(filepath, asset_name, blend_filepath):
    return json.dumps(
        [_prefab_reference_filepath(filepath, blend_filepath), asset_name],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def prefab_asset_reference(filepath, asset_name):
    """Create one stable serialized reference to an object asset."""
    return _prefab_asset_reference(filepath, asset_name, bpy.data.filepath)


def prefab_asset_reference_parts(asset_reference):
    """Decode an asset reference into an absolute library path and asset name."""
    if not asset_reference:
        return None
    try:
        values = json.loads(asset_reference)
    except (TypeError, ValueError):
        return None
    if not isinstance(values, list) or len(values) != 2:
        return None
    filepath, asset_name = values
    if not isinstance(filepath, str) or not isinstance(asset_name, str):
        return None
    filepath = normalize_path(filepath)
    if not filepath or not asset_name:
        return None
    return filepath, asset_name


def set_prefab_asset_reference(obj, asset_reference):
    """Store only the source asset reference on a placed prefab object."""
    obj[_PREFAB_ASSET_REFERENCE_PROP] = asset_reference
    for property_name in _LEGACY_PREFAB_PROPERTIES:
        if property_name in obj:
            del obj[property_name]
    if obj.get("_aw_mode") == 'PREFAB':
        del obj["_aw_mode"]


def get_prefab_asset_reference(obj):
    """Return a placed object's source asset reference, if present and valid."""
    asset_reference = obj.get(_PREFAB_ASSET_REFERENCE_PROP, "")
    if prefab_asset_reference_parts(asset_reference) is None:
        return ""
    return asset_reference


def find_prefab_asset_reference_owner(obj):
    """Find an object or parent carrying a prefab asset reference."""
    current = obj
    while current is not None:
        if _PREFAB_ASSET_REFERENCE_PROP in current:
            return current
        current = current.parent
    return None


def make_prefab_asset_references_relative(blend_filepath):
    """Rewrite stored prefab references relative to the current blend file."""
    if not blend_filepath:
        return 0

    updated_count = 0
    for obj in bpy.data.objects:
        if obj.library is not None:
            continue
        asset_reference = obj.get(_PREFAB_ASSET_REFERENCE_PROP, "")
        parts = prefab_asset_reference_parts(asset_reference)
        if parts is None:
            continue
        filepath, asset_name = parts
        relative_reference = _prefab_asset_reference(
            filepath,
            asset_name,
            blend_filepath,
        )
        if relative_reference == asset_reference:
            continue
        obj[_PREFAB_ASSET_REFERENCE_PROP] = relative_reference
        updated_count += 1
    return updated_count


def invalidate_prefab_dependency_reference_cache():
    """Clear all cached linked-dependency to prefab-asset mappings."""
    _prefab_dependency_reference_cache.clear()


def invalidate_prefab_dependency_reference_cache_for_library(filepath):
    """Clear a cached mapping for one prefab library."""
    _prefab_dependency_reference_cache.pop(normalize_path(filepath), None)


def _object_children_by_parent():
    children_by_parent = {}
    for obj in bpy.data.objects:
        if obj.parent is None:
            continue
        children_by_parent.setdefault(obj.parent.as_pointer(), []).append(obj)
    return children_by_parent


def _object_hierarchy(root, children_by_parent):
    stack = [root]
    visited = set()
    while stack:
        obj = stack.pop()
        pointer = obj.as_pointer()
        if pointer in visited:
            continue
        visited.add(pointer)
        yield obj
        stack.extend(children_by_parent.get(pointer, ()))


def _prefab_dependency_ids(root, children_by_parent):
    object_stack = list(_object_hierarchy(root, children_by_parent))
    visited_objects = set()
    visited_collections = set()
    dependencies = []

    while object_stack:
        obj = object_stack.pop()
        object_pointer = obj.as_pointer()
        if object_pointer in visited_objects:
            continue
        visited_objects.add(object_pointer)

        instance_collection = getattr(obj, "instance_collection", None)
        if instance_collection is not None:
            collection_pointer = instance_collection.as_pointer()
            if collection_pointer not in visited_collections:
                visited_collections.add(collection_pointer)
                dependencies.append(instance_collection)
                object_stack.extend(instance_collection.all_objects)

        data = getattr(obj, "data", None)
        if data is not None:
            dependencies.append(data)

    return dependencies


def _linked_dependency_key(id_data):
    library = getattr(id_data, "library", None)
    if library is None:
        return None
    return (
        type(id_data).__name__,
        normalize_path(library.filepath),
        id_data.name,
    )


def linked_prefab_dependency_keys(root):
    """Return linked mesh/collection identities reachable from a prefab root."""
    return _linked_prefab_dependency_keys(root, _object_children_by_parent())


def _linked_prefab_dependency_keys(root, children_by_parent):
    keys = set()
    for id_data in _prefab_dependency_ids(root, children_by_parent):
        key = _linked_dependency_key(id_data)
        if key is not None:
            keys.add(key)
    return keys


def _local_prefab_dependency_keys(root, children_by_parent):
    return {
        (type(id_data).__name__, id_data.as_pointer())
        for id_data in _prefab_dependency_ids(root, children_by_parent)
    }


def prefab_dependency_conflicts(objects):
    """Return asset names that share a mesh or collection with another asset."""
    assets_by_dependency = {}
    children_by_parent = _object_children_by_parent()
    for obj in objects:
        for dependency_key in _local_prefab_dependency_keys(obj, children_by_parent):
            assets_by_dependency.setdefault(dependency_key, set()).add(obj.name)

    conflicts = {}
    for asset_names in assets_by_dependency.values():
        if len(asset_names) < 2:
            continue
        for asset_name in asset_names:
            conflicts.setdefault(asset_name, set()).update(asset_names - {asset_name})
    return conflicts


def _is_free_prefab_root(obj):
    if obj.parent is not None:
        return False
    if _has_asset_object_ancestor(obj):
        return False
    if _object_has_asset_descendant(obj):
        return False
    return True


def prospective_prefab_asset_objects(scene):
    """Return objects that Make All Free Objects Assets would treat as prefabs."""
    return [obj for obj in scene.collection.all_objects if _is_free_prefab_root(obj)]


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
    invalidate_prefab_dependency_reference_cache_for_library(lib_entry.filepath)
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
            invalidate_prefab_dependency_reference_cache_for_library(abs_path)
            return True
        print(f"Anvil Level Design: lib_reload returned {result} for {abs_path}", flush=True)
    except RuntimeError as exc:
        print(f"Anvil Level Design: lib_reload failed for {abs_path}: {exc}", flush=True)
    return False


def find_existing_linked_object(abs_path, obj_name):
    library = find_loaded_library(abs_path)
    if library is None:
        return None
    return bpy.data.objects.get((obj_name, library.filepath))


def link_prefab_object(abs_path, obj_name):
    with bpy.data.libraries.load(abs_path, link=True, assets_only=True) as (data_from, data_to):
        if obj_name not in data_from.objects:
            return None
        data_to.objects = [obj_name]
    if not data_to.objects:
        return None
    return data_to.objects[0]


def resolve_prefab_linked_object(filepath, object_name):
    """Resolve an object asset without relying on a scene library index."""
    abs_path = normalize_path(filepath)
    if not os.path.isfile(abs_path):
        return None, abs_path, False, f"Library not found: {abs_path}"
    if not object_name:
        return None, abs_path, False, "No prefab name supplied"

    linked_asset = find_existing_linked_object(abs_path, object_name)
    reused_linked_asset = linked_asset is not None
    if linked_asset is None:
        linked_asset = link_prefab_object(abs_path, object_name)
        if linked_asset is None:
            return None, abs_path, False, f"Object '{object_name}' not found in {abs_path}"

    return linked_asset, abs_path, reused_linked_asset, ""


def _scene_prefab_library_index(scene, filepath):
    target = normalize_path(filepath)
    for library_index, lib_entry in enumerate(scene.anvil_prefab_libraries):
        if normalize_path(lib_entry.filepath) == target:
            return library_index
    return -1


def resolve_prefab_asset_reference(scene, asset_reference):
    """Resolve a stored asset reference against the scene's prefab libraries."""
    parts = prefab_asset_reference_parts(asset_reference)
    if parts is None:
        return None, "Invalid prefab asset reference"

    filepath, asset_name = parts
    library_index = _scene_prefab_library_index(scene, filepath)
    if library_index < 0:
        return None, f"Prefab library is not configured: {filepath}"

    linked_asset, _abs_path, _reused, error = resolve_prefab_linked_object(
        filepath,
        asset_name,
    )
    if linked_asset is None:
        return None, error

    return {
        "asset_reference": asset_reference,
        "library_index": library_index,
        "asset_name": asset_name,
        "linked_asset": linked_asset,
    }, ""


def _build_prefab_dependency_reference_cache(scene, filepath):
    abs_path = normalize_path(filepath)
    library_index = _scene_prefab_library_index(scene, abs_path)
    if library_index < 0:
        _prefab_dependency_reference_cache[abs_path] = {}
        return {}

    source_assets = []
    lib_entry = scene.anvil_prefab_libraries[library_index]
    for asset_item in lib_entry.objects:
        if asset_item.asset_type != 'OBJECT':
            continue
        linked_asset, _path, _reused, _error = resolve_prefab_linked_object(
            abs_path,
            asset_item.name,
        )
        if linked_asset is None:
            continue
        asset_reference = prefab_asset_reference(abs_path, asset_item.name)
        source_assets.append((linked_asset, asset_reference))

    dependency_references = {}
    children_by_parent = _object_children_by_parent()
    for linked_asset, asset_reference in source_assets:
        for dependency_key in _linked_prefab_dependency_keys(
                linked_asset,
                children_by_parent):
            dependency_references.setdefault(dependency_key, set()).add(asset_reference)

    _prefab_dependency_reference_cache[abs_path] = dependency_references
    return dependency_references


def _prefab_dependency_references(scene, dependency_key):
    filepath = dependency_key[1]
    dependency_references = _prefab_dependency_reference_cache.get(filepath)
    if dependency_references is None:
        dependency_references = _build_prefab_dependency_reference_cache(scene, filepath)
    return dependency_references.get(dependency_key, set())


def _linked_prefab_root(obj):
    current = obj
    while current.library is not None and current.parent is not None:
        current = current.parent
    return current


def object_has_linked_prefab_dependency(obj):
    """Return whether an object or its ancestors expose linked prefab data.

    This is an ancestry-only presentation check. It never scans the scene,
    builds the dependency cache, or links a library.
    """
    current = obj
    while current is not None:
        for id_data in (
                current,
                getattr(current, "data", None),
                getattr(current, "instance_collection", None)):
            if id_data is not None and getattr(id_data, "library", None) is not None:
                return True
        current = current.parent
    return False


def resolve_prefab_from_object(scene, obj):
    """Resolve a selected placed prefab from its reference or unique linked data."""
    if obj is None:
        return None, "No object selected"

    reference_error = ""
    reference_owner = find_prefab_asset_reference_owner(obj)
    if reference_owner is not None:
        asset_reference = get_prefab_asset_reference(reference_owner)
        resolved, error = resolve_prefab_asset_reference(scene, asset_reference)
        if resolved is not None:
            resolved["repeat_object"] = reference_owner
            return resolved, ""
        reference_error = error

    repeat_object = _linked_prefab_root(obj)
    dependency_keys = linked_prefab_dependency_keys(repeat_object)
    candidate_references = set()
    for dependency_key in dependency_keys:
        candidate_references.update(
            _prefab_dependency_references(scene, dependency_key),
        )

    if not candidate_references:
        if reference_error:
            return None, reference_error
        return None, "Selected object is not a resolvable linked prefab"
    if len(candidate_references) > 1:
        return None, "Selected object matches multiple prefab assets"

    asset_reference = next(iter(candidate_references))
    resolved, error = resolve_prefab_asset_reference(scene, asset_reference)
    if resolved is None:
        return None, error
    resolved["repeat_object"] = repeat_object
    return resolved, ""


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
    for obj in prospective_prefab_asset_objects(scene):
        if obj.asset_data is not None:
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
