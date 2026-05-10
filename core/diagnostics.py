import os

import bpy
import gpu


def _safe_get(description, getter):
    try:
        return getter()
    except Exception as exc:
        return f"<{description} unavailable: {type(exc).__name__}: {exc}>"


def _format_active_viewports(screen):
    if screen is None:
        return "<no screen>"

    viewports = []
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for space in area.spaces:
            if space.type != 'VIEW_3D':
                continue
            shading = space.shading
            viewports.append(
                "VIEW_3D("
                f"shading={shading.type}, "
                f"scene_lights={getattr(shading, 'use_scene_lights_render', '<unknown>')}, "
                f"scene_world={getattr(shading, 'use_scene_world_render', '<unknown>')}"
                ")"
            )
    if not viewports:
        return "<no 3D viewports>"
    return "; ".join(viewports)


def _format_lights():
    if not bpy.data.lights:
        return "<none>"
    return ", ".join(
        f"{light.name}:{light.type}"
        for light in bpy.data.lights
    )


def _format_gpu_platform():
    vendor = _safe_get("GPU vendor", lambda: gpu.platform.vendor_get())
    renderer = _safe_get("GPU renderer", lambda: gpu.platform.renderer_get())
    version = _safe_get("GPU version", lambda: gpu.platform.version_get())
    return f"vendor={vendor!r}, renderer={renderer!r}, version={version!r}"


def _format_addon_candidates():
    candidates = []
    for addon_path in bpy.utils.script_paths(subdir="addons"):
        if not os.path.isdir(addon_path):
            continue
        try:
            names = os.listdir(addon_path)
        except OSError:
            continue
        matches = [
            name for name in names
            if "anvil" in name.lower() or "level-design" in name.lower()
        ]
        if matches:
            candidates.append(f"{addon_path}: {', '.join(sorted(matches))}")
    if not candidates:
        return "<none found>"
    return " | ".join(candidates)


def print_eevee_crash_diagnostics(label, include_addon_candidates):
    scene = _safe_get("scene", lambda: bpy.context.scene)
    screen = _safe_get("screen", lambda: bpy.context.screen)
    workspace = _safe_get("workspace", lambda: bpy.context.workspace)
    render_engine = _safe_get("render engine", lambda: scene.render.engine)
    workspace_name = getattr(workspace, "name", workspace)
    screen_name = getattr(screen, "name", screen)
    viewports = _safe_get("active viewports", lambda: _format_active_viewports(screen))
    lights = _safe_get("lights", _format_lights)
    gpu_platform = _safe_get("GPU platform", _format_gpu_platform)

    print(
        "Anvil Level Design: Eevee crash diagnostics"
        f" [{label}]"
        f" render_engine={render_engine!r}"
        f" workspace={workspace_name!r}"
        f" screen={screen_name!r}"
        f" viewports={viewports}"
        f" lights={lights}",
        flush=True,
    )
    print(
        "Anvil Level Design: Eevee crash diagnostics"
        f" [{label}] GPU {gpu_platform}",
        flush=True,
    )
    if include_addon_candidates:
        addon_candidates = _safe_get("addon candidates", _format_addon_candidates)
        if addon_candidates != "<none found>":
            print(
                "Anvil Level Design: Eevee crash diagnostics"
                f" [{label}] addon_folder_candidates={addon_candidates}",
                flush=True,
            )
