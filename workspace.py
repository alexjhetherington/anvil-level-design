import bpy
from bpy.app.handlers import persistent
from mathutils import Quaternion

WORKSPACE_NAME = "Level Design"


def workspace_exists():
    """Check if the Level Design workspace already exists"""
    return WORKSPACE_NAME in bpy.data.workspaces


def create_level_design_workspace():
    """Create the Level Design workspace if it doesn't exist.

    Layout:
    +------------------+------------------+--------+
    |                  |                  | OUTLNR |
    |   VIEW_3D (L)    |   VIEW_3D (R)    +--------+
    |   Mat Preview    |   Wireframe      | PROPS  |
    |   N-panel open   |   Top-down ortho |        |
    +------------------+------------------+        |
    |      FILE_BROWSER                   |        |
    +-------------------------------------+--------+

    - Left 3D Viewport: Material preview, N-panel open (Level Design tab)
    - Right 3D Viewport: Wireframe, N-panel closed, toolbar hidden, top-down orthographic
    - File Browser below the viewports only (not under right column)
    - Outliner and Properties on far right (full height)
    """
    if workspace_exists():
        return False

    # Store current workspace to restore later
    original_workspace = bpy.context.window.workspace
    original_name = original_workspace.name

    try:
        # Get workspace names before duplicate
        existing_names = set(bpy.data.workspaces.keys())

        # Create new workspace by duplicating current one
        bpy.ops.workspace.duplicate()

        # Find the new workspace by checking what's new in the list
        # (Don't rely on it becoming active - that's not guaranteed)
        new_workspace = None
        for ws in bpy.data.workspaces:
            if ws.name not in existing_names:
                new_workspace = ws
                break

        # Fallback: look for the .001 suffix pattern
        if not new_workspace:
            for suffix in ['.001', '.002', '.003']:
                candidate_name = original_name + suffix
                if candidate_name in bpy.data.workspaces:
                    new_workspace = bpy.data.workspaces[candidate_name]
                    break

        if not new_workspace:
            print(f"Anvil Level Design: Could not find duplicated workspace")
            return False

        new_workspace.name = WORKSPACE_NAME

        # Schedule the layout configuration
        bpy.app.timers.register(
            lambda: _setup_workspace_deferred(original_workspace),
            first_interval=0.1
        )

    except Exception as e:
        print(f"Anvil Level Design: Error creating workspace: {e}")
        return False

    return True


def _setup_workspace_deferred(original_workspace):
    """Deferred workspace setup to ensure Blender is ready"""
    try:
        # Make sure we're on the Level Design workspace
        workspace = bpy.data.workspaces.get(WORKSPACE_NAME)
        if not workspace:
            return None

        # Switch to the Level Design workspace to configure it
        bpy.context.window.workspace = workspace

        # Start by closing areas until only one remains
        bpy.app.timers.register(_close_areas_step, first_interval=0.1)

    except Exception as e:
        print(f"Anvil Level Design: Error in deferred setup: {e}")

    return None


def _close_areas_step():
    """Close areas until only one remains."""
    try:
        screen = bpy.context.window.screen
        areas = list(screen.areas)

        if len(areas) <= 1:
            # Done closing, now start splitting
            bpy.app.timers.register(_configure_layout_step1, first_interval=0.1)
            return None

        # Try to close the first area that isn't the largest
        largest = max(areas, key=lambda a: a.width * a.height)

        for area in areas:
            if area != largest:
                try:
                    region = None
                    for r in area.regions:
                        if r.type == 'WINDOW':
                            region = r
                            break

                    if region:
                        with bpy.context.temp_override(area=area, region=region):
                            bpy.ops.screen.area_close()

                        # Schedule next close
                        bpy.app.timers.register(_close_areas_step, first_interval=0.1)
                        return None
                except Exception as e:
                    print(f"Anvil Level Design: Close failed for area: {e}")
                    continue

        # If we get here, couldn't close any areas - proceed anyway
        print("Anvil Level Design: Could not close all areas, proceeding with current layout")
        bpy.app.timers.register(_configure_layout_step1, first_interval=0.1)

    except Exception as e:
        print(f"Anvil Level Design: Error in close step: {e}")
        bpy.app.timers.register(_configure_layout_step1, first_interval=0.1)

    return None


def _get_areas_by_type(screen, area_type):
    """Get all areas of a specific type"""
    return [a for a in screen.areas if a.type == area_type]


def _configure_layout_step1():
    """Step 1: Find/create the main VIEW_3D and split for right column"""
    try:
        workspace = bpy.data.workspaces.get(WORKSPACE_NAME)
        if not workspace or bpy.context.window.workspace != workspace:
            return None

        screen = bpy.context.window.screen
        areas = list(screen.areas)

        # If we have one area, great. Otherwise find the largest one.
        if len(areas) == 1:
            main_area = areas[0]
            main_area.type = 'VIEW_3D'
        else:
            # Find the largest area and use it
            main_area = max(areas, key=lambda a: a.width * a.height)
            main_area.type = 'VIEW_3D'

        # Split vertically: 85% left (viewports+browser), 15% right (outliner+props)
        _split_area(main_area, 'VERTICAL', 0.85)

        bpy.app.timers.register(_configure_layout_step2, first_interval=0.1)

    except Exception as e:
        print(f"Anvil Level Design: Error in layout step 1: {e}")

    return None


def _configure_layout_step2():
    """Step 2: Mark right column as OUTLINER, then split it for Properties"""
    try:
        screen = bpy.context.window.screen

        # Find the rightmost VIEW_3D area - that's our right column
        view3d_areas = _get_areas_by_type(screen, 'VIEW_3D')
        if len(view3d_areas) < 2:
            # Maybe only one area, try to continue
            bpy.app.timers.register(_configure_layout_step3, first_interval=0.1)
            return None

        # Sort by x, rightmost is our right column
        view3d_areas.sort(key=lambda a: a.x, reverse=True)
        right_column = view3d_areas[0]

        # Mark it as OUTLINER so we can identify it later
        right_column.type = 'OUTLINER'

        # Split it horizontally: 30% top (Outliner), 70% bottom (Properties)
        _split_area(right_column, 'HORIZONTAL', 0.3)

        bpy.app.timers.register(_configure_layout_step3, first_interval=0.1)

    except Exception as e:
        print(f"Anvil Level Design: Error in layout step 2: {e}")

    return None


def _configure_layout_step3():
    """Step 3: Set bottom of right column to PROPERTIES"""
    try:
        screen = bpy.context.window.screen

        # Find OUTLINER areas (from the split in step 2)
        outliner_areas = _get_areas_by_type(screen, 'OUTLINER')

        if len(outliner_areas) >= 2:
            # Sort by y - lowest one becomes PROPERTIES
            outliner_areas.sort(key=lambda a: a.y)
            outliner_areas[0].type = 'PROPERTIES'

        bpy.app.timers.register(_configure_layout_step4, first_interval=0.1)

    except Exception as e:
        print(f"Anvil Level Design: Error in layout step 3: {e}")

    return None


def _configure_layout_step4():
    """Step 4: Split left area horizontally for file browser at bottom"""
    try:
        screen = bpy.context.window.screen

        # Find VIEW_3D areas (should only be on the left side now)
        view3d_areas = _get_areas_by_type(screen, 'VIEW_3D')
        if not view3d_areas:
            bpy.app.timers.register(_configure_layout_step5, first_interval=0.1)
            return None

        # Get the largest VIEW_3D (our main left area)
        main_area = max(view3d_areas, key=lambda a: a.width * a.height)

        # Split horizontally: 85% top (viewports), 15% bottom (file browser)
        _split_area(main_area, 'HORIZONTAL', 0.20)

        bpy.app.timers.register(_configure_layout_step5, first_interval=0.1)

    except Exception as e:
        print(f"Anvil Level Design: Error in layout step 4: {e}")

    return None


def _configure_layout_step5():
    """Step 5: Mark bottom left as FILE_BROWSER, split top for two viewports"""
    try:
        screen = bpy.context.window.screen

        # Find VIEW_3D areas (left side only now)
        view3d_areas = _get_areas_by_type(screen, 'VIEW_3D')
        if not view3d_areas:
            bpy.app.timers.register(_configure_layout_step6, first_interval=0.1)
            return None

        # Sort by y - lowest becomes file browser
        view3d_areas.sort(key=lambda a: a.y)
        view3d_areas[0].type = 'FILE_BROWSER'

        # The remaining VIEW_3D (top area) needs to be split for two viewports
        remaining_view3d = _get_areas_by_type(screen, 'VIEW_3D')
        if remaining_view3d:
            viewport_area = max(remaining_view3d, key=lambda a: a.width * a.height)
            # Split vertically: 50/50 for left and right viewports
            _split_area(viewport_area, 'VERTICAL', 0.5)

        bpy.app.timers.register(_configure_layout_step6, first_interval=0.1)

    except Exception as e:
        print(f"Anvil Level Design: Error in layout step 5: {e}")

    return None


def _configure_layout_step6():
    """Step 6: Configure all areas with their final settings"""
    try:
        screen = bpy.context.window.screen
        scene = bpy.context.scene

        # Configure scene-level settings
        # Set unit system to None
        scene.unit_settings.system = 'NONE'
        print("Anvil Level Design: Set unit system to None")

        # Enable grid snapping
        scene.tool_settings.use_snap = True
        scene.tool_settings.snap_elements = {'INCREMENT'}
        print("Anvil Level Design: Enabled grid snapping")

        # Configure the two viewports
        view3d_areas = _get_areas_by_type(screen, 'VIEW_3D')
        view3d_areas.sort(key=lambda a: a.x)  # Sort left to right

        if len(view3d_areas) >= 1:
            _configure_left_viewport(view3d_areas[0])
        if len(view3d_areas) >= 2:
            _configure_right_viewport(view3d_areas[1])

        # Configure file browser
        file_browsers = _get_areas_by_type(screen, 'FILE_BROWSER')
        for area in file_browsers:
            for space in area.spaces:
                if space.type == 'FILE_BROWSER':
                    space.params.display_type = 'THUMBNAIL'
                    break

        print(f"Anvil Level Design: Created '{WORKSPACE_NAME}' workspace")

    except Exception as e:
        print(f"Anvil Level Design: Error in layout step 6: {e}")

    return None


def _split_area(area, direction, factor):
    """Split an area using operator with override context.

    Args:
        area: The area to split
        direction: 'VERTICAL' or 'HORIZONTAL'
        factor: Split position (0.0 to 1.0)
    """
    # Find a region in the area to use for context
    region = None
    for r in area.regions:
        if r.type == 'WINDOW':
            region = r
            break

    if not region:
        return False

    try:
        with bpy.context.temp_override(area=area, region=region):
            bpy.ops.screen.area_split(direction=direction, factor=factor)
        return True
    except Exception as e:
        print(f"Anvil Level Design: Error splitting area: {e}")
        return False


def _configure_left_viewport(area):
    """Configure the left 3D viewport: Material preview, N-panel open"""
    for space in area.spaces:
        if space.type == 'VIEW_3D':
            # Set shading to Material Preview
            space.shading.type = 'MATERIAL'

            # Show the sidebar (N-panel) - Level Design panel will be here
            space.show_region_ui = True

            # Show the toolbar (T-panel)
            space.show_region_toolbar = True

            # Set grid subdivisions to 1
            space.overlay.grid_subdivisions = 1
            print("Anvil Level Design: Set left viewport grid subdivisions to 1")

            break


def _configure_right_viewport(area):
    """Configure the right 3D viewport: Wireframe, no panels, top-down ortho, locked rotation"""
    for space in area.spaces:
        if space.type == 'VIEW_3D':
            # Set shading to Wireframe
            space.shading.type = 'WIREFRAME'

            # Hide the sidebar (N-panel)
            space.show_region_ui = False

            # Hide the toolbar (T-panel)
            space.show_region_toolbar = False

            # Set grid subdivisions to 1
            space.overlay.grid_subdivisions = 1
            print("Anvil Level Design: Set right viewport grid subdivisions to 1")

            # Set to orthographic top-down view
            region_3d = space.region_3d
            if region_3d:
                region_3d.view_perspective = 'ORTHO'
                # Top-down view: looking down -Z axis
                region_3d.view_rotation = Quaternion((1, 0, 0, 0))
                # Lock rotation for orthographic view
                region_3d.lock_rotation = True
                print("Anvil Level Design: Locked rotation in right viewport")

            break


@persistent
def on_load_post(dummy):
    """Handler called after a .blend file is loaded"""
    # Use a timer to defer workspace creation until Blender is fully ready
    def deferred_workspace_check():
        try:
            if not workspace_exists():
                create_level_design_workspace()
        except Exception as e:
            print(f"Anvil Level Design: Error checking workspace on load: {e}")
        return None  # Don't repeat

    bpy.app.timers.register(deferred_workspace_check, first_interval=0.5)


def ensure_workspace_exists():
    """Ensure the Level Design workspace exists. Called on addon enable."""
    def deferred_create():
        try:
            if not workspace_exists():
                create_level_design_workspace()
        except Exception as e:
            print(f"Anvil Level Design: Error ensuring workspace: {e}")
        return None

    # Defer creation to ensure Blender is ready
    bpy.app.timers.register(deferred_create, first_interval=0.2)


def register():
    # Create workspace on addon enable
    ensure_workspace_exists()

    # Register handler for when blend files are loaded
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)


def unregister():
    # Remove the load handler
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)

    # Note: We don't delete the workspace on unregister
    # as the user may have customized it
