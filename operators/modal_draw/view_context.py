"""Shared 3D View resolution for modal draw tools."""


class View3DTarget:
    """A concrete 3D viewport target used by modal placement tools."""

    def __init__(self, window, screen, area, region, space_data):
        self.window = window
        self.screen = screen
        self.area = area
        self.region = region
        self.space_data = space_data

    def override_kwargs(self):
        return {
            "window": self.window,
            "screen": self.screen,
            "area": self.area,
            "region": self.region,
            "space_data": self.space_data,
        }

    def matches(self, other):
        if other is None:
            return False
        try:
            return (
                self.window == other.window
                and self.screen == other.screen
                and self.area == other.area
                and self.region == other.region
                and self.space_data == other.space_data
            )
        except (AttributeError, ReferenceError):
            return False

    def is_live(self):
        try:
            if self.window.screen != self.screen:
                return False
            if not any(area == self.area for area in self.screen.areas):
                return False
            if self.area.type != 'VIEW_3D':
                return False
            if self.region.type != 'WINDOW':
                return False
            if not any(region == self.region for region in self.area.regions):
                return False
            if not any(space == self.space_data for space in self.area.spaces):
                return False
            return self.space_data.region_3d is not None
        except (AttributeError, ReferenceError):
            return False


class ViewMouseEvent:
    """Mouse event data converted to a target region's local coordinates."""

    def __init__(self, mouse_x, mouse_y, mouse_region_x, mouse_region_y, ctrl, shift, alt):
        self.mouse_x = mouse_x
        self.mouse_y = mouse_y
        self.mouse_region_x = mouse_region_x
        self.mouse_region_y = mouse_region_y
        self.ctrl = ctrl
        self.shift = shift
        self.alt = alt


def _window_region_for_area(area):
    try:
        for region in area.regions:
            if region.type == 'WINDOW':
                return region
    except (AttributeError, ReferenceError):
        return None
    return None


def _active_3d_space_for_area(area):
    try:
        if area.type != 'VIEW_3D':
            return None
        space = area.spaces.active
        if space is None or space.type != 'VIEW_3D':
            return None
        if space.region_3d is None:
            return None
        return space
    except (AttributeError, ReferenceError):
        return None


def _mouse_in_region(region, mouse_x, mouse_y):
    try:
        return (
            mouse_x >= region.x
            and mouse_x < region.x + region.width
            and mouse_y >= region.y
            and mouse_y < region.y + region.height
        )
    except (AttributeError, ReferenceError):
        return False


def view_target_for_area(window, screen, area, preferred_region):
    space = _active_3d_space_for_area(area)
    if space is None:
        return None

    region = preferred_region
    try:
        if region is None or region.type != 'WINDOW' or not any(region == candidate for candidate in area.regions):
            region = _window_region_for_area(area)
    except (AttributeError, ReferenceError):
        return None

    if region is None:
        return None

    return View3DTarget(window, screen, area, region, space)


def view_target_under_mouse(window, mouse_x, mouse_y):
    try:
        screen = window.screen
        areas = list(screen.areas)
    except (AttributeError, ReferenceError):
        return None

    for area in areas:
        space = _active_3d_space_for_area(area)
        if space is None:
            continue

        region = _window_region_for_area(area)
        if region is None:
            continue

        if _mouse_in_region(region, mouse_x, mouse_y):
            return View3DTarget(window, screen, area, region, space)

    return None


def view_mouse_event_for_target(event, target):
    mouse_x = event.mouse_x
    mouse_y = event.mouse_y
    try:
        mouse_region_x = mouse_x - target.region.x
        mouse_region_y = mouse_y - target.region.y
    except (AttributeError, ReferenceError):
        mouse_region_x = event.mouse_region_x
        mouse_region_y = event.mouse_region_y

    return ViewMouseEvent(
        mouse_x,
        mouse_y,
        mouse_region_x,
        mouse_region_y,
        event.ctrl,
        event.shift,
        event.alt,
    )


def synthetic_mouse_event(mouse_x, mouse_y, target, ctrl, shift, alt):
    try:
        mouse_region_x = mouse_x - target.region.x
        mouse_region_y = mouse_y - target.region.y
    except (AttributeError, ReferenceError):
        mouse_region_x = 0
        mouse_region_y = 0

    return ViewMouseEvent(
        mouse_x,
        mouse_y,
        mouse_region_x,
        mouse_region_y,
        ctrl,
        shift,
        alt,
    )
