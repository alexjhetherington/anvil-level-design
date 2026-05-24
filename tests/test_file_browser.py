import os

from ..core.materials import get_selected_image_path
from .base_test import AnvilTestCase
from .helpers import TEXTURE_PATH


HOTSPOT_TEXTURE_PATH = os.path.join(os.path.dirname(TEXTURE_PATH), "dev_hotspot.png")


class _FakeParams:
    def __init__(self, directory, filename):
        self.directory = directory.encode('utf-8')
        self.filename = filename


class _FakeActiveSpace:
    def __init__(self, params):
        self.params = params


class _FakeSpaces:
    def __init__(self, active):
        self.active = active


class _FakeArea:
    def __init__(self, area_type, directory, filename):
        self.type = area_type
        self.spaces = _FakeSpaces(_FakeActiveSpace(_FakeParams(directory, filename)))


class _FakeScreen:
    def __init__(self, areas):
        self.areas = areas


def _file_browser_area(filepath):
    directory = os.path.dirname(filepath) + os.sep
    filename = os.path.basename(filepath)
    return _FakeArea('FILE_BROWSER', directory, filename)


def _normal_path(filepath):
    return os.path.normcase(os.path.abspath(filepath))


class FileBrowserSelectionTest(AnvilTestCase):
    def test_file_browser_selected_image_path_skips_non_file_selection_and_uses_later_image_file(self):
        missing_area = _FakeArea(
            'FILE_BROWSER',
            os.path.dirname(TEXTURE_PATH) + os.sep,
            "brick_4.blend\\Material\\brick_4",
        )
        valid_area = _file_browser_area(TEXTURE_PATH)
        screen = _FakeScreen([missing_area, valid_area])

        result = get_selected_image_path(screen, None)

        self.assertEqual(_normal_path(result), _normal_path(TEXTURE_PATH))

    def test_file_browser_selected_image_path_prefers_active_file_browser_area(self):
        first_area = _file_browser_area(TEXTURE_PATH)
        active_area = _file_browser_area(HOTSPOT_TEXTURE_PATH)
        screen = _FakeScreen([first_area, active_area])

        result = get_selected_image_path(screen, active_area)

        self.assertEqual(_normal_path(result), _normal_path(HOTSPOT_TEXTURE_PATH))
