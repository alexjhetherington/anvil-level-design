import os
import shutil
import tempfile

import bpy

from ..core.materials import create_material_with_image, find_material_with_image
from ..handlers import file_browser
from ..handlers.file_browser import consolidate_duplicate_materials
from .base_test import AnvilTestCase
from .helpers import TEXTURE_PATH


class MaterialLookupTest(AnvilTestCase):
    """Verify materials are keyed by image identity, not just by filename."""

    def setUp(self):
        super().setUp()
        file_browser._last_material_count = 0
        self._tmp_dir = tempfile.mkdtemp(prefix="anvil_material_lookup_")
        self._dark_path = os.path.join(self._tmp_dir, "Dark", "texture_01.png")
        self._green_path = os.path.join(self._tmp_dir, "Green", "texture_01.png")
        os.makedirs(os.path.dirname(self._dark_path))
        os.makedirs(os.path.dirname(self._green_path))
        shutil.copyfile(TEXTURE_PATH, self._dark_path)
        shutil.copyfile(TEXTURE_PATH, self._green_path)

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_find_material_returns_none_for_cross_folder_same_name(self):
        """find_material_with_image distinguishes images with shared basename."""
        create_material_with_image(bpy.data.images.load(self._dark_path, check_existing=True))
        green_image = bpy.data.images.load(self._green_path, check_existing=True)

        self.assertIsNone(find_material_with_image(green_image))

    def test_find_material_reuses_existing_for_same_image(self):
        """Reloading the same file returns the same material."""
        image = bpy.data.images.load(self._dark_path, check_existing=True)
        material = create_material_with_image(image)
        reloaded = bpy.data.images.load(self._dark_path, check_existing=True)

        self.assertIs(find_material_with_image(reloaded), material)

    def test_consolidate_keeps_cross_folder_materials_separate(self):
        """consolidate must not merge IMG_X and IMG_X.001 referencing different images."""
        create_material_with_image(bpy.data.images.load(self._dark_path, check_existing=True))
        create_material_with_image(bpy.data.images.load(self._green_path, check_existing=True))

        consolidate_duplicate_materials()

        self.assertIn("IMG_texture_01.png", bpy.data.materials)
        self.assertIn("IMG_texture_01.png.001", bpy.data.materials)

    def test_consolidate_still_merges_object_duplicates(self):
        """consolidate still merges IMG_X.001 with IMG_X when both reference the same image."""
        image = bpy.data.images.load(self._dark_path, check_existing=True)
        canonical = create_material_with_image(image)
        canonical.copy()

        consolidate_duplicate_materials()

        self.assertIn("IMG_texture_01.png", bpy.data.materials)
        self.assertNotIn("IMG_texture_01.png.001", bpy.data.materials)
