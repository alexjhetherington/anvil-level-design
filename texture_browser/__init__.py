"""Texture browser feature package."""

from . import browser


def register():
    browser.register()


def unregister():
    browser.unregister()
