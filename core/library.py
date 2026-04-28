def is_library_object(obj):
    """Return True when the object or its data comes from a library."""
    if obj is None:
        return False

    if getattr(obj, "library", None) is not None:
        return True

    data = getattr(obj, "data", None)
    return data is not None and getattr(data, "library", None) is not None
