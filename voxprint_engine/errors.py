"""Errors the worker can show to the user without a traceback."""
from __future__ import annotations


class ImageJobError(ValueError):
    """The request is not an image job this worker can finish."""
