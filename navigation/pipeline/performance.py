"""Performance utilities - DEPRECATED: Use navigation.utils.image_processing instead.

This module is kept for backward compatibility and re-exports functions from
navigation.utils.image_processing. Direct imports from this module are deprecated.
"""

from __future__ import annotations

import warnings

# Re-export from new location for backward compatibility
from navigation.utils.image_processing import (
    resize_for_inference,
    upscale_class_map,
)

__all__ = [
    "resize_for_inference",
    "upscale_class_map",
]

# Warn on first import
warnings.warn(
    "navigation.pipeline.performance is deprecated. "
    "Import from navigation.utils.image_processing instead.",
    DeprecationWarning,
    stacklevel=2,
)
