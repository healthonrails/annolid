"""
Interfaces package for Annolid GUI.

This package contains abstract interfaces for dependency injection,
testing, and clean architecture implementation.
"""

from .services import (
    IAnnotationService,
    IInferenceService,
    IProjectService,
    IVideoService,
)

__all__ = [
    "IAnnotationService",
    "IInferenceService",
    "IProjectService",
    "IVideoService",
]
