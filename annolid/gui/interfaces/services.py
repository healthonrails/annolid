"""
Interfaces for Annolid GUI Services.

Defines abstract interfaces for dependency injection and testing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path


class IAnnotationService(ABC):
    """Interface for annotation service operations."""

    @abstractmethod
    def validate_annotation_file(
        self, file_path: Union[str, Path]
    ) -> Tuple[bool, List[str]]:
        """Validate an annotation file."""
        pass

    @abstractmethod
    def convert_annotation_format(
        self, source_format: str, target_format: str, input_data: Any, **kwargs
    ) -> Any:
        """Convert annotation between formats."""
        pass

    @abstractmethod
    def merge_annotations(
        self, annotations_list: List[Dict[str, Any]], merge_strategy: str = "combine"
    ) -> Dict[str, Any]:
        """Merge multiple annotations."""
        pass


class IVideoService(ABC):
    """Interface for video service operations."""

    @abstractmethod
    def load_video(self, video_path: Union[str, Path]) -> Tuple[bool, str]:
        """Load a video file."""
        pass

    @abstractmethod
    def get_video_metadata(
        self, video_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """Get video metadata."""
        pass

    @abstractmethod
    def extract_frame_at_index(
        self, video_path: Union[str, Path], frame_index: int
    ) -> Optional[Any]:
        """Extract a specific frame from video."""
        pass


class IInferenceService(ABC):
    """Interface for inference service operations."""

    @abstractmethod
    def validate_model_config(
        self, model_config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate a model configuration."""
        pass

    @abstractmethod
    def process_inference_results(
        self,
        model_type: str,
        raw_results: Any,
        model_config: Dict[str, Any],
        postprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process raw inference results."""
        pass

    @abstractmethod
    def convert_results_to_labelme_format(
        self, results: Dict[str, Any], image_path: str, image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Convert inference results to LabelMe format."""
        pass


class IProjectService(ABC):
    """Interface for project service operations."""

    @abstractmethod
    def create_project(
        self,
        project_path: Union[str, Path],
        project_name: str,
        project_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Create a new project."""
        pass

    @abstractmethod
    def load_project(self, project_path: Union[str, Path]) -> Tuple[bool, str]:
        """Load an existing project."""
        pass

    @abstractmethod
    def save_project_config(self) -> Tuple[bool, str]:
        """Save project configuration."""
        pass

    @abstractmethod
    def get_project_info(self) -> Optional[Dict[str, Any]]:
        """Get project information."""
        pass
