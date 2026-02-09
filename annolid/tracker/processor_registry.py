from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable

ProcessorLoader = Callable[[], type]
ModelNamePredicate = Callable[[str], bool]

CUTIE_BACKEND = "cutie"
COWTRACKER_BACKEND = "cowtracker"
COTRACKER_BACKEND = "cotracker"
DEFAULT_BACKEND = "default"


@dataclass(frozen=True)
class TrackingBackendSpec:
    """Registry entry describing how to route model names to a processor class."""

    name: str
    predicate: ModelNamePredicate
    loader: ProcessorLoader


def normalize_model_name(model_name: str | None) -> str:
    return str(model_name or "").strip().lower()


def model_name_contains(token: str) -> ModelNamePredicate:
    token_norm = token.strip().lower()

    def _predicate(model_name: str) -> bool:
        return token_norm in model_name

    return _predicate


class TrackingProcessorRegistry:
    """Mutable registry for tracking processor backends."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._specs: list[TrackingBackendSpec] = []

    def register(
        self,
        spec: TrackingBackendSpec,
        *,
        replace: bool = False,
        prepend: bool = False,
    ) -> None:
        with self._lock:
            index = next(
                (
                    idx
                    for idx, existing in enumerate(self._specs)
                    if existing.name == spec.name
                ),
                None,
            )
            if index is not None:
                if not replace:
                    raise ValueError(
                        f"Tracking backend '{spec.name}' is already registered."
                    )
                self._specs.pop(index)
            if prepend:
                self._specs.insert(0, spec)
            else:
                self._specs.append(spec)

    def specs(self) -> list[TrackingBackendSpec]:
        with self._lock:
            return list(self._specs)

    def has_match(self, model_name: str | None, backend_name: str) -> bool:
        name_norm = normalize_model_name(model_name)
        with self._lock:
            for spec in self._specs:
                if spec.name == backend_name:
                    return bool(spec.predicate(name_norm))
        return False

    def resolve_processor_class(self, model_name: str | None) -> type:
        name_norm = normalize_model_name(model_name)
        with self._lock:
            for spec in self._specs:
                if spec.predicate(name_norm):
                    return spec.loader()

        raise RuntimeError(
            "Tracking processor registry does not define a default backend."
        )


def _load_cutie_processor() -> type:
    from annolid.segmentation.cutie_vos.video_processor import CutieVideoProcessor

    return CutieVideoProcessor


def _load_cowtracker_processor() -> type:
    from annolid.tracker.cowtracker.track import CoWTrackerProcessor

    return CoWTrackerProcessor


def _load_cotracker_processor() -> type:
    from annolid.tracker.cotracker.track import CoTrackerProcessor

    return CoTrackerProcessor


def _load_default_processor() -> type:
    from annolid.segmentation.SAM.edge_sam_bg import VideoProcessor

    return VideoProcessor


TRACKING_PROCESSOR_REGISTRY = TrackingProcessorRegistry()
TRACKING_PROCESSOR_REGISTRY.register(
    TrackingBackendSpec(
        name=CUTIE_BACKEND,
        predicate=model_name_contains(CUTIE_BACKEND),
        loader=_load_cutie_processor,
    )
)
TRACKING_PROCESSOR_REGISTRY.register(
    TrackingBackendSpec(
        name=COWTRACKER_BACKEND,
        predicate=model_name_contains(COWTRACKER_BACKEND),
        loader=_load_cowtracker_processor,
    )
)
TRACKING_PROCESSOR_REGISTRY.register(
    TrackingBackendSpec(
        name=COTRACKER_BACKEND,
        predicate=model_name_contains(COTRACKER_BACKEND),
        loader=_load_cotracker_processor,
    )
)
TRACKING_PROCESSOR_REGISTRY.register(
    TrackingBackendSpec(
        name=DEFAULT_BACKEND,
        predicate=lambda _: True,
        loader=_load_default_processor,
    )
)


def register_tracking_backend(
    *,
    name: str,
    predicate: ModelNamePredicate,
    loader: ProcessorLoader,
    replace: bool = False,
    prepend: bool = True,
) -> None:
    TRACKING_PROCESSOR_REGISTRY.register(
        TrackingBackendSpec(name=name, predicate=predicate, loader=loader),
        replace=replace,
        prepend=prepend,
    )


def resolve_tracking_processor_class(model_name: str | None) -> type:
    return TRACKING_PROCESSOR_REGISTRY.resolve_processor_class(model_name)
