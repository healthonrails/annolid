from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from annolid.postprocessing.zone_schema import ZoneShapeSpec


@dataclass(frozen=True)
class ZoneAssayProfile:
    name: str
    title: str
    description: str
    allowed_access_states: frozenset[str] | None = None
    included_zone_kinds: frozenset[str] | None = None
    always_include_zone_kinds: frozenset[str] = field(
        default_factory=lambda: frozenset({"barrier_edge", "interaction_zone"})
    )
    respect_zone_phase: bool = False

    def _normalize(self, value: str | None) -> str:
        return str(value or "").strip().lower()

    def includes(self, spec: ZoneShapeSpec) -> bool:
        zone_kind = self._normalize(spec.zone_kind)
        access_state = self._normalize(spec.access_state)
        phase = self._normalize(spec.phase)

        if (
            self.included_zone_kinds is not None
            and zone_kind not in self.included_zone_kinds
        ):
            if zone_kind not in self.always_include_zone_kinds:
                return False

        if self.respect_zone_phase and phase and phase not in {self.name, "custom"}:
            return False

        if self.allowed_access_states is not None:
            if zone_kind in self.always_include_zone_kinds:
                return True
            if access_state not in self.allowed_access_states:
                return False

        return True

    def filter_zone_specs(
        self, zone_specs: Sequence[ZoneShapeSpec]
    ) -> list[ZoneShapeSpec]:
        return [spec for spec in zone_specs if self.includes(spec)]


PROFILE_GENERIC = ZoneAssayProfile(
    name="generic",
    title="Generic",
    description="Analyze every explicit zone without applying access rules.",
    allowed_access_states=None,
    included_zone_kinds=None,
    respect_zone_phase=False,
)

PROFILE_PHASE_1 = ZoneAssayProfile(
    name="phase_1",
    title="Phase 1",
    description=(
        "Treat open chambers and open passageways as accessible while preserving "
        "barrier-edge and interaction zones."
    ),
    allowed_access_states=frozenset({"open", "accessible", "tethered", "unknown"}),
    included_zone_kinds=frozenset(
        {
            "chamber",
            "doorway",
            "barrier_edge",
            "interaction_zone",
            "custom",
        }
    ),
    respect_zone_phase=False,
)

PROFILE_PHASE_2 = ZoneAssayProfile(
    name="phase_2",
    title="Phase 2",
    description="Treat all chambers and passageways as accessible while preserving barrier zones.",
    allowed_access_states=None,
    included_zone_kinds=frozenset(
        {
            "chamber",
            "doorway",
            "barrier_edge",
            "interaction_zone",
            "custom",
        }
    ),
    respect_zone_phase=False,
)


PROFILE_REGISTRY: dict[str, ZoneAssayProfile] = {
    profile.name: profile
    for profile in (PROFILE_GENERIC, PROFILE_PHASE_1, PROFILE_PHASE_2)
}


def available_assay_profiles() -> list[ZoneAssayProfile]:
    return [PROFILE_REGISTRY[key] for key in ("generic", "phase_1", "phase_2")]


def resolve_assay_profile(profile: str | ZoneAssayProfile | None) -> ZoneAssayProfile:
    if isinstance(profile, ZoneAssayProfile):
        return profile
    key = str(profile or "generic").strip().lower()
    return PROFILE_REGISTRY.get(key, PROFILE_GENERIC)


def filter_zone_specs_by_assay_profile(
    zone_specs: Sequence[ZoneShapeSpec],
    profile: str | ZoneAssayProfile | None = None,
) -> list[ZoneShapeSpec]:
    resolved = resolve_assay_profile(profile)
    return resolved.filter_zone_specs(zone_specs)
