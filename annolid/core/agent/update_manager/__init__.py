from .auto_update import AutoUpdatePolicy
from .canary import CanaryPolicy, CanaryResult, evaluate_canary
from .history import (
    last_known_good_version,
    load_auto_update_state,
    load_update_history,
    record_update_history,
    save_auto_update_state,
)
from .manager import SignedUpdateManager, SignedUpdatePlan
from .manifest import UpdateManifest, fetch_channel_manifest, release_page_url
from .rollback import RollbackPlan, build_rollback_plan, execute_rollback
from .service import UpdateManagerService
from .verify import VerificationResult, verify_manifest

__all__ = [
    "AutoUpdatePolicy",
    "CanaryPolicy",
    "CanaryResult",
    "evaluate_canary",
    "SignedUpdateManager",
    "SignedUpdatePlan",
    "UpdateManifest",
    "fetch_channel_manifest",
    "release_page_url",
    "VerificationResult",
    "verify_manifest",
    "RollbackPlan",
    "build_rollback_plan",
    "execute_rollback",
    "UpdateManagerService",
    "load_update_history",
    "record_update_history",
    "last_known_good_version",
    "load_auto_update_state",
    "save_auto_update_state",
]
