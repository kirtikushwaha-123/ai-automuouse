"""RootCause AI verification engine and patch generation package."""

from rootcause.verification.patch_engine import (
    apply_diff_to_worktree,
    generate_candidate_patch,
    parse_diff_target_files,
    validate_patch_safety,
)
from rootcause.verification.verifier import (
    VerificationService,
    apply_verified_patch_to_original,
)

__all__ = [
    "parse_diff_target_files",
    "validate_patch_safety",
    "apply_diff_to_worktree",
    "generate_candidate_patch",
    "VerificationService",
    "apply_verified_patch_to_original",
]
