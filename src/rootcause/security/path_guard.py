"""Path traversal protection and symlink validation."""

import os
from pathlib import Path
from typing import Union

from rootcause.core.exceptions import SecurityViolation


def validate_safe_path(
    root: Union[Path, str],
    target: Union[Path, str],
    allow_nonexistent: bool = False,
) -> Path:
    """Validate that target path is safely contained within root.
    
    Prevents directory traversal (e.g. '../', absolute paths pointing elsewhere),
    null-byte injections, and symlink escapes.
    
    Args:
        root: The allowed root directory.
        target: The target file or directory path (relative or absolute).
        allow_nonexistent: If True, allows paths that do not exist yet (e.g. for creating new files).
        
    Returns:
        The fully resolved, safe Path.
        
    Raises:
        SecurityViolation: If path attempts to escape the root directory or contains null bytes.
        FileNotFoundError: If path does not exist and allow_nonexistent is False.
    """
    root_str = str(root)
    target_str = str(target)
    
    if "\x00" in root_str or "\x00" in target_str:
        raise SecurityViolation("Null byte detected in path.")
    
    resolved_root = Path(root).resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Configured root directory does not exist: {resolved_root}")
    
    target_path = Path(target)
    if target_path.is_absolute():
        candidate = target_path
    else:
        candidate = resolved_root / target_path

    # Resolve symlinks and parent navigation (..)
    resolved_candidate = candidate.resolve()
    
    # Verify candidate is contained within root
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError:
        raise SecurityViolation(
            f"Path traversal detected: '{target}' resolves to '{resolved_candidate}', "
            f"which escapes root directory '{resolved_root}'."
        )
    
    if not allow_nonexistent and not resolved_candidate.exists():
        raise FileNotFoundError(f"Target path does not exist: {resolved_candidate}")
        
    return resolved_candidate


def is_path_safe(
    root: Union[Path, str],
    target: Union[Path, str],
    allow_nonexistent: bool = False,
) -> bool:
    """Check if target path is safe without raising exceptions."""
    try:
        validate_safe_path(root, target, allow_nonexistent=allow_nonexistent)
        return True
    except (SecurityViolation, FileNotFoundError, ValueError):
        return False
