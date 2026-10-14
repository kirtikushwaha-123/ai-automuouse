"""Safe repository file listing and reading tools. Never executes repository code."""

import os
from pathlib import Path
from typing import List, Optional, Set, Union

from rootcause.core.models import FileInfo, FileReadResult
from rootcause.security.path_guard import validate_safe_path

DEFAULT_EXCLUDES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "build",
    "dist",
    ".eggs",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".whl",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".pyc", ".pyo", ".pyd",
    ".pdf", ".sqlite", ".db",
}

MAX_SAFE_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def _is_binary_file(path: Path) -> bool:
    """Detect whether a file is binary using extension and initial byte inspection."""
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with path.open("rb") as f:
            chunk = f.read(4096)
            return b"\x00" in chunk
    except Exception:
        return True


def list_repo_files(
    repo_path: Union[Path, str],
    max_files: int = 1000,
    exclude_patterns: Optional[List[str]] = None,
    max_file_size_bytes: int = MAX_SAFE_FILE_SIZE_BYTES,
) -> List[FileInfo]:
    """List repository files safely without executing any code.
    
    Args:
        repo_path: Absolute or relative path to repo root.
        max_files: Maximum number of files to return.
        exclude_patterns: Additional folder/file patterns to exclude.
        max_file_size_bytes: Skip files exceeding this size limit.
        
    Returns:
        List of FileInfo models with relative paths, sizes, and binary flags.
    """
    resolved_root = Path(repo_path).resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {resolved_root}")

    excludes = set(DEFAULT_EXCLUDES)
    if exclude_patterns:
        excludes.update(exclude_patterns)

    file_list: List[FileInfo] = []

    for root, dirs, files in os.walk(str(resolved_root)):
        # Prune excluded directories in-place
        dirs[:] = [
            d for d in dirs
            if d not in excludes and not any(d.endswith(x) for x in [".egg-info"])
        ]

        for filename in sorted(files):
            if filename.endswith(".pyc") or filename.endswith(".pyo"):
                continue
            full_path = Path(root) / filename
            try:
                safe_path = validate_safe_path(resolved_root, full_path)
                rel_path = str(safe_path.relative_to(resolved_root)).replace("\\", "/")
                file_stat = safe_path.stat()

                if file_stat.st_size > max_file_size_bytes:
                    continue

                is_bin = _is_binary_file(safe_path)

                file_list.append(FileInfo(
                    path=rel_path,
                    size_bytes=file_stat.st_size,
                    is_file=True,
                    is_binary=is_bin,
                ))
                if len(file_list) >= max_files:
                    return file_list
            except Exception:
                continue

    return file_list


def safe_read_file(
    repo_path: Union[Path, str],
    relative_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    max_lines: int = 500,
    max_file_size_bytes: int = MAX_SAFE_FILE_SIZE_BYTES,
) -> FileReadResult:
    """Read file content with line ranges and bounds checking without executing code.
    
    Args:
        repo_path: Root of the repository.
        relative_path: Relative path to the file.
        start_line: 1-indexed starting line number.
        end_line: 1-indexed ending line number.
        max_lines: Safety limit on total lines returned.
        max_file_size_bytes: Safety limit on file size in bytes.
        
    Returns:
        FileReadResult containing file content, line counts, and line-numbered formatting.
    """
    safe_path = validate_safe_path(repo_path, relative_path)
    
    if not safe_path.is_file():
        raise FileNotFoundError(f"Path is not a regular file: {safe_path}")

    file_size = safe_path.stat().st_size
    is_binary = _is_binary_file(safe_path)

    if is_binary:
        return FileReadResult(
            file_path=relative_path.replace("\\", "/"),
            total_lines=0,
            start_line=1,
            end_line=1,
            content="[Binary file not displayed]",
            numbered_content="   1 | [Binary file not displayed]",
            truncated=False,
            file_size_bytes=file_size,
            is_binary=True,
        )

    # Check file size limit
    truncated = False
    if file_size > max_file_size_bytes:
        truncated = True

    with safe_path.open("r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
        
    total_lines = len(all_lines)
    
    if total_lines == 0:
        return FileReadResult(
            file_path=relative_path.replace("\\", "/"),
            total_lines=0,
            start_line=1,
            end_line=0,
            content="",
            numbered_content="",
            truncated=truncated,
            file_size_bytes=file_size,
            is_binary=False,
        )

    # Determine bounds (1-indexed)
    s_line = max(1, start_line) if start_line is not None else 1
    e_line = min(total_lines, end_line) if end_line is not None else total_lines
    
    if s_line > total_lines:
        s_line = total_lines
    if e_line < s_line:
        e_line = s_line

    selected_lines = all_lines[s_line - 1 : e_line]
    
    if len(selected_lines) > max_lines:
        selected_lines = selected_lines[:max_lines]
        e_line = s_line + max_lines - 1
        truncated = True

    numbered_lines = [
        f"{s_line + idx:4d} | {line}"
        for idx, line in enumerate(selected_lines)
    ]

    return FileReadResult(
        file_path=relative_path.replace("\\", "/"),
        total_lines=total_lines,
        start_line=s_line,
        end_line=e_line,
        content="".join(selected_lines),
        numbered_content="".join(numbered_lines),
        truncated=truncated,
        file_size_bytes=file_size,
        is_binary=False,
    )
