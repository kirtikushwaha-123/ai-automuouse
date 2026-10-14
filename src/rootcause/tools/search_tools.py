"""Code and text search tools across repository files. Never executes code."""

import os
from pathlib import Path
import re
from typing import List, Optional, Union

from rootcause.core.models import SearchMatch
from rootcause.security.path_guard import validate_safe_path
from rootcause.tools.file_tools import DEFAULT_EXCLUDES, _is_binary_file

MAX_SEARCH_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2MB


def search_in_files(
    repo_path: Union[Path, str],
    query: str,
    is_regex: bool = False,
    max_results: int = 50,
    file_extensions: Optional[List[str]] = None,
    case_sensitive: bool = False,
) -> List[SearchMatch]:
    """Search for a text pattern or regex across repository files.
    
    Args:
        repo_path: Root of the repository.
        query: Search string or regular expression.
        is_regex: If True, treats query as regex.
        max_results: Maximum matching lines to return.
        file_extensions: Optional list of extensions to include (e.g. ['.py', '.json']).
        case_sensitive: Whether match is case sensitive.
        
    Returns:
        List of SearchMatch models with file paths, line numbers, and line contents.
        
    Raises:
        ValueError: If is_regex is True and query is an invalid regular expression.
        FileNotFoundError: If repo_path does not exist.
    """
    resolved_root = Path(repo_path).resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {resolved_root}")

    if not query:
        return []

    flags = 0 if case_sensitive else re.IGNORECASE
    if is_regex:
        try:
            pattern = re.compile(query, flags)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression pattern '{query}': {exc}") from exc
    else:
        escaped = re.escape(query)
        pattern = re.compile(escaped, flags)

    allowed_exts = set(file_extensions) if file_extensions else None

    results: List[SearchMatch] = []

    for root, dirs, files in os.walk(str(resolved_root)):
        dirs[:] = [
            d for d in dirs
            if d not in DEFAULT_EXCLUDES and not d.endswith(".egg-info")
        ]

        for filename in sorted(files):
            if filename.endswith(".pyc") or filename.endswith(".pyo"):
                continue
            if allowed_exts:
                if not any(filename.endswith(ext) for ext in allowed_exts):
                    continue

            full_path = Path(root) / filename
            try:
                safe_path = validate_safe_path(resolved_root, full_path)
                rel_path = str(safe_path.relative_to(resolved_root)).replace("\\", "/")
                
                # Skip binary files and files larger than limit
                if safe_path.stat().st_size > MAX_SEARCH_FILE_SIZE_BYTES or _is_binary_file(safe_path):
                    continue

                with safe_path.open("r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, start=1):
                        if pattern.search(line):
                            results.append(SearchMatch(
                                file_path=rel_path,
                                line_number=line_no,
                                line_content=line.rstrip("\r\n"),
                            ))
                            if len(results) >= max_results:
                                return results
            except Exception:
                continue

    return results
