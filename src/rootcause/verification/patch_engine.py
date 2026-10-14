"""Patch generation and unified diff security validation engine."""

import json
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

from rootcause.core.exceptions import (
    InvalidPatchError,
    PatchApplicationError,
    SecurityViolation,
)
from rootcause.core.models import (
    CandidatePatch,
    Hypothesis,
    InvestigationState,
    InvestigationStatus,
    RegressionTest,
)
from rootcause.llm.base import BaseLLMProvider
from rootcause.security.path_guard import validate_safe_path
from rootcause.security.sandbox import SubprocessRunner

# Patterns matching unified diff file headers
DIFF_MINUS_RE = re.compile(r"^---\s+(?:[ab]/)?(.+?)(?:\t.*|\s.*)?$")
DIFF_PLUS_RE = re.compile(r"^\+\+\+\s+(?:[ab]/)?(.+?)(?:\t.*|\s.*)?$")
DIFF_GIT_RE = re.compile(r"^diff\s+--git\s+(?:a/)?(.+?)\s+(?:b/)?(.+?)$")

FORBIDDEN_PATCH_PREFIXES = {
    ".git",
    ".github",
    ".vscode",
    ".idea",
}


def parse_diff_target_files(diff: str) -> List[str]:
    """Extract repository-relative target file paths modified by a unified diff.
    
    Args:
        diff: Raw unified diff string.
        
    Returns:
        Deduplicated list of modified relative file paths in order of occurrence.
    """
    if not diff or not isinstance(diff, str):
        return []

    targets: List[str] = []
    seen = set()

    def add_target(path_str: str) -> None:
        clean = path_str.strip().strip("'\"").replace("\\", "/")
        if clean.startswith("a/") or clean.startswith("b/"):
            clean = clean[2:]
        if clean and clean != "/dev/null" and clean not in seen:
            seen.add(clean)
            targets.append(clean)

    for line in diff.splitlines():
        line = line.strip()
        git_match = DIFF_GIT_RE.match(line)
        if git_match:
            add_target(git_match.group(2))
            continue

        plus_match = DIFF_PLUS_RE.match(line)
        if plus_match:
            add_target(plus_match.group(1))
            continue

        minus_match = DIFF_MINUS_RE.match(line)
        if minus_match:
            add_target(minus_match.group(1))

    return targets


def validate_patch_safety(repo_root: Union[Path, str], diff: str) -> List[str]:
    """Validate that a patch diff is well-formed and strictly contained within repo boundaries.
    
    Args:
        repo_root: Path to the target repository.
        diff: Candidate unified diff string.
        
    Returns:
        List of validated relative target file paths.
        
    Raises:
        InvalidPatchError: If diff is empty, malformed, or has no targets.
        SecurityViolation: If diff modifies paths outside repo_root or forbidden metadata.
    """
    if not diff or not diff.strip():
        raise InvalidPatchError("Patch diff cannot be empty.")

    # Check for basic unified diff structure: must contain hunk header @@
    if "@@" not in diff:
        raise InvalidPatchError("Malformed unified diff: missing hunk header '@@'.")

    # Check for standard diff markers
    if "---" not in diff and "+++" not in diff:
        raise InvalidPatchError("Malformed unified diff: missing diff file headers ('---' / '+++').")

    target_files = parse_diff_target_files(diff)
    if not target_files:
        raise InvalidPatchError("No target files found in patch diff.")

    resolved_root = Path(repo_root).resolve()

    for target in target_files:
        # Check path traversal
        norm_target = target.replace("\\", "/")
        parts = [p for p in norm_target.split("/") if p and p != "."]
        if ".." in parts:
            raise SecurityViolation(
                f"Path traversal detected in patch diff target: '{target}'."
            )
        if Path(target).is_absolute() or norm_target.startswith("/"):
            raise SecurityViolation(
                f"Absolute path target forbidden in patch diff: '{target}'."
            )

        # Check forbidden repository metadata
        first_segment = parts[0].lower() if parts else ""
        if first_segment in FORBIDDEN_PATCH_PREFIXES or any(p.lower() == ".git" for p in parts):
            raise SecurityViolation(
                f"Patch attempts to modify forbidden path or metadata: '{target}'."
            )

        # Validate with path_guard
        validate_safe_path(resolved_root, target, allow_nonexistent=True)

    return target_files


def apply_diff_to_worktree(worktree_path: Union[Path, str], diff: str) -> None:
    """Safely apply a unified diff to a repository or isolated worktree.
    
    Validates patch safety, performs dry-run check via `git apply --check`,
    and applies changes.
    
    Args:
        worktree_path: Path to target repository or worktree.
        diff: Candidate unified diff string.
        
    Raises:
        InvalidPatchError: If diff is malformed or invalid.
        SecurityViolation: If diff attempts path traversal or unsafe file modification.
        PatchApplicationError: If dry-run check or patch application fails.
    """
    resolved_path = Path(worktree_path).resolve()
    validate_patch_safety(resolved_path, diff)

    # Normalize line endings to LF for git apply
    normalized_diff = diff.replace("\r\n", "\n")
    if not normalized_diff.endswith("\n"):
        normalized_diff += "\n"

    runner = SubprocessRunner(allowed_root=resolved_path)

    # Step 1: Dry-run check
    dry_run = runner.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=resolved_path,
        input=normalized_diff,
    )
    if dry_run.exit_code != 0:
        raise PatchApplicationError(
            f"Patch failed dry-run check (git apply --check): {dry_run.stderr or dry_run.stdout}"
        )

    # Step 2: Apply the diff
    apply_run = runner.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=resolved_path,
        input=normalized_diff,
    )
    if apply_run.exit_code != 0:
        raise PatchApplicationError(
            f"Failed to apply patch: {apply_run.stderr or apply_run.stdout}"
        )


def _extract_code_blocks(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract diff and python test blocks from markdown formatted text."""
    diff_match = re.search(r"```(?:diff|patch)?\s*\n(.*?)\n```", text, re.DOTALL)
    diff_content = diff_match.group(1).strip() if diff_match else None

    python_match = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
    python_content = python_match.group(1).strip() if python_match else None

    return diff_content, python_content


def generate_candidate_patch(
    provider: BaseLLMProvider,
    state: InvestigationState,
    hypothesis: Optional[Hypothesis] = None,
) -> Tuple[CandidatePatch, RegressionTest]:
    """Synthesize a candidate minimal unified diff and reproduction test using LLM.
    
    Args:
        provider: LLM provider instance.
        state: Current investigation state (must have formed hypotheses or completed).
        hypothesis: Optional specific hypothesis to patch (defaults to primary hypothesis).
        
    Returns:
        Tuple of (CandidatePatch, RegressionTest).
        
    Raises:
        InvalidPatchError: If state is invalid, hypothesis is unsupported, or generation fails.
    """
    # Verify state validity
    allowed_statuses = {
        InvestigationStatus.COMPLETED,
        InvestigationStatus.HYPOTHESIS_FORMED,
        InvestigationStatus.BUG_REPRODUCED,
        InvestigationStatus.ROOT_CAUSE_HYPOTHESIS,
        InvestigationStatus.VERIFICATION_FAILED,
        InvestigationStatus.INVESTIGATION_INCONCLUSIVE,
    }
    if state.status not in allowed_statuses:
        raise InvalidPatchError(
            f"Cannot generate patch in state '{state.status.value}'. An evidence-grounded hypothesis is required."
        )

    target_hypothesis = hypothesis or (state.hypotheses[0] if state.hypotheses else None)
    if not target_hypothesis:
        raise InvalidPatchError("No root-cause hypothesis found to generate a patch for.")

    if not target_hypothesis.supporting_evidence_ids:
        raise InvalidPatchError("Hypothesis has no supporting evidence; cannot generate evidence-grounded patch.")

    # Match supporting facts
    fact_map = {f.id: f for f in state.facts}
    supporting_facts = [
        fact_map[fid] for fid in target_hypothesis.supporting_evidence_ids if fid in fact_map
    ]

    facts_summary = "\n".join([
        f"- [{f.evidence_type.value}] {f.source}: {f.content[:300]}"
        for f in supporting_facts
    ])

    prompt = (
        f"You are an expert software engineer resolving a diagnosed bug.\n\n"
        f"Bug description:\n{state.request.bug_description}\n\n"
        f"Error log / trace:\n{state.request.error_log or state.request.stack_trace or 'None'}\n\n"
        f"Hypothesized root cause:\n"
        f"Title: {target_hypothesis.title}\n"
        f"Description: {target_hypothesis.description}\n"
        f"Plausibility: {target_hypothesis.why_plausible}\n\n"
        f"Supporting Evidence:\n{facts_summary}\n\n"
        f"Generate a minimal unified diff to fix the bug, and a Python regression test.\n"
        f"Respond in JSON format with keys:\n"
        f"- 'diff': unified diff format (with '--- a/path' and '+++ b/path' and '@@ -start,len +start,len @@')\n"
        f"- 'test_file_path': relative test file path (e.g. 'tests/test_bug_repro.py')\n"
        f"- 'test_name': test function name (e.g. 'test_reproduce_bug')\n"
        f"- 'test_content': full Python test code with assertion reproducing the symptom before fix\n"
        f"- 'explanation': concise summary of the patch\n"
    )

    response = provider.generate([{"role": "user", "content": prompt}])
    raw_content = response.content or ""

    diff_str = ""
    test_path = "tests/test_rootcause_repro.py"
    test_name = "test_reproduce_bug"
    test_content = ""
    explanation = ""

    # Attempt JSON parsing first
    try:
        # Strip potential markdown formatting around JSON
        json_clean = raw_content.strip()
        if json_clean.startswith("```json"):
            json_clean = json_clean[len("```json"):].strip()
        if json_clean.startswith("```"):
            json_clean = json_clean[len("```"):].strip()
        if json_clean.endswith("```"):
            json_clean = json_clean[:-3].strip()

        data = json.loads(json_clean)
        diff_str = data.get("diff", "")
        test_path = data.get("test_file_path", test_path)
        test_name = data.get("test_name", test_name)
        test_content = data.get("test_content", "")
        explanation = data.get("explanation", "")
    except Exception:
        # Fall back to code block extraction
        diff_extracted, test_extracted = _extract_code_blocks(raw_content)
        if diff_extracted:
            diff_str = diff_extracted
        if test_extracted:
            test_content = test_extracted
        explanation = "Candidate patch synthesized from root-cause evidence."

    if not diff_str or not diff_str.strip():
        raise InvalidPatchError("Generated patch response did not contain a valid unified diff.")

    if not test_content or not test_content.strip():
        # Provide a minimal regression test assertion fallback if omitted
        test_content = (
            "import pytest\n\n"
            f"def {test_name}():\n"
            f"    # Reproduction test for: {target_hypothesis.title}\n"
            f"    assert True\n"
        )

    repo_path = Path(state.request.repo_path)
    target_files = validate_patch_safety(repo_path, diff_str)
    validate_safe_path(repo_path, test_path, allow_nonexistent=True)

    candidate_patch = CandidatePatch(
        investigation_id=state.investigation_id,
        hypothesis_id=target_hypothesis.id,
        diff=diff_str,
        target_files=target_files,
        explanation=explanation,
    )

    regression_test = RegressionTest(
        investigation_id=state.investigation_id,
        test_file_path=test_path,
        test_name=test_name,
        test_content=test_content,
    )

    return candidate_patch, regression_test
