"""Tool registry and argument validator for RootCause AI."""

import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from rootcause.core.exceptions import (
    ToolArgumentValidationError,
    UnregisteredToolError,
)
from rootcause.core.models import Evidence, EvidenceType
from rootcause.tools.file_tools import list_repo_files, safe_read_file
from rootcause.tools.git_tools import (
    get_commit_diff,
    get_git_status,
    get_recent_commits,
    get_working_tree_diff,
)
from rootcause.tools.search_tools import search_in_files
from rootcause.tools.test_runner import check_test_reproduces_bug, discover_tests, run_pytest
from rootcause.tools.trace_parser import parse_python_traceback

# Sensitive argument keys that must be redacted in audit logs
SENSITIVE_KEY_RE = re.compile(r"(?i)(key|secret|token|password|auth|cred)")


def redact_arguments(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of arguments with sensitive values redacted."""
    redacted: Dict[str, Any] = {}
    for k, v in args.items():
        if SENSITIVE_KEY_RE.search(k):
            redacted[k] = "[REDACTED]"
        elif isinstance(v, dict):
            redacted[k] = redact_arguments(v)
        else:
            redacted[k] = v
    return redacted


# --- Pydantic Argument Schemas for Tools ---

class ListFilesArgs(BaseModel):
    max_files: int = Field(500, ge=1, le=2000, description="Maximum files to return")
    exclude_patterns: Optional[List[str]] = Field(None, description="Optional extra directory/file exclusion patterns")


class ReadFileArgs(BaseModel):
    relative_path: str = Field(..., description="Repository-relative file path")
    start_line: Optional[int] = Field(None, ge=1, description="1-indexed starting line number")
    end_line: Optional[int] = Field(None, ge=1, description="1-indexed ending line number")
    max_lines: int = Field(200, ge=1, le=1000, description="Maximum lines to read")


class SearchCodeArgs(BaseModel):
    query: str = Field(..., description="Text query or regular expression to search")
    is_regex: bool = Field(False, description="Whether query should be treated as a regex")
    max_results: int = Field(30, ge=1, le=100, description="Maximum matching lines")
    file_extensions: Optional[List[str]] = Field(None, description="Optional extensions filter (e.g. ['.py'])")
    case_sensitive: bool = Field(False, description="Case-sensitive search")


class GitStatusArgs(BaseModel):
    pass


class RecentCommitsArgs(BaseModel):
    max_count: int = Field(10, ge=1, le=50, description="Maximum recent commits to inspect")


class CommitDiffArgs(BaseModel):
    commit_ref: str = Field(..., description="Commit hash or reference (e.g. HEAD, HEAD~1)")


class WorkingTreeDiffArgs(BaseModel):
    relative_file: Optional[str] = Field(None, description="Optional relative file path to limit diff")


class ParseTracebackArgs(BaseModel):
    trace_text: str = Field(..., description="Raw Python stack trace or test failure output")


class DiscoverTestsArgs(BaseModel):
    test_target: Optional[str] = Field(None, description="Optional subfolder or test file")


class RunPytestArgs(BaseModel):
    test_target: Optional[str] = Field(None, description="Optional specific test file or test node id")
    timeout: float = Field(30.0, ge=1.0, le=120.0, description="Execution timeout in seconds")


class ToolExecutionResult(BaseModel):
    """Structured output from executing a registered tool."""
    tool_name: str
    success: bool
    output_summary: str
    raw_output: Any
    evidence_items: List[Evidence] = Field(default_factory=list)
    error_message: Optional[str] = None


class ToolDefinition:
    """Descriptor and execution binding for a registered tool."""

    def __init__(
        self,
        name: str,
        description: str,
        schema: Type[BaseModel],
        handler: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.schema = schema
        self.handler = handler

    def to_openai_tool(self) -> Dict[str, Any]:
        """Produce OpenAI-compatible function definition."""
        json_schema = self.schema.model_json_schema()
        # Clean internal metadata
        parameters = {
            "type": "object",
            "properties": json_schema.get("properties", {}),
            "required": json_schema.get("required", []),
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


class ToolRegistry:
    """Registry managing available investigation tools with validated execution."""

    def __init__(
        self,
        repo_path: Union[Path, str],
        bug_description: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.bug_description = bug_description or ""
        self.stack_trace = stack_trace or ""
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def register(
        self,
        name: str,
        description: str,
        schema: Type[BaseModel],
        handler: Callable[..., Any],
    ) -> None:
        """Register a new tool."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            schema=schema,
            handler=handler,
        )

    def get_tool(self, name: str) -> ToolDefinition:
        """Get a registered tool definition."""
        if name not in self._tools:
            raise UnregisteredToolError(f"Tool '{name}' is not in registered tools: {list(self._tools.keys())}")
        return self._tools[name]

    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return all tools in OpenAI-compatible function format."""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def execute(
        self,
        tool_name: str,
        raw_args: Dict[str, Any],
        step_index: int = 0,
    ) -> ToolExecutionResult:
        """Validate arguments and execute the requested tool, synthesizing evidence.
        
        Args:
            tool_name: Registered tool name.
            raw_args: Dictionary of arguments.
            step_index: Current step in investigation.
            
        Returns:
            ToolExecutionResult with bounded output and structured Evidence items.
            
        Raises:
            UnregisteredToolError: If tool_name is not registered.
            ToolArgumentValidationError: If arguments fail Pydantic validation.
        """
        try:
            tool = self.get_tool(tool_name)
        except UnregisteredToolError as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                output_summary=f"UnregisteredToolError: {str(exc)}",
                raw_output=None,
                error_message=str(exc),
            )

        # Validate arguments against Pydantic schema
        try:
            validated_args = tool.schema.model_validate(raw_args or {})
        except ValidationError as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                output_summary=f"ToolArgumentValidationError: {str(exc)}",
                raw_output=None,
                error_message=str(exc),
            )

        # Execute handler
        try:
            output = tool.handler(self.repo_path, validated_args)
            evidence_items = self._extract_evidence(tool_name, validated_args, output, step_index)
            summary = self._summarize_output(tool_name, output)

            return ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                output_summary=summary,
                raw_output=output,
                evidence_items=evidence_items,
            )
        except Exception as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                output_summary=f"Execution error: {type(exc).__name__}: {str(exc)}",
                raw_output=None,
                error_message=f"{type(exc).__name__}: {str(exc)}",
            )

    def _summarize_output(self, tool_name: str, output: Any) -> str:
        """Produce a bounded textual summary of tool output suitable for LLM context."""
        if output is None:
            return "No output."
        if isinstance(output, str):
            if len(output) > 2000:
                return output[:2000] + f"\n... [truncated {len(output) - 2000} chars]"
            return output
        if isinstance(output, list):
            items_str = json.dumps([item.model_dump() if hasattr(item, "model_dump") else item for item in output[:10]], default=str)
            if len(output) > 10:
                return items_str + f"\n... [total {len(output)} items, truncated]"
            return items_str
        if hasattr(output, "model_dump"):
            dump = output.model_dump()
            # If large content, truncate content field
            if "content" in dump and isinstance(dump["content"], str) and len(dump["content"]) > 2000:
                dump["content"] = dump["content"][:2000] + "\n... [content truncated]"
            return json.dumps(dump, default=str)
        return str(output)[:2000]

    def _extract_evidence(
        self,
        tool_name: str,
        validated_args: BaseModel,
        output: Any,
        step_index: int,
    ) -> List[Evidence]:
        """Convert tool outputs into structured Evidence items."""
        evidence_list: List[Evidence] = []

        if tool_name == "read_repo_file" and hasattr(output, "content") and output.content:
            args = validated_args
            evidence_list.append(Evidence(
                evidence_type=EvidenceType.SOURCE_CODE,
                source=f"{output.file_path}:{output.start_line}-{output.end_line}",
                file_path=output.file_path,
                line_start=output.start_line,
                line_end=output.end_line,
                content=output.content,
                why_relevant=f"Inspected source file {output.file_path} from lines {output.start_line} to {output.end_line}",
                source_tool="read_repo_file",
                step_index=step_index,
            ))

        elif tool_name == "search_code" and isinstance(output, list) and output:
            for match in output[:5]:  # Top 5 matches as evidence items
                evidence_list.append(Evidence(
                    evidence_type=EvidenceType.SOURCE_CODE,
                    source=f"{match.file_path}:{match.line_number}",
                    file_path=match.file_path,
                    line_start=match.line_number,
                    line_end=match.line_number,
                    content=match.line_content,
                    why_relevant=f"Matched search query '{getattr(validated_args, 'query', '')}' in {match.file_path}",
                    source_tool="search_code",
                    step_index=step_index,
                ))

        elif tool_name == "get_commit_diff" and isinstance(output, str) and output:
            ref = getattr(validated_args, "commit_ref", "HEAD")
            evidence_list.append(Evidence(
                evidence_type=EvidenceType.GIT_DIFF,
                source=f"git show {ref}",
                content=output[:3000],
                why_relevant=f"Inspected git diff for commit {ref}",
                source_tool="get_commit_diff",
                step_index=step_index,
            ))

        elif tool_name == "run_pytest" and hasattr(output, "output"):
            passed_cnt = output.passed_count if hasattr(output, "passed_count") and output.passed_count else (output.tests_ran - output.failures - output.errors)
            failed_cnt = output.failures + output.errors

            failing_tests = []
            for line in output.failure_details:
                if line.startswith("FAILED "):
                    failing_tests.append(line.replace("FAILED ", "").split(" - ")[0].strip())
                elif line.startswith("ERROR "):
                    failing_tests.append(line.replace("ERROR ", "").split(" - ")[0].strip())
            if not failing_tests and output.failure_details:
                failing_tests = output.failure_details[:3]

            reproduces, repro_reason = check_test_reproduces_bug(
                output,
                bug_description=self.bug_description,
                stack_trace=self.stack_trace,
            )

            fail_str = f"Failing test(s): {', '.join(failing_tests)}" if failing_tests else "No failing tests"
            repro_str = f"Bug reproduced: {'YES' if reproduces else 'NO'} ({repro_reason})"

            why_relevant = (
                f"Pytest execution: {passed_cnt} passed, {failed_cnt} failed ({output.tests_ran} total). "
                f"{fail_str}. {repro_str}"
            )

            evidence_list.append(Evidence(
                evidence_type=EvidenceType.TEST_OUTPUT,
                source=f"pytest {getattr(validated_args, 'test_target', '') or 'suite'}",
                content=output.output[:3000],
                why_relevant=why_relevant,
                source_tool="run_pytest",
                step_index=step_index,
                metadata={
                    "passed": output.passed,
                    "exit_code": output.exit_code,
                    "tests_ran": output.tests_ran,
                    "passed_count": passed_cnt,
                    "failures": output.failures,
                    "errors": output.errors,
                    "failing_tests": failing_tests,
                    "reproduces_bug": reproduces,
                    "reproduction_reason": repro_reason,
                    "timed_out": output.timed_out,
                    "failure_details": output.failure_details,
                },
            ))

        elif tool_name == "get_recent_commits" and isinstance(output, list) and output:
            commits_summary = "\n".join(f"{c.hash[:8]} - {c.author} ({c.date}): {c.message}" for c in output[:5])
            evidence_list.append(Evidence(
                evidence_type=EvidenceType.GIT_COMMIT,
                source="git log",
                content=commits_summary,
                why_relevant=f"Inspected {len(output)} recent commits in repository history",
                source_tool="get_recent_commits",
                step_index=step_index,
            ))

        elif tool_name == "get_working_tree_diff" and isinstance(output, str) and output:
            evidence_list.append(Evidence(
                evidence_type=EvidenceType.GIT_DIFF,
                source="git diff",
                content=output[:3000],
                why_relevant="Inspected uncommitted changes in git working tree",
                source_tool="get_working_tree_diff",
                step_index=step_index,
            ))

        elif tool_name == "list_repo_files" and isinstance(output, list) and output:
            file_paths = [f.relative_path for f in output[:30]]
            evidence_list.append(Evidence(
                evidence_type=EvidenceType.SOURCE_CODE,
                source="repo file list",
                content="\n".join(file_paths),
                why_relevant=f"Enumerated repository structure ({len(output)} files discovered)",
                source_tool="list_repo_files",
                step_index=step_index,
            ))

        elif tool_name == "parse_traceback" and hasattr(output, "frames"):
            for frame in output.frames[-3:]:  # Innermost frames
                evidence_list.append(Evidence(
                    evidence_type=EvidenceType.STACK_TRACE,
                    source=f"{frame.file_path}:{frame.line_number}",
                    file_path=frame.file_path,
                    line_start=frame.line_number,
                    line_end=frame.line_number,
                    content=f"in {frame.function_name}: {frame.code_line or output.error_type}",
                    why_relevant=f"Active stack frame in {output.error_type}: {output.error_message}",
                    source_tool="parse_traceback",
                    step_index=step_index,
                ))

        return evidence_list

    def _register_default_tools(self) -> None:
        """Register the default suite of investigation tools."""
        self.register(
            name="list_repo_files",
            description="List repository files safely. Excludes .git and build caches.",
            schema=ListFilesArgs,
            handler=lambda repo, args: list_repo_files(repo, max_files=args.max_files, exclude_patterns=args.exclude_patterns),
        )
        self.register(
            name="read_repo_file",
            description="Read content and line numbers of a repository file within line bounds.",
            schema=ReadFileArgs,
            handler=lambda repo, args: safe_read_file(repo, args.relative_path, start_line=args.start_line, end_line=args.end_line, max_lines=args.max_lines),
        )
        self.register(
            name="search_code",
            description="Search repository files for text or regex patterns. Returns matching lines and line numbers.",
            schema=SearchCodeArgs,
            handler=lambda repo, args: search_in_files(repo, args.query, is_regex=args.is_regex, max_results=args.max_results, file_extensions=args.file_extensions, case_sensitive=args.case_sensitive),
        )
        self.register(
            name="get_git_status",
            description="Inspect git working tree status (uncommitted changes or untracked files).",
            schema=GitStatusArgs,
            handler=lambda repo, args: get_git_status(repo),
        )
        self.register(
            name="get_recent_commits",
            description="Retrieve recent git commits with commit hash, author, date, and message.",
            schema=RecentCommitsArgs,
            handler=lambda repo, args: get_recent_commits(repo, max_count=args.max_count),
        )
        self.register(
            name="get_commit_diff",
            description="Inspect the full patch and diff of a specific Git commit.",
            schema=CommitDiffArgs,
            handler=lambda repo, args: get_commit_diff(repo, args.commit_ref),
        )
        self.register(
            name="get_working_tree_diff",
            description="Inspect uncommitted diffs against HEAD.",
            schema=WorkingTreeDiffArgs,
            handler=lambda repo, args: get_working_tree_diff(repo, relative_file=args.relative_file),
        )
        self.register(
            name="parse_traceback",
            description="Parse a Python stack trace to extract source files, line numbers, and error messages.",
            schema=ParseTracebackArgs,
            handler=lambda repo, args: parse_python_traceback(args.trace_text),
        )
        self.register(
            name="discover_tests",
            description="Discover available pytest test targets without executing code.",
            schema=DiscoverTestsArgs,
            handler=lambda repo, args: discover_tests(repo, test_target=args.test_target),
        )
        self.register(
            name="run_pytest",
            description="Execute pytest under controlled sandbox constraints and parse structured results.",
            schema=RunPytestArgs,
            handler=lambda repo, args: run_pytest(repo, test_target=args.test_target, timeout=args.timeout),
        )
