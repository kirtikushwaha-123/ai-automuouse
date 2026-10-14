"""Stack-trace parsing and source-location extraction for Python tracebacks."""

import re
from typing import List, Optional

from rootcause.core.exceptions import TraceParseError
from rootcause.core.models import ParsedFrame, ParsedTraceback

FRAME_RE = re.compile(
    r'^(?:[E>]\s+)?\s*File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>.+))?',
)

EXCEPTION_RE = re.compile(
    r"^(?:[E>]\s+)?(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Failure|Interrupt|Exit)?)(?::\s*(?P<msg>.*))?$"
)

# Common Python built-in and standard exceptions
KNOWN_EXCEPTIONS = {
    "AssertionError", "AttributeError", "EOFError", "FloatingPointError",
    "GeneratorExit", "ImportError", "IndexError", "KeyError",
    "KeyboardInterrupt", "MemoryError", "NameError", "NotImplementedError",
    "OSError", "OverflowError", "RecursionError", "ReferenceError",
    "RuntimeError", "StopIteration", "SyntaxError", "IndentationError",
    "TabError", "SystemError", "SystemExit", "TypeError",
    "UnboundLocalError", "UnicodeError", "ValueError", "ZeroDivisionError",
    "FileNotFoundError", "PermissionError", "IsADirectoryError",
    "ConnectionError", "TimeoutError",
}


def parse_python_traceback(trace_text: str) -> ParsedTraceback:
    """Parse a Python traceback into structured frames and exception details.
    
    Extracts filenames, line numbers, function names, and error messages without
    inventing data for malformed or missing components. Supports standard tracebacks
    as well as pytest-prefixed (`E `) failure logs.
    
    Args:
        trace_text: Raw traceback or test failure text.
        
    Returns:
        ParsedTraceback instance.
        
    Raises:
        TraceParseError: If text is empty or cannot be parsed as a traceback.
    """
    if not trace_text or not trace_text.strip():
        raise TraceParseError("Traceback text is empty.")

    lines = trace_text.strip().splitlines()
    frames: List[ParsedFrame] = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = FRAME_RE.match(line)
        if match:
            file_path = match.group("file")
            line_number = int(match.group("line"))
            function_name = match.group("func").strip() if match.group("func") else "<module>"
            
            # Check if next line contains the code snippet
            code_line = None
            if i + 1 < len(lines):
                next_raw = lines[i + 1].strip()
                # Remove pytest E/ > prefix if present
                clean_next = re.sub(r"^[E>]\s+", "", next_raw).strip()
                if not FRAME_RE.match(clean_next) and not clean_next.startswith("^") and not EXCEPTION_RE.match(clean_next):
                    code_line = clean_next
                    i += 1

            frames.append(ParsedFrame(
                file_path=file_path,
                line_number=line_number,
                function_name=function_name,
                code_line=code_line,
            ))
        i += 1

    # Find the exception line (searching backwards from bottom)
    error_type = "UnknownError"
    error_message = ""
    found_exception = False

    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx].strip()
        if not line:
            continue
        # Skip caret/syntax indicator lines
        clean_caret = re.sub(r"^[E>]\s+", "", line)
        if set(clean_caret) <= {"^", "~", " "}:
            continue

        exc_match = EXCEPTION_RE.match(line)
        if exc_match:
            cand_type = exc_match.group("type")
            if cand_type and (
                cand_type.endswith("Error")
                or cand_type.endswith("Exception")
                or cand_type in KNOWN_EXCEPTIONS
                or (cand_type.isidentifier() and not cand_type.startswith("File"))
            ):
                error_type = cand_type
                first_msg = exc_match.group("msg").strip() if exc_match.group("msg") else ""
                
                # Check for multiline continuation messages
                msg_parts = [first_msg] if first_msg else []
                for cont_idx in range(idx + 1, len(lines)):
                    cont_line = lines[cont_idx].strip()
                    if not cont_line or cont_line.startswith("File ") or FRAME_RE.match(cont_line):
                        break
                    clean_cont = re.sub(r"^[E>]\s+", "", cont_line).strip()
                    if clean_cont and not EXCEPTION_RE.match(clean_cont):
                        msg_parts.append(clean_cont)
                    else:
                        break

                error_message = "\n".join(msg_parts).strip()
                found_exception = True
                break

    if not frames and not found_exception:
        raise TraceParseError("No traceback frames or recognizable exceptions found in input.")

    innermost_frame = frames[-1] if frames else None

    return ParsedTraceback(
        error_type=error_type,
        error_message=error_message,
        frames=frames,
        raw_traceback=trace_text,
        innermost_frame=innermost_frame,
    )
