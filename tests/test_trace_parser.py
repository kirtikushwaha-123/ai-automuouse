"""Tests for Python stack-trace parsing and source-location extraction."""

import pytest

from rootcause.core.exceptions import TraceParseError
from rootcause.tools.trace_parser import parse_python_traceback

SAMPLE_TRACE_SIMPLE = """\
Traceback (most recent call last):
  File "src/app/calculator.py", line 6, in divide
    return a / b
ZeroDivisionError: division by zero
"""

SAMPLE_TRACE_MULTIFRAME = """\
Traceback (most recent call last):
  File "tests/test_api.py", line 22, in test_endpoint
    response = client.get("/users/1")
  File "src/app/router.py", line 14, in get_user
    return service.fetch_user(user_id)
  File "src/app/service.py", line 45, in fetch_user
    raise ValueError(f"User {user_id} not found in database")
ValueError: User 1 not found in database
"""

SAMPLE_TRACE_CHAINED = """\
Traceback (most recent call last):
  File "src/app/db.py", line 10, in connect
    raise ConnectionRefusedError("Could not connect to port 5432")
ConnectionRefusedError: Could not connect to port 5432

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "src/app/main.py", line 30, in start_app
    connect()
  File "src/app/main.py", line 32, in start_app
    raise RuntimeError("Application failed to initialize database")
RuntimeError: Application failed to initialize database
"""

SAMPLE_SYNTAX_ERROR = """\
  File "src/app/broken.py", line 15
    def invalid_syntax(
                       ^
SyntaxError: unexpected EOF while parsing
"""

SAMPLE_PYTEST_PREFIXED = """\
    def test_divide_zero():
>       divide(10, 0)

tests/test_calc.py:12: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
E   File "src/app/calculator.py", line 8, in divide
E       raise ZeroDivisionError("division by zero")
E   ZeroDivisionError: division by zero
"""

SAMPLE_MULTILINE_EXCEPTION = """\
Traceback (most recent call last):
  File "app/validate.py", line 20, in check_schema
    raise AssertionError(
AssertionError: Schema validation failed:
  - 'name' is a required property
  - 'age' must be greater than or equal to 0
"""


class TestTraceParser:
    """Verify robust traceback parsing, source frame extraction, and error handling."""

    def test_parse_simple_traceback(self):
        parsed = parse_python_traceback(SAMPLE_TRACE_SIMPLE)
        assert parsed.error_type == "ZeroDivisionError"
        assert parsed.error_message == "division by zero"
        assert len(parsed.frames) == 1
        
        frame = parsed.frames[0]
        assert frame.file_path == "src/app/calculator.py"
        assert frame.line_number == 6
        assert frame.function_name == "divide"
        assert frame.code_line == "return a / b"
        assert parsed.innermost_frame == frame

    def test_parse_multiframe_traceback(self):
        parsed = parse_python_traceback(SAMPLE_TRACE_MULTIFRAME)
        assert parsed.error_type == "ValueError"
        assert parsed.error_message == "User 1 not found in database"
        assert len(parsed.frames) == 3
        
        innermost = parsed.innermost_frame
        assert innermost is not None
        assert innermost.file_path == "src/app/service.py"
        assert innermost.line_number == 45
        assert innermost.function_name == "fetch_user"

    def test_parse_chained_traceback(self):
        parsed = parse_python_traceback(SAMPLE_TRACE_CHAINED)
        # Should detect the final unhandled exception in the chain
        assert parsed.error_type == "RuntimeError"
        assert parsed.error_message == "Application failed to initialize database"
        assert len(parsed.frames) >= 2

    def test_parse_syntax_error(self):
        parsed = parse_python_traceback(SAMPLE_SYNTAX_ERROR)
        assert parsed.error_type == "SyntaxError"
        assert "unexpected EOF" in parsed.error_message
        assert len(parsed.frames) == 1
        assert parsed.frames[0].line_number == 15

    def test_parse_pytest_prefixed_traceback(self):
        parsed = parse_python_traceback(SAMPLE_PYTEST_PREFIXED)
        assert parsed.error_type == "ZeroDivisionError"
        assert parsed.error_message == "division by zero"
        assert len(parsed.frames) == 1
        assert parsed.frames[0].file_path == "src/app/calculator.py"
        assert parsed.frames[0].line_number == 8

    def test_parse_multiline_exception_message(self):
        parsed = parse_python_traceback(SAMPLE_MULTILINE_EXCEPTION)
        assert parsed.error_type == "AssertionError"
        assert "Schema validation failed" in parsed.error_message
        assert "'name' is a required property" in parsed.error_message
        assert "'age' must be greater than or equal to 0" in parsed.error_message

    def test_parse_exception_without_frames(self):
        text = "ZeroDivisionError: division by zero"
        parsed = parse_python_traceback(text)
        assert parsed.error_type == "ZeroDivisionError"
        assert parsed.error_message == "division by zero"
        assert len(parsed.frames) == 0
        assert parsed.innermost_frame is None

    def test_parse_frames_only_without_exception(self):
        text = '  File "src/app/worker.py", line 42, in process_task\n    do_work()\n'
        parsed = parse_python_traceback(text)
        assert parsed.error_type == "UnknownError"
        assert len(parsed.frames) == 1
        assert parsed.frames[0].file_path == "src/app/worker.py"
        assert parsed.frames[0].line_number == 42

    def test_malformed_incomplete_frame_ignored(self):
        # Line with missing line number should not match as valid frame
        text = 'File "corrupted.py", missing line info\nValueError: Invalid state\n'
        parsed = parse_python_traceback(text)
        assert parsed.error_type == "ValueError"
        assert parsed.error_message == "Invalid state"
        # Did not invent a frame for the corrupted line
        assert len(parsed.frames) == 0

    def test_empty_traceback_raises_error(self):
        with pytest.raises(TraceParseError, match="Traceback text is empty"):
            parse_python_traceback("")

        with pytest.raises(TraceParseError, match="Traceback text is empty"):
            parse_python_traceback("   \n\t  ")

    def test_malformed_unrelated_text_raises_error(self):
        with pytest.raises(TraceParseError, match="No traceback frames or recognizable exceptions found"):
            parse_python_traceback("Some random log line: 2026-10-13 12:00:00 INFO Server started successfully")
