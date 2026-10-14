"""Controlled subprocess runner with strict command allowlisting, timeouts, and env scrubbing."""

import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Union

from rootcause.core.exceptions import (
    CommandExecutionError,
    CommandTimeoutError,
    SecurityViolation,
)
from rootcause.core.models import CommandExecutionResult
from rootcause.security.path_guard import validate_safe_path

# Allowed executable names (lowercase stems)
ALLOWED_EXECUTABLES = {"git", "pytest", "python", "python3"}

# Disallowed Git subcommands and dangerous options
FORBIDDEN_GIT_SUBCOMMANDS = {
    "clone",
    "push",
    "remote",
    "config",
    "alias",
    "credential",
    "daemon",
    "upload-pack",
    "receive-pack",
}

# Safe system environment variables to preserve
SAFE_ENV_ALLOWLIST = {
    "PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "LANG",
    "LC_ALL",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "WINDIR",
}

# Regex pattern identifying sensitive keys that must never be exposed to subprocesses
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)(key|secret|token|password|auth|cred|openai|anthropic|gemini|aws|github|gitlab)"
)


class SubprocessRunner:
    """Executes external processes in a strictly controlled, sandboxed manner."""

    def __init__(self, allowed_root: Union[Path, str], default_timeout: float = 30.0):
        self.allowed_root = Path(allowed_root).resolve()
        if not self.allowed_root.exists():
            raise FileNotFoundError(f"Allowed root directory does not exist: {self.allowed_root}")
        self.default_timeout = default_timeout

    def _validate_command(self, command: Sequence[str]) -> None:
        """Verify command adheres to strict security allowlist."""
        if not command or not isinstance(command, (list, tuple)):
            raise SecurityViolation("Command must be a non-empty sequence of string arguments.")

        for arg in command:
            if not isinstance(arg, str):
                raise SecurityViolation(f"All command arguments must be strings, got: {type(arg)}")
            if "\x00" in arg:
                raise SecurityViolation("Command argument contains null byte.")

        executable = Path(command[0]).stem.lower()
        if executable not in ALLOWED_EXECUTABLES:
            raise SecurityViolation(
                f"Executable '{command[0]}' is not in the allowlist {ALLOWED_EXECUTABLES}."
            )

        # Validate python invocations: only allow -m pytest or -m unittest
        if executable in {"python", "python3"}:
            if len(command) < 3 or command[1] != "-m":
                raise SecurityViolation(
                    f"Python can only be invoked with '-m pytest' or '-m unittest', got: {' '.join(command)}"
                )
            module = command[2].lower()
            if module not in {"pytest", "unittest"}:
                raise SecurityViolation(
                    f"Python module '{module}' is not allowed. Only 'pytest' and 'unittest' are permitted."
                )

        # Validate git invocations
        if executable == "git":
            if len(command) > 1:
                subcommand = command[1].lower()
                # Check for forbidden subcommands
                if subcommand in FORBIDDEN_GIT_SUBCOMMANDS:
                    raise SecurityViolation(
                        f"Git subcommand '{subcommand}' is forbidden by security policy."
                    )
                # Disallow inline config overrides via git -c
                if subcommand == "-c" or any(arg.startswith("-c") for arg in command[1:]):
                    raise SecurityViolation("Inline git '-c' configuration overrides are forbidden.")
                # Disallow dangerous flags
                for arg in command:
                    if arg.startswith("--exec-path") or arg.startswith("--upload-pack"):
                        raise SecurityViolation(f"Dangerous git flag '{arg}' is forbidden.")

    def _sanitize_env(self, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Produce a clean environment scrubbed of sensitive secrets and tokens."""
        clean_env: Dict[str, str] = {}

        # Copy only safe system variables from host
        for key, val in os.environ.items():
            if key.upper() in SAFE_ENV_ALLOWLIST:
                if not SENSITIVE_KEY_PATTERN.search(key):
                    clean_env[key] = val

        # Ensure Python unbuffered / utf-8 defaults
        clean_env["PYTHONUNBUFFERED"] = "1"
        clean_env["PYTHONIOENCODING"] = "utf-8"

        # Apply any explicit extra environment variables if not sensitive
        if extra_env:
            for key, val in extra_env.items():
                if SENSITIVE_KEY_PATTERN.search(key):
                    raise SecurityViolation(
                        f"Attempted to inject sensitive environment variable '{key}' into subprocess."
                    )
                clean_env[key] = val

        return clean_env

    def run(
        self,
        command: Sequence[str],
        cwd: Optional[Union[Path, str]] = None,
        timeout: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
        check: bool = False,
        input: Optional[str] = None,
    ) -> CommandExecutionResult:
        """Execute a controlled command within the allowed working directory boundary.
        
        Args:
            command: Command argument list.
            cwd: Working directory (must be contained within self.allowed_root).
            timeout: Maximum execution time in seconds.
            extra_env: Additional non-sensitive environment variables.
            check: If True, raise CommandExecutionError on non-zero exit code.
            input: Optional string input passed to stdin.
            
        Returns:
            CommandExecutionResult with execution details.
            
        Raises:
            SecurityViolation: If command, arguments, or cwd violate security rules.
            CommandTimeoutError: If execution exceeds timeout limit.
            CommandExecutionError: If check=True and command returns non-zero code.
        """
        self._validate_command(command)

        # Validate working directory
        target_cwd = cwd if cwd is not None else self.allowed_root
        safe_cwd = validate_safe_path(self.allowed_root, target_cwd)

        # Prepare sanitized environment
        env = self._sanitize_env(extra_env)

        timeout_sec = timeout if timeout is not None else self.default_timeout

        cmd_list = list(command)
        executable = Path(cmd_list[0]).stem.lower()
        if executable in {"python", "python3"} and cmd_list[0].lower() in {"python", "python3", "python.exe", "python3.exe"}:
            cmd_list[0] = sys.executable

        start_time = time.perf_counter()
        try:
            process = subprocess.run(
                cmd_list,
                input=input,
                cwd=str(safe_cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                shell=False,
                encoding="utf-8",
                errors="replace",
            )
            duration = time.perf_counter() - start_time
            result = CommandExecutionResult(
                command=list(command),
                exit_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                duration_seconds=round(duration, 4),
                timed_out=False,
                cwd=str(safe_cwd),
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start_time
            raise CommandTimeoutError(
                f"Command '{' '.join(command)}' timed out after {timeout_sec:.1f}s",
                timeout_seconds=timeout_sec,
            ) from exc

        if check and result.exit_code != 0:
            raise CommandExecutionError(
                f"Command '{' '.join(command)}' failed with exit code {result.exit_code}: {result.stderr or result.stdout}",
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        return result
