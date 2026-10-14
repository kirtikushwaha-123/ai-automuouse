"""Custom exceptions for RootCause AI."""


class RootCauseException(Exception):
    """Base exception for all RootCause AI errors."""
    pass


class SecurityViolation(RootCauseException):
    """Raised when a security guardrail is violated (path traversal, disallowed command, symlink escape)."""
    pass


class CommandExecutionError(RootCauseException):
    """Raised when a controlled subprocess command fails unexpectedly."""

    def __init__(self, message: str, exit_code: int = -1, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class CommandTimeoutError(CommandExecutionError):
    """Raised when a subprocess execution exceeds its hard timeout."""

    def __init__(self, message: str, timeout_seconds: float):
        super().__init__(message, exit_code=-1)
        self.timeout_seconds = timeout_seconds


class WorktreeError(RootCauseException):
    """Raised when git worktree operations fail."""
    pass


class InvalidRepositoryError(RootCauseException):
    """Raised when a directory is not a valid git repository."""
    pass


class TraceParseError(RootCauseException):
    """Raised when a traceback cannot be parsed."""
    pass


class InvalidStateTransitionError(RootCauseException):
    """Raised when an illegal investigation state transition is attempted."""

    def __init__(self, current_status: str, target_status: str):
        super().__init__(
            f"Invalid state transition from '{current_status}' to '{target_status}'."
        )
        self.current_status = current_status
        self.target_status = target_status


class InvalidEvidenceReferenceError(RootCauseException):
    """Raised when a hypothesis references fabricated or non-existent evidence IDs."""
    pass


class ConfigurationError(RootCauseException):
    """Raised when required configuration or credentials are missing or invalid."""
    pass


class AuthenticationError(ConfigurationError):
    """Raised when provided credentials are rejected or unauthorized."""
    pass


class MissingCredentialsError(ConfigurationError):
    """Raised when required LLM credentials are missing in live mode."""
    pass


class LLMProviderError(RootCauseException):
    """Raised when the LLM provider fails (network, rate limit, timeout, malformed payload)."""
    pass


class RateLimitError(LLMProviderError):
    """Raised when provider returns HTTP 429 rate limit exceeded."""
    pass


class ToolArgumentValidationError(RootCauseException):
    """Raised when tool arguments fail Pydantic schema validation."""
    pass


class UnregisteredToolError(RootCauseException):
    """Raised when the model attempts to call an unregistered tool name."""
    pass


class InvalidPatchError(RootCauseException):
    """Raised when a candidate patch is empty, malformed, or unparseable."""
    pass


class PatchApplicationError(RootCauseException):
    """Raised when patch application fails or conflicts occur."""
    pass


class VerificationError(RootCauseException):
    """Raised when verification encounters an execution failure."""
    pass


class ConfirmationRequiredError(RootCauseException):
    """Raised when application to original repository is attempted without explicit human confirmation."""
    pass


class StaleVerificationError(RootCauseException):
    """Raised when attempting to use or verify against stale results or mismatched repository state."""
    pass

