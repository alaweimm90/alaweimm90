"""
Custom exceptions for the ATLAS SDK.
"""


class AtlasError(Exception):
    """Base exception for all ATLAS SDK errors."""
    pass


class AuthenticationError(AtlasError):
    """Raised when API authentication fails."""
    pass


class ValidationError(AtlasError):
    """Raised when request parameters are invalid."""
    pass


class RateLimitError(AtlasError):
    """Raised when API rate limit is exceeded."""
    pass


class ServerError(AtlasError):
    """Raised when the server returns an error."""
    pass


class TaskNotFoundError(AtlasError):
    """Raised when a requested task is not found."""
    pass


class AgentNotFoundError(AtlasError):
    """Raised when a requested agent is not found."""
    pass