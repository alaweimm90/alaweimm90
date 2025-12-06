"""
Structured Logging for TalAI Turing Challenge System.

Provides JSON-formatted logging with contextual information for debugging and monitoring.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        # Add context if present
        if hasattr(record, "context"):
            log_data["context"] = record.context

        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """Filter to add contextual information to log records."""

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        record.context = self.context
        return True


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = True,
) -> None:
    """
    Set up logging configuration for TalAI.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        json_format: Use JSON formatting (default: True for structured logs)
    """
    log_level = getattr(logging, level.upper())

    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    handlers = [console_handler]

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )

    # Set third-party loggers to WARNING
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with optional context.

    Args:
        name: Logger name (usually __name__)
        context: Optional context dictionary to include in all logs

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__, {"feature": "agent-tournaments"})
        >>> logger.info("Tournament started", extra={"participants": 8})
    """
    logger = logging.getLogger(name)

    if context:
        logger.addFilter(ContextFilter(context))

    return logger


# Convenience loggers for each Turing Challenge feature
tournament_logger = get_logger("talair.tournaments", {"feature": "agent-tournaments"})
advocate_logger = get_logger("talair.advocate", {"feature": "devils-advocate"})
swarm_logger = get_logger("talair.swarm", {"feature": "swarm-voting"})
emergent_logger = get_logger("talair.emergent", {"feature": "emergent-behavior"})
system_logger = get_logger("talair.system", {"feature": "turing-system"})


# Example usage
if __name__ == "__main__":
    # Set up logging
    setup_logging(level="DEBUG", json_format=True)

    # Test different log levels
    logger = get_logger(__name__, {"test": True})

    logger.debug("Debug message", extra={"detail": "verbose"})
    logger.info("Info message", extra={"count": 42})
    logger.warning("Warning message", extra={"threshold": 80})
    logger.error("Error message", extra={"code": "E001"})

    try:
        raise ValueError("Test exception")
    except Exception:
        logger.exception("Exception occurred", extra={"context": "testing"})
