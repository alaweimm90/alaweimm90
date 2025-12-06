"""
Structured Logging Configuration

Provides:
- JSON structured logging
- Multiple log levels
- File and console handlers
- Request/response logging
- Performance logging
"""

import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
import sys


# ============================================================================
# JSON FORMATTER
# ============================================================================

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            log_data["exception_type"] = record.exc_info[0].__name__

        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


# ============================================================================
# CONTEXT LOGGER
# ============================================================================

class ContextLogger:
    """Logger with context management."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context: Dict[str, Any] = {}

    def set_context(self, **kwargs):
        """Set context fields for subsequent logs."""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear all context."""
        self.context.clear()

    def log(self, level: int, msg: str, **extra):
        """Log with context."""
        # Merge context with extra fields
        fields = {**self.context, **extra}

        # Create LogRecord with extra fields
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(unknown file)",
            0,
            msg,
            (),
            None,
        )
        record.extra_fields = fields

        self.logger.handle(record)

    def debug(self, msg: str, **extra):
        """Debug level log."""
        self.log(logging.DEBUG, msg, **extra)

    def info(self, msg: str, **extra):
        """Info level log."""
        self.log(logging.INFO, msg, **extra)

    def warning(self, msg: str, **extra):
        """Warning level log."""
        self.log(logging.WARNING, msg, **extra)

    def error(self, msg: str, **extra):
        """Error level log."""
        self.log(logging.ERROR, msg, **extra)

    def critical(self, msg: str, **extra):
        """Critical level log."""
        self.log(logging.CRITICAL, msg, **extra)


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_format: Use JSON formatting if True
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.error(f"Failed to create file handler: {e}")

    # Set specific loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)


# ============================================================================
# LOGGERS CACHE
# ============================================================================

_loggers: Dict[str, ContextLogger] = {}


def get_logger(name: str) -> ContextLogger:
    """
    Get or create a context logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        ContextLogger instance
    """
    if name not in _loggers:
        base_logger = logging.getLogger(name)
        _loggers[name] = ContextLogger(base_logger)

    return _loggers[name]


# ============================================================================
# PERFORMANCE LOGGING
# ============================================================================

class PerformanceLogger:
    """Context manager for performance logging."""

    def __init__(self, logger: ContextLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(f"Starting {self.operation}", **self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                duration_seconds=elapsed,
                status="success",
                **self.context
            )
        else:
            self.logger.error(
                f"Failed {self.operation}: {exc_val}",
                duration_seconds=elapsed,
                status="failed",
                exception_type=exc_type.__name__,
                **self.context
            )


# ============================================================================
# REQUEST LOGGING HELPER
# ============================================================================

def log_request(
    logger: ContextLogger,
    method: str,
    path: str,
    status_code: int,
    duration: float,
    **extra
):
    """Log HTTP request."""
    logger.info(
        f"{method} {path}",
        http_method=method,
        http_path=path,
        http_status=status_code,
        duration_ms=duration * 1000,
        **extra
    )


def log_algorithm_execution(
    logger: ContextLogger,
    algorithm: str,
    domain: str,
    duration: float,
    quality: float,
    speedup: float,
    status: str = "success",
):
    """Log algorithm execution."""
    logger.info(
        f"Algorithm {algorithm} executed",
        algorithm=algorithm,
        domain=domain,
        duration_seconds=duration,
        quality=quality,
        speedup=speedup,
        status=status,
    )


def log_validation(
    logger: ContextLogger,
    hypothesis_id: str,
    domain: str,
    duration: float,
    score: float,
    result: str = "valid",
):
    """Log hypothesis validation."""
    logger.info(
        f"Hypothesis {hypothesis_id} validated",
        hypothesis_id=hypothesis_id,
        domain=domain,
        duration_seconds=duration,
        validation_score=score,
        result=result,
    )
