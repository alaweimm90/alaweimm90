"""
Logging infrastructure for Librex.QAP.

Provides centralized logging configuration with support for:
- File and console output
- Multiple log levels
- Formatted messages
- Performance tracking
"""

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    verbose: bool = False,
) -> logging.Logger:
    """
    Configure logging for Librex.QAP.

    Parameters
    ----------
    log_file : str, optional
        Path to log file. If None, only console logging.
    level : str, default='INFO'
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    verbose : bool, default=False
        Enable verbose output

    Returns
    -------
    logger : logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger("Librex.QAP")
    logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    logger.handlers = []

    # Format string
    if verbose:
        fmt = (
            "%(asctime)s | %(name)s | %(levelname)-8s | "
            "%(filename)s:%(lineno)d | %(funcName)s() | %(message)s"
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(message)s"

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path}")

    return logger


class PerformanceTracker:
    """Track and log performance metrics."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize performance tracker.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance. If None, uses 'Librex.QAP' logger.
        """
        self.logger = logger or logging.getLogger("Librex.QAP")
        self.metrics = {}

    def start(self, name: str) -> None:
        """
        Start timing a section.

        Parameters
        ----------
        name : str
            Section name
        """
        self.metrics[name] = {"start": datetime.now(), "end": None}
        self.logger.debug(f"Started: {name}")

    def end(self, name: str) -> float:
        """
        End timing a section.

        Parameters
        ----------
        name : str
            Section name

        Returns
        -------
        elapsed : float
            Elapsed time in seconds
        """
        if name not in self.metrics:
            self.logger.warning(f"Stopping non-existent timer: {name}")
            return 0.0

        self.metrics[name]["end"] = datetime.now()
        elapsed = (self.metrics[name]["end"] - self.metrics[name]["start"]).total_seconds()

        self.logger.debug(f"Completed: {name} ({elapsed:.3f}s)")
        return elapsed

    def log_summary(self) -> None:
        """Log summary of all tracked metrics."""
        if not self.metrics:
            self.logger.info("No metrics tracked")
            return

        self.logger.info("=" * 60)
        self.logger.info("PERFORMANCE SUMMARY")
        self.logger.info("=" * 60)

        total_time = 0
        for name, times in sorted(self.metrics.items()):
            if times["end"] is None:
                status = "(INCOMPLETE)"
                elapsed = 0
            else:
                elapsed = (times["end"] - times["start"]).total_seconds()
                status = ""
                total_time += elapsed

            self.logger.info(f"  {name:<30} {elapsed:>8.3f}s  {status}")

        self.logger.info("=" * 60)
        self.logger.info(f"  {'TOTAL':<30} {total_time:>8.3f}s")
        self.logger.info("=" * 60)

    def get_elapsed(self, name: str) -> Optional[float]:
        """
        Get elapsed time for a metric.

        Parameters
        ----------
        name : str
            Metric name

        Returns
        -------
        elapsed : float or None
            Elapsed time in seconds, or None if not found/incomplete
        """
        if name not in self.metrics:
            return None

        if self.metrics[name]["end"] is None:
            return None

        return (self.metrics[name]["end"] - self.metrics[name]["start"]).total_seconds()


# Global logger instance
_global_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """
    Get global logger instance.

    Returns
    -------
    logger : logging.Logger
        Global Librex.QAP logger
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = setup_logging(level="INFO")

    return _global_logger


def log_info(message: str) -> None:
    """Log info message."""
    get_logger().info(message)


def log_warning(message: str) -> None:
    """Log warning message."""
    get_logger().warning(message)


def log_error(message: str) -> None:
    """Log error message."""
    get_logger().error(message)


def log_debug(message: str) -> None:
    """Log debug message."""
    get_logger().debug(message)


__all__ = [
    "setup_logging",
    "PerformanceTracker",
    "get_logger",
    "log_info",
    "log_warning",
    "log_error",
    "log_debug",
]
