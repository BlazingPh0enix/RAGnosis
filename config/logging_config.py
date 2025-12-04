"""
Centralized logging configuration for DocuLens.

This module provides a unified logging setup with:
- Console output with colored formatting
- File logging with rotation
- Per-module loggers with configurable levels
- Performance timing decorators
"""

import logging
import logging.handlers
import os
import sys
import time
import functools
from pathlib import Path
from typing import Optional, Callable, Any
from datetime import datetime


# Default log settings
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "doculens.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        # Save original levelname
        levelname = record.levelname
        
        # Add color if available
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = levelname
        
        return result


class LoggingConfig:
    """
    Centralized logging configuration manager.
    
    Usage:
        from config.logging import get_logger, setup_logging
        
        # Initialize logging (call once at app startup)
        setup_logging(log_level=logging.DEBUG)
        
        # Get a logger for your module
        logger = get_logger(__name__)
        logger.info("Starting process...")
    """
    
    _instance: Optional["LoggingConfig"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "LoggingConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not LoggingConfig._initialized:
            self.log_level = DEFAULT_LOG_LEVEL
            self.log_dir = Path(DEFAULT_LOG_DIR)
            self.log_file = DEFAULT_LOG_FILE
            self.console_handler: Optional[logging.Handler] = None
            self.file_handler: Optional[logging.Handler] = None
            self._loggers: dict[str, logging.Logger] = {}
            LoggingConfig._initialized = True
    
    def setup(
        self,
        log_level: int = DEFAULT_LOG_LEVEL,
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        colored_console: bool = True,
    ) -> None:
        """
        Configure the logging system.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            log_file: Log filename
            enable_console: Enable console output
            enable_file: Enable file logging
            colored_console: Use colored output for console
        """
        self.log_level = log_level
        
        if log_dir:
            self.log_dir = Path(log_dir)
        if log_file:
            self.log_file = log_file
        
        # Create log directory if needed
        if enable_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if enable_console:
            self.console_handler = logging.StreamHandler(sys.stdout)
            self.console_handler.setLevel(log_level)
            
            if colored_console and sys.stdout.isatty():
                formatter = ColoredFormatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
            else:
                formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
            
            self.console_handler.setFormatter(formatter)
            root_logger.addHandler(self.console_handler)
        
        # File handler with rotation
        if enable_file:
            log_path = self.log_dir / self.log_file
            self.file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=MAX_LOG_SIZE,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            self.file_handler.setLevel(log_level)
            self.file_handler.setFormatter(
                logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
            )
            root_logger.addHandler(self.file_handler)
        
        # Set levels for noisy third-party libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("qdrant_client").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger for the given module name.
        
        Args:
            name: Module name (typically __name__)
            
        Returns:
            Configured logger instance
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            # Simplify long module names
            if name.startswith("doculens."):
                name = name[9:]  # Remove 'doculens.' prefix
            self._loggers[name] = logger
        return self._loggers.get(name, logging.getLogger(name))
    
    def set_level(self, level: int, logger_name: Optional[str] = None) -> None:
        """
        Set log level for a specific logger or all loggers.
        
        Args:
            level: Logging level
            logger_name: Specific logger name, or None for root logger
        """
        if logger_name:
            logging.getLogger(logger_name).setLevel(level)
        else:
            logging.getLogger().setLevel(level)
            if self.console_handler:
                self.console_handler.setLevel(level)
            if self.file_handler:
                self.file_handler.setLevel(level)


# Global config instance
_config = LoggingConfig()


def setup_logging(
    log_level: int = DEFAULT_LOG_LEVEL,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    colored_console: bool = True,
) -> None:
    """
    Initialize the logging system.
    
    Call this once at application startup.
    
    Args:
        log_level: Logging level (use logging.DEBUG, logging.INFO, etc.)
        log_dir: Directory for log files (default: 'logs/')
        log_file: Log filename (default: 'doculens.log')
        enable_console: Enable console output
        enable_file: Enable file logging with rotation
        colored_console: Use colored output for console (auto-disabled if not TTY)
    """
    _config.setup(
        log_level=log_level,
        log_dir=log_dir,
        log_file=log_file,
        enable_console=enable_console,
        enable_file=enable_file,
        colored_console=colored_console,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module.
    
    Args:
        name: Module name (typically pass __name__)
        
    Returns:
        Logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Processing document...")
    """
    return _config.get_logger(name)


def set_log_level(level: int, logger_name: Optional[str] = None) -> None:
    """
    Change log level at runtime.
    
    Args:
        level: New logging level
        logger_name: Specific logger, or None for global
    """
    _config.set_level(level, logger_name)


# Timing decorator for performance logging
def log_execution_time(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
) -> Callable:
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger to use (defaults to function's module logger)
        level: Log level for timing messages
        
    Example:
        @log_execution_time()
        def slow_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.log(level, f"{func.__name__} completed in {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.log(
                    logging.ERROR,
                    f"{func.__name__} failed after {elapsed:.2f}ms: {e}"
                )
                raise
        return wrapper
    return decorator


# Context manager for timing code blocks
class LogTimer:
    """
    Context manager for timing code blocks.
    
    Example:
        with LogTimer(logger, "Database query"):
            results = db.query(...)
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.DEBUG,
    ):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time: float = 0
    
    def __enter__(self) -> "LogTimer":
        self.start_time = time.perf_counter()
        self.logger.log(self.level, f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = (time.perf_counter() - self.start_time) * 1000
        
        if exc_type is None:
            self.logger.log(
                self.level,
                f"Completed: {self.operation} ({elapsed:.2f}ms)"
            )
        else:
            self.logger.error(
                f"Failed: {self.operation} ({elapsed:.2f}ms) - {exc_val}"
            )


# Convenience function for structured logging
def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """
    Log a structured event with key-value pairs.
    
    Args:
        logger: Logger instance
        event: Event name
        level: Log level
        **kwargs: Additional key-value pairs to log
        
    Example:
        log_event(logger, "document_parsed", 
                  document="report.pdf", pages=10, time_ms=523)
    """
    parts = [f"[{event}]"]
    for key, value in kwargs.items():
        parts.append(f"{key}={value}")
    
    logger.log(level, " ".join(parts))
