"""
Centralized logging configuration for RAGOps.
Provides consistent, structured console and file logging.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "ragops", log_file: Optional[str] = "logs/ragops.log") -> logging.Logger:
    """
    Configure and return a standardized logger instance.

    Args:
        name: Name of the logger.
        log_file: Path to write log files (optional).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if already configured
    if logger.handlers:
        return logger

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# Default application logger instance
logger = get_logger("ragops")
