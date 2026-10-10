"""Structured logging configuration for Industrial Incident Investigator."""

import logging
import sys
from typing import Optional


def configure_logging(level: str = "INFO") -> None:
    """Configures root logger with standardized formatting."""
    log_format = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s"
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Reduce verbosity of chatty 3rd party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Returns a module logger with the given name."""
    return logging.getLogger(name)
