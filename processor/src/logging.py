"""Logging configuration for the application."""

import logging

def configure_logging():
    """Configure logging with standard format and levels."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)7s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)