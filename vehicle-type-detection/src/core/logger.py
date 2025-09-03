import logging
import os
import sys


def setup_logger(name: str = "vehicle_detection_api", level: int = None) -> logging.Logger:
    """
    Set up a logger with console output
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Get log level from environment or use default
        if level is None:
            log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
            level = getattr(logging, log_level_str, logging.INFO)

        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger
