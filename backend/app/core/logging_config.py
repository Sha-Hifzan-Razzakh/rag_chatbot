import logging
import logging.config
from typing import Any, Dict

from app.core.config import settings


def get_logging_config() -> Dict[str, Any]:
    """
    Build a dictConfig-compatible logging configuration.

    Adjusts log level based on settings.DEBUG / ENVIRONMENT.
    """
    level = "DEBUG" if settings.DEBUG else "INFO"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": (
                    "%(asctime)s | %(levelname)s | %(name)s | "
                    "%(funcName)s:%(lineno)d | %(message)s"
                ),
            },
            "uvicorn": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
            },
            "uvicorn": {
                "class": "logging.StreamHandler",
                "formatter": "uvicorn",
                "level": level,
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console"],
                "level": level,
            },
            "uvicorn": {
                "handlers": ["uvicorn"],
                "level": level,
                "propagate": False,
            },
            "uvicorn.error": {
                "level": level,
            },
            "uvicorn.access": {
                "handlers": ["uvicorn"],
                "level": level,
                "propagate": False,
            },
            "app": {  # our app's namespace
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
        },
    }


def setup_logging() -> None:
    """
    Apply the logging configuration.

    Call this once on application startup (we do it in main.py lifespan).
    """
    config = get_logging_config()
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper to get a logger in other modules.
    """
    return logging.getLogger(name)
