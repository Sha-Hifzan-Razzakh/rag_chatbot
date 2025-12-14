import logging
import logging.config
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, cast

from app.core.config import settings


# -----------------------------
# Context-aware logger adapter
# -----------------------------

class AgentContextAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that injects agent/runtime context fields.

    Usage:
      base = get_logger(__name__)
      logger = AgentContextAdapter(base, {"conversation_id": "...", "request_id": "..."})
      logger.info("msg", extra={"tool_name": "search_docs"})
    """

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> Tuple[str, MutableMapping[str, Any]]:
        extra_any = kwargs.get("extra") or {}

        # Pylance-safe: ensure both sides are real dicts before ** unpacking.
        extra_dict: Dict[str, Any] = dict(extra_any) if isinstance(extra_any, Mapping) else {}

        adapter_extra_any = getattr(self, "extra", {})  # LoggerAdapter.extra is loosely typed
        adapter_extra: Dict[str, Any] = dict(cast(Any, adapter_extra_any)) if isinstance(adapter_extra_any, Mapping) else {}

        merged: Dict[str, Any] = {**adapter_extra, **extra_dict}
        kwargs["extra"] = merged
        return msg, kwargs


def get_logging_config() -> Dict[str, Any]:
    """
    Build a dictConfig-compatible logging configuration.

    Adjusts log level based on settings.DEBUG / ENVIRONMENT.
    """
    level = "DEBUG" if settings.DEBUG else "INFO"

    # Include context keys safely via %(key)s (they'll be present via defaults filter)
    base_fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(funcName)s:%(lineno)d | "
        "conversation_id=%(conversation_id)s request_id=%(request_id)s "
        "tool=%(tool_name)s stop=%(stop_reason)s | %(message)s"
    )

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "context_defaults": {
                "()": "app.core.logging_config.ContextDefaultsFilter",
            }
        },
        "formatters": {
            "default": {
                "format": base_fmt,
            },
            "uvicorn": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "filters": ["context_defaults"],
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


class ContextDefaultsFilter(logging.Filter):
    """
    Ensures missing contextual fields don't break formatting.
    """

    DEFAULTS = {
        "conversation_id": "-",
        "request_id": "-",
        "tool_name": "-",
        "stop_reason": "-",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        for k, v in self.DEFAULTS.items():
            if not hasattr(record, k):
                setattr(record, k, v)
        return True


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


def get_agent_logger(
    name: str,
    *,
    conversation_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> AgentContextAdapter:
    """
    Convenience wrapper to get a LoggerAdapter pre-loaded with agent context.
    """
    base = get_logger(name)
    return AgentContextAdapter(
        base,
        {
            "conversation_id": conversation_id or "-",
            "request_id": request_id or "-",
        },
    )
