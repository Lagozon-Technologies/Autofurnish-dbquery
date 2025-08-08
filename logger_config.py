import logging
from logging.config import dictConfig
import time
from contextlib import contextmanager


# Configure logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG"  # <--- THIS LINE CHANGED
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "app.log",
            "formatter": "json",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "level": "DEBUG"
        }
    },
    "loggers": {
        "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": [], "level": "CRITICAL", "propagate": False},
        "app": {"handlers": ["console", "file"], "level": "DEBUG", "propagate": False},
        "azure.core.pipeline.policies.http_logging_policy": {
            "handlers": ["console", "file"],
            "level": "ERROR",
            "propagate": False
        }
    },
    "root": {"handlers": ["console"], "level": "INFO"}
}



def configure_logging():
    """Initialize the logging configuration"""
    dictConfig(LOGGING_CONFIG)
    # Suppress noisy loggers
    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
# Context manager for timing operations
@contextmanager
def log_execution_time(operation_name, logger=None):
    """
    Context manager for logging operation execution time.
    If no logger is provided, creates one with the caller's module name.
    """
    if logger is None:
        # Get the caller's module name
        import inspect
        caller_frame = inspect.stack()[1]
        module = inspect.getmodule(caller_frame[0])
        logger = logging.getLogger(module.__name__ if module else __name__)
    
    start_time = time.time()
    logger.info(f"Starting operation: {operation_name}")
    try:
        yield
    except Exception as e:
        logger.error(f"Operation failed: {operation_name}", exc_info=True)
        raise
    finally:
        duration = time.time() - start_time
        logger.info(f"Completed operation: {operation_name} in {duration:.4f} seconds")
