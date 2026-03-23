import logging
from typing import Any, List, Union

try:
    from colorlog import ColoredFormatter
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT_COLOR = "%(log_color)s%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
LOG_FORMAT_PLAIN = "%(asctime)s | %(levelname)-8s | %(message)s"
LEVEL_MAP = {
    10: 'DEBUG',
    20: 'INFO',
    30: 'WARNING',
    40: 'ERROR',
    50: 'CRITICAL',
}

logger = logging.getLogger("augmed")

def config(level: str = "info") -> None:
    """Configure the augmed logger level and handler."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Logging level '{level}' not valid.")
    logger.setLevel(numeric_level)

    # Remove existing handlers.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler.
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)

    if _HAS_COLORLOG:
        formatter = ColoredFormatter(LOG_FORMAT_COLOR, DATE_FORMAT)
    else:
        formatter = logging.Formatter(LOG_FORMAT_PLAIN, DATE_FORMAT)
    ch.setFormatter(formatter)

    logger.addHandler(ch)


def level() -> str:
    return LEVEL_MAP[logger.level]

def arg_log(
    action: str,
    arg_names: Union[str, List[str]],
    arg_vals: Union[Any, List[Any]],
) -> None:
    message = action + ' with ' + ', '.join(
        f"{arg_name}={arg_val}" for arg_name, arg_val in zip(arg_names, arg_vals)
    ) + '.'
    logger.info(message)

# Default config.
config("info")
