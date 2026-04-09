from colorlog import ColoredFormatter
import inspect
import logging as python_logging
import numpy as np
from typing import Any

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LEVEL_MAP = {
    10: 'DEBUG',
    20: 'INFO',
    30: 'WARNING',
    40: 'ERROR',
    50: 'CRITICAL',
}
LOG_FORMAT_COLOR = "%(log_color)s%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
LOG_FORMAT_PLAIN = "%(asctime)s | %(levelname)-8s | %(message)s"

class Logger:
    def __init__(self) -> None:
        self._logger: python_logging.Logger | None = None

    def configure(self, level: str) -> None:
        # Create logger and set level.
        self._logger = python_logging.getLogger('AugMed')
        numeric_level = getattr(python_logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Logging level '{level}' not valid.")
        self._logger.setLevel(numeric_level)

        # Create console handler and set level.
        ch = python_logging.StreamHandler()
        ch.setLevel(numeric_level)

        # Add formatter to console handler.
        formatter = ColoredFormatter(LOG_FORMAT_COLOR, DATE_FORMAT)
        ch.setFormatter(formatter)

        # Remove old handlers.
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)
        
        # Add console handler to logger.
        self._logger.addHandler(ch)

    def critical(self, *args, **kwargs):
        return self._logger.critical(*args, **kwargs)

    def debug(self, *args, **kwargs):
        return self._logger.debug(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self._logger.error(*args, **kwargs)

    @staticmethod
    def _format_numpy(val: Any) -> str:
        if isinstance(val, np.ndarray):
            return f"np.ndarray(shape={val.shape}, dtype={val.dtype})"
        return repr(val)

    def info(self, *args, **kwargs):
        return self._logger.info(*args, **kwargs)

    @property
    def level(self) -> str:
        return LEVEL_MAP[self._logger.level]

    def log_method(
        self,
        message: str | None = None,
        ) -> None:
        frame = inspect.currentframe().f_back
        func_name = frame.f_code.co_name
        arg_info = inspect.getargvalues(frame)
        parts = []
        for name in arg_info.args:
            parts.append(f"{name}={self._format_numpy(arg_info.locals[name])}")
        if arg_info.varargs and arg_info.locals.get(arg_info.varargs):
            for val in arg_info.locals[arg_info.varargs]:
                parts.append(self._format_numpy(val))
        if arg_info.keywords and arg_info.locals.get(arg_info.keywords):
            for k, v in arg_info.locals[arg_info.keywords].items():
                parts.append(f"{k}={self._format_numpy(v)}")
        fn_str = f"{func_name}({', '.join(parts)})"
        if message:
            self.info(f"{fn_str}: {message}")
        else:
            self.info(fn_str)

    def warn(self, *args, **kwargs):
        return self._logger.warning(*args, **kwargs)

logger = Logger()

# Default config.
logger.configure('info')
