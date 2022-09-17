import logging
import sys
from types import FrameType
from typing import Optional, Union

from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level: Union[str, int] = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame: Optional[FrameType] = sys._getframe(6)
        depth = 6
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logger() -> None:
    flair_logger = logging.getLogger("flair")
    logging.basicConfig(handlers=[InterceptHandler()], force=True)
    for handler in flair_logger.handlers:
        logging.getLogger("flair").removeHandler(handler)
    logging.getLogger("flair").addHandler(InterceptHandler())
