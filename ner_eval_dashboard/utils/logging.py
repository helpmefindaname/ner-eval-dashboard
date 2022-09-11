import logging
import sys

import flair.file_utils
from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logger():
    flair_logger =  logging.getLogger("flair")
    logging.basicConfig(handlers=[InterceptHandler()], force=True)
    for handler in flair_logger.handlers:
        logging.getLogger("flair").removeHandler(handler)
    logging.getLogger("flair").addHandler(InterceptHandler())