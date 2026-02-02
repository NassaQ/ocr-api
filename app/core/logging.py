import logging
import sys

logging.getLogger("ppocr").setLevel(logging.ERROR)

logger = logging.getLogger("ocr")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(levelprefix)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


class LevelFormatter(logging.Formatter):
    """Formatter that adds colored level prefix like uvicorn."""

    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, "")
        record.levelprefix = f"{color}{record.levelname}:{self.RESET}    "
        return super().format(record)


handler.setFormatter(LevelFormatter(fmt="%(levelprefix)s %(message)s"))
logger.addHandler(handler)
logger.propagate = False
