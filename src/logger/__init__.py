import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Log directory at project root
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    )

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = configure_logger()
