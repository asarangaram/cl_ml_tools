import sys
from loguru import logger
from .config import Config

logger.remove()
# Add a new handler to sys.stderr with the level from the config
logger.add(sys.stderr, level=Config.LOG_LEVEL)

