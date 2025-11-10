import sys
import os
from loguru import logger

logger.remove()
# Get log level from environment variable, default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO")
# Add a new handler to sys.stderr that only shows INFO level messages and higher
logger.add(sys.stderr, level=log_level)

