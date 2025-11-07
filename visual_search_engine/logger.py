import sys
from loguru import logger

logger.remove()
# Add a new handler to sys.stderr that only shows INFO level messages and higher
logger.add(sys.stderr, level="INFO")
