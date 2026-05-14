"""
Logging configuration.
"""
import logging

def setup_logging():
    # TODO 1: Set up basic logging with level INFO using logging.basicConfig()
    logging.basicConfig(level=logging.INFO)
    # TODO 2: Create a named logger using logging.getLogger() and return it
    return logging.getLogger(__name__)
