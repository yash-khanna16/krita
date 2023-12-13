import logging
from config import LOG_LEVEL

# Initiating the logger
logger = logging.getLogger()


def init_logger(app):
    """
    This method is used to initiate the logger.
    """
    # Setting the log level
    log_level = getattr(logging, LOG_LEVEL)
    logger.setLevel(log_level)

    #  Stream Handler
    stream_handler = logging.StreamHandler()

    stream_formatter = logging.Formatter(
        '%(asctime)-15s %(levelname)-8s '
        '%(filename)s %(lineno)d '
        '%(message)s')

    # Setting up the formatter with handlers and
    # adding the handlers to the logger
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
