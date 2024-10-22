import os
import datetime
import logging


class LoggerConfig:
    log_dir = ".logs/"
    log_file = None
    is_logging_enabled = False

    @classmethod
    def setup(cls):
        if cls.is_logging_enabled:
            if not os.path.exists(cls.log_dir):
                os.makedirs(cls.log_dir)

            now = datetime.datetime.now()
            cls.log_file = os.path.join(cls.log_dir, f'run_gp_{now.strftime("%Y-%m-%dT%H-%M-%S")}.log')

            logging.basicConfig(
                filename=cls.log_file,
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] (%(filename)s): %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )


def enable_logging(enable: bool):
    """Включает или выключает логирование."""
    LoggerConfig.is_logging_enabled = enable
    if enable:
        LoggerConfig.setup()
        info("Logging has been enabled.")
    else:
        info("Logging has been disabled.")


def info(message: str):
    if LoggerConfig.is_logging_enabled:
        logging.getLogger(__name__).info(message, stacklevel=2)


def error(message: str):
    if LoggerConfig.is_logging_enabled:
        logging.getLogger(__name__).error(message, stacklevel=2)


def warning(message: str):
    if LoggerConfig.is_logging_enabled:
        logging.getLogger(__name__).warning(message, stacklevel=2)


def success(message: str):
    if LoggerConfig.is_logging_enabled:
        logging.getLogger(__name__).info(message, stacklevel=2)


def debug(message: str):
    if LoggerConfig.is_logging_enabled:
        logging.getLogger(__name__).debug(message, stacklevel=2)


def critical(message: str):
    if LoggerConfig.is_logging_enabled:
        logging.getLogger(__name__).critical(message, stacklevel=2)


def set_level(level: str):
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    logging.getLogger().setLevel(levels.get(level.upper(), logging.INFO))
    info(f"Log level set to {level.upper()}")
