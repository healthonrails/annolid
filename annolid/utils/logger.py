import datetime
import logging
import os
import sys

from pathlib import Path
from typing import Union

from annolid.utils.log_paths import resolve_annolid_logs_root

__appname__ = "annolid"

try:
    import termcolor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    termcolor = None


def resolve_path(path: Union[Path, str]) -> Path:
    return Path(path).expanduser().resolve()


def _bool_from_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_log_file_path() -> Path:
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file_name = f"{__appname__}_{current_date}.log"
    return resolve_path(resolve_annolid_logs_root() / "app" / log_file_name)


def _init_windows_colorama() -> None:
    if os.name != "nt":
        return
    try:
        import colorama
    except ModuleNotFoundError:  # pragma: no cover
        return
    colorama.init()


COLORS = {
    "WARNING": "yellow",
    "INFO": "white",
    "DEBUG": "blue",
    "CRITICAL": "red",
    "ERROR": "red",
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, use_color=True):
        logging.Formatter.__init__(self, fmt)
        self.use_color = bool(use_color and termcolor is not None)

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:

            def colored(text):
                assert termcolor is not None
                return termcolor.colored(
                    text,
                    color=COLORS[levelname],
                    attrs=["bold"],
                )

            record.levelname2 = colored("{:<7}".format(record.levelname))
            record.message2 = colored(record.getMessage())

            asctime2 = datetime.datetime.fromtimestamp(record.created)
            record.asctime2 = colored(str(asctime2))

            record.module2 = termcolor.colored(record.module, color="cyan")
            record.funcName2 = termcolor.colored(record.funcName, color="cyan")
            record.lineno2 = termcolor.colored(record.lineno, color="cyan")
        else:
            record.levelname2 = "{:<7}".format(record.levelname)
            record.message2 = record.getMessage()
            record.asctime2 = str(datetime.datetime.fromtimestamp(record.created))
            record.module2 = record.module
            record.funcName2 = record.funcName
            record.lineno2 = record.lineno
        return logging.Formatter.format(self, record)


logger = logging.getLogger(__appname__)
logger.propagate = False
_is_configured = False


def configure_logging(
    *,
    level: int = logging.INFO,
    enable_console: bool = True,
    enable_file_logging: Union[bool, None] = None,
    log_file: Union[str, Path, None] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure Annolid logging once, from runtime entrypoints only."""
    global _is_configured

    if _is_configured and not force:
        return logger

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        _is_configured = False

    logger.setLevel(level)
    _init_windows_colorama()

    if enable_console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(
            ColoredFormatter(
                "%(asctime)s [%(levelname2)s] %(module2)s:%(funcName2)s:%(lineno2)s- %(message2)s"
            )
        )
        logger.addHandler(stream_handler)

    if enable_file_logging is None:
        enable_file_logging = _bool_from_env(os.getenv("ANNOLID_LOG_TO_FILE", "0"))

    if enable_file_logging:
        try:
            log_path = resolve_path(log_file) if log_file else _default_log_file_path()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(module)s:%(funcName)s:%(lineno)d - %(message)s"
                )
            )
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("File logging disabled: unable to open log file (%s)", exc)

    _is_configured = True
    return logger
