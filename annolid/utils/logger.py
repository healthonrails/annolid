import datetime
import logging
import os
import sys

from pathlib import Path
from typing import Union

__appname__ = "annolid"

try:
    import termcolor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    termcolor = None


def resolve_path(path: Union[Path, str]) -> Path:
    return Path(path).expanduser().resolve()


# Get the current user's home directory
home_dir = Path.home()

# Define the directory name for logs
logs_dir_name = f"{__appname__}_logs"

# Create the path for the logs directory
logs_dir_path = home_dir / logs_dir_name

# Resolve the path to the logs directory
resolved_logs_dir_path = resolve_path(logs_dir_path)

# Create the logs directory if it doesn't exist
if not resolved_logs_dir_path.exists():
    resolved_logs_dir_path.mkdir(parents=True)

# Get the current date
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Define the log file name with the current date
log_file_name = f"{__appname__}_{current_date}.log"

# Create the path for the log file
log_file_path = resolved_logs_dir_path / log_file_name

# Resolve the path to the log file
resolved_log_file_path = resolve_path(log_file_path)

if os.name == "nt":  # Windows
    import colorama

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
logger.setLevel(logging.INFO)

# Configure logging to write to stderr
stream_handler = logging.StreamHandler(sys.stderr)
handler_format = ColoredFormatter(
    "%(asctime)s [%(levelname2)s] %(module2)s:%(funcName2)s:%(lineno2)s- %(message2)s"
)
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Configure logging to write to a log file
log_file = resolved_log_file_path  # Set the path to your log file
file_handler = logging.FileHandler(log_file)
file_format = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(module)s:%(funcName)s:%(lineno)d - %(message)s"
)
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)
