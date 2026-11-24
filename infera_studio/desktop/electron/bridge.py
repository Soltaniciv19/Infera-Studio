"""Start the private Electron backend process."""

import fcntl
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from infera_studio.config.preferences import xdg
from infera_studio.persistence.sqlite import Store

from .operations import Bridge
from .protocol import serve
from .voice import VoiceService


def configure_logging() -> None:
    directory = xdg("XDG_STATE_HOME", ".local/state")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "infera-studio.log"
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
    handler = RotatingFileHandler(path, maxBytes=512 * 1024, backupCount=2, encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )


def main() -> None:
    configure_logging()
    data = xdg("XDG_DATA_HOME", ".local/share")
    data.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (data / "electron.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Infera Studio is already open for this data folder.") from None
        voice = VoiceService(Path(sys.argv[1]), Path(sys.argv[2]))
        serve(Bridge(Store(data / "conversations.sqlite3"), xdg("XDG_CONFIG_HOME", ".config") / "settings.json"), voice)


if __name__ == "__main__":
    main()
