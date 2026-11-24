"""Launch the Electron desktop from a source checkout or the Debian package."""

import argparse
import os
from pathlib import Path

VERSION = "1.3.0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Infera Studio local LLM desktop")
    parser.add_argument("--version", action="version", version=f"Infera Studio {VERSION}")
    _, flags = parser.parse_known_args()
    root = Path(__file__).resolve().parents[2]
    if root.parent.name == "resources":
        electron = root.parents[1] / "electron"
        args = [str(electron), *flags]
    else:
        electron = root / "node_modules" / ".bin" / "electron"
        args = [str(electron), str(root), *flags]
    if not electron.is_file():
        parser.exit(1, "Electron dependencies are missing. Run npm ci from the Infera Studio source directory.\n")
    os.execv(electron, args)
