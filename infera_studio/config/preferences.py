"""XDG paths and durable application preferences."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .settings import BACKENDS

STAT_LABELS = {
    "gpu_load": "GPU load",
    "gpu_vram": "VRAM usage",
    "gpu_temp": "GPU temperature",
    "cpu_load": "CPU load",
    "cpu_temp": "CPU temperature",
    "ram": "RAM usage",
}
DEFAULT_STATS = tuple(STAT_LABELS)


def xdg(variable: str, default: str) -> Path:
    configured = Path(os.getenv(variable, "")).expanduser()
    root = configured if configured.is_absolute() else Path.home() / default
    current = root / "infera-studio"
    legacy = root / "cortex-local"
    return legacy if not current.exists() and legacy.is_dir() else current


def default_model_dirs() -> tuple[str, ...]:
    return tuple(
        str(Path.home() / "AI" / item)
        for item in (
            "models",
            "lama-cpp/models",
            "lama-cpp/llama.cpp/models",
        )
    )


@dataclass(frozen=True)
class Preferences:
    model_dirs: tuple[str, ...] = ()
    binaries: dict[str, str] | None = None
    backend: str = "Vulkan"
    theme: str = "Dark"
    show_reasoning: bool = False
    setup_complete: bool = False
    visible_stats: tuple[str, ...] = DEFAULT_STATS
    tavily_api_key: str = field(default="", repr=False)

    @classmethod
    def load(cls, path: Path) -> "Preferences":
        if not path.exists():
            return cls(model_dirs=default_model_dirs())
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Preferences":
        if not isinstance(data, dict):
            raise ValueError("Settings must be a JSON object.")
        dirs = data.get("model_dirs", list(default_model_dirs()))
        binaries = data.get("binaries") or {}
        if not isinstance(dirs, list) or any(not isinstance(p, str) for p in dirs):
            raise ValueError("Model directories must be a list of paths.")
        if not isinstance(binaries, dict) or any(
            name not in BACKENDS or not isinstance(path, str) for name, path in binaries.items()
        ):
            raise ValueError("Inference executables must be configured by backend.")
        backend, theme = data.get("backend", "Vulkan"), data.get("theme", "Dark")
        if backend not in BACKENDS or theme not in ("Dark", "Light"):
            raise ValueError("Settings contain an unknown backend or theme.")
        visible_stats = data.get("visible_stats", list(DEFAULT_STATS))
        if not isinstance(visible_stats, list) or any(
            not isinstance(stat, str) or stat not in STAT_LABELS for stat in visible_stats
        ):
            raise ValueError("Settings contain an unknown toolbar statistic.")
        key = data.get("tavily_api_key", "")
        if not isinstance(key, str):
            raise ValueError("Enter a valid Tavily API key.")
        key = key.strip()
        if len(key) > 512 or any(not "!" <= char <= "~" for char in key):
            raise ValueError("Enter a valid Tavily API key.")
        return cls(
            tuple(dirs),
            binaries,
            backend,
            theme,
            data.get("show_reasoning") is True,
            data.get("setup_complete") is True,
            tuple(stat for stat in DEFAULT_STATS if stat in visible_stats),
            key,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=".settings-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(asdict(self), stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(name).unlink(missing_ok=True)
