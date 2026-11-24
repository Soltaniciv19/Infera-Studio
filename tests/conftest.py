import shutil
import struct
from pathlib import Path

import pytest

from infera_studio.catalog.gguf import scan
from infera_studio.config.preferences import Preferences
from infera_studio.config.settings import InferenceSettings
from infera_studio.inference.client import Cancellation
from infera_studio.inference.server import ServerManager

TINY_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


def string(value: str, endian: str = "<") -> bytes:
    binary = value.encode()
    return struct.pack(f"{endian}Q", len(binary)) + binary


def gguf(path: Path, metadata: dict | None = None, weights: bool = True, endian: str = "<") -> Path:
    data = {
        "general.architecture": "llama",
        "general.name": "Example GGUF",
        "general.file_type": 15,
        "llama.context_length": 16384,
        "llama.block_count": 4,
        "llama.embedding_length": 64,
        "llama.attention.head_count": 4,
        "llama.attention.head_count_kv": 2,
    }
    data.update(metadata or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(b"GGUF" + struct.pack(f"{endian}IQQ", 3, int(weights), len(data)))
        for key, value in data.items():
            file.write(string(key, endian))
            if isinstance(value, str):
                file.write(struct.pack(f"{endian}I", 8) + string(value, endian))
            elif isinstance(value, bool):
                file.write(struct.pack(f"{endian}I?", 7, value))
            else:
                file.write(struct.pack(f"{endian}II", 4, value))
        if weights:
            file.write(string("token_embd.weight", endian))
            file.write(struct.pack(f"{endian}IQQIQ", 2, 128, 32, 0, 0))
            file.write(b"\0" * 128)
    return path


@pytest.fixture
def mock_model(tmp_path):
    gguf(tmp_path / "models" / "example.gguf")
    return scan((str(tmp_path / "models"),))[0][0]


@pytest.fixture
def mock_prefs(tmp_path):
    server = tmp_path / "llama-server with spaces"
    shutil.copyfile(Path(__file__).with_name("fake_server.py"), server)
    server.chmod(0o755)
    return Preferences(binaries={"CPU": str(server), "Vulkan": str(server), "HIP": str(server)})


@pytest.fixture
def settings():
    return InferenceSettings(backend="CPU", context=4096, output=32)


@pytest.fixture
def server(mock_model, mock_prefs, settings):
    supervisor = ServerManager(startup_timeout=6)
    try:
        info = supervisor.load(mock_model, settings, mock_prefs, Cancellation(), lambda *_: None)
        yield supervisor, info
    finally:
        supervisor.stop()
