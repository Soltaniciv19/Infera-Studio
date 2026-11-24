import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from conftest import gguf

from infera_studio.application import hugging_face
from infera_studio.catalog.gguf import scan
from infera_studio.desktop.electron.operations import Bridge
from infera_studio.inference.client import Cancellation, Cancelled
from infera_studio.persistence.sqlite import Store

REPO = "publisher/example-GGUF"
REVISION = "a" * 40
FIRST = "Q4_K_M/example-00001-of-00002.gguf"
SECOND = "Q4_K_M/example-00002-of-00002.gguf"


def test_lists_only_complete_safe_variants() -> None:
    siblings = [
        {"rfilename": name, "size": 100}
        for name in (
            "example-IQ3_XS.gguf", SECOND, FIRST, "mmproj-F16.gguf",
            "example-Q8_0-00001-of-00002.gguf", "../escape.gguf", "/absolute.gguf",
            "nested\\escape.gguf", "nested/./escape.gguf", "readme.md",
        )
    ]
    options = hugging_face.collect_variants(siblings)
    assert [(option.path, option.quant, option.size) for option in options] == [
        ("example-IQ3_XS.gguf", "IQ3_XS", 100),
        (FIRST, "Q4_K_M", 200),
        ("mmproj-F16.gguf", "Vision projector", 100),
    ]
    assert [file.path for file in options[1].files] == [FIRST, SECOND]


@pytest.fixture
def hub_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, bytes], list[str]]:
    folder = tmp_path / "models"
    folder.mkdir()
    files = {
        name: gguf(tmp_path / Path(name).name, {"split.no": index, "split.count": 2}).read_bytes()
        for index, name in enumerate((FIRST, SECOND))
    }
    served: list[str] = []

    def metadata(url: str, cancel: Cancellation) -> dict[str, Any]:
        cancel.check()
        assert url == f"{hugging_face.HUB}/api/models/{REPO}/revision/{REVISION}?blobs=true"
        return {"sha": REVISION, "siblings": [
            {"rfilename": name, "size": len(data), "lfs": {"sha256": hashlib.sha256(data).hexdigest()}}
            for name, data in files.items()
        ]}

    @contextmanager
    def request(url: str, cancel: Cancellation) -> Iterator[BytesIO]:
        cancel.check()
        prefix = f"{hugging_face.HUB}/{REPO}/resolve/{REVISION}/"
        assert url.startswith(prefix)
        name = url.removeprefix(prefix)
        served.append(name)
        yield BytesIO(files[name])

    monkeypatch.setattr(hugging_face, "get_json", metadata)
    monkeypatch.setattr(hugging_face, "request", request)
    return folder, files, served


def test_download_publishes_the_split_variant_together(
    hub_files: tuple[Path, dict[str, bytes], list[str]],
) -> None:
    folder, files, served = hub_files
    events: list[tuple[str, Any]] = []

    def emit(event: str, value: Any) -> None:
        assert scan((str(folder),)) == ([], [])
        events.append((event, value))

    saved = Path(hugging_face.download(REPO, REVISION, FIRST, str(folder), Cancellation(), emit))
    assert saved.parent == folder
    assert {file.name: file.read_bytes() for file in saved.iterdir()} == {
        Path(name).name: data for name, data in files.items()
    }
    assert served == [FIRST, SECOND]
    assert events[0][0] == "progress" and events[0][1]["part"] == 1
    models, errors = scan((str(folder),))
    assert not errors and len(models) == 1 and models[0].available and len(models[0].shards) == 2
    with pytest.raises(ValueError, match="already has a download folder"):
        hugging_face.download(REPO, REVISION, FIRST, str(folder), Cancellation(), emit)
    assert served == [FIRST, SECOND]


def test_cancel_removes_staging_and_retry_succeeds(
    hub_files: tuple[Path, dict[str, bytes], list[str]],
) -> None:
    folder, _, served = hub_files
    cancel = Cancellation()

    def stop_at_second_part(event: str, value: Any) -> None:
        if value["part"] == 2:
            cancel.cancel()

    with pytest.raises(Cancelled):
        hugging_face.download(REPO, REVISION, FIRST, str(folder), cancel, stop_at_second_part)
    assert served == [FIRST]
    assert list(folder.iterdir()) == []
    saved = hugging_face.download(REPO, REVISION, FIRST, str(folder), Cancellation(), lambda *_: None)
    assert Path(saved).is_dir()


def test_checksum_failure_leaves_no_model(hub_files: tuple[Path, dict[str, bytes], list[str]]) -> None:
    folder, files, _ = hub_files

    def corrupt_second_part(event: str, value: Any) -> None:
        if value["part"] == 2:
            files[SECOND] = bytes(len(files[SECOND]))

    with pytest.raises(ValueError, match="failed its checksum"):
        hugging_face.download(REPO, REVISION, FIRST, str(folder), Cancellation(), corrupt_second_part)
    assert list(folder.iterdir()) == []


def test_rejects_unconfigured_download_folder(tmp_path: Path) -> None:
    bridge = Bridge(Store(tmp_path / "chats.sqlite3"), tmp_path / "settings.json")
    with pytest.raises(ValueError, match="Choose a saved model folder"):
        bridge.download_model(REPO, REVISION, FIRST, str(tmp_path), Cancellation(), lambda *_: None)
