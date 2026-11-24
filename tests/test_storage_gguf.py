import json
import os
import sqlite3
import struct
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from conftest import gguf, string

from infera_studio.catalog.gguf import read_header, scan
from infera_studio.config.preferences import DEFAULT_STATS, Preferences, xdg
from infera_studio.config.settings import InferenceSettings
from infera_studio.persistence.migrations import MIGRATIONS
from infera_studio.persistence.sqlite import Store


def test_metadata_and_recursive_deduplication(tmp_path):
    original = gguf(tmp_path / "nested" / "weights.GGUF", endian=">")
    (tmp_path / "link.gguf").symlink_to(original)
    gguf(tmp_path / "vocabulary.gguf", weights=False)
    gguf(tmp_path / "mmproj-test.gguf", {"general.architecture": "clip"})
    (tmp_path / "bad.gguf").write_text("oops")
    models, errors = scan((str(tmp_path), str(tmp_path / "nested"), str(tmp_path / "gone")))
    assert len(models) == 4
    assert len(errors) == 1
    model = next(item for item in models if item.available)
    assert (model.path, model.architecture, model.parameters, model.quant, model.context) == (
        str(original),
        "llama",
        4096,
        "Q4_K_M",
        16384,
    )
    assert read_header(original).tensors == 1
    assert model.size == original.stat().st_size
    assert any("Vocabulary-only" in item.issue for item in models)
    assert any("standalone" in item.issue for item in models)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_skip_tokenizer_strings(tmp_path: Path, endian: str) -> None:
    path = tmp_path / "tokenizer.gguf"
    prefix = b"GGUF" + struct.pack(f"{endian}IQQ", 3, 0, 2)
    prefix += string("tokenizer.ggml.tokens", endian) + struct.pack(f"{endian}IIQ", 9, 8, 3)
    tokens = b"".join(string(token, endian) for token in ("", "token", "λ" * 10_000))
    name = string("general.name", endian) + struct.pack(f"{endian}I", 8) + string("Example", endian)
    path.write_bytes(prefix + tokens + name)
    assert read_header(path).metadata == {"general.name": "Example"}

    path.write_bytes(prefix + tokens[:-1])
    with pytest.raises(ValueError, match="Truncated GGUF header"):
        read_header(path)


def test_shards_and_missing_model(tmp_path):
    one = gguf(tmp_path / "model-00001-of-00002.gguf", {"split.no": 0, "split.count": 2})
    two = gguf(tmp_path / "model-00002-of-00002.gguf", {"split.no": 1, "split.count": 2})
    models, _ = scan((str(tmp_path),))
    (model,) = models
    assert model.available and model.path == str(one)
    assert model.size == one.stat().st_size + two.stat().st_size
    assert model.parameters == 8192
    store = Store(tmp_path / "conversations")
    store.initialize()
    store.save_models(models)
    one.unlink()
    assert not model.available
    incomplete, _ = scan((str(tmp_path),))
    assert incomplete[0].path == str(one)
    assert "Incomplete" in incomplete[0].issue
    two.rename(tmp_path / "moved.gguf")
    store.save_models([])
    assert "Missing" in store.catalog()[str(one)].issue


def test_chat_search_fork_context_and_deletion(tmp_path, mock_model, settings):
    path = tmp_path / "history.db"
    store = Store(path)
    store.initialize()
    chat = store.create(mock_model.path, settings)
    reply = store.start_turn(chat, "Find 100% of _this_ message's text.")
    store.save_response(reply, "Good answer", "hidden", "complete", 20.0)
    second = store.start_turn(chat, "Next user turn")
    store.save_response(second, "partial", "", "stopped")
    original, msgs = Store(path).get(chat)
    assert original.settings == settings and original.model == mock_model.path
    assert msgs[1].reasoning == "hidden"
    assert Store(path).search("100% of _this_")[0].id == chat
    assert not Store(path).search("100_ ")
    assert not Store(path).search("' OR 1=1 --")
    store.rename(chat, "Renamed")
    cut = msgs[2].id
    store.set_context(chat, cut, "older turns")
    fork = store.fork(chat, second)
    branch, branch_messages = store.get(fork)
    assert len(branch_messages) == 3
    assert branch_messages[1].generation_per_second == 20.0
    assert branch_messages[-1].id == branch.active_from
    assert branch.summary == "older turns"
    assert store.get(chat)[0].title == "Renamed"
    store.delete(chat)
    with pytest.raises(ValueError, match="deleted"):
        store.get(chat)
    assert os.path.isfile(mock_model.path)
    store.delete()
    assert store.search() == []
    with store.connection() as conn:
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


def test_migrate_and_recover_partial_response(tmp_path, settings):
    path = tmp_path / "v1.db"
    with sqlite3.connect(path) as conn:
        for sql in MIGRATIONS[0]:
            conn.execute(sql)
        conn.execute("PRAGMA user_version = 1")
        conn.execute(
            "INSERT INTO chats VALUES (1,'v1','original',?,'then','then')",
            (json.dumps(asdict(settings) | {"thinking": "Model default"}),),
        )
        conn.execute("INSERT INTO messages VALUES (1,1,'assistant','partial','then','','streaming')")
    store = Store(path)
    assert store.initialize() == 1
    chat, messages = store.get(1)
    assert chat.active_from == 0 and chat.summary == "" and chat.settings == settings
    assert chat.pinned_at is None
    assert messages[0].state == "interrupted" and messages[0].content == "partial"
    assert messages[0].generation_per_second is None
    assert messages[0].performance is None
    assert store.initialize() == 0


def test_pin_limit_persists_and_frees_slots(tmp_path: Path, settings: InferenceSettings) -> None:
    path = tmp_path / "pins.db"
    store = Store(path)
    store.initialize()
    ids = [store.create("", settings, f"Chat {index}") for index in range(12)]
    first_pin = store.set_pinned(ids[0], True)
    assert first_pin is not None
    for chat_id in ids[1:10]:
        store.set_pinned(chat_id, True)

    reopened = Store(path)
    assert reopened.get(ids[0])[0].pinned_at == first_pin
    assert reopened.set_pinned(ids[0], True) == first_pin
    assert reopened.search()[0].id == ids[-1]
    assert sum(chat.pinned_at is not None for chat in reopened.search()) == 10
    with pytest.raises(ValueError, match="up to 10"):
        reopened.set_pinned(ids[10], True)

    assert reopened.set_pinned(ids[0], False) is None
    assert reopened.get(ids[0])[0].pinned_at is None
    assert reopened.set_pinned(ids[10], True) is not None
    reopened.delete(ids[1])
    assert reopened.set_pinned(ids[11], True) is not None


def test_visible_stats_preferences(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"setup_complete":true}')
    assert Preferences.load(path).visible_stats == DEFAULT_STATS
    for selection in (("cpu_load", "ram"), ()):
        replace(Preferences.load(path), visible_stats=selection).save(path)
        assert Preferences.load(path).visible_stats == selection
    with pytest.raises(ValueError, match="toolbar statistic"):
        Preferences.from_dict({"visible_stats": ["unknown"]})


def test_xdg_and_atomic_settings(tmp_path, monkeypatch, mock_prefs):
    path = tmp_path / "config" / "settings.json"
    prefs = replace(
        mock_prefs, model_dirs=("/tmp/with spaces",), backend="HIP", theme="Light", setup_complete=True
    )
    prefs.save(path)
    assert Preferences.load(path) == prefs
    with monkeypatch.context() as override:

        def failed(*_):
            raise OSError("failure before replacement")

        override.setattr(os, "replace", failed)
        with pytest.raises(OSError):
            replace(prefs, theme="Dark").save(path)
    assert Preferences.load(path) == prefs
    assert list(path.parent.iterdir()) == [path]
    assert path.stat().st_mode & 0o777 == 0o600
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert xdg("XDG_DATA_HOME", ".local/share") == tmp_path / "infera-studio"
    legacy = tmp_path / "cortex-local"
    legacy.mkdir()
    assert xdg("XDG_DATA_HOME", ".local/share") == legacy
    current = tmp_path / "infera-studio"
    current.mkdir()
    assert xdg("XDG_DATA_HOME", ".local/share") == current
