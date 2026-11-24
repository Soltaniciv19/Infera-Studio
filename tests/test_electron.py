import base64
import json
import math
import os
import struct
import subprocess
import wave
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import TINY_PNG, gguf

from infera_studio.catalog.gguf import Model
from infera_studio.config.preferences import Preferences
from infera_studio.config.settings import InferenceSettings
from infera_studio.host.hardware import SystemMonitor, cpu_temperature
from infera_studio.persistence.sqlite import Store

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("INFERA_TEST_ELECTRON") != "1",
        reason="Set INFERA_TEST_ELECTRON=1 with Electron dependencies installed.",
    ),
]


def test_electron_dictates_into_the_draft(tmp_path: Path, mock_prefs: Preferences, mock_model: Model) -> None:
    config, data, state = (tmp_path / name for name in ("config", "data", "state"))
    replace(
        mock_prefs, model_dirs=(str(Path(mock_model.path).parent),), backend="CPU", setup_complete=True,
    ).save(config / "infera-studio/settings.json")
    binary = tmp_path / "whisper-cli"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import io, sys, time, wave\n"
        "assert sys.argv[3:7] == ['-f', '-', '--output-file', 'transcript']\n"
        "assert sys.argv[7:] == ['--language', 'auto', '--no-timestamps', '--no-gpu', '--suppress-nst']\n"
        "with wave.open(io.BytesIO(sys.stdin.buffer.read())) as audio:\n"
        "    assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)\n"
        "    frames = audio.getnframes()\n"
        "    assert frames >= 8000 and any(audio.readframes(-1))\n"
        "time.sleep(0.25)\n"
        "if frames <= 32000:\n"
        "    assert frames in (16000, 32000)\n"
        "print({16000: 'Hello from', 32000: 'Hello from voice'}.get(frames, 'Hello from voice input.'))\n",
        encoding="utf-8",
    )
    binary.chmod(0o700)
    weights = tmp_path / "ggml-base.bin"
    weights.touch()
    profile = state / "infera-studio/electron"
    profile.mkdir(parents=True)
    (profile / "voice-input.json").write_text(json.dumps({"executable": str(binary), "model": str(weights)}))
    recording = tmp_path / "speech-fixture.wav"
    with wave.open(str(recording), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        samples = [round(4096 * math.sin(2 * math.pi * 440 * index / 16000)) for index in range(32000)]
        audio.writeframes(struct.pack("<32000h", *samples))
    env = os.environ | {
        "XDG_CONFIG_HOME": str(config),
        "XDG_DATA_HOME": str(data),
        "XDG_STATE_HOME": str(state),
        "INFERA_TEST_VOICE_AUDIO": str(recording),
    }
    subprocess.run(
        ["node", "scripts/test-electron.cjs"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        timeout=45,
        check=True,
    )
    store = Store(data / "infera-studio/conversations.sqlite3")
    [chat] = store.search("Hello from voice input")
    assert store.get(chat.id)[1][0].content == "Existing draft. Hello from voice input."


def test_electron_chat_flow(tmp_path: Path, mock_prefs: Preferences, mock_model: Model) -> None:
    config = tmp_path / "config" / "infera-studio" / "settings.json"
    gguf(Path(mock_model.path), {"llama.nextn_predict_layers": 1})
    projector = gguf(Path(mock_model.path).parent / "mmproj.gguf", {
        "general.architecture": "clip", "clip.has_vision_encoder": True,
    })
    image = tmp_path / "pixel.png"
    image.write_bytes(base64.b64decode(TINY_PNG))
    attachment = tmp_path / "attachment.md"
    attachment.write_text("<b>Local attachment snapshot</b>\n", encoding="utf-8")
    replace(
        mock_prefs, model_dirs=(str(Path(mock_model.path).parent),), backend="CPU", setup_complete=True
    ).save(config)
    env = os.environ | {
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "INFERA_TEST_ATTACHMENT": str(attachment),
        "INFERA_TEST_IMAGE": str(image),
        "INFERA_TEST_PROJECTOR": str(projector),
        "INFERA_EXPECT_GPU_STATS": "1" if SystemMonitor().sample().gpu else "0",
        "INFERA_EXPECT_CPU_TEMP": "1" if cpu_temperature() is not None else "0",
    }
    result = subprocess.run(
        ["node", "scripts/test-electron.cjs"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        timeout=95,
        check=False,
    )
    assert result.returncode == 0
    store = Store(tmp_path / "data" / "infera-studio" / "conversations.sqlite3")
    chats = store.search("Hello through Electron")
    assert len(chats) == 2
    source = next(chat for chat in chats if not chat.title.endswith(" · branch"))
    branch = next(chat for chat in chats if chat.id != source.id)
    _, turns = store.get(source.id)
    assert turns[0].attachments[0].content == "<b>Local attachment snapshot</b>\n"
    assert turns[0].attachments[1].content == TINY_PNG
    assert store.get(branch.id)[1][0].attachments == turns[0].attachments
    assert turns[1].content == "Hello world"
    assert turns[1].generation_per_second == 20.0
    assert turns[1].performance is not None
    assert turns[1].performance.kv_allocated_mib == 64.0
    assert turns[1].performance.cached_prompt_tokens == 2
    assert turns[1].performance.generation_seconds == 0.1
    samples = turns[1].performance.samples
    assert len(samples) >= 3
    elapsed = [sample["elapsed_seconds"] for sample in samples]
    assert sorted(set(elapsed)) == elapsed
    assert samples[-1]["context_tokens"] == 6 and samples[-1]["generated_tokens"] == 2
    assert samples[0]["kv_allocated_mib"] == 64.0
    assert samples[-1]["prompt_tokens"] == 4
    assert samples[-1]["processed_prompt_tokens"] == 2 and samples[-1]["cached_prompt_tokens"] == 2
    assert turns[3].content == "Hello world" and turns[3].generation_per_second == 20.0
    assert turns[-1].state == "stopped"
    assert turns[-1].generation_per_second is None
    assert turns[-1].performance is not None and turns[-1].performance.stage == "stopped"
    assert branch.settings.context == 4096 and branch.settings.output == 512
    assert branch.settings.mtp == "Off"
    assert branch.settings.thinking == "On" and branch.settings.vision_projector == str(projector)
    assert Preferences.load(config).theme == "Light"
    assert Preferences.load(config).visible_stats == ("cpu_load", "ram")
    assert Preferences.load(config).tavily_api_key == "tvly-electron-fixture"


def test_electron_reopens_legacy_profile(tmp_path: Path) -> None:
    config, data, state = (tmp_path / name for name in ("config", "data", "state"))
    settings = config / "cortex-local/settings.json"
    Preferences(backend="CPU", setup_complete=True).save(settings)
    store = Store(data / "cortex-local/conversations.sqlite3")
    store.initialize()
    title = "Saved before the rename"
    chat_id = store.create("", InferenceSettings(backend="CPU"), title)
    project = tmp_path / "example-project"
    project.mkdir()
    store.set_workspace(chat_id, str(project))
    profile = state / "cortex-local/electron"
    profile.mkdir(parents=True)
    env = os.environ | {
        "XDG_CONFIG_HOME": str(config),
        "XDG_DATA_HOME": str(data),
        "XDG_STATE_HOME": str(state),
        "INFERA_TEST_SAVED_CHAT": title,
        "INFERA_TEST_SAVED_WORKSPACE": project.name,
        "INFERA_TEST_STARTUP_ONLY": "1",
    }
    result = subprocess.run(
        ["node", "scripts/test-electron.cjs"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        timeout=55,
        check=False,
    )
    assert result.returncode == 0
    assert not store.get(chat_id)[0].workspace
    assert (profile / "Preferences").is_file()
    assert (state / "cortex-local/infera-studio.log").is_file()
    assert not any((root / "infera-studio").exists() for root in (config, data, state))


def test_electron_streams_agent_activity(tmp_path: Path, mock_prefs: Preferences, mock_model: Model) -> None:
    config, data, state = (tmp_path / name for name in ("config", "data", "state"))
    replace(
        mock_prefs, model_dirs=(str(Path(mock_model.path).parent),), backend="CPU", setup_complete=True,
    ).save(config / "infera-studio/settings.json")
    store = Store(data / "infera-studio/conversations.sqlite3")
    store.initialize()
    chat_id = store.create(mock_model.path, InferenceSettings(backend="CPU"), "Agent fixture")
    saved_message = "Saved before closing Infera Studio"
    message_id = store.start_turn(chat_id, "Remember this conversation")
    store.save_response(message_id, saved_message, "", "complete")
    project = tmp_path / "project"
    project.mkdir()
    (project / "seed.txt").write_text("Read me\n")
    store.set_workspace(chat_id, str(project))
    env = os.environ | {
        "XDG_CONFIG_HOME": str(config),
        "XDG_DATA_HOME": str(data),
        "XDG_STATE_HOME": str(state),
        "INFERA_TEST_AGENT": "1",
        "INFERA_TEST_SAVED_CHAT": "Agent fixture",
        "INFERA_TEST_SAVED_MESSAGE": saved_message,
        "FAKE_INFERA_READY": "2.5",
        "FAKE_INFERA_SLOW_TOOL": "1",
    }
    result = subprocess.run(
        ["node", "scripts/test-electron.cjs"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        timeout=55,
        check=False,
    )
    assert result.returncode == 0
    assert (project / "src/result.txt").read_text() == "Project from the seed\n"
