import io
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import TINY_PNG, gguf

from infera_studio.application.runtime import BudgetExceeded, Runtime
from infera_studio.catalog.gguf import Model, scan
from infera_studio.config.preferences import Preferences
from infera_studio.config.settings import InferenceSettings
from infera_studio.conversations.metrics import Metrics
from infera_studio.conversations.records import Message
from infera_studio.desktop.electron.attachments import parse_attachments
from infera_studio.desktop.electron.operations import Bridge
from infera_studio.host.hardware import (
    GIB,
    GPU,
    GPUStats,
    SystemMonitor,
    amd_gpus,
    cpu_temperature,
    gpu_stats,
    nvidia_gpu_stats,
)
from infera_studio.inference.client import Cancellation, Cancelled, Client, HttpError, InferenceError, sse
from infera_studio.inference.memory import estimate, recommend
from infera_studio.inference.prompts import chat_payload, request_messages
from infera_studio.inference.server import ServerManager, command, detect, inspect_binary
from infera_studio.inference.telemetry import update_metrics
from infera_studio.persistence.sqlite import Store


def test_backend_command_is_private_and_independent(mock_model, mock_prefs, settings):
    for backend in ("Vulkan", "HIP", "CPU"):
        selected = replace(settings, backend=backend, flash="on", cache_k="q8_0")
        exe = detect(mock_prefs)[backend]
        caps = inspect_binary(exe, backend)
        args = command(exe, mock_model, selected, caps, recommend(mock_model, selected, None), 7351, 15)
        assert args[:3] == [str(exe), "-m", mock_model.path]
        assert args[args.index("--host") + 1] == "127.0.0.1"
        assert args[args.index("--parallel") + 1] == "1"
        assert (
            args[args.index("--device") + 1] == {"Vulkan": "Vulkan0", "HIP": "ROCm0", "CPU": "none"}[backend]
        )
        assert args[args.index("--api-key-file") + 1] == "/proc/self/fd/15"
        assert args[args.index("--cache-type-k") + 1] == "q8_0"
        assert args[args.index("--flash-attn") + 1] == "on"
        assert "--no-context-shift" in args
        assert "--metrics" in args
        assert args[args.index("--log-verbosity") + 1] == "4"
    with pytest.raises(ValueError):
        replace(settings, output=0)


def test_startup_readiness_crash_and_recovery(server, mock_model, mock_prefs, settings):
    manager, info = server
    assert info.context == 4096 and manager.child.poll() is None
    assert info.client.kv_occupancy(Cancellation()) == 0.125
    with pytest.raises(HttpError, match="401"):
        Client(info.client.port, "wrong").json("/props")
    pid = info.client.json("/props")["pid"]
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        os.kill(pid, signal.SIGKILL)
        manager.child.wait(timeout=5)
        assert "unexpectedly" in manager.health()
        restored = manager.load(mock_model, settings, mock_prefs, Cancellation(), lambda *_: None)
        assert restored.client.json("/props")["pid"] != pid
        manager.stop()
        assert manager.child is None
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize(("support", "mode", "expected"), [
    ("supported", "Automatic", "draft-mtp"),
    ("unsupported", "Automatic", None),
    ("broken", "Automatic", None),
    ("supported", "Off", None),
])
def test_mtp_startup(
    mock_model: Model, mock_prefs: Preferences, settings: InferenceSettings, monkeypatch: pytest.MonkeyPatch,
    support: str, mode: str, expected: str | None,
) -> None:
    monkeypatch.setenv("FAKE_INFERA_MTP", support)
    model = replace(mock_model, metadata=mock_model.metadata | {"llama.nextn_predict_layers": 1})
    config = replace(settings, mtp=mode)
    manager = ServerManager(startup_timeout=6)
    try:
        info = manager.load(model, config, mock_prefs, Cancellation(), lambda *_: None)
        assert info.client.json("/props")["spec_type"] == expected
        assert ("enabled" in info.mtp) == bool(expected)
        if support == "broken":
            assert "MTP startup failed" in manager.diagnostics()
        assert manager.load(model, config, mock_prefs, Cancellation(), lambda *_: None) is info
    finally:
        manager.stop()


def test_mtp_uses_selected_companion(
    tmp_path: Path, mock_prefs: Preferences, settings: InferenceSettings,
) -> None:
    model_file = gguf(tmp_path / "target.gguf", {"general.architecture": "gemma4", "gemma4.embedding_length": 64})
    draft_file = gguf(tmp_path / "assistant head.gguf", {
        "general.architecture": "gemma4-assistant", "gemma4-assistant.embedding_length_out": 64,
    })
    models = {item.path: item for item in scan((str(tmp_path),))[0]}
    model, draft = models[str(model_file)], models[str(draft_file)]
    assert model.available and not draft.available and draft.is_mtp_head_for(model)
    manager = ServerManager(startup_timeout=6)
    try:
        config = replace(settings, mtp_draft=draft.path)
        info = manager.load(model, config, mock_prefs, Cancellation(), lambda *_: None, mtp_draft=draft)
        assert info.client.json("/props")["spec_draft_model"] == draft.path
        draft_file.unlink()
        restored = manager.load(model, config, mock_prefs, Cancellation(), lambda *_: None, mtp_draft=draft)
        assert restored.client.json("/props")["spec_type"] is None
        assert "unavailable" in restored.mtp
    finally:
        manager.stop()


def test_startup_timeout_does_not_leave_child(mock_model, mock_prefs, settings, monkeypatch):
    monkeypatch.setenv("FAKE_INFERA_READY", "30")
    manager = ServerManager(startup_timeout=0.55)
    with pytest.raises(InferenceError, match="Timed out"):
        manager.load(mock_model, settings, mock_prefs, Cancellation(), lambda *_: None)
    assert manager.child is None


def test_disk_missing_model_stops_server(server, mock_model, mock_prefs, settings):
    manager, _ = server
    Path(mock_model.path).unlink()
    with pytest.raises(InferenceError, match="missing"):
        manager.load(mock_model, settings, mock_prefs, Cancellation(), lambda *_: None)
    assert manager.child is None


def test_guardian_reaps_on_owner_crash(tmp_path, mock_model, mock_prefs):
    output = tmp_path / "pid"
    code = """
import os
import time
from pathlib import Path
from infera_studio.config.preferences import Preferences
from infera_studio.config.settings import InferenceSettings
from infera_studio.catalog.gguf import scan
from infera_studio.inference.client import Cancellation
from infera_studio.inference.server import ServerManager
model = scan((os.environ["TEST_MODEL_DIR"],))[0][0]
manager = ServerManager()
info = manager.load(model, InferenceSettings(backend="CPU"),
                    Preferences(binaries={"CPU": os.environ["TEST_BINARY"]}), Cancellation(), lambda *_: None)
Path(os.environ["TEST_PID"]).write_text(str(info.client.json("/props")["pid"]))
time.sleep(20)
"""
    env = os.environ | {
        "TEST_MODEL_DIR": str(Path(mock_model.path).parent),
        "TEST_BINARY": mock_prefs.binaries["CPU"],
        "TEST_PID": str(output),
    }
    app = subprocess.Popen([sys.executable, "-c", code], env=env)
    try:
        deadline = time.monotonic() + 6
        while not output.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        pid = int(output.read_text())
        app.kill()
        deadline = time.monotonic() + 6
        while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not Path(f"/proc/{pid}").exists()
    finally:
        if app.poll() is None:
            app.kill()
        app.wait(timeout=5)


def test_minimal_network_request_stream_and_metrics(server, settings):
    _, info = server
    wire = request_messages([], settings, draft="Hello")
    changes = []
    result = info.client.stream(
        chat_payload(wire, settings),
        Cancellation(),
        lambda text, reasoning: changes.append((text, reasoning)),
    )
    body = info.client.json("/last")
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert set(body) == {"messages", "model", "stream", "max_tokens", "stream_options"}
    assert changes == [("Hello", "private"), (" world", "")]
    assert result.generated_tokens == 2 and result.generation_per_second == 20
    assert result.prompt_tokens == 4 and result.cached_prompt_tokens == 2
    assert result.prompt_seconds == 0.025 and result.generation_seconds == 0.1
    assert result.first_token_seconds is not None
    with pytest.raises(InferenceError, match="early"):
        list(sse(io.BytesIO(b'data: {"choices": []}\n\n')))


def test_partial_metrics_preserve_known_values_and_ignore_invalid_fields() -> None:
    metrics = Metrics()
    update_metrics(metrics, {"prompt_progress": {"total": 100, "processed": 30, "cache": 10, "time_ms": 10}})
    assert metrics.prompt_tokens == 100 and metrics.cached_prompt_tokens == 10
    assert metrics.prompt_progress_tokens == 30 and metrics.prompt_per_second == 2000
    update_metrics(metrics, {
        "timings": {"prompt_n": 90, "predicted_n": 5, "predicted_per_second": 25},
        "usage": None,
    })
    update_metrics(metrics, {
        "timings": {"predicted_per_second": float("nan")},
        "usage": {"completion_tokens": True, "prompt_tokens_details": {"cached_tokens": -1}},
    })
    assert metrics.prompt_tokens == 100 and metrics.processed_prompt_tokens == 90
    assert metrics.cached_prompt_tokens == 10
    assert metrics.generated_tokens == 5 and metrics.generation_per_second == 25


def test_response_cancellation_interrupts_socket(server, settings):
    _, info = server
    cancel = Cancellation()
    before = time.monotonic()
    with pytest.raises(Cancelled):
        info.client.stream(
            chat_payload(request_messages([], settings, draft="slow"), settings),
            cancel,
            lambda *_: cancel.cancel(),
        )
    assert time.monotonic() - before < 2
    assert info.client.json("/health")["status"] == "ok"


def test_context_budget_matches_templated_request(server, settings):
    _, info = server
    config = replace(settings, system="Use few words", thinking="Off")
    turns = [
        Message(1, "user", "old turn", "", "", "complete"),
        Message(2, "assistant", "old answer", "", "", "complete"),
        Message(3, "user", "new turn", "", "", "complete"),
    ]
    wire = request_messages(turns, config, active_from=3, draft="follow up")
    budget = info.client.count(wire, config, info.context, True, Cancellation())
    assert wire == [
        {"role": "system", "content": "Use few words"},
        {"role": "user", "content": "new turn\n\nfollow up"},
    ]
    assert budget.system == 3 and budget.conversation == 5 and not budget.estimated
    assert budget.remaining == 4096 - 8 - config.output
    completed = request_messages(turns[:2], settings)
    assert info.client.count(completed, settings, info.context, True, Cancellation()).prompt == 4


def test_over_budget_never_writes_or_truncates(mock_model, mock_prefs, settings, tmp_path):
    store = Store(tmp_path / "history.db")
    store.initialize()
    settings = replace(settings, context=32, output=32)
    chat, messages = store.get(store.create(mock_model.path, settings))
    runtime = Runtime(store)
    try:
        with pytest.raises(BudgetExceeded):
            runtime.generate(
                mock_model,
                mock_prefs,
                chat,
                messages,
                "test prompt",
                Cancellation(),
                *(lambda *_: None for _ in range(5)),
                attachments=parse_attachments([{"name": "note.txt", "content": "attachment text"}]),
            )
        assert store.get(chat.id)[1] == []
    finally:
        runtime.server.stop()


def test_attachment_round_trip(
    tmp_path: Path, mock_model: Model, mock_prefs: Preferences, settings: InferenceSettings,
) -> None:
    store = Store(tmp_path / "attachment.db")
    store.initialize()
    chat, messages = store.get(store.create(mock_model.path, settings))
    files = parse_attachments([{"name": "note.md", "content": "<em>Keep this snapshot</em>"}])
    with pytest.raises(ValueError, match="UTF-8"):
        parse_attachments([{"name": "bad.txt", "content": "\x00"}])
    runtime = Runtime(store)
    try:
        budgets = []
        performance: list[Metrics] = []
        result = runtime.generate(
            mock_model, mock_prefs, chat, messages, "", Cancellation(),
            lambda *_: None, lambda *_: None, budgets.append, lambda *_: None, lambda *_: None,
            attachments=files,
            performance=performance.append,
        )
        info = runtime.server.info
        assert info is not None
        _, turns = store.get(chat.id)
        assert turns[0].content == "" and turns[0].attachments == files
        wire = info.client.json("/last")["messages"]
        assert wire == [{"role": "user", "content": 'Attached file "note.md":\n<em>Keep this snapshot</em>'}]
        assert budgets[0].prompt == len(wire[0]["content"].split()) + 1
        assert [event.stage for event in performance[:4]] == ["model", "token_check", "prompt", "prompt"]
        assert any(event.stage == "generation" for event in performance)
        assert turns[1].performance == result.metrics
        assert result.metrics.kv_allocated_mib == 64.0
        assert result.metrics.request_seconds >= result.metrics.seconds
        with_response = store.fork(chat.id, turns[1].id + 1)
        assert store.get(with_response)[1][1].performance == result.metrics
        _, branch = store.get(store.fork(chat.id, turns[1].id))
        assert request_messages(branch, settings) == wire
        assert request_messages(turns, settings, active_from=turns[1].id) == [
            {"role": "assistant", "content": "Hello world"}
        ]
    finally:
        runtime.server.stop()


def test_vision_counts_and_preserves_image_turns(
    tmp_path: Path, mock_model: Model, mock_prefs: Preferences, settings: InferenceSettings,
) -> None:
    folder = Path(mock_model.path).parent
    projector = gguf(folder / "mmproj vision.gguf", {"general.architecture": "clip", "clip.has_vision_encoder": True})
    audio = gguf(folder / "mmproj audio.gguf", {"general.architecture": "clip", "clip.has_vision_encoder": False})
    store = Store(tmp_path / "vision.db")
    config = tmp_path / "settings.json"
    replace(mock_prefs, model_dirs=(str(folder),)).save(config)
    bridge = Bridge(store, config)
    bridge.initialize()
    chat_id = store.create(mock_model.path, settings)
    image = {"name": "pixel.png", "content": TINY_PNG, "media_type": "image/png"}
    def emit(_event: str, _value: object) -> None:
        pass

    with pytest.raises(ValueError, match="projector"):
        bridge.configure(chat_id, settings={"backend": "CPU", "vision_projector": str(audio)})
    with pytest.raises(ValueError, match="valid PNG or JPEG"):
        parse_attachments([image | {"media_type": "image/jpeg"}])
    try:
        with pytest.raises(InferenceError, match="matching vision projector"):
            bridge.generate(chat_id, "", Cancellation(), emit, attachments=[image])
        assert store.get(chat_id)[1] == []
        bridge.configure(chat_id, settings={"backend": "CPU", "vision_projector": str(projector)})
        bridge.open_chat(chat_id, Cancellation(), emit)
        chat, _ = store.get(chat_id)
        info = bridge.current_info(chat)
        assert info and info.vision_supported and info.client.json("/props")["projector"] == str(projector)
        assert bridge.budget(chat_id, "", Cancellation(), attachments=[image]).prompt == 131
        bridge.generate(chat_id, "", Cancellation(), emit, attachments=[image])
        messages = info.client.json("/last")["messages"]
        assert messages == [{"role": "user", "content": [
            {"type": "text", "text": 'Attached image "pixel.png"'},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_PNG}"}},
        ]}]
        _, turns = store.get(chat_id)
        _, branch = store.get(store.fork(chat_id, turns[1].id))
        assert branch[0].attachments == turns[0].attachments == parse_attachments([image])
        assert request_messages(branch, chat.settings, vision=True) == messages
        assert request_messages(branch, chat.settings) == [{"role": "user", "content": 'Attached image "pixel.png"'}]
    finally:
        bridge.runtime.server.stop()


def test_gpu_stats_keep_available_metrics_and_pick_the_largest_amd_gpu(tmp_path: Path) -> None:
    for index, vendor, vram in ((0, "0x10de", 32), (1, "0x1002", 1), (2, "0x1002", 16)):
        pci = tmp_path / f"0000:0{index}:00.0"
        sensor = pci / "hwmon" / "hwmon0"
        sensor.mkdir(parents=True)
        (pci / "vendor").write_text(vendor)
        (pci / "mem_info_vram_total").write_text(str(vram * GIB))
        (pci / "mem_info_vram_used").write_text(str(GIB))
        (pci / "gpu_busy_percent").write_text("0")
        (sensor / "temp1_input").write_text("42500")
        card = tmp_path / f"card{index}"
        card.mkdir()
        (card / "device").symlink_to(pci)

    assert gpu_stats(tmp_path) == GPUStats("0000:02:00.0", 0, GIB, 16 * GIB, 42.5)
    primary = tmp_path / "0000:02:00.0"
    (primary / "gpu_busy_percent").write_text("101")
    (primary / "hwmon" / "hwmon0" / "temp1_input").unlink()
    assert gpu_stats(tmp_path) == GPUStats(primary.name, None, GIB, 16 * GIB, None)
    assert gpu_stats(tmp_path / "missing") is None


def test_nvidia_telemetry_keeps_partial_device_readings(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, 0, "00000000:06:00.0, GeForce, 42, 2048, 16384, 59\n"
            "00000000:07:00.0, Other GPU, [N/A], 0, 8192, [N/A]\n"
            "invalid row\n",
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert list(nvidia_gpu_stats("/usr/bin/nvidia-smi")) == [
        GPUStats("00000000:06:00.0", 42, 2 * GIB, 16 * GIB, 59, "NVIDIA", "GeForce"),
        GPUStats("00000000:07:00.0", None, 0, 8 * GIB, None, "NVIDIA", "Other GPU"),
    ]


def test_system_stats_count_cpu_deltas_and_available_ram(tmp_path: Path) -> None:
    stat = tmp_path / "stat"
    stat.write_text("cpu 100 10 20 500 5 0 0 0 5 2\n")
    (tmp_path / "meminfo").write_text("MemTotal: 16777216 kB\nMemFree: 1 kB\nMemAvailable: 12582912 kB\n")
    monitor = SystemMonitor(proc=tmp_path, hwmon=tmp_path, drm=tmp_path)
    first = monitor.sample()
    assert first.cpu_load is None and first.cpu_temperature_c is None
    assert first.ram_used == 4 * GIB and first.ram_total == 16 * GIB
    stat.write_text("cpu 120 10 30 540 15 0 0 0 25 2\n")
    assert monitor.sample().cpu_load == 37.5
    assert monitor.sample().cpu_load is None
    stat.write_text("cpu 10 10 20 50 5 0 0 0 5 2\n")
    assert monitor.sample().cpu_load is None


def test_cpu_temperature_uses_package_sensors(tmp_path: Path) -> None:
    for index, name, values in (
        (0, "amdgpu", {"temp1_input": "92000"}),
        (1, "k10temp", {"temp1_input": "68000", "temp2_input": "55000", "temp3_input": "75000"}),
        (2, "coretemp", {"temp1_input": "60000", "temp2_input": "65000"}),
    ):
        root = tmp_path / f"hwmon{index}"
        root.mkdir()
        (root / "name").write_text(name)
        for key, value in values.items():
            (root / key).write_text(value)
    assert cpu_temperature(tmp_path) == 60
    (tmp_path / "hwmon2/name").unlink()
    assert cpu_temperature(tmp_path) == 55
    (tmp_path / "hwmon1/temp2_input").unlink()
    assert cpu_temperature(tmp_path) == 68


def test_hybrid_memory_estimation_and_amd_sysfs(tmp_path, mock_model, settings):
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    for name, value in (
        ("vendor", "0x1002"),
        ("device", "0x73bf"),
        ("mem_info_vram_total", str(16 * GIB)),
        ("mem_info_vram_used", str(2 * GIB)),
    ):
        (device / name).write_text(value)
    (observed,) = amd_gpus(tmp_path)
    assert observed.free == 14 * GIB and observed.pci_id == "0x73bf"
    model = replace(
        mock_model,
        architecture="qwen35",
        size=7 * GIB,
        context=262144,
        metadata={
            "qwen35.block_count": 32,
            "qwen35.full_attention_interval": 4,
            "qwen35.embedding_length": 4096,
            "qwen35.attention.head_count": 16,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.ssm.state_size": 128,
            "qwen35.ssm.time_step_rank": 32,
            "qwen35.ssm.group_count": 16,
            "qwen35.ssm.conv_kernel": 4,
            "qwen35.ssm.inner_size": 4096,
        },
    )
    config = replace(settings, backend="Vulkan", context=0)
    first = estimate(model, config, 4096, observed)
    second = estimate(model, config, 8192, observed)
    assert first.kv * 2 == pytest.approx(second.kv, abs=1)
    assert first.recurrent == second.recurrent and first.recurrent > 0
    assert estimate(model, config, 8192, observed, parallel=2).kv == pytest.approx(second.kv * 2, abs=1)
    constrained = GPU("card0", 16 * GIB, 13 * GIB)
    assert recommend(model, config, constrained).context == 4096
    assert recommend(model, config, observed).context > 4096
