import os
from pathlib import Path

import pytest

from infera_studio.catalog.gguf import scan
from infera_studio.config.preferences import Preferences
from infera_studio.config.settings import InferenceSettings
from infera_studio.inference.client import Cancellation, Cancelled, Client, HttpError
from infera_studio.inference.prompts import chat_payload, request_messages
from infera_studio.inference.server import ServerManager


@pytest.mark.integration
@pytest.mark.parametrize("backend", ("Vulkan", "HIP", "CPU"))
def test_real_local_llama_streams(backend):
    model_file = os.environ.get("INFERA_TEST_MODEL")
    executable = os.environ.get(f"INFERA_TEST_{backend.upper()}")
    if not model_file or not executable:
        pytest.skip(f"Set INFERA_TEST_MODEL and INFERA_TEST_{backend.upper()} to run this integration.")
    path = Path(model_file).expanduser().resolve()
    model = next((item for item in scan((str(path.parent),))[0] if item.path == str(path)), None)
    assert model is not None and model.available
    manager = ServerManager()
    ctx = int(os.getenv("INFERA_TEST_CONTEXT", "4096"))
    layers = int(os.getenv("INFERA_TEST_GPU_LAYERS", "-1"))
    settings = InferenceSettings(backend=backend, context=ctx, output=48, gpu_layers=layers, seed=42)
    try:
        info = manager.load(
            model, settings, Preferences(binaries={backend: executable}), Cancellation(), lambda *_: None
        )
        assert info.context > 0 and (ctx == 0 or info.context <= ctx)
        wire = request_messages([], settings, draft="Answer in one sentence: what is the capital of France?")
        count = info.client.count(wire, settings, info.context, info.thinking_supported, Cancellation())
        assert count.remaining >= 0 and count.system == 0
        received = []
        with pytest.raises(HttpError, match="401"):
            Client(info.client.port, "").json("/tokenize", {"content": "authentication test"})
        metrics = info.client.stream(
            chat_payload(wire, settings, info.thinking_supported, info.samplers),
            Cancellation(),
            lambda text, _: received.append(text),
        )
        assert "".join(received).strip()
        assert metrics.generated_tokens is None or metrics.generated_tokens > 0
        assert count.estimated or metrics.prompt_tokens in (None, count.prompt)
        cancellation = Cancellation()
        with pytest.raises(Cancelled):
            info.client.stream(
                chat_payload(wire, settings, info.thinking_supported, info.samplers),
                cancellation,
                lambda *_: cancellation.cancel(),
            )
        assert info.client.json("/health")["status"] == "ok"
    finally:
        manager.stop()
    assert manager.child is None
