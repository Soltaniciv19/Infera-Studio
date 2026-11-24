"""JSON-lines transport and worker scheduling for Electron requests."""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from infera_studio.application.runtime import BudgetExceeded
from infera_studio.inference.client import Budget, Cancellation, Cancelled, InferenceError

from .operations import Bridge
from .voice import BATCH_MS, MAX_SECONDS, RATE, VoiceService, VoiceSession

LOG = logging.getLogger(__name__)


def encode(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Budget):
        return asdict(value) | {"remaining": value.remaining, "conversation": value.conversation}
    return asdict(value)


def serve(bridge: Bridge, voice: VoiceService) -> None:
    operations: dict[int, Cancellation] = {}
    active_voice: tuple[int, VoiceSession] | None = None
    lock = Lock()
    output_lock = Lock()

    def send(message: dict[str, Any]) -> None:
        with output_lock:
            print(json.dumps(message, default=encode), flush=True)

    def run(request_id: int, action: str, params: dict[str, Any], cancel: Cancellation) -> None:
        def emit(event: str, value: Any) -> None:
            send({"id": request_id, "event": event, "value": value})

        try:
            cancel.check()
            if action == "voice_status":
                files = voice.configuration()
                result = {
                    "source": files.source, "sample_rate": RATE, "batch_ms": BATCH_MS, "max_seconds": MAX_SECONDS,
                } if files else None
            else:
                result = bridge.dispatch(action, params, cancel, emit)
            send({"id": request_id, "ok": True, "value": result})
        except Cancelled:
            send({"id": request_id, "ok": False, "cancelled": True})
        except BudgetExceeded as exc:
            emit("budget", exc.budget)
            send({"id": request_id, "ok": False, "error": str(exc)})
        except Exception as exc:
            LOG.warning("Operation %s failed error_type=%s", action, type(exc).__name__)
            send({"id": request_id, "ok": False, "error": str(exc)})
        finally:
            with lock:
                operations.pop(request_id, None)

    def dictate(request_id: int, session: VoiceSession) -> None:
        nonlocal active_voice
        try:
            text = voice.transcribe(session, lambda value: send(
                {"id": request_id, "event": "transcript", "value": value},
            ))
            session.cancel.check()
            response = {"id": request_id, "ok": True, "value": text}
        except Cancelled:
            response = {"id": request_id, "ok": False, "cancelled": True}
        except Exception as exc:
            LOG.warning("Dictation failed error_type=%s", type(exc).__name__)
            response = {"id": request_id, "ok": False, "error": str(exc)}
        with lock:
            active_voice = None
        send(response)

    inference = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infera-studio-runtime")
    lookup = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infera-studio-history")
    telemetry = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infera-studio-monitor")
    hub = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infera-studio-hub")
    dictation = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infera-studio-dictation")
    try:
        for line in sys.stdin:
            request = json.loads(line)
            request_id = request["id"]
            action = request["action"]
            with lock:
                if action == "voice_start":
                    if active_voice:
                        send({"id": request_id, "ok": False, "error": "Voice input is still finishing. Try again."})
                    else:
                        voice_session = VoiceSession(Cancellation())
                        active_voice = request_id, voice_session
                        dictation.submit(dictate, request_id, voice_session)
                    continue
                if active_voice and active_voice[0] == request_id:
                    voice_session = active_voice[1]
                    if action == "voice_audio":
                        try:
                            voice_session.push(request.get("audio", ""))
                        except InferenceError:
                            voice_session.abort()
                        continue
                    if action == "voice_finish":
                        voice_session.finish()
                        continue
                    if action == "cancel":
                        voice_session.abort()
                        continue
                if action in {"voice_audio", "voice_finish"}:
                    continue
                if action == "cancel":
                    if current := operations.get(request_id):
                        current.cancel()
                    continue
                cancel = Cancellation()
                operations[request_id] = cancel
            if action == "system_stats":
                executor = telemetry
            elif action in {"search", "conversation", "new", "set_pinned", "voice_status"}:
                executor = lookup
            elif action in {"hub_search", "hub_variants", "hub_download"}:
                executor = hub
            else:
                executor = inference
            executor.submit(run, request_id, action, request.get("params", {}), cancel)
    finally:
        with lock:
            if active_voice:
                active_voice[1].abort()
            for cancel in operations.values():
                cancel.cancel()
        dictation.shutdown(wait=True, cancel_futures=True)
        lookup.shutdown(wait=True, cancel_futures=True)
        telemetry.shutdown(wait=True, cancel_futures=True)
        hub.shutdown(wait=True, cancel_futures=True)
        inference.shutdown(wait=True, cancel_futures=True)
        bridge.runtime.server.stop()
