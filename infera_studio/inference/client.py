"""Loopback HTTP and OpenAI-style streaming for llama.cpp."""

import http.client
import json
import logging
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from infera_studio.config.settings import InferenceSettings
from infera_studio.conversations.metrics import Metrics

from .prompts import ToolCall, WireMessage, template_args
from .telemetry import number, update_metrics

LOG = logging.getLogger(__name__)
MAX_LINE = 16 * 1024 * 1024


class InferenceError(RuntimeError):
    pass


class Cancelled(InferenceError):
    pass


class Cancellation:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled("Operation stopped.")

    def wait(self, seconds: float) -> bool:
        return self._event.wait(seconds)

    def bind(self, sock: socket.socket | None) -> None:
        with self._lock:
            self._socket = sock
            if self.cancelled:
                self._interrupt()

    def _interrupt(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                LOG.debug("Cancellation socket already closed")

    def cancel(self) -> None:
        with self._lock:
            self._event.set()
            self._interrupt()


class HttpError(InferenceError):
    def __init__(self, status: int, message: str):
        super().__init__(f"llama.cpp HTTP {status}: {message}")
        self.status = status


@dataclass(frozen=True)
class Budget:
    context: int
    prompt: int
    system: int
    reserved: int
    estimated: bool

    @property
    def remaining(self) -> int:
        return self.context - self.prompt - self.reserved

    @property
    def conversation(self) -> int:
        return max(0, self.prompt - self.system)


@dataclass
class ToolCallDelta:
    id: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)
    size: int = 0

    def append(self, delta: dict[str, Any]) -> None:
        identifier = delta.get("id", "")
        function = delta.get("function", {})
        if (not isinstance(identifier, str) or delta.get("type", "function") != "function"
                or not isinstance(function, dict)):
            raise InferenceError("The model returned an invalid tool call.")
        name, arguments = function.get("name", ""), function.get("arguments", "")
        if not isinstance(name, str) or not isinstance(arguments, str):
            raise InferenceError("The model returned invalid tool arguments.")
        if identifier and self.id != identifier:
            self.id += identifier
        self.name += name
        self.size += len(arguments)
        if len(self.id) > 128 or len(self.name) > 128 or self.size > 1_600_000:
            raise InferenceError("The model returned an oversized tool call.")
        if arguments:
            self.arguments.append(arguments)

    def complete(self) -> ToolCall:
        if not self.id or not self.name or not self.arguments:
            raise InferenceError("The model returned an incomplete tool call.")
        return {"id": self.id, "type": "function", "function": {
            "name": self.name, "arguments": "".join(self.arguments),
        }}


def sse(response: Any) -> Iterator[dict]:
    lines: list[bytes] = []
    while line := response.readline(MAX_LINE + 1):
        if len(line) > MAX_LINE:
            raise InferenceError("Inference returned an oversized streaming event.")
        if line.startswith(b"data:"):
            lines.append(line[5:].strip())
        elif not line.strip() and lines:
            data = b"\n".join(lines)
            lines.clear()
            if data == b"[DONE]":
                return
            try:
                event = json.loads(data)
            except (UnicodeError, ValueError) as exc:
                raise InferenceError("Inference returned a malformed streaming event.") from exc
            if not isinstance(event, dict):
                raise InferenceError("Inference returned an invalid streaming event.")
            yield event
    raise InferenceError("The inference stream ended early.")


class Client:
    def __init__(self, port: int, key: str):
        self.port, self.key = port, key

    def _request(
        self,
        path: str,
        data: dict | None,
        cancel: Cancellation,
        on_response: Callable[[http.client.HTTPResponse], Any],
        timeout: float,
    ) -> Any:
        cancel.check()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        try:
            conn.connect()
            cancel.bind(conn.sock)
            cancel.check()
            conn.request(
                "GET" if data is None else "POST",
                path,
                body=None if data is None else json.dumps(data).encode(),
                headers=headers,
            )
            with conn.getresponse() as response:
                if response.status != 200:
                    raw = response.read(65536).decode("utf-8", "replace")
                    try:
                        raw = json.loads(raw)["error"]["message"]
                    except (KeyError, TypeError, ValueError):
                        pass
                    raise HttpError(response.status, str(raw)[:1600] or response.reason)
                result = on_response(response)
            cancel.check()
            return result
        except (OSError, http.client.HTTPException) as exc:
            cancel.check()
            raise InferenceError(f"Could not communicate with the local inference server ({exc}).") from exc
        except InferenceError:
            cancel.check()
            raise
        finally:
            cancel.bind(None)
            conn.close()

    def json(
        self, path: str, data: dict | None = None, cancel: Cancellation | None = None, timeout: float = 12
    ) -> dict:
        def read(response: http.client.HTTPResponse) -> dict:
            try:
                value = json.loads(response.read(MAX_LINE))
            except (UnicodeError, ValueError) as exc:
                raise InferenceError("Inference returned invalid JSON.") from exc
            if not isinstance(value, dict):
                raise InferenceError("Inference returned an unexpected JSON response.")
            return value

        return self._request(path, data, cancel or Cancellation(), read, timeout)

    def kv_occupancy(self, cancel: Cancellation) -> float | None:
        def read(response: http.client.HTTPResponse) -> float | None:
            for line in response.read(256 * 1024).splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == b"llamacpp:kv_cache_usage_ratio":
                    try:
                        value = number(float(parts[1]))
                    except ValueError:
                        return None
                    return value if value is not None and value <= 1 else None
            return None

        return self._request("/metrics", None, cancel, read, timeout=0.75)

    def count(
        self,
        messages: list[WireMessage],
        settings: InferenceSettings,
        context: int,
        thinking_supported: bool,
        cancel: Cancellation,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Budget:
        options = template_args(settings, thinking_supported)
        if tools:
            options["tools"] = tools
        if messages and messages[-1]["role"] == "assistant":
            options["add_generation_prompt"] = False
        if any(isinstance(message["content"], list) for message in messages):
            def count_images(items: list[WireMessage]) -> int:
                try:
                    result = self.json(
                        "/v1/chat/completions/input_tokens",
                        {"model": "infera-studio", "messages": items, **options},
                        cancel,
                        timeout=30,
                    ).get("input_tokens")
                except HttpError as exc:
                    if exc.status not in (404, 405, 501):
                        raise
                    raise InferenceError("Vision requires a llama-server build that can count image tokens.") from exc
                if type(result) is not int or result <= 0:
                    raise InferenceError("llama-server did not report an image token count.")
                return result

            count = count_images(messages)
            system = max(0, count - count_images(messages[1:])) if messages[0]["role"] == "system" else 0
            return Budget(context, count, system, settings.output, False)
        if not any(message["role"] == "user" for message in messages):
            if not messages:
                return Budget(context, 0, 0, settings.output, False)
            system_text = messages[0]["content"]
            assert isinstance(system_text, str)
            try:
                result = self.json("/tokenize", {"content": system_text, "parse_special": False}, cancel)
                length = len(result["tokens"])
                return Budget(context, length, length, settings.output, True)
            except (HttpError, KeyError, TypeError):
                size = len(system_text.encode())
                return Budget(context, size, size, settings.output, True)
        try:
            prompt = self.json("/apply-template", {"messages": messages, **options}, cancel)["prompt"]
            count = len(self.json("/tokenize", {"content": prompt, "add_special": True}, cancel)["tokens"])
            system = 0
            if messages[0]["role"] == "system":
                without_system = self.json("/apply-template", {"messages": messages[1:], **options}, cancel)[
                    "prompt"
                ]
                baseline = len(
                    self.json("/tokenize", {"content": without_system, "add_special": True}, cancel)["tokens"]
                )
                system = max(0, count - baseline)
            return Budget(context, count, system, settings.output, False)
        except HttpError as exc:
            if exc.status not in (404, 405, 501):
                raise
            LOG.info("Tokenizer endpoints unavailable, using a conservative context estimate")
        except (KeyError, TypeError):
            LOG.warning("Tokenizer endpoint returned an unexpected structure")
        sizes = [
            len(json.dumps(message, ensure_ascii=False).encode("utf-8", "replace")) + 24
            if tools else len(str(message["content"]).encode()) + 24
            for message in messages
        ]
        count = sum(sizes) + 16 + (len(json.dumps(tools).encode()) if tools else 0)
        system = sizes[0] if messages[0]["role"] == "system" else 0
        return Budget(context, count, system, settings.output, True)

    def stream(
        self,
        payload: dict,
        cancel: Cancellation,
        on_delta: Callable[[str, str], None],
        metrics: Metrics | None = None,
        on_metrics: Callable[[Metrics], None] | None = None,
        *,
        tool_calls: list[ToolCall] | None = None,
        on_tool_request: Callable[[str], None] | None = None,
    ) -> Metrics:
        metrics = metrics if metrics is not None else Metrics()
        started = time.monotonic()
        pending: dict[int, ToolCallDelta] = {}
        metrics.finish_reason = None

        def read(response: http.client.HTTPResponse) -> None:
            last_report = started
            reported_tool: str | None = None
            for event in sse(response):
                cancel.check()
                if "error" in event:
                    error = event["error"]
                    raise InferenceError(
                        str(error.get("message", error) if isinstance(error, dict) else error)[:1600]
                    )
                now = time.monotonic()
                first = False
                had_progress = metrics.prompt_progress_tokens is not None
                update_metrics(metrics, event)
                for choice in event.get("choices") or []:
                    reason = choice.get("finish_reason")
                    if reason in ("stop", "length", "content_filter", "tool_calls", "function_call"):
                        metrics.finish_reason = reason
                    delta = choice.get("delta") or {}
                    content, reasoning = delta.get("content") or "", delta.get("reasoning_content") or ""
                    calls = delta.get("tool_calls")
                    tool_request: ToolCallDelta | None = None
                    if tool_calls is not None and calls:
                        if not isinstance(calls, list):
                            raise InferenceError("The model returned invalid tool calls.")
                        for call in calls:
                            index = call.get("index") if isinstance(call, dict) else None
                            if type(index) is not int or not 0 <= index < 4:
                                raise InferenceError("The model exceeded the tool calls allowed per round.")
                            tool_request = pending.setdefault(index, ToolCallDelta())
                            tool_request.append(call)
                    if content or reasoning or tool_request is not None:
                        if metrics.first_token_seconds is None:
                            metrics.first_token_seconds = now - started
                            metrics.stage = "generation"
                            first = True
                        if content or reasoning:
                            on_delta(content, reasoning)
                        if tool_request is not None and on_tool_request and tool_request.name != reported_tool:
                            reported_tool = tool_request.name
                            on_tool_request(reported_tool)
                first_progress = not had_progress and metrics.prompt_progress_tokens is not None
                if on_metrics and (first or first_progress or now - last_report >= 0.25):
                    metrics.seconds = now - started
                    on_metrics(metrics)
                    last_report = now

        try:
            self._request("/v1/chat/completions", payload, cancel, read, 180)
            if tool_calls is not None and (pending or metrics.finish_reason == "tool_calls"):
                if not pending or metrics.finish_reason != "tool_calls":
                    raise InferenceError("The model stopped before completing its tool calls.")
                completed = [pending[index].complete() for index in sorted(pending)]
                if len({call["id"] for call in completed}) != len(completed):
                    raise InferenceError("The model returned duplicate tool call IDs.")
                tool_calls.extend(completed)
        finally:
            metrics.seconds = time.monotonic() - started
        return metrics
