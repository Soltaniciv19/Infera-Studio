#!/usr/bin/python3
"""Deterministic llama-server fixture with delayed health and streaming."""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if "--help" in sys.argv:
    print(
        "--host --port --api-key-file --parallel --no-context-shift --alias --device "
        "--list-devices --fit --no-ui --jinja --mmproj --no-mmproj --no-mmproj-offload --log-colors --reasoning-format "
        "--flash-attn [on|off|auto] --cache-type-k --cache-type-v --metrics"
    )
    print("--log-verbosity N\n    - 4: trace (more info)")
    if os.getenv("FAKE_INFERA_MTP") != "unsupported":
        print("--spec-type none,draft-mtp --spec-draft-model FNAME")
    raise SystemExit(0)
if "--list-devices" in sys.argv:
    print("  Vulkan0: Fake GPU\n  ROCm0: Fake GPU")
    raise SystemExit(0)

parser = argparse.ArgumentParser()
parser.add_argument("-m")
parser.add_argument("--host")
parser.add_argument("--port", type=int)
parser.add_argument("--ctx-size", type=int)
parser.add_argument("--api-key-file")
parser.add_argument("--spec-type")
parser.add_argument("--spec-draft-model")
parser.add_argument("--mmproj")
parser.add_argument("--no-mmproj", action="store_true")
args, _ = parser.parse_known_args()
if args.spec_type and os.getenv("FAKE_INFERA_MTP") == "broken":
    print("Unable to create the MTP context", file=sys.stderr)
    raise SystemExit(1)
print("0.00.000.123 I llama_kv_cache:        CPU KV buffer size = 64.00 MiB", flush=True)
key = Path(args.api_key_file).read_text().strip()
started = time.monotonic()
last: dict = {}


def text_content(message: dict) -> str:
    content = message["content"]
    return content if isinstance(content, str) else " ".join(part["text"] for part in content if part["type"] == "text")


PROJECT_TOOLS = (
    ("search_files", {"query": "seed", "kind": "name"}),
    ("read_file", {"path": "seed.txt"}),
    ("create_file", {"path": "src/result.txt", "content": "Project from the seed\n"}),
)
PROJECT_UPDATES = ("I’ll locate the seed file 🔎.", "I’ll read the seed.", "I’ll write the result file.")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_: object) -> None:
        pass

    def reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self) -> bool:
        if self.headers.get("Authorization") == f"Bearer {key}":
            return True
        self.reply(401, {"error": {"message": "Unauthorized"}})
        return False

    def do_GET(self) -> None:
        if self.path == "/health":
            if time.monotonic() - started > float(os.getenv("FAKE_INFERA_READY", "0.25")):
                self.reply(200, {"status": "ok"})
            else:
                self.reply(503, {"status": "loading"})
        elif self.authorized():
            if self.path == "/props":
                self.reply(
                    200,
                    {
                        "model_path": args.m,
                        "chat_template": "enable_thinking",
                        "default_generation_settings": {"n_ctx": args.ctx_size, "params": {"min_p": 0.0}},
                        "pid": os.getpid(),
                        "spec_type": args.spec_type,
                        "spec_draft_model": args.spec_draft_model,
                        "modalities": {"vision": bool(args.mmproj and not args.no_mmproj)},
                        "projector": args.mmproj,
                    },
                )
            if self.path == "/last":
                self.reply(200, last)
            if self.path == "/metrics":
                body = b"# TYPE llamacpp:kv_cache_usage_ratio gauge\nllamacpp:kv_cache_usage_ratio 0.125\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    def do_POST(self) -> None:
        if not self.authorized():
            return
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        if self.path == "/apply-template":
            prompt = " ".join(m["content"] for m in data["messages"])
            if data.get("tools"):
                prompt += " " + " ".join(item["function"]["name"] for item in data["tools"])
            self.reply(200, {"prompt": prompt + (" assistant" if data.get("add_generation_prompt", True) else "")})
        elif self.path == "/tokenize":
            self.reply(200, {"tokens": list(range(len(data["content"].split())))})
        elif self.path == "/v1/chat/completions/input_tokens":
            images = sum(
                part["type"] == "image_url" for message in data["messages"]
                if isinstance(message["content"], list) for part in message["content"]
            )
            self.reply(200, {"input_tokens": 128 * images + sum(len(text_content(m).split()) for m in data["messages"])})
        elif self.path == "/v1/chat/completions":
            last.clear()
            last.update(data)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            messages = data["messages"]
            latest = messages[-1]
            if any(item["function"]["name"] == "web_search" for item in data.get("tools", [])):
                if "Search the fixture web" in text_content(latest) and latest["role"] == "user":
                    self.event({"choices": [{"delta": {"tool_calls": [
                        {"index": 0, "id": "search", "type": "function", "function": {
                            "name": "web_search", "arguments": '{"query":"fixture weather"}',
                        }},
                    ]}, "finish_reason": "tool_calls"}]})
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                if latest.get("tool_call_id") == "search":
                    result = json.loads(latest["content"])
                    content = result["results"][0]["content"] if result.get("results") else result["error"]
                    self.event({"choices": [{"delta": {"content": content}, "finish_reason": "stop"}]})
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
            if data.get("tools") and "Interrupted file tool" in text_content(messages[-1]):
                self.event({"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "partial_call", "type": "function", "function": {
                        "name": "create_file", "arguments": '{"path":"partial.txt","content":""}',
                    }},
                ]}}]})
                self.event({"choices": [{"delta": {}, "finish_reason": "length"}]})
                self.wfile.write(b"data: [DONE]\n\n")
                return
            if data.get("tools") and (
                messages[-1]["role"] == "tool" or "Create the fixture project" in text_content(messages[-1])
            ):
                self.project(messages)
                return
            if any("reasoning-only" in text_content(message) for message in messages) and data.get(
                "chat_template_kwargs", {}
            ).get("enable_thinking", True):
                self.event({"choices": [{"delta": {"reasoning_content": "Still thinking"}}]})
                self.event(
                    {
                        "choices": [{"delta": {}, "finish_reason": "length"}],
                        "usage": {"completion_tokens": data["max_tokens"]},
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
                return
            slow = any("slow" in text_content(message) for message in messages)
            first = {"choices": [{"delta": {"reasoning_content": "private", "content": "Hello"}}]}
            if data.get("return_progress"):
                self.event({"choices": [], "prompt_progress": {"total": 4, "cache": 2, "processed": 3, "time_ms": 20}})
                if slow:
                    time.sleep(0.6)
            if data.get("timings_per_token"):
                first["timings"] = {
                    "prompt_n": 2, "cache_n": 2, "prompt_ms": 25, "prompt_per_second": 80,
                    "predicted_n": 1, "predicted_ms": 50, "predicted_per_second": 20,
                }
            self.event(first)
            if slow:
                time.sleep(5)
            self.event(
                {
                    "choices": [{"delta": {"content": " world"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 2}},
                    "timings": {
                        "prompt_n": 2, "cache_n": 2, "prompt_ms": 25, "prompt_per_second": 80,
                        "predicted_n": 2, "predicted_ms": 100, "predicted_per_second": 20,
                    },
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")

    def project(self, messages: list[dict]) -> None:
        results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
        if len(results) < len(PROJECT_TOOLS):
            index = len(results)
            name, params = PROJECT_TOOLS[index]
            self.event({"choices": [{"delta": {
                "reasoning_content": f"Project step {index + 1} →", "content": PROJECT_UPDATES[index][:-1],
            }}]})
            self.event({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": f"call_{index}", "type": "function", "function": {"name": name}},
            ], "content": "."}}]})
            if name == "create_file" and os.getenv("FAKE_INFERA_SLOW_TOOL"):
                time.sleep(1.5)
            for arguments in ("{", json.dumps(params)[1:]):
                self.event({"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": arguments}},
                ]}}]})
            self.event({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        else:
            success = results[0].get("matches") == [{"path": "seed.txt"}] and (
                results[1].get("content") == "Read me\n" and results[2].get("path") == "src/result.txt"
            )
            content = "Project created." if success else "Tool results were incorrect."
            self.event({"choices": [{"delta": {"content": content}, "finish_reason": "stop"}]})
        self.wfile.write(b"data: [DONE]\n\n")

    def event(self, data: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
        self.wfile.flush()


ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
