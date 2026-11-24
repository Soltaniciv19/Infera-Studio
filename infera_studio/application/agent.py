"""Run explicit model tool calls and feed results into the current response."""

import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from infera_studio.conversations.metrics import Metrics
from infera_studio.conversations.records import ToolActivity
from infera_studio.inference.client import Cancellation, Client, InferenceError
from infera_studio.inference.prompts import ToolCall, WireMessage

from .web_search import WEB_SEARCH_TOOL, run_web_search
from .workspace import TOOLS, open_workspace

MAX_TOOL_CALLS = 12
DEFAULT_PROGRESS = ("Preparing tool call", "Running tool call")
TOOL_PROGRESS = {
    "read_file": ("Preparing file read", "Reading file"),
    "create_file": ("Generating file content", "Creating file"),
    "search_files": ("Preparing project search", "Searching project"),
    "web_search": ("Preparing web search", "Searching the web with Tavily"),
}


def tool_definitions(workspace: bool, web_search: bool) -> list[dict[str, Any]]:
    return (TOOLS if workspace else []) + ([WEB_SEARCH_TOOL] if web_search else [])


def stream_agent(
    client: Client,
    workspace: str,
    wire: list[WireMessage],
    payload: dict[str, Any],
    cancel: Cancellation,
    metrics: Metrics,
    on_delta: Callable[[str, str], None],
    on_metrics: Callable[[Metrics], None] | None,
    on_activity: Callable[[ToolActivity], None],
    on_progress: Callable[[str], None],
    before_next_round: Callable[[], None],
    *,
    tavily_api_key: str = "",
) -> None:
    started = time.monotonic()
    finished_calls = 0

    def preparing(name: str) -> None:
        on_progress(TOOL_PROGRESS.get(name, DEFAULT_PROGRESS)[0])

    try:
        with (open_workspace(workspace, cancel) if workspace else nullcontext()) as project:
            while True:
                content: list[str] = []
                reasoning: list[str] = []
                calls: list[ToolCall] = []

                def receive(text: str, thoughts: str) -> None:
                    content.append(text)
                    reasoning.append(thoughts)
                    on_delta(text, thoughts)

                client.stream(
                    payload, cancel, receive, metrics, on_metrics,
                    tool_calls=calls, on_tool_request=preparing,
                )
                if not calls:
                    return
                if finished_calls + len(calls) > MAX_TOOL_CALLS:
                    raise InferenceError("The tool call limit for this message was reached.")
                finished_calls += len(calls)
                assistant: WireMessage = {"role": "assistant", "content": "".join(content), "tool_calls": calls}
                if thoughts := "".join(reasoning):
                    assistant["reasoning_content"] = thoughts
                wire.append(assistant)
                for call in calls:
                    function = call["function"]
                    name, arguments = function["name"], function["arguments"]
                    cancel.check()
                    on_progress(TOOL_PROGRESS.get(name, DEFAULT_PROGRESS)[1])
                    if name == "web_search":
                        result, activity = run_web_search(tavily_api_key, arguments, cancel)
                    elif project is not None:
                        result, activity = project.execute(name, arguments)
                    else:
                        result = {"error": "This tool is unavailable without a project folder."}
                        activity = ToolActivity(name.encode("utf-8", "replace").decode(), "", False)
                    on_activity(activity)
                    wire.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, ensure_ascii=False)})
                before_next_round()
                if assistant["content"]:
                    on_delta("\n\n", "")
    finally:
        metrics.seconds = time.monotonic() - started
