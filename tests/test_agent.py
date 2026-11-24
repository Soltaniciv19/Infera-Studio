import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from infera_studio.application.workspace import open_workspace
from infera_studio.catalog.gguf import Model
from infera_studio.config.preferences import Preferences
from infera_studio.config.settings import InferenceSettings
from infera_studio.conversations.records import ToolActivity
from infera_studio.desktop.electron.operations import Bridge
from infera_studio.inference.client import Cancellation, InferenceError
from infera_studio.persistence.sqlite import Store


def test_agent_runs_the_file_tools_only_in_the_enabled_chat(
    tmp_path: Path, mock_model: Model, mock_prefs: Preferences, settings: InferenceSettings,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "seed.txt").write_text("Read me\n")
    config = tmp_path / "settings.json"
    replace(mock_prefs, model_dirs=(str(Path(mock_model.path).parent),)).save(config)
    store = Store(tmp_path / "conversations.db")
    bridge = Bridge(store, config)
    events: list[tuple[str, Any]] = []

    def emit(name: str, value: Any) -> None:
        events.append((name, value))

    cancel = Cancellation()
    try:
        bridge.initialize()
        agent = store.create(mock_model.path, settings)
        ask = store.create(mock_model.path, settings)
        bridge.workspace(agent, str(project))
        bridge.open_chat(agent, cancel, emit)
        info = bridge.runtime.server.info
        assert info is not None
        prompt = "Create the fixture project"
        assert bridge.budget(agent, prompt, cancel).prompt == bridge.budget(ask, prompt, cancel).prompt + 3
        bridge.generate(ask, prompt, cancel, emit)
        assert not (project / "src").exists()
        assert "tools" not in info.client.json("/last")
        events.clear()
        _, turns, result = bridge.generate(agent, prompt, cancel, emit)
        assert result.state == "complete"
        updates = ("I’ll locate the seed file 🔎.", "I’ll read the seed.", "I’ll write the result file.")
        assert turns[-1].content == "\n\n".join((*updates, "Project created."))
        assert (project / "src/result.txt").read_text() == "Project from the seed\n"
        expected_tools = (("search_files", "."), ("read_file", "seed.txt"), ("create_file", "src/result.txt"))
        expected_activity = tuple(
            ToolActivity(name, path, True, len("\n\n".join(updates[:index]).encode("utf-16-le")) // 2,
                         len("".join(f"Project step {i} →" for i in range(1, index + 1))))
            for index, (name, path) in enumerate(expected_tools, start=1)
        )
        assert turns[-1].tool_activity == expected_activity
        latest_delta = ("", "")
        completed = 0
        for name, value in events:
            if name == "delta":
                latest_delta = value
            elif name == "tool_activity":
                completed += 1
                assert value == expected_activity[:completed]
                assert latest_delta[0].endswith(updates[completed - 1])
        assert completed == 3
        phases = [value[1] for name, value in events if name == "status"]
        assert phases.index("Generating file content") < phases.index("Creating file")
        request = info.client.json("/last")
        assert [message["role"] for message in request["messages"]] == [
            "user", "assistant", "tool", "assistant", "tool", "assistant", "tool",
        ]
        assert request["messages"][-1]["tool_call_id"] == "call_2"
        assert Store(store.path).get(agent)[0].workspace == str(project)
        branch = store.fork(agent, turns[-1].id + 1)
        assert store.get(branch)[0].workspace == ""
        assert store.get(branch)[1][-1].tool_activity == turns[-1].tool_activity
        _, turns, interrupted = bridge.generate(agent, "Interrupted file tool", cancel, emit)
        assert interrupted.state == "error" and "before completing" in interrupted.error
        assert not turns[-1].tool_activity and not (project / "partial.txt").exists()
        bridge.workspace(agent, "")
        chat, saved = store.get(agent)
        assert not chat.workspace and saved == turns
        bridge.generate(agent, "Tell me about the project", cancel, emit)
        assert "tools" not in info.client.json("/last")
    finally:
        bridge.runtime.server.stop()


def test_workspace_never_overwrites_or_follows_links(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("keep this outside")
    (project / "symlink.txt").symlink_to(external)
    (project / "outside").symlink_to(tmp_path, target_is_directory=True)
    os.link(external, project / "hardlink.txt")
    os.mkfifo(project / "pipe")
    cancel = Cancellation()
    with open_workspace(str(project), cancel) as workspace:
        assert workspace.create("src/new.txt", "First\nSecond\nthird\n")["bytes_written"] == 19
        assert workspace.read("src/new.txt", 2, 1) == {
            "path": "src/new.txt", "start_line": 2, "next_line": 3, "content": "Second\n",
            "truncated_file": False, "truncated_line": False,
        }
        assert workspace.search("SECOND", kind="content")["matches"] == [
            {"path": "src/new.txt", "line": 2, "text": "Second"},
        ]
        for path in (str(external), "../external.txt", "symlink.txt", "outside/external.txt",
                     "hardlink.txt", "pipe", ".git/config", "model.gguf"):
            assert "error" in workspace.execute("read_file", json.dumps({"path": path}))[0]
        for path in ("src/new.txt", "outside/escaped.txt", "symlink.txt", "../escaped.txt"):
            assert "error" in workspace.execute("create_file", json.dumps({"path": path, "content": "bad"}))[0]
        assert workspace.search("", kind="name")["matches"] == [{"path": "src/new.txt"}]
        cancel.cancel()
        with pytest.raises(InferenceError, match="stopped"):
            workspace.execute("create_file", '{"path":"cancelled.txt","content":""}')
    assert (project / "src/new.txt").read_text() == "First\nSecond\nthird\n"
    assert list((project / "src").iterdir()) == [project / "src/new.txt"]
    assert external.read_text() == "keep this outside"
    assert not (tmp_path / "escaped.txt").exists()


def test_web_search_uses_tavily_only_when_enabled(
    tmp_path: Path, mock_model: Model, mock_prefs: Preferences, settings: InferenceSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "settings.json"
    replace(mock_prefs, model_dirs=(str(Path(mock_model.path).parent),)).save(config)
    store = Store(tmp_path / "conversations.db")
    bridge = Bridge(store, config)
    connect = MagicMock()
    monkeypatch.setattr("infera_studio.application.web_search.http.client.HTTPSConnection", connect)
    response = connect.return_value.getresponse.return_value.__enter__.return_value
    response.status = 200
    source = {"title": "Example weather", "url": "https://example.test/weather", "content": "Sunny in the fixture."}
    response.read.return_value = json.dumps({"results": [source | {"raw_content": "ignored"}]}).encode()

    def emit(_name: str, _value: Any) -> None:
        pass

    key = "tvly-fixture-key"
    cancel = Cancellation()
    try:
        settings_for_ui = json.loads(json.dumps(bridge.initialize()["preferences"]))
        assert settings_for_ui["has_tavily_api_key"] is False
        updated = bridge.save_preferences(settings_for_ui | {"tavily_api_key": f" {key} "})["preferences"]
        assert updated["has_tavily_api_key"] is True
        assert "tavily_api_key" not in updated
        bridge.save_preferences(settings_for_ui | {"theme": "Light"})
        assert Preferences.load(config).tavily_api_key == key
        assert config.stat().st_mode & 0o077 == 0

        chat = store.create(mock_model.path, settings)
        bridge.open_chat(chat, cancel, emit)
        info = bridge.runtime.server.info
        assert info is not None
        draft = "Search the fixture web"
        normal_budget = bridge.budget(chat, draft, cancel)
        search_budget = bridge.budget(chat, draft, cancel, web_search=True)
        assert search_budget.prompt == normal_budget.prompt + 1
        bridge.generate(chat, draft, cancel, emit)
        connect.assert_not_called()
        assert "tools" not in info.client.json("/last")

        _, turns, result = bridge.generate(chat, draft, cancel, emit, web_search=True)
        assert result.state == "complete"
        assert turns[-1].content == source["content"]
        assert turns[-1].tool_activity == (ToolActivity("web_search", "fixture weather", True, 0, 0),)
        assert store.get(chat)[1][-1].tool_activity == turns[-1].tool_activity
        connect.assert_called_once_with("api.tavily.com", timeout=10)
        request = connect.return_value.request.call_args
        assert request.args == ("POST", "/search")
        assert request.kwargs["headers"]["Authorization"] == f"Bearer {key}"
        assert json.loads(request.kwargs["body"]) == {
            "query": "fixture weather", "search_depth": "basic", "max_results": 5,
        }
        inference_request = info.client.json("/last")
        assert [item["function"]["name"] for item in inference_request["tools"]] == ["web_search"]
        assert json.loads(inference_request["messages"][-1]["content"]) == {"results": [source]}
        assert key not in json.dumps(inference_request)

        response.status = 401
        _, turns, result = bridge.generate(chat, draft, cancel, emit, web_search=True)
        assert result.state == "complete"
        assert turns[-1].tool_activity == (ToolActivity("web_search", "fixture weather", False, 0, 0),)
        assert "Tavily rejected the API key" in turns[-1].content
        assert key not in turns[-1].content

        bridge.save_preferences(settings_for_ui | {"tavily_api_key": ""})
        assert Preferences.load(config).tavily_api_key == ""
        with pytest.raises(ValueError, match="Add a Tavily API key"):
            bridge.generate(chat, draft, cancel, emit, web_search=True)
        assert connect.call_count == 2
    finally:
        bridge.runtime.server.stop()
