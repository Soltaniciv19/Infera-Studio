"""Desktop operations exposed to Electron through the private bridge."""

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from infera_studio.application import hugging_face
from infera_studio.application.runtime import Runtime
from infera_studio.application.workspace import project_path
from infera_studio.catalog.gguf import Model, scan
from infera_studio.config.preferences import STAT_LABELS, Preferences, default_model_dirs
from infera_studio.config.settings import BACKENDS, CACHE_TYPES, CONTEXT_SIZES, InferenceSettings
from infera_studio.conversations.records import Chat
from infera_studio.host.hardware import SystemMonitor, amd_gpus
from infera_studio.inference.client import Cancellation, Client, HttpError, InferenceError
from infera_studio.inference.memory import recommend
from infera_studio.inference.server import SAMPLERS, ServerInfo, detect
from infera_studio.persistence.sqlite import PINNED_CHAT_LIMIT, Store

from .attachments import parse_attachments

Event = Callable[[str, Any], None]


class Bridge:
    def __init__(self, store: Store, config: Path) -> None:
        self.store = store
        self.config = config
        self.runtime = Runtime(store)
        self.monitor = SystemMonitor()
        self._kv_unsupported: Client | None = None
        self.prefs = Preferences()
        self.models: dict[str, Model] = {}
        self.errors: list[str] = []

    def initialize(self) -> dict[str, Any]:
        self.store.initialize()
        note = ""
        try:
            self.prefs = Preferences.load(self.config)
        except (OSError, ValueError) as exc:
            self.prefs = Preferences(model_dirs=default_model_dirs(), setup_complete=True)
            note = f"Settings could not be read. Open Settings to replace the file. {exc}"
        return self.rescan() | {"note": note, "history": self.store.search(), "pinned_limit": PINNED_CHAT_LIMIT}

    def rescan(self) -> dict[str, Any]:
        models, self.errors = scan(self.prefs.model_dirs)
        self.store.save_models(models)
        self.models = self.store.catalog()
        preferences = asdict(self.prefs)
        preferences.pop("tavily_api_key")
        return {
            "preferences": preferences | {"has_tavily_api_key": bool(self.prefs.tavily_api_key)},
            "models": [
                {
                    "path": model.path, "name": model.name, "architecture": model.architecture,
                    "quant": model.quant, "parameters": model.parameters, "context": model.context,
                    "size": model.size, "shards": model.shards, "issue": model.issue,
                    "mtp": model.has_mtp_layers,
                    "thinking": model.has_thinking_switch,
                    "visionProjector": model.is_vision_projector,
                }
                for model in sorted(self.models.values(), key=lambda model: (model.name.casefold(), model.path))
            ],
            "errors": self.errors,
            "discovered": detect(self.prefs),
            "backends": BACKENDS,
            "cacheTypes": CACHE_TYPES,
            "contextSizes": CONTEXT_SIZES,
            "statLabels": STAT_LABELS,
        }

    def save_preferences(self, preferences: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(preferences, dict):
            raise ValueError("Settings must be a JSON object.")
        values = Preferences.from_dict({"tavily_api_key": self.prefs.tavily_api_key} | preferences)
        if values.binaries != self.prefs.binaries:
            self.runtime.server.stop()
        values.save(self.config)
        self.prefs = values
        return self.rescan()

    def download_model(self, repo: str, revision: str, path: str, folder: str, cancel: Cancellation, emit: Event) -> str:
        if not isinstance(folder, str) or folder not in self.prefs.model_dirs:
            raise ValueError("Choose a saved model folder in Settings.")
        return hugging_face.download(repo, revision, path, folder, cancel, emit)

    def selected(self, path: str) -> Model:
        model = self.models.get(path)
        if not model or not model.available:
            raise InferenceError("Choose an available GGUF model from the model picker.")
        return model

    def create(self, source: int | None = None, model: str | None = None) -> int:
        previous = self.store.get(source)[0] if source is not None else None
        settings = previous.settings if previous else InferenceSettings(backend=self.prefs.backend)
        models = self.models
        if model:
            self.selected(model)
        elif previous and (existing := models.get(previous.model)) and existing.available:
            model = existing.path
        else:
            model = next((item.path for item in models.values() if item.available), "")
        if not previous or previous.model != model:
            settings = replace(settings, mtp_draft="", vision_projector="")
        return self.store.create(model, replace(settings, system=""))

    def configure(self, chat_id: int, model: str | None = None, settings: dict[str, Any] | None = None) -> None:
        chat, _ = self.store.get(chat_id)
        if model is not None:
            self.selected(model)
        values = InferenceSettings.from_dict(settings) if settings is not None else chat.settings
        target = chat.model if model is None else model
        if target != chat.model and settings is None:
            values = replace(values, mtp_draft="", vision_projector="")
        if values.mtp_draft and values.mtp_draft != chat.settings.mtp_draft:
            if not self.mtp_draft(self.selected(target), values):
                raise ValueError("Choose a compatible MTP draft from the local model library.")
        if values.vision_projector and (target != chat.model or values.vision_projector != chat.settings.vision_projector):
            self.selected(target)
            if not self.vision_projector(values):
                raise ValueError("Choose a vision projector from the local model library.")
        self.store.configure(chat_id, target, values)

    def mtp_draft(self, target: Model, settings: InferenceSettings) -> Model | None:
        draft = self.models.get(settings.mtp_draft)
        return draft if draft and draft.is_mtp_head_for(target) else None

    def vision_projector(self, settings: InferenceSettings) -> Model | None:
        projector = self.models.get(settings.vision_projector)
        return projector if projector and projector.is_vision_projector else None

    def fork(self, chat_id: int, message_id: int) -> int:
        _, messages = self.store.get(chat_id)
        if not any(message.id == message_id for message in messages):
            raise ValueError("This message is no longer in the conversation.")
        return self.store.fork(chat_id, message_id)

    def load(self, chat: Chat, cancel: Cancellation, emit: Event) -> ServerInfo:
        try:
            model = self.selected(chat.model)
        except InferenceError:
            self.runtime.server.stop()
            raise
        info = self.runtime.server.load(
            model, chat.settings, self.prefs, cancel, lambda state, detail: emit("status", [state, detail]),
            mtp_draft=self.mtp_draft(model, chat.settings),
            vision_projector=self.vision_projector(chat.settings),
        )
        emit("features", self.features(info))
        return info

    @staticmethod
    def features(info: ServerInfo) -> dict[str, bool | str]:
        return {"vision": info.vision_supported, "thinking": info.thinking_supported,
                "mtp": info.mtp, "mtpAvailable": info.mtp_supported, "mtpEnabled": info.mtp_enabled}

    def open_chat(self, chat_id: int, cancel: Cancellation, emit: Event) -> None:
        chat, _ = self.store.get(chat_id)
        if not chat.model:
            self.runtime.server.stop()
            return
        if model := self.models.get(chat.model):
            if model.available:
                self.load(chat, cancel, emit)
                return
        self.runtime.server.stop()
        emit("notice", "The selected model is unavailable. Choose a replacement to keep chatting.")

    def workspace(self, chat_id: int, path: str) -> str:
        root = project_path(path) if path else ""
        self.store.set_workspace(chat_id, root)
        return root

    def web_search_key(self, enabled: bool) -> str:
        if type(enabled) is not bool:
            raise ValueError("Invalid web search selection.")
        if enabled and not self.prefs.tavily_api_key:
            raise ValueError("Add a Tavily API key in Settings → Web search.")
        return self.prefs.tavily_api_key if enabled else ""

    def generate(
        self, chat_id: int, draft: str | None, cancel: Cancellation, emit: Event,
        attachments: list[dict[str, str]] | None = None,
        *,
        web_search: bool = False,
    ) -> Any:
        tavily_api_key = self.web_search_key(web_search)
        files = parse_attachments(attachments)
        if draft is None and files:
            raise ValueError("Send attachments with a user message.")
        if draft is not None and not draft.strip() and not files:
            raise ValueError("Enter a message.")
        chat, messages = self.store.get(chat_id)
        model = self.selected(chat.model)
        result = self.runtime.generate(
            model, self.prefs, chat, messages, draft, cancel,
            lambda state, detail: emit("status", [state, detail]),
            lambda info: emit("features", self.features(info)),
            lambda budget: emit("budget", budget),
            lambda message_id: emit("started", [*self.store.get(chat_id), message_id]),
            lambda content, reasoning: emit("delta", [content, reasoning]),
            attachments=files,
            mtp_draft=self.mtp_draft(model, chat.settings),
            vision_projector=self.vision_projector(chat.settings),
            performance=lambda metrics: emit("performance", metrics),
            workspace=chat.workspace,
            tool_event=lambda activity: emit("tool_activity", activity),
            tavily_api_key=tavily_api_key,
        )
        return [*self.store.get(chat_id), result]

    def current_info(self, chat: Chat) -> ServerInfo | None:
        server = self.runtime.server
        if (
            server.signature is None
            or server.signature[0] != chat.model
            or server.signature[-1] != chat.settings.launch_signature()
        ):
            return None
        return server.info

    def system_stats(self, cancel: Cancellation) -> dict[str, Any]:
        stats = asdict(self.monitor.sample())
        server = self.runtime.server
        info = server.info
        occupancy = None
        if info and info.metrics_supported and info.client is not self._kv_unsupported:
            try:
                occupancy = info.client.kv_occupancy(cancel)
            except HttpError as exc:
                if exc.status in (404, 405, 501):
                    self._kv_unsupported = info.client
            except InferenceError:
                cancel.check()
            else:
                if occupancy is None:
                    self._kv_unsupported = info.client
        return stats | {"kv_occupancy": occupancy if info is server.info else None}

    def budget(
        self, chat_id: int, draft: str, cancel: Cancellation, attachments: list[dict[str, str]] | None = None,
        *,
        web_search: bool = False,
    ) -> Any:
        tavily_api_key = self.web_search_key(web_search)
        files = parse_attachments(attachments)
        chat, messages = self.store.get(chat_id)
        if info := self.current_info(chat):
            return self.runtime.budget(
                info, chat, messages, draft if draft.strip() or files else "", cancel, files,
                workspace=bool(chat.workspace), web_search=bool(tavily_api_key),
            )
        return None

    def context(self, chat_id: int, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        chat, _ = self.store.get(chat_id)
        info = self.current_info(chat)
        model = self.models.get(chat.model)
        gpu = max(amd_gpus(), key=lambda device: device.free, default=None)
        values = InferenceSettings.from_dict(settings) if settings is not None else chat.settings
        if gpu and info and info.plan.gpu and gpu.device == info.plan.gpu.device and info.observed_vram is not None:
            baseline = replace(gpu, used=max(info.plan.gpu.used, gpu.used - max(0, info.observed_vram)))
        else:
            baseline = gpu
        plan = info.plan if info and settings is None else (recommend(model, values, baseline) if model else None)
        return {
            "gpu": gpu,
            "plan": (asdict(plan) | {"required": plan.required, "headroom": plan.headroom}) if plan else None,
            "slot": info.context if info else None,
            "observed": info.observed_vram if info else None,
            "kvAllocatedMib": info.kv_allocated_mib if info else None,
            "allocations": info.allocations if info else (),
            "samplers": sorted(info.samplers if info else SAMPLERS - {"min_p"}),
            "thinking": bool((info and info.thinking_supported) or (model and model.has_thinking_switch)),
            "mtp": info.mtp if info else "",
            "mtpDrafts": [
                [draft.path, draft.name] for draft in self.models.values() if model and draft.is_mtp_head_for(model)
            ],
        }

    def change_context(
        self, chat_id: int, start: int, summarize: bool, cancel: Cancellation, emit: Event
    ) -> None:
        if not summarize:
            self.store.set_context(chat_id, start)
            return
        chat, messages = self.store.get(chat_id)
        first_turn = next((message.id for message in messages if message.role == "user"), 0)
        if start <= max(chat.active_from, first_turn) or not any(
            message.id == start and message.role == "user" for message in messages
        ):
            raise ValueError("Choose a later user turn before generating a summary.")
        info = self.current_info(chat)
        if not info:
            raise InferenceError("Load the model before summarizing earlier turns.")
        emit("status", ["Generating", "Summarizing earlier turns"])
        self.runtime.summarize(info, chat, messages, start, cancel)

    def delete(self, chat_id: int | None, unload: bool) -> None:
        if unload:
            self.runtime.server.stop()
        self.store.delete(chat_id)

    def dispatch(self, action: str, params: dict[str, Any], cancel: Cancellation, emit: Event) -> Any:
        methods: dict[str, Callable[..., Any]] = {
            "initialize": self.initialize,
            "preferences": self.save_preferences,
            "rescan": self.rescan,
            "hub_search": hugging_face.search,
            "hub_variants": hugging_face.variants,
            "hub_download": self.download_model,
            "search": self.store.search,
            "conversation": self.store.get,
            "open": self.open_chat,
            "new": self.create,
            "configure": self.configure,
            "workspace": self.workspace,
            "fork": self.fork,
            "rename": self.store.rename,
            "set_pinned": self.store.set_pinned,
            "delete": self.delete,
            "generate": self.generate,
            "budget": self.budget,
            "context": self.context,
            "change_context": self.change_context,
            "unload": self.runtime.server.stop,
            "health": self.runtime.server.health,
            "system_stats": self.system_stats,
            "diagnostics": self.runtime.server.diagnostics,
        }
        if action not in methods:
            raise ValueError("Unknown desktop operation.")
        if action in {"open", "generate", "budget", "change_context", "system_stats",
                      "hub_search", "hub_variants", "hub_download"}:
            params["cancel"] = cancel
        if action in {"open", "generate", "change_context", "hub_download"}:
            params["emit"] = emit
        return methods[action](**params)
