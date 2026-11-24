"""Application operations serialized by the inference worker."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from infera_studio.catalog.gguf import Model
from infera_studio.config.preferences import Preferences
from infera_studio.config.settings import InferenceSettings
from infera_studio.conversations.metrics import Metrics
from infera_studio.conversations.records import Attachment, Chat, Message, ToolActivity
from infera_studio.inference.client import Budget, Cancellation, Cancelled, InferenceError
from infera_studio.inference.prompts import (
    Content,
    WireMessage,
    chat_payload,
    image_content,
    join_content,
    request_messages,
)
from infera_studio.inference.server import ServerInfo, ServerManager
from infera_studio.persistence.sqlite import Store

from .agent import stream_agent, tool_definitions

LOG = logging.getLogger(__name__)


class BudgetExceeded(InferenceError):
    def __init__(self, budget: Budget):
        self.budget = budget
        super().__init__(
            f"The prompt and reserved output exceed the loaded context by {-budget.remaining:,} tokens."
        )


@dataclass(frozen=True)
class Generation:
    metrics: Metrics
    state: str
    error: str = ""
    notice: str = ""


class Runtime:
    def __init__(self, store: Store):
        self.store = store
        self.server = ServerManager()

    def budget(
        self, info: ServerInfo, chat: Chat, messages: list[Message], draft: str, cancel: Cancellation,
        attachments: tuple[Attachment, ...] = (),
        *,
        workspace: bool = False,
        web_search: bool = False,
    ) -> Budget:
        if any(file.media_type for file in attachments) and not info.vision_supported:
            raise InferenceError("Load a matching vision projector before sending images.")
        wire = request_messages(
            messages, chat.settings, chat.active_from, chat.summary, draft, attachments, vision=info.vision_supported,
        )
        return info.client.count(
            wire, chat.settings, info.context, info.thinking_supported, cancel,
            tools=tool_definitions(workspace, web_search),
        )

    def generate(
        self,
        model: Model,
        prefs: Preferences,
        chat: Chat,
        messages: list[Message],
        draft: str | None,
        cancel: Cancellation,
        status: Callable[[str, str], None],
        ready: Callable[[ServerInfo], None],
        counted: Callable[[Budget], None],
        started: Callable[[int], None],
        updated: Callable[[str, str], None],
        attachments: tuple[Attachment, ...] = (),
        mtp_draft: Model | None = None,
        vision_projector: Model | None = None,
        performance: Callable[[Metrics], None] | None = None,
        workspace: str = "",
        tool_event: Callable[[tuple[ToolActivity, ...]], None] | None = None,
        *,
        tavily_api_key: str = "",
    ) -> Generation:
        request_start = time.monotonic()
        metrics = Metrics()

        def report(value: Metrics) -> None:
            elapsed = time.monotonic() - request_start
            value.request_seconds = elapsed
            if performance:
                value.record_sample(elapsed)
                performance(replace(value, samples=value.samples.copy()))

        report(metrics)
        try:
            info = self.server.load(
                model, chat.settings, prefs, cancel, status, mtp_draft=mtp_draft,
                vision_projector=vision_projector,
            )
            token_check_start = time.monotonic()
            metrics.model_seconds = token_check_start - request_start
            metrics.context = info.context
            metrics.kv_allocated_mib = info.kv_allocated_mib
            metrics.stage = "token_check"
            report(metrics)
            ready(info)
            if any(file.media_type for file in attachments) and not info.vision_supported:
                raise InferenceError("Load a matching vision projector before sending images.")
            wire = request_messages(
                messages, chat.settings, chat.active_from, chat.summary, draft or "", attachments,
                vision=info.vision_supported,
            )
            if not wire or wire[-1]["role"] != "user":
                raise InferenceError("A response needs an active user message.")
            tools = tool_definitions(bool(workspace), bool(tavily_api_key))
            budget = info.client.count(wire, chat.settings, info.context, info.thinking_supported, cancel, tools=tools)
            turn_setup_start = time.monotonic()
            metrics.token_check_seconds = turn_setup_start - token_check_start
            metrics.budget_prompt = budget.prompt
            metrics.budget_estimated = budget.estimated
            counted(budget)
            if budget.remaining < 0:
                raise BudgetExceeded(budget)
            cancel.check()
            message_id = self.store.start_turn(chat.id, draft, attachments)
        except Exception as exc:
            metrics.stage = "stopped" if isinstance(exc, Cancelled) else "error"
            report(metrics)
            raise
        started(message_id)
        status("Generating", "Streaming response")
        content: list[str] = []
        reasoning: list[str] = []
        last_update = last_save = time.monotonic()
        agent_progress = "Streaming response"

        def progress(detail: str) -> None:
            nonlocal agent_progress, last_update
            if detail != agent_progress:
                updated("".join(content), "".join(reasoning))
                last_update = time.monotonic()
                agent_progress = detail
                status("Generating", detail)

        def receive(text: str, thoughts: str) -> None:
            nonlocal last_update, last_save
            content.append(text)
            reasoning.append(thoughts)
            if agent_progress == "Reading tool results" and (text.strip() or thoughts.strip()):
                progress("Streaming response")
            current = time.monotonic()
            if len(content) == 1 or current - last_update >= 0.07:
                updated("".join(content), "".join(reasoning))
                last_update = current
            if current - last_save >= 1.5:
                self.store.save_response(
                    message_id, "".join(content), "".join(reasoning), "streaming", performance=replace(
                        metrics, request_seconds=current - request_start,
                    ),
                )
                last_save = current

        keys = {
            "temperature": "temp",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
            "repeat_penalty": "penalty_repeat",
        }
        defaults = {}
        for setting, key in keys.items():
            value = model.metadata.get(f"general.sampling.{key}")
            if type(value) in (int, float) and getattr(chat.settings, setting) is None:
                try:
                    replace(chat.settings, **{setting: value})
                except ValueError:
                    LOG.warning("Ignoring invalid GGUF sampler default name=%s", setting)
                else:
                    defaults[setting] = value
        sampling = replace(chat.settings, **defaults)
        state, error, notice = "complete", "", ""
        metrics.turn_setup_seconds = time.monotonic() - turn_setup_start
        metrics.stage = "prompt"
        report(metrics)
        try:
            payload = chat_payload(wire, sampling, info.thinking_supported, info.samplers)
            if performance:
                payload.update(timings_per_token=True, return_progress=True)
            if tools:
                activity: list[ToolActivity] = []
                payload["tools"] = tools

                def tool_used(item: ToolActivity) -> None:
                    text, thoughts = "".join(content), "".join(reasoning)
                    activity.append(replace(
                        item, content_end=len(text.encode("utf-16-le")) // 2,
                        reasoning_end=len(thoughts.encode("utf-16-le")) // 2,
                    ))
                    self.store.save_tool_activity(message_id, activity, text, thoughts)
                    if tool_event:
                        tool_event(tuple(activity))

                def before_next_round() -> None:
                    progress("Reading tool results")
                    followup = info.client.count(
                        wire, sampling, info.context, info.thinking_supported, cancel, tools=tools,
                    )
                    metrics.budget_prompt = followup.prompt
                    metrics.budget_estimated = followup.estimated
                    counted(followup)
                    if followup.remaining < 0:
                        raise InferenceError("The tool results exceeded the loaded context. "
                                             "Try a narrower request or increase the context.")

                stream_agent(
                    info.client, workspace, wire, payload, cancel, metrics, receive,
                    report if performance else None, tool_used, progress, before_next_round,
                    tavily_api_key=tavily_api_key,
                )
            else:
                info.client.stream(payload, cancel, receive, metrics, report if performance else None)
        except Cancelled:
            state = "stopped"
        except Exception as exc:
            state, error = "error", str(exc)
            # API errors may include excerpts of the prompt. Only the exception type goes to logs.
            LOG.warning("Generation failed error_type=%s", type(exc).__name__)
        finally:
            text, thoughts = "".join(content), "".join(reasoning)
            if state == "complete" and not text.strip():
                state = "no_answer"
                if thoughts.strip() and metrics.finish_reason == "length":
                    notice = "The model reached the output token limit while thinking."
                elif thoughts.strip():
                    notice = "The model generated reasoning without a final answer."
                elif metrics.finish_reason == "length":
                    notice = "The model reached the output token limit before returning an answer."
                else:
                    notice = "The model returned an empty answer."
                if info.thinking_supported and sampling.thinking == "On":
                    notice += " Turn Thinking off or increase the output limit, then regenerate."
            speed = metrics.generation_per_second if state == "complete" else None
            metrics.stage = state
            elapsed = time.monotonic() - request_start
            metrics.request_seconds = elapsed
            if performance:
                metrics.record_sample(elapsed)
            self.store.save_response(message_id, text, thoughts, state, speed, metrics)
            updated(text, thoughts)
        return Generation(metrics, state, error, notice)

    def summarize(
        self, info: ServerInfo, chat: Chat, messages: list[Message], start: int, cancel: Cancellation
    ) -> None:
        settings = replace(InferenceSettings(), output=min(768, chat.settings.output), thinking="Off")
        prompt: Content = (
            "Summarize the conversation below for use as context in later turns. "
            "Keep relevant facts, decisions, and open questions. Return only the summary."
        )
        if chat.summary:
            prompt = join_content(prompt, "Previous summary\n" + chat.summary)
        for message in messages:
            if chat.active_from <= message.id < start and (message.content or message.attachments):
                turn = f"{message.role.title()}\n{message.prompt_text()}"
                prompt = join_content(prompt, image_content(turn, message.attachments, info.vision_supported))
        wire: list[WireMessage] = [{"role": "user", "content": prompt}]
        budget = info.client.count(wire, settings, info.context, info.thinking_supported, cancel)
        if budget.remaining < 0:
            raise InferenceError(
                "The selected turns are too large to summarize in one request. "
                "Choose fewer turns or increase the context."
            )
        text: list[str] = []
        info.client.stream(
            chat_payload(wire, settings, info.thinking_supported, info.samplers),
            cancel,
            lambda delta, _: text.append(delta),
        )
        summary = "".join(text).strip()
        if not summary:
            raise InferenceError("The model returned an empty summary. Active context was preserved.")
        cancel.check()
        self.store.set_context(chat.id, start, summary)
