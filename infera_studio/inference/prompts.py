"""Construct llama.cpp requests from conversation turns and explicit settings."""

from typing import Any, Literal, NotRequired, TypedDict

from infera_studio.config.settings import InferenceSettings
from infera_studio.conversations.records import Attachment, Message, turn_text


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image_url"]
    image_url: dict[str, str]


type Content = str | list[TextPart | ImagePart]


class ToolFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolFunction


class WireMessage(TypedDict):
    role: str
    content: Content
    reasoning_content: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_id: NotRequired[str]


def content_parts(content: Content) -> list[TextPart | ImagePart]:
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


def image_content(text: str, attachments: tuple[Attachment, ...], vision: bool) -> Content:
    images: list[TextPart | ImagePart] = [
        {"type": "image_url", "image_url": {"url": f"data:{file.media_type};base64,{file.content}"}}
        for file in attachments if vision and file.media_type in ("image/png", "image/jpeg")
    ]
    return content_parts(text) + images if images else text


def join_content(first: Content, second: Content) -> Content:
    if isinstance(first, str) and isinstance(second, str):
        return first + "\n\n" + second
    return content_parts(first) + content_parts(second)


def request_messages(
    turns: list[Message],
    settings: InferenceSettings,
    active_from: int = 0,
    summary: str = "",
    draft: str = "",
    attachments: tuple[Attachment, ...] = (),
    *,
    vision: bool = False,
) -> list[WireMessage]:
    messages: list[WireMessage] = [
        {"role": message.role, "content": image_content(message.prompt_text(), message.attachments, vision)}
        for message in turns
        if message.id >= active_from and (message.content or message.attachments)
    ]
    if draft or attachments:
        messages.append({"role": "user", "content": image_content(turn_text(draft, attachments), attachments, vision)})
    compact: list[WireMessage] = []
    for message in messages:
        if compact and message["role"] == compact[-1]["role"]:
            compact[-1]["content"] = join_content(compact[-1]["content"], message["content"])
        else:
            compact.append(message)
    messages = compact
    if summary:
        for message in messages:
            if message["role"] == "user":
                message["content"] = join_content("Summary of earlier turns:\n" + summary, message["content"])
                break
    if settings.system.strip():
        messages.insert(0, {"role": "system", "content": settings.system})
    return messages


def template_args(settings: InferenceSettings, thinking_supported: bool) -> dict[str, Any]:
    if thinking_supported:
        return {"chat_template_kwargs": {"enable_thinking": settings.thinking == "On"}}
    return {}


def chat_payload(
    messages: list[WireMessage],
    settings: InferenceSettings,
    thinking_supported: bool = False,
    supported: frozenset[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": messages,
        "model": "infera-studio",
        "stream": True,
        "max_tokens": settings.output,
        "stream_options": {"include_usage": True},
    }
    for name in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "seed"):
        value = getattr(settings, name)
        if value is not None and (supported is None or name in supported):
            payload[name] = value
    return payload | template_args(settings, thinking_supported)
