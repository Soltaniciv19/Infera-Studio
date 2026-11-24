"""Conversation, message, and attachment records."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from infera_studio.config.settings import InferenceSettings

from .metrics import Metrics


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Chat:
    id: int
    title: str
    model: str
    settings: InferenceSettings
    created: str
    updated: str
    active_from: int
    summary: str
    workspace: str = ""
    pinned_at: str | None = None


@dataclass(frozen=True)
class Attachment:
    name: str
    content: str
    media_type: str = ""

    def prompt_text(self) -> str:
        if self.media_type:
            return f"Attached image {json.dumps(self.name, ensure_ascii=False)}"
        return f"Attached file {json.dumps(self.name, ensure_ascii=False)}:\n{self.content}"


@dataclass(frozen=True)
class ToolActivity:
    name: str
    path: str
    ok: bool
    # UTF-16 offsets into the saved response and reasoning, for transcript rendering.
    content_end: int | None = None
    reasoning_end: int | None = None


@dataclass(frozen=True)
class Message:
    id: int
    role: str
    content: str
    created: str
    reasoning: str
    state: str
    generation_per_second: float | None = None
    attachments: tuple[Attachment, ...] = ()
    performance: Metrics | None = None
    tool_activity: tuple[ToolActivity, ...] = ()

    def prompt_text(self) -> str:
        return turn_text(self.content, self.attachments)


def turn_text(content: str, attachments: tuple[Attachment, ...]) -> str:
    return "\n\n".join(part for part in (content, *(file.prompt_text() for file in attachments)) if part)
