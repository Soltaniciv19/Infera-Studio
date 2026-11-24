"""Transactional local conversation and catalog storage."""

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from infera_studio.catalog.gguf import Model
from infera_studio.config.settings import InferenceSettings
from infera_studio.conversations.metrics import Metrics
from infera_studio.conversations.records import Attachment, Chat, Message, ToolActivity, now

from .migrations import MIGRATIONS

PINNED_CHAT_LIMIT = 10


class Store:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=8, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA secure_delete = ON")
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> int:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        conn = sqlite3.connect(self.path, timeout=8)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        finally:
            conn.close()
        with self.connection(write=True) as conn:
            if conn.execute("PRAGMA quick_check").fetchall()[0][0] != "ok":
                raise ValueError("The conversation database failed its SQLite integrity check.")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > len(MIGRATIONS):
                raise ValueError("The conversation database belongs to a newer Infera Studio version.")
            for number, statements in enumerate(MIGRATIONS[version:], version + 1):
                for statement in statements:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {number}")
            changed = conn.execute("UPDATE messages SET state='interrupted' WHERE state='streaming'")
            return changed.rowcount

    @staticmethod
    def _chat(row: sqlite3.Row) -> Chat:
        values = dict(row)
        settings = InferenceSettings.from_dict(json.loads(values.pop("config")))
        return Chat(settings=settings, **values)

    def get(self, chat_id: int) -> tuple[Chat, list[Message]]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
            if row is None:
                raise ValueError("This conversation has been deleted.")
            messages = conn.execute(
                """SELECT id, role, content, created, reasoning, state, generation_per_second, attachments,
                performance, tool_activity
                FROM messages WHERE chat_id=? ORDER BY id""",
                (chat_id,),
            )
            turns = []
            for message in messages:
                values = dict(message)
                attachments = tuple(Attachment(**file) for file in json.loads(values.pop("attachments")))
                performance = values.pop("performance")
                activity = tuple(ToolActivity(**item) for item in json.loads(values.pop("tool_activity")))
                turns.append(Message(
                    **values, attachments=attachments, tool_activity=activity,
                    performance=Metrics(**json.loads(performance)) if performance is not None else None,
                ))
            return self._chat(row), turns

    def search(self, query: str = "") -> list[Chat]:
        pattern = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self.connection() as conn:
            rows = conn.execute(
                r"""SELECT * FROM chats c WHERE title LIKE ? ESCAPE '\'
                OR EXISTS (SELECT 1 FROM messages m WHERE m.chat_id=c.id AND m.content LIKE ? ESCAPE '\')
                ORDER BY updated DESC, id DESC""",
                (pattern, pattern),
            )
            return [self._chat(row) for row in rows]

    def create(self, model: str, settings: InferenceSettings, title: str = "New chat") -> int:
        timestamp = now()
        with self.connection(write=True) as conn:
            cursor = conn.execute(
                "INSERT INTO chats(title,model,config,created,updated) VALUES (?,?,?,?,?)",
                (title, model, json.dumps(asdict(settings)), timestamp, timestamp),
            )
            return int(cursor.lastrowid)

    def rename(self, chat_id: int, title: str) -> None:
        with self.connection(write=True) as conn:
            conn.execute("UPDATE chats SET title=?, updated=? WHERE id=?", (title, now(), chat_id))

    def set_pinned(self, chat_id: int, pinned: bool) -> str | None:
        with self.connection(write=True) as conn:
            chat = conn.execute("SELECT pinned_at FROM chats WHERE id=?", (chat_id,)).fetchone()
            if chat is None:
                raise ValueError("This conversation has been deleted.")
            pinned_at: str | None = chat["pinned_at"]
            if pinned == (pinned_at is not None):
                return pinned_at
            if pinned:
                count = conn.execute("SELECT count(*) FROM chats WHERE pinned_at IS NOT NULL").fetchone()[0]
                if count >= PINNED_CHAT_LIMIT:
                    raise ValueError(f"You can pin up to {PINNED_CHAT_LIMIT} chats. Unpin a chat to make room.")
            pinned_at = now() if pinned else None
            conn.execute("UPDATE chats SET pinned_at=? WHERE id=?", (pinned_at, chat_id))
            return pinned_at

    def configure(self, chat_id: int, model: str, settings: InferenceSettings) -> None:
        with self.connection(write=True) as conn:
            conn.execute(
                "UPDATE chats SET model=?, config=?, updated=? WHERE id=?",
                (model, json.dumps(asdict(settings)), now(), chat_id),
            )

    def delete(self, chat_id: int | None = None) -> None:
        with self.connection(write=True) as conn:
            if chat_id is not None:
                conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
            else:
                conn.execute("DELETE FROM chats")

    def set_workspace(self, chat_id: int, workspace: str) -> None:
        with self.connection(write=True) as conn:
            changed = conn.execute("UPDATE chats SET workspace=? WHERE id=?", (workspace, chat_id))
            if not changed.rowcount:
                raise ValueError("This conversation has been deleted.")

    def start_turn(self, chat_id: int, prompt: str | None, attachments: tuple[Attachment, ...] = ()) -> int:
        with self.connection(write=True) as conn:
            timestamp = now()
            if prompt is not None:
                title = prompt if prompt.strip() else (attachments[0].name if attachments else "New chat")
                first_line = " ".join(title.split())[:64] or "New chat"
                conn.execute(
                    """UPDATE chats SET title=? WHERE id=? AND title='New chat'
                    AND NOT EXISTS (SELECT 1 FROM messages WHERE chat_id=?)""",
                    (first_line, chat_id, chat_id),
                )
                conn.execute(
                    "INSERT INTO messages(chat_id,role,content,created,attachments) VALUES (?,'user',?,?,?)",
                    (chat_id, prompt, timestamp, json.dumps([asdict(file) for file in attachments])),
                )
            cursor = conn.execute(
                "INSERT INTO messages(chat_id,role,content,created,state) VALUES (?,'assistant','',?,'streaming')",
                (chat_id, timestamp),
            )
            conn.execute("UPDATE chats SET updated=? WHERE id=?", (timestamp, chat_id))
            return int(cursor.lastrowid)

    def save_response(
        self, message_id: int, content: str, reasoning: str, state: str,
        generation_per_second: float | None = None,
        performance: Metrics | None = None,
    ) -> None:
        with self.connection(write=True) as conn:
            conn.execute(
                """UPDATE messages SET content=?, reasoning=?, state=?, generation_per_second=?,
                performance=? WHERE id=?""",
                (content, reasoning, state, generation_per_second,
                 json.dumps(asdict(performance), allow_nan=False) if performance is not None else None, message_id),
            )

    def fork(self, chat_id: int, before: int) -> int:
        with self.connection(write=True) as conn:
            source = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
            if source is None:
                raise ValueError("This conversation has been deleted.")
            timestamp = now()
            new_id = int(
                conn.execute(
                    "INSERT INTO chats(title,model,config,created,updated) VALUES (?,?,?,?,?)",
                    (source["title"] + " · branch", source["model"], source["config"], timestamp, timestamp),
                ).lastrowid
            )
            copied_start = 0
            last_copied = 0
            for row in conn.execute(
                "SELECT * FROM messages WHERE chat_id=? AND id<? ORDER BY id", (chat_id, before)
            ):
                copied = conn.execute(
                    """INSERT INTO messages(chat_id,role,content,created,reasoning,state,
                    generation_per_second,attachments,performance,tool_activity) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        new_id, row["role"], row["content"], row["created"], row["reasoning"], row["state"],
                        row["generation_per_second"], row["attachments"], row["performance"], row["tool_activity"],
                    ),
                )
                last_copied = int(copied.lastrowid)
                if row["id"] == source["active_from"]:
                    copied_start = last_copied
            if before == source["active_from"]:
                # Editing the first retained turn will give the replacement a new message ID.
                copied_start = last_copied + 1
            if copied_start:
                conn.execute(
                    "UPDATE chats SET active_from=?, summary=? WHERE id=?",
                    (copied_start, source["summary"], new_id),
                )
            return new_id

    def save_tool_activity(
        self, message_id: int, activity: list[ToolActivity], content: str, reasoning: str,
    ) -> None:
        with self.connection(write=True) as conn:
            conn.execute(
                "UPDATE messages SET tool_activity=?, content=?, reasoning=? WHERE id=?",
                (json.dumps([asdict(item) for item in activity]), content, reasoning, message_id),
            )

    def set_context(self, chat_id: int, start: int, summary: str = "") -> None:
        with self.connection(write=True) as conn:
            if (
                start
                and not conn.execute(
                    "SELECT 1 FROM messages WHERE id=? AND chat_id=? AND role='user'",
                    (start, chat_id),
                ).fetchone()
            ):
                raise ValueError("Active context must start with a user message in this conversation.")
            conn.execute(
                "UPDATE chats SET active_from=?, summary=?, updated=? WHERE id=?",
                (start, summary if start else "", now(), chat_id),
            )

    def save_models(self, models: list[Model]) -> None:
        with self.connection(write=True) as conn:
            conn.execute("UPDATE models SET present=0")
            conn.executemany(
                """INSERT INTO models VALUES (?,?,1) ON CONFLICT(path)
                DO UPDATE SET record=excluded.record,present=1""",
                (
                    (
                        model.path,
                        json.dumps(model.record()),
                    )
                    for model in models
                ),
            )

    def catalog(self) -> dict[str, Model]:
        with self.connection() as conn:
            result = {}
            for row in conn.execute("SELECT * FROM models"):
                data = json.loads(row["record"])
                if not row["present"]:
                    data["issue"] = "Missing file or outside the current model directories."
                result[row["path"]] = Model.from_record(data)
            return result
