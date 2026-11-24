"""Ordered SQLite migrations for the local application database."""

MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        """CREATE TABLE chats (
            id INTEGER PRIMARY KEY, title TEXT NOT NULL, model TEXT NOT NULL,
            config TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL
        )""",
        """CREATE TABLE messages (
            id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL, created TEXT NOT NULL, reasoning TEXT NOT NULL DEFAULT '',
            state TEXT NOT NULL DEFAULT 'complete'
        )""",
        "CREATE INDEX messages_by_chat ON messages(chat_id, id)",
    ),
    (
        "ALTER TABLE chats ADD COLUMN active_from INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE chats ADD COLUMN summary TEXT NOT NULL DEFAULT ''",
        "CREATE TABLE models (path TEXT PRIMARY KEY, record TEXT NOT NULL, present INTEGER NOT NULL)",
    ),
    ("ALTER TABLE messages ADD COLUMN generation_per_second REAL",),
    ("ALTER TABLE messages ADD COLUMN attachments TEXT NOT NULL DEFAULT '[]'",),
    ("ALTER TABLE messages ADD COLUMN performance TEXT",),
    (
        "ALTER TABLE chats ADD COLUMN workspace TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE messages ADD COLUMN tool_activity TEXT NOT NULL DEFAULT '[]'",
    ),
    ("ALTER TABLE chats ADD COLUMN pinned_at TEXT",),
)
