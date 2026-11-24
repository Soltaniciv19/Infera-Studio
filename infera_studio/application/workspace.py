"""Bounded project file operations for explicitly enabled Agent conversations."""

import codecs
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from infera_studio.conversations.records import ToolActivity
from infera_studio.inference.client import Cancellation

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
READ_LIMIT = 512 * 1024
CREATE_LIMIT = 256 * 1024
TEXT_LIMIT = 6000
SEARCH_FILE_LIMIT = 64 * 1024
SEARCH_BYTES_LIMIT = 16 * 1024 * 1024
SEARCH_ENTRIES_LIMIT = 12_000
SEARCH_SKIPPED = {".git", ".hg", ".svn", ".venv", "node_modules", "__pycache__"}
VCS_DIRS = {".git", ".hg", ".svn"}
PATH = {"type": "string", "description": "A path relative to the selected project folder."}


def tool(name: str, description: str, properties: dict[str, Any], *required: str) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(required),
                       "additionalProperties": False},
    }}


TOOLS = [
    tool("read_file", "Read UTF-8 text from a project file. Returns line numbers and a next line to read.",
         {"path": PATH, "start_line": {"type": "integer", "minimum": 1},
          "max_lines": {"type": "integer", "minimum": 1, "maximum": 200}}, "path"),
    tool("create_file",
         "Create a new UTF-8 text file and any missing parent folders. Existing files are protected.",
         {"path": PATH, "content": {"type": "string", "description": "The entire new file, including newlines."}},
         "path", "content"),
    tool("search_files", "Find files by name or search UTF-8 file contents for literal text, ignoring case. "
         "Use an empty query with kind 'name' to list files.",
         {"query": {"type": "string"}, "path": PATH | {"description": "A project subfolder, defaulting to '.'."},
          "kind": {"type": "string", "enum": ["name", "content", "both"]}}, "query"),
]


def project_path(value: str) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("Choose an absolute project folder.")
    try:
        path = Path(value).resolve(strict=True)
        if not path.is_dir():
            raise ValueError("Choose a project folder.")
    except (OSError, RuntimeError) as exc:
        raise ValueError("The project folder is unavailable.") from exc
    if path == Path(path.anchor):
        raise ValueError("Choose a specific project folder.")
    return str(path)


def path_parts(path: str, allow_root: bool = False) -> tuple[str, ...]:
    if not isinstance(path, str):
        raise ValueError("Provide a project-relative path.")
    if allow_root and path == ".":
        return ()
    parts = tuple(path.split("/"))
    if (len(path.encode()) > 1024 or len(parts) > 24 or
            any(part in {"", ".", ".."} | VCS_DIRS or any(ord(char) < 32 for char in part)
                for part in parts) or parts[-1].lower().endswith(".gguf")):
        raise ValueError("Provide a valid project-relative path.")
    return parts


@contextmanager
def directory(root: int, parts: tuple[str, ...], *, create: bool = False) -> Iterator[int]:
    current = os.dup(root)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, dir_fd=current)
                except FileExistsError:
                    pass
            child = os.open(part, DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = child
        yield current
    finally:
        os.close(current)


@contextmanager
def open_workspace(path: str, cancel: Cancellation) -> Iterator["Workspace"]:
    selected = Path(path)
    if not selected.is_absolute() or len(selected.parts) < 2 or ".." in selected.parts:
        raise ValueError("Choose a specific project folder.")
    root = os.open("/", DIR_FLAGS)
    try:
        with directory(root, selected.parts[1:]) as fd:
            yield Workspace(fd, cancel)
    finally:
        os.close(root)


def read_text(parent: int, name: str, limit: int) -> tuple[str, bool, int]:
    # Inspect an inode before opening it for I/O, even if the directory entry changes.
    handle = os.open(name, os.O_PATH | os.O_NOFOLLOW, dir_fd=parent)
    try:
        details = os.fstat(handle)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise ValueError("Only regular files with a single hard link are supported.")
        with open(f"/proc/self/fd/{handle}", "rb") as file:
            data = file.read(limit + 1)
    finally:
        os.close(handle)
    truncated = len(data) > limit
    buffer = data[:limit]
    if b"\0" in buffer or buffer.startswith(b"GGUF"):
        raise ValueError("The file is binary.")
    text = codecs.getincrementaldecoder("utf-8")().decode(buffer, final=not truncated)
    if truncated and "\n" in text:
        text = text[:text.rfind("\n") + 1]
    return text, truncated, len(data)


class Workspace:
    def __init__(self, fd: int, cancel: Cancellation) -> None:
        self.fd = fd
        self.cancel = cancel

    def execute(self, name: str, arguments: str) -> tuple[dict[str, Any], ToolActivity]:
        self.cancel.check()
        operations = {"read_file": self.read, "create_file": self.create, "search_files": self.search}
        target = ""
        try:
            params = json.loads(arguments)
            if not isinstance(params, dict) or name not in operations:
                raise ValueError("Unknown file tool or invalid arguments.")
            value = params.get("path", ".")
            if isinstance(value, str):
                target = value[:250].encode("utf-8", "replace").decode()
            result = operations[name](**params)
        except UnicodeError:
            result = {"error": "File tools require valid UTF-8 text."}
        except (TypeError, RecursionError):
            result = {"error": "Invalid arguments for this file tool."}
        except ValueError as exc:
            result = {"error": str(exc)}
        except FileExistsError:
            result = {"error": "The file already exists."}
        except OSError as exc:
            result = {"error": exc.strerror or "The file operation failed."}
        return result, ToolActivity(name.encode("utf-8", "replace").decode(), target, "error" not in result)

    def read(self, path: str, start_line: int = 1, max_lines: int = 100) -> dict[str, Any]:
        parts = path_parts(path)
        if (type(start_line) is not int or start_line < 1 or
                type(max_lines) is not int or not 1 <= max_lines <= 200):
            raise ValueError("Use a start line of at least 1 and a line limit from 1 to 200.")
        with directory(self.fd, parts[:-1]) as fd:
            text, truncated, _ = read_text(fd, parts[-1], READ_LIMIT)
        lines = text.splitlines(keepends=True)
        selected: list[str] = []
        remaining = TEXT_LIMIT
        clipped = False
        for index in range(start_line - 1, min(len(lines), start_line - 1 + max_lines)):
            line = lines[index]
            if selected and len(line) > remaining:
                break
            clipped = len(line) > remaining
            selected.append(line[:remaining - 1] + "…" if clipped else line)
            remaining -= len(selected[-1])
        following = start_line + len(selected)
        return {"path": path, "start_line": start_line, "content": "".join(selected),
                "next_line": following if following <= len(lines) else None,
                "truncated_file": truncated, "truncated_line": clipped}

    def create(self, path: str, content: str) -> dict[str, Any]:
        parts = path_parts(path)
        if not isinstance(content, str) or "\0" in content:
            raise ValueError("Provide UTF-8 text to create a file.")
        data = content.encode()
        if len(data) > CREATE_LIMIT:
            raise ValueError("A new file can contain up to 256 KiB of UTF-8 text.")
        temp = f".infera-{uuid4().hex}.tmp"
        with directory(self.fd, parts[:-1], create=True) as fd:
            file = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=fd)
            try:
                with os.fdopen(file, "wb") as output:
                    output.write(data)
                self.cancel.check()
                os.link(temp, parts[-1], src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            finally:
                os.unlink(temp, dir_fd=fd)
        return {"path": path, "bytes_written": len(data)}

    def search(self, query: str, path: str = ".", kind: str = "both") -> dict[str, Any]:
        parts = path_parts(path, allow_root=True)
        if not isinstance(query, str) or len(query) > 250 or kind not in ("name", "content", "both"):
            raise ValueError("Provide search text up to 250 characters and a valid search kind.")
        query = query.casefold()
        if not query:
            kind = "name"
        results: list[dict[str, Any]] = []
        entries = scanned = result_size = 0
        truncated = False

        def record(match: dict[str, Any]) -> bool:
            nonlocal result_size
            result_size += len(json.dumps(match, ensure_ascii=False))
            if result_size > TEXT_LIMIT or len(results) >= 40:
                return False
            results.append(match)
            return True

        def walk(fd: int, relative: tuple[str, ...]) -> bool:
            nonlocal entries, scanned, truncated
            with os.scandir(fd) as children:
                for child in children:
                    self.cancel.check()
                    entries += 1
                    if entries > SEARCH_ENTRIES_LIMIT or scanned >= SEARCH_BYTES_LIMIT:
                        return False
                    name = child.name
                    if name in SEARCH_SKIPPED:
                        continue
                    child_path = "/".join((*relative, name))
                    try:
                        path_parts(child_path)
                        if child.is_dir(follow_symlinks=False):
                            if len(relative) < 23:
                                with directory(fd, (name,)) as nested:
                                    if not walk(nested, (*relative, name)):
                                        return False
                            else:
                                truncated = True
                            continue
                        if not child.is_file(follow_symlinks=False):
                            continue
                        if child.stat(follow_symlinks=False).st_nlink != 1:
                            continue
                        if kind != "content" and query in child_path.casefold():
                            if not record({"path": child_path}):
                                return False
                        if kind != "name":
                            limit = min(SEARCH_FILE_LIMIT, SEARCH_BYTES_LIMIT - scanned)
                            try:
                                text, partial, size = read_text(fd, name, limit)
                            except ValueError:
                                scanned += limit
                                continue
                            scanned += size
                            truncated |= partial
                            for line, value in enumerate(text.splitlines(), 1):
                                position = value.casefold().find(query)
                                if position >= 0:
                                    snippet = value[max(0, position - 80):max(0, position - 80) + 220]
                                    if not record({"path": child_path, "line": line, "text": snippet}):
                                        return False
                    except (OSError, ValueError):
                        continue
            return True

        with directory(self.fd, parts) as fd:
            complete = walk(fd, parts)
        return {"matches": results, "truncated": not complete or truncated}
