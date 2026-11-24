"""Search the public Hugging Face Hub and download complete GGUF variants."""

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPException, HTTPResponse
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from infera_studio.catalog.gguf import DOWNLOAD_STAGING_PREFIX, SHARD, read_header
from infera_studio.inference.client import Cancellation

HUB = "https://huggingface.co"
REPO_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,95}")
COMMIT = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
QUANT = re.compile(
    r"(?<![a-z0-9])((?:I?Q[1-8]|TQ[12]|BF16|FP?(?:8|16|32)|MXFP4|NVFP4)(?:_[a-z0-9]+)*)(?![a-z0-9])",
    re.IGNORECASE,
)
METADATA_LIMIT = 8 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class File:
    path: str
    size: int
    sha256: str = ""

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Variant:
    path: str
    quant: str
    files: tuple[File, ...]

    @property
    def size(self) -> int:
        return sum(file.size for file in self.files)

    def record(self) -> dict[str, str | int]:
        return {"path": self.path, "quant": self.quant, "size": self.size, "parts": len(self.files)}


def valid_repo(repo: Any) -> bool:
    return isinstance(repo, str) and 1 <= len(parts := repo.split("/")) <= 2 and all(
        REPO_PART.fullmatch(part) and not part.endswith((".", "-")) for part in parts
    )


def safe_file(path: Any) -> bool:
    if not isinstance(path, str) or not path.lower().endswith(".gguf") or len(path) > 2048:
        return False
    return all(
        part not in {"", ".", ".."} and len(part.encode("utf-8", "replace")) <= 255
        and not any(ord(char) < 32 or ord(char) == 127 or char == "\\" or 0xD800 <= ord(char) <= 0xDFFF
                    for char in part)
        for part in path.split("/")
    )


class HubRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: HTTPResponse, code: int, msg: str, headers: Any, newurl: str,
    ) -> Request | None:
        url = urlsplit(newurl)
        host = url.hostname or ""
        if (
            url.scheme != "https" or url.username is not None or url.port not in (None, 443)
            or not (host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co")))
        ):
            raise ValueError("Hugging Face redirected to an unexpected download host.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@contextmanager
def request(url: str, cancel: Cancellation) -> Iterator[HTTPResponse]:
    cancel.check()
    try:
        with build_opener(HubRedirect()).open(
            Request(url, headers={"User-Agent": "InferaStudio", "Accept-Encoding": "identity"}), timeout=12,
        ) as response:
            yield response
    except HTTPError as exc:
        exc.close()
        cancel.check()
        if exc.code in (401, 403):
            raise ValueError("Hugging Face denied access. Choose a publicly downloadable model.") from None
        if exc.code == 429:
            raise ValueError("Hugging Face is rate limiting requests. Try again shortly.") from None
        raise ValueError(f"Hugging Face returned HTTP {exc.code}.") from None
    except (URLError, HTTPException, ConnectionError, TimeoutError):
        cancel.check()
        raise ValueError("The connection to Hugging Face was interrupted. Try again.") from None


def get_json(url: str, cancel: Cancellation) -> Any:
    with request(url, cancel) as response:
        data = bytearray()
        while chunk := response.read1(64 * 1024):
            cancel.check()
            data.extend(chunk)
            if len(data) > METADATA_LIMIT:
                raise ValueError("The Hugging Face response is too large.")
    cancel.check()
    try:
        return json.loads(data)
    except (ValueError, RecursionError):
        raise ValueError("Hugging Face returned unreadable model information.") from None


def search(query: str, cancel: Cancellation) -> list[str]:
    if not isinstance(query, str) or len(query) > 200:
        raise ValueError("Enter a model name or repository ID up to 200 characters.")
    params = urlencode({
        "search": query.strip(), "filter": "gguf", "sort": "downloads", "direction": -1, "limit": 50,
        "expand[]": ["gated", "private", "disabled"],
    }, doseq=True)
    records = get_json(f"{HUB}/api/models?{params}", cancel)
    if not isinstance(records, list):
        raise ValueError("Hugging Face returned unreadable search results.")
    return list(dict.fromkeys(
        item["id"] for item in records
        if isinstance(item, dict) and valid_repo(item.get("id"))
        and not any(item.get(key) for key in ("gated", "private", "disabled"))
    ))


def quantization(path: str, stem: str) -> str:
    if Path(path).name.casefold().startswith("mmproj"):
        return "Vision projector"
    tags = QUANT.findall(stem) or QUANT.findall(path.rpartition("/")[0])
    return tags[-1].upper() if tags else "GGUF"


def collect_variants(siblings: list[Any]) -> list[Variant]:
    variants: list[Variant] = []
    split_files: dict[tuple[str, str, int], dict[int, File]] = {}
    seen: set[str] = set()
    for item in siblings:
        if not isinstance(item, dict) or not safe_file(path := item.get("rfilename")):
            continue
        size = item.get("size")
        if path in seen or type(size) is not int or not 24 <= size < 2**53:
            continue
        seen.add(path)
        lfs = item.get("lfs")
        digest = lfs.get("sha256") if isinstance(lfs, dict) else None
        file = File(path, size, digest if isinstance(digest, str) and SHA256.fullmatch(digest) else "")
        parent, _, name = path.rpartition("/")
        if match := SHARD.fullmatch(name):
            stem, index, count = match[1], int(match[2]), int(match[3])
            if 1 <= index <= count:
                split_files.setdefault((parent, stem, count), {})[index] = file
        else:
            variants.append(Variant(path, quantization(path, name[:-5]), (file,)))
    for (_, stem, count), parts in split_files.items():
        if len(parts) == count:
            files = tuple(parts[index] for index in range(1, count + 1))
            variants.append(Variant(files[0].path, quantization(files[0].path, stem), files))
    return sorted(variants, key=lambda variant: (variant.quant == "Vision projector", variant.size, variant.path))


def repository(repo: str, cancel: Cancellation, revision: str = "") -> tuple[str, list[Variant]]:
    if not valid_repo(repo):
        raise ValueError("Choose a valid Hugging Face repository.")
    if not isinstance(revision, str) or (revision and not COMMIT.fullmatch(revision)):
        raise ValueError("Choose a valid Hugging Face model revision.")
    url = f"{HUB}/api/models/{repo}"
    if revision:
        url += f"/revision/{revision}"
    details = get_json(url + "?blobs=true", cancel)
    if not isinstance(details, dict):
        raise ValueError("Hugging Face returned unreadable repository information.")
    if any(details.get(key) for key in ("gated", "private", "disabled")):
        raise ValueError("Choose a publicly downloadable Hugging Face model.")
    commit, siblings = details.get("sha"), details.get("siblings")
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit) or not isinstance(siblings, list):
        raise ValueError("Hugging Face returned incomplete repository information.")
    if revision and revision != commit:
        raise ValueError("Hugging Face returned a different model revision. Choose the model again.")
    cancel.check()
    return commit, collect_variants(siblings)


def variants(repo: str, cancel: Cancellation) -> dict[str, Any]:
    revision, options = repository(repo, cancel)
    return {"revision": revision, "variants": [variant.record() for variant in options]}


def transfer(file: File, url: str, output: Path, cancel: Cancellation, progress: Callable[[int], None]) -> None:
    digest = hashlib.sha256() if file.sha256 else None
    received = 0
    with request(url, cancel) as response, output.open("xb") as stream:
        while chunk := response.read1(CHUNK_SIZE):
            cancel.check()
            received += len(chunk)
            if received > file.size:
                raise ValueError(f"{file.name} exceeded its advertised size.")
            stream.write(chunk)
            if digest is not None:
                digest.update(chunk)
            progress(received)
        cancel.check()
        if received != file.size:
            raise ValueError(f"{file.name} ended before its advertised size. Try downloading again.")
        if digest is not None and digest.hexdigest() != file.sha256:
            raise ValueError(f"{file.name} failed its checksum. Try downloading again.")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        read_header(output)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{file.name} has an invalid GGUF header. {exc}") from exc


def download(
    repo: str, revision: str, path: str, folder: str, cancel: Cancellation,
    emit: Callable[[str, Any], None],
) -> str:
    if not isinstance(revision, str) or not COMMIT.fullmatch(revision) or not safe_file(path):
        raise ValueError("Choose a GGUF variant to download.")
    revision, options = repository(repo, cancel, revision)
    variant = next((option for option in options if option.path == path), None)
    if variant is None:
        raise ValueError("This GGUF variant is no longer available. Choose the model again.")
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("The model folder is unavailable. Choose an existing folder in Settings.")
    key = hashlib.sha256(f"{repo}/{path}".encode()).hexdigest()[:16]
    destination = f"{repo.replace('/', '--')[:96]}--{variant.quant.replace(' ', '-')[:32]}--{key}"
    # Anchor writes to the chosen directory, even if it is renamed during the transfer.
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        anchor = Path(f"/proc/self/fd/{directory}")
        if os.path.lexists(anchor / destination):
            raise ValueError("This variant already has a download folder in the selected model folder.")
        total = variant.size
        if shutil.disk_usage(anchor).free < total:
            raise ValueError("The selected model folder does not have enough free space for this variant.")
        completed = 0
        last_update = 0.0
        with tempfile.TemporaryDirectory(prefix=DOWNLOAD_STAGING_PREFIX, dir=anchor) as temporary:
            staging = Path(temporary)
            for index, file in enumerate(variant.files, 1):
                cancel.check()

                def progress(received: int) -> None:
                    nonlocal last_update
                    now = time.monotonic()
                    if not received or now - last_update >= 0.2:
                        emit("progress", {"received": completed + received, "total": total,
                                          "part": index, "parts": len(variant.files)})
                        last_update = now

                progress(0)
                url = f"{HUB}/{repo}/resolve/{revision}/{quote(file.path)}"
                transfer(file, url, staging / file.name, cancel, progress)
                completed += file.size
            cancel.check()
            if os.path.lexists(anchor / destination):
                raise ValueError("This variant already has a download folder in the selected model folder.")
            os.rename(staging, anchor / destination)
        return str(Path(os.readlink(anchor)) / destination)
    finally:
        os.close(directory)
