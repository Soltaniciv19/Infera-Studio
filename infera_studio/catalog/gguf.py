"""Read-only GGUF discovery, metadata parsing and shard validation."""

import logging
import math
import os
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

LOG = logging.getLogger(__name__)
SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf", re.IGNORECASE)
DOWNLOAD_STAGING_PREFIX = ".infera-download-"
QUANTS = dict(
    enumerate(
        "F32 F16 Q4_0 Q4_1 ? ? ? Q8_0 Q5_0 Q5_1 Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M "
        "Q5_K_S Q5_K_M Q6_K IQ2_XXS IQ2_XS Q2_K_S IQ3_XS IQ3_XXS IQ1_S IQ4_NL IQ3_S IQ3_M "
        "IQ2_S IQ2_M IQ4_XS IQ1_M BF16 ? ? ? TQ1_0 TQ2_0 MXFP4_MOE NVFP4 Q1_0 Q2_0 F8_E4M3".split()
    )
)
FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
MTP_HEAD_ISSUE = "MTP draft head. Select it in Model settings for a compatible chat model."


@dataclass(frozen=True)
class Header:
    metadata: dict
    tensors: int
    parameters: int


class Reader:
    def __init__(self, stream: BinaryIO, size: int):
        self.stream = stream
        self.size = size
        self.endian = "<"

    def read(self, count: int, keep: bool = True) -> bytes:
        if count > self.size - self.stream.tell():
            raise ValueError("Truncated GGUF header.")
        if not keep:
            self.stream.seek(count, os.SEEK_CUR)
            return b""
        if count > 8 * 1024 * 1024:
            raise ValueError("GGUF metadata field is too large.")
        data = self.stream.read(count)
        if len(data) != count:
            raise ValueError("Truncated GGUF header.")
        return data

    def number(self, fmt: str) -> int | float | bool:
        return struct.unpack(self.endian + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, keep: bool = True) -> str:
        return self.read(int(self.number("Q")), keep).decode("utf-8", errors="replace")

    def skip_strings(self, count: int) -> None:
        # Tokenizer arrays can contain hundreds of thousands of strings. Track the
        # offset locally to avoid calling tell() for every length and payload.
        length = struct.Struct(self.endian + "Q")
        position = self.stream.tell()
        for _ in range(count):
            prefix = self.stream.read(length.size)
            if len(prefix) != length.size:
                raise ValueError("Truncated GGUF header.")
            size = length.unpack(prefix)[0]
            position += length.size + size
            if position > self.size:
                raise ValueError("Truncated GGUF header.")
            if size:
                self.stream.seek(size, os.SEEK_CUR)

    def value(self, kind: int, keep: bool) -> object:
        if kind in FORMATS:
            fmt = FORMATS[kind]
            if keep:
                return self.number(fmt)
            self.read(struct.calcsize(fmt), False)
            return None
        if kind == 8:
            return self.string(keep)
        if kind != 9:
            raise ValueError(f"Unknown GGUF metadata type {kind}.")
        item, count = int(self.number("I")), int(self.number("Q"))
        if count > 10_000_000 or item == 9:
            raise ValueError("Unsupported GGUF array.")
        if item in FORMATS and (not keep or count > 4096):
            self.read(struct.calcsize(FORMATS[item]) * count, False)
            return None
        if item == 8:
            self.skip_strings(count)
            return None
        if not keep or count > 4096:
            for _ in range(count):
                self.value(item, False)
            return None
        return [self.value(item, True) for _ in range(count)]


def read_header(path: Path) -> Header:
    with path.open("rb") as stream:
        reader = Reader(stream, os.fstat(stream.fileno()).st_size)
        if reader.read(4) != b"GGUF":
            raise ValueError("Invalid GGUF signature.")
        version = int(reader.number("I"))
        if version == 3 << 24:
            reader.endian, version = ">", 3
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}.")
        tensors, count = int(reader.number("Q")), int(reader.number("Q"))
        if tensors > 1_000_000 or count > 100_000:
            raise ValueError("GGUF header exceeds supported metadata limits.")
        metadata = {}
        for _ in range(count):
            key, kind = reader.string(), int(reader.number("I"))
            keep = (
                not key.startswith("tokenizer.") or key == "tokenizer.chat_template"
            ) and not key.startswith(("general.license", "general.description"))
            value = reader.value(kind, keep)
            if keep and value is not None:
                metadata[key] = value
        parameters = 0
        for _ in range(tensors):
            reader.string(False)
            dimensions = int(reader.number("I"))
            if not 1 <= dimensions <= 4:
                raise ValueError("Invalid GGUF tensor rank.")
            shape = [int(reader.number("Q")) for _ in range(dimensions)]
            if any(not 0 < dimension < 2**40 for dimension in shape):
                raise ValueError("Invalid GGUF tensor dimension.")
            parameters += math.prod(shape)
            reader.read(12, False)  # Tensor type and offset.
    return Header(metadata, tensors, parameters)


@dataclass(frozen=True)
class Model:
    path: str
    name: str
    architecture: str
    quant: str
    parameters: int
    context: int
    size: int
    shards: tuple[str, ...]
    metadata: dict
    issue: str = ""

    @property
    def available(self) -> bool:
        return not self.issue and all(Path(name).is_file() for name in self.shards)

    @property
    def has_thinking_switch(self) -> bool:
        template = self.metadata.get("tokenizer.chat_template")
        return isinstance(template, str) and "enable_thinking" in template

    @property
    def has_mtp_layers(self) -> bool:
        layers = self.metadata.get(f"{self.architecture}.nextn_predict_layers")
        return type(layers) is int and layers > 0

    @property
    def is_vision_projector(self) -> bool:
        return (
            self.architecture in {"clip", "mmproj"}
            and self.metadata.get("clip.has_vision_encoder") is True
            and self.parameters > 0
            and len(self.shards) == 1
            and Path(self.path).is_file()
        )

    def is_mtp_head_for(self, target: "Model") -> bool:
        if self.issue != MTP_HEAD_ISSUE or not all(Path(name).is_file() for name in self.shards):
            return False
        if self.architecture == "gemma4-assistant":
            output_size = self.metadata.get("gemma4-assistant.embedding_length_out")
            target_size = target.metadata.get("gemma4.embedding_length")
            return target.architecture == "gemma4" and (
                output_size is None or target_size is None or output_size == target_size
            )
        return self.architecture == target.architecture

    def record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, data: dict) -> "Model":
        return cls(**{**data, "shards": tuple(data["shards"])})


def model_from_files(files: tuple[Path, ...], expected: int) -> Model:
    first = files[0]
    names = tuple(map(str, files))
    size = 0
    meta: dict = {}
    parameters = 0
    issue = ""
    try:
        size = sum(file.stat().st_size for file in files)
        headers = tuple(read_header(file) for file in files)
        meta = headers[0].metadata
        parameters = sum(header.parameters for header in headers)
        if expected > 1 or meta.get("split.count", 1) > 1:
            if (
                len(files) != expected
                or any(
                    (match := SHARD.fullmatch(file.name)) is None
                    or int(match[2]) != index + 1
                    or header.metadata.get("split.no", index) != index
                    or header.metadata.get("split.count", expected) != expected
                    for index, (file, header) in enumerate(zip(files, headers))
                )
                or expected != meta.get("split.count", expected)
            ):
                issue = f"Incomplete or inconsistent GGUF shards ({len(files)} of {max(expected, meta.get('split.count', 1))})."
        arch = meta.get("general.architecture", "")
        if not issue:
            if not parameters:
                issue = "Vocabulary-only GGUF. No model weights are present."
            elif arch == "gemma4-assistant" or (
                first.name.lower().startswith("mtp-")
                and type(layers := meta.get(f"{arch}.nextn_predict_layers")) is int
                and layers > 0
            ):
                issue = MTP_HEAD_ISSUE
            elif (
                not arch
                or arch in {"clip", "mmproj", "whisper", "wavtokenizer-dec", "bert", "nomic-bert"}
                or (
                    str(meta.get("general.type", "")).lower() not in {"", "model"}
                    or first.name.lower().startswith(("mmproj", "mtp-"))
                )
            ):
                issue = "This GGUF is not a standalone text chat model."
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        LOG.warning("GGUF scan failed for %s: %s", first, exc)
        issue = f"Unable to inspect GGUF ({exc})."
    arch = str(meta.get("general.architecture", ""))
    ctx = meta.get(f"{arch}.context_length")
    quant_number = meta.get("general.file_type")
    quant_match = re.search(r"\b((?:I?Q[1-8]|BF16|F16|F32)[\w]*)", first.stem, re.IGNORECASE)
    quant = QUANTS.get(quant_number, "?") if isinstance(quant_number, int) else "?"
    if quant == "?" and quant_match:
        quant = quant_match[1].upper()
    path = first
    if (match := SHARD.fullmatch(first.name)) and int(match[2]) != 1:
        path = first.with_name(f"{match[1]}-00001-of-{match[3]}{first.suffix}")
    return Model(
        str(path),
        str(meta.get("general.name") or first.stem),
        arch,
        quant,
        parameters,
        ctx if isinstance(ctx, int) and ctx > 0 else 0,
        size,
        names,
        meta,
        issue,
    )


def scan(directories: tuple[str, ...]) -> tuple[list[Model], list[str]]:
    found: set[Path] = set()
    errors: list[str] = []

    def failed(exc: OSError) -> None:
        errors.append(str(exc))

    for name in directories:
        root = Path(name).expanduser()
        if not root.is_dir():
            errors.append(f"Directory unavailable: {root}")
            continue
        for parent, folders, files in os.walk(root, onerror=failed):
            folders[:] = sorted(name for name in folders if not name.startswith(DOWNLOAD_STAGING_PREFIX))
            for file in files:
                if not file.lower().endswith(".gguf"):
                    continue
                try:
                    path = (Path(parent) / file).resolve(strict=True)
                    if path.is_file():
                        found.add(path)
                except (OSError, RuntimeError) as exc:
                    errors.append(str(exc))
    groups: dict[tuple[Path, str, int], list[Path]] = {}
    models = []
    for file in sorted(found):
        match = SHARD.fullmatch(file.name)
        if match:
            groups.setdefault((file.parent, match[1], int(match[3])), []).append(file)
        else:
            models.append(model_from_files((file,), 1))
    models.extend(model_from_files(tuple(files), count) for (_, _, count), files in groups.items())
    return sorted(models, key=lambda item: (item.name.casefold(), item.path)), errors
