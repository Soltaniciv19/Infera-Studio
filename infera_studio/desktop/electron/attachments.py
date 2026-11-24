"""Validate untrusted attachment snapshots from the Electron renderer."""

import base64
import binascii
import re

from infera_studio.conversations.records import Attachment

MAX_ATTACHMENT_COUNT = 8
MAX_ATTACHMENT_BYTES = 256 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 512 * 1024
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 8 * 1024 * 1024
NON_TEXT = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")


def image_dimensions(data: bytes, media_type: str) -> tuple[int, int]:
    if media_type == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"):
        return int.from_bytes(data[16:20]), int.from_bytes(data[20:24])
    if media_type == "image/jpeg" and data.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 4 <= len(data) and data[offset] == 0xFF:
            marker = data[offset + 1]
            if marker == 0xFF:
                offset += 1
                continue
            if marker in (0xD9, 0xDA):
                break
            size = int.from_bytes(data[offset + 2:offset + 4])
            if size < 2 or offset + 2 + size > len(data):
                break
            if marker in (0xC0, 0xC1, 0xC2) and size >= 8:
                return int.from_bytes(data[offset + 7:offset + 9]), int.from_bytes(data[offset + 5:offset + 7])
            offset += 2 + size
    return 0, 0


def parse_attachments(values: list[dict[str, str]] | None) -> tuple[Attachment, ...]:
    if values is None:
        return ()
    if not isinstance(values, list) or len(values) > MAX_ATTACHMENT_COUNT + MAX_IMAGES:
        raise ValueError("Attach up to eight text files and four images per message.")
    files: list[Attachment] = []
    total = text_count = image_total = image_count = 0
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Invalid attachment.")
        name, content = value.get("name"), value.get("content")
        if (
            not isinstance(name, str) or not name or len(name) > 255 or name in {".", ".."}
            or "/" in name or NON_TEXT.search(name)
            or not isinstance(content, str)
        ):
            raise ValueError("Invalid attachment name or content.")
        media_type = value.get("media_type", "")
        if media_type != "":
            if media_type not in ("image/png", "image/jpeg"):
                raise ValueError("Vision accepts PNG and JPEG images.")
            image_count += 1
            if image_count > MAX_IMAGES:
                raise ValueError("Attach up to four images per message.")
            if len(content) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
                raise ValueError("Each image can be up to 4 MiB.")
            try:
                data = base64.b64decode(content, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Could not read the attached image.") from exc
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("Each image can be up to 4 MiB.")
            image_total += len(data)
            if image_total > MAX_TOTAL_IMAGE_BYTES:
                raise ValueError("Images can total up to 8 MiB per message.")
            width, height = image_dimensions(data, media_type)
            if not (0 < width <= 8192 and 0 < height <= 8192 and width * height <= 25_000_000):
                raise ValueError("Choose a valid PNG or JPEG up to 8192 pixels per side and 25 megapixels.")
            files.append(Attachment(name, content, media_type))
            continue
        text_count += 1
        if text_count > MAX_ATTACHMENT_COUNT:
            raise ValueError("Attach up to eight text files per message.")
        if NON_TEXT.search(content):
            raise ValueError("Attachments must be UTF-8 text files.")
        size = len(content.encode("utf-8"))
        if size > MAX_ATTACHMENT_BYTES:
            raise ValueError("Each attachment can be up to 256 KiB.")
        total += size
        if total > MAX_TOTAL_ATTACHMENT_BYTES:
            raise ValueError("Attachments can total up to 512 KiB per message.")
        files.append(Attachment(name, content))
    return tuple(files)
