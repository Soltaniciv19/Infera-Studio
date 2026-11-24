"""Own dictation sessions and stream audio windows through local Whisper inference."""

import base64
import json
import math
import os
import select
import signal
import struct
import subprocess
import sys
import time
from array import array
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Condition

from infera_studio.inference.client import Cancellation, InferenceError

RATE = 16_000
BATCH_MS = 250
MAX_SECONDS = 300
BATCH_BYTES = RATE * BATCH_MS // 1000 * 4
MAX_AUDIO = MAX_SECONDS * RATE * 4
PREVIEW_INTERVAL = 4 * BATCH_BYTES
WINDOW = 12 * RATE * 4
OVERLAP = 3 * RATE * 4
MAX_TEXT = 32_000
TIMEOUT = 120
SIZE = struct.Struct("<I")
SEGMENT = struct.Struct("<III")


@dataclass(frozen=True)
class VoiceFiles:
    executable: Path
    model: Path
    source: str


class VoiceSession:
    def __init__(self, cancel: Cancellation) -> None:
        self.cancel = cancel
        self.changed = Condition()
        self.audio = bytearray()
        self.finished = False
        self.deadline = time.monotonic() + MAX_SECONDS + TIMEOUT

    def push(self, encoded: str) -> None:
        if not isinstance(encoded, str) or len(encoded) > 4 * ((BATCH_BYTES + 2) // 3):
            raise InferenceError("Invalid microphone audio batch.")
        try:
            samples = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise InferenceError("Invalid microphone audio batch.") from exc
        if not samples or len(samples) % 4 or len(samples) > BATCH_BYTES:
            raise InferenceError("Invalid microphone audio batch.")
        with self.changed:
            if not self.finished:
                self.audio.extend(samples[:MAX_AUDIO - len(self.audio)])
                self.changed.notify()

    def finish(self) -> None:
        with self.changed:
            self.finished = True
            self.changed.notify()

    def abort(self) -> None:
        self.cancel.cancel()
        self.finish()

    def snapshot(self, start: int, previous: int, limit: int) -> tuple[bytes, bool]:
        threshold = previous + PREVIEW_INTERVAL
        with self.changed:
            while True:
                self.cancel.check()
                if time.monotonic() >= self.deadline:
                    raise InferenceError("Transcription timed out. Try a shorter recording.")
                end = min(len(self.audio), limit)
                final = self.finished and end == len(self.audio)
                if final or (end > previous and end >= min(threshold, limit)):
                    return bytes(self.audio[start:end]), final
                self.changed.wait(0.2)


def to_pcm(raw: bytes) -> bytes:
    samples = array("f")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    pcm = array("h")
    peak = 0
    for sample in samples:
        value = round(max(-1.0, min(1.0, sample)) * 32767) if math.isfinite(sample) else 0
        pcm.append(value)
        peak = max(peak, abs(value))
    if len(pcm) < RATE // 4 or peak < 32:
        return b""
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm.tobytes()


@contextmanager
def whisper_process(args: list[str]) -> Iterator[subprocess.Popen[bytes]]:
    try:
        process = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise InferenceError("Could not start Whisper. Check Settings → Voice input.") from exc
    try:
        yield process
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()


class WhisperPipe:
    def __init__(self, process: subprocess.Popen[bytes], cancel: Cancellation) -> None:
        assert process.stdin is not None and process.stdout is not None
        self.process = process
        self.input = process.stdin.fileno()
        self.output = process.stdout.fileno()
        os.set_blocking(self.input, False)
        self.cancel = cancel
        self.deadline = time.monotonic() + TIMEOUT

    def check(self) -> None:
        self.cancel.check()
        if time.monotonic() >= self.deadline:
            raise InferenceError("Transcription timed out. Try a shorter recording or a smaller Whisper model.")

    def write(self, data: bytes) -> None:
        pending = memoryview(data)
        while pending:
            self.check()
            if select.select([], [self.input], [], 0.1)[1]:
                try:
                    pending = pending[os.write(self.input, pending):]
                except BlockingIOError:
                    continue
                except BrokenPipeError as exc:
                    raise InferenceError("Whisper closed before reading the recording.") from exc

    def read(self, size: int, *, until_eof: bool = False) -> bytes:
        result = bytearray()
        while len(result) < size:
            self.check()
            if select.select([self.output], [], [], 0.1)[0]:
                chunk = os.read(self.output, size - len(result))
                if not chunk:
                    if until_eof:
                        break
                    raise InferenceError("Whisper stopped unexpectedly. Check Settings → Voice input.")
                result.extend(chunk)
        return bytes(result)

    def segments(self, pcm: bytes) -> list[tuple[int, str]]:
        self.deadline = time.monotonic() + TIMEOUT
        self.write(SIZE.pack(len(pcm) // 2) + pcm)
        count, = SIZE.unpack(self.read(SIZE.size))
        if count > 512:
            raise InferenceError("Whisper returned too many transcript segments.")
        text_size = 0
        segments: list[tuple[int, str]] = []
        for _ in range(count):
            _, end, size = SEGMENT.unpack(self.read(SEGMENT.size))
            text_size += size
            if text_size > MAX_TEXT:
                raise InferenceError("Whisper returned too much text.")
            text = self.read(size).decode("utf-8", errors="replace")
            if text.strip():
                # Whisper timestamps are in 10 ms units. Float audio uses four bytes per sample.
                segments.append((min(end * RATE * 4 // 100, len(pcm) * 2), text))
        return segments


def supports_avx2_worker() -> bool:
    required = {"avx", "avx2", "bmi2", "f16c", "fma", "sse4_2"}
    try:
        flags = [line.partition(":")[2].split() for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.partition(":")[0].strip() == "flags"]
    except (OSError, UnicodeError):
        return False
    return bool(flags) and all(required.issubset(core) for core in flags)


class VoiceService:
    def __init__(self, settings: Path, bundle: Path) -> None:
        self.settings = settings
        self.bundle = bundle

    def configuration(self) -> VoiceFiles | None:
        try:
            custom = json.loads(self.settings.read_text(encoding="utf-8"))
            executable, model = Path(custom["executable"]), Path(custom["model"])
            if (executable.is_absolute() and model.is_absolute() and executable.is_file()
                    and model.is_file() and os.access(executable, os.X_OK) and os.access(model, os.R_OK)):
                return VoiceFiles(executable, model, "custom")
        except (OSError, UnicodeError, ValueError, TypeError, KeyError):
            pass

        executable, model = self.bundle / "infera-whisper-worker", self.bundle / "ggml-base-q5_1.bin"
        accelerated = self.bundle / "infera-whisper-worker-avx2"
        if accelerated.is_file() and os.access(accelerated, os.X_OK) and supports_avx2_worker():
            executable = accelerated
        if executable.is_file() and model.is_file() and os.access(executable, os.X_OK) and os.access(model, os.R_OK):
            return VoiceFiles(executable, model, "bundled")
        return None

    def transcribe(self, session: VoiceSession, preview: Callable[[str], None]) -> str:
        files = self.configuration()
        if files is None:
            raise InferenceError("Whisper files are unavailable. Open Settings → Voice input.")
        session.cancel.check()
        if files.source == "custom":
            return self.custom_transcript(files, session, preview)

        with whisper_process([str(files.executable), str(files.model)]) as process:
            pipe = WhisperPipe(process, session.cancel)
            if pipe.read(1) != b"R":
                raise InferenceError("Whisper could not load the bundled speech model.")

            prefix: list[str] = []
            start = previous = 0
            segments: list[tuple[int, str]] = []
            shown = ""
            while True:
                audio, final = session.snapshot(start, previous, start + WINDOW)
                end = start + len(audio)
                if end != previous:
                    pcm = to_pcm(audio)
                    segments = pipe.segments(pcm) if pcm else []
                text = "".join(prefix + [text for _, text in segments]).strip()
                if len(text) > MAX_TEXT:
                    raise InferenceError("Whisper returned too much text.")
                if final:
                    return text
                if text != shown:
                    preview(text)
                    shown = text

                previous = end
                if len(audio) == WINDOW:
                    cutoff = WINDOW - OVERLAP
                    stable = [(offset, text) for offset, text in segments if 0 < offset <= cutoff]
                    if not stable and segments:
                        # A sustained phrase may have no complete segment before the cutoff.
                        stable = segments[:1]
                    if stable:
                        prefix.extend(text for _, text in stable)
                    advance = stable[-1][0] if stable else cutoff
                    start += advance or cutoff
                    previous = start
                    segments = []

    @staticmethod
    def custom_transcript(files: VoiceFiles, session: VoiceSession, preview: Callable[[str], None]) -> str:
        # A user-supplied CLI only understands complete WAV inputs. Replay snapshots for previews.
        args = [str(files.executable), "-m", str(files.model), "-f", "-", "--output-file", "transcript",
                "--language", "auto", "--no-timestamps", "--no-gpu", "--suppress-nst"]
        previous = 0
        text = ""
        while True:
            audio, final = session.snapshot(0, previous, MAX_AUDIO)
            if len(audio) != previous:
                pcm = to_pcm(audio)
                if pcm:
                    header = struct.pack("<4sI8sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVEfmt ", 16,
                                         1, 1, RATE, RATE * 2, 2, 16, b"data", len(pcm))
                    with whisper_process(args) as process:
                        pipe = WhisperPipe(process, session.cancel)
                        pipe.write(header + pcm)
                        assert process.stdin is not None
                        process.stdin.close()
                        output = pipe.read(MAX_TEXT * 4 + 1, until_eof=True)
                        if len(output) > MAX_TEXT * 4:
                            raise InferenceError("Whisper returned too much text.")
                        while process.poll() is None:
                            pipe.check()
                            session.cancel.wait(0.05)
                        if process.returncode:
                            raise InferenceError("Whisper could not transcribe the recording. Check Voice input settings.")
                        text = output.decode("utf-8", errors="replace").strip()
                        if len(text) > MAX_TEXT:
                            raise InferenceError("Whisper returned too much text.")
                previous = len(audio)
                if not final:
                    preview(text)
            if final:
                return text
