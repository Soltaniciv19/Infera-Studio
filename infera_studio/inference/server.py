"""Detect compatible executables and supervise a private llama.cpp instance."""

import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from infera_studio.catalog.gguf import Model
from infera_studio.config.preferences import Preferences
from infera_studio.config.settings import BACKENDS, InferenceSettings
from infera_studio.host.hardware import GIB, MIB, amd_gpus

from .client import Cancellation, Client, HttpError, InferenceError
from .memory import QWEN_HYBRIDS, MemoryEstimate, recommend
from .telemetry import number

LOG = logging.getLogger(__name__)
DEVICE_PREFIX = {"Vulkan": "Vulkan", "HIP": "ROCm"}
ALLOC = re.compile(
    r"\b(Vulkan\d*|ROCm\d*|HIP\d*)\s+(model|KV|RS|compute) buffer size\s*=\s*(\d+(?:\.\d+)?) MiB"
)
KV_ALLOC = re.compile(
    r"^(?:\d+(?:\.\d+)+\s+)?(?:I\s+)?llama_kv_cache(?:_unified)?:\s+\S+\s+"
    r"KV buffer size\s*=\s*(\d+(?:\.\d+)?) MiB\b"
)
SPLIT_PROGRESS = re.compile(r"offloaded (\d+)/(\d+) layers")
SAMPLERS = frozenset(("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "seed"))


def environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("LLAMA_ARG_", "LLAMA_CACHE", "LLAMA_LOG"))
    }


def executable(value: str) -> Path | None:
    path = Path(value).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return path.absolute()
    return None


def detect(prefs: Preferences) -> dict[str, Path | None]:
    root = Path.home() / "AI/lama-cpp/llama.cpp"
    binaries = prefs.binaries or {}
    detected = {}
    for backend in BACKENDS:
        configured = binaries.get(backend)
        if configured:
            detected[backend] = executable(configured)
            continue
        double = backend.lower()
        candidates = [shutil.which(f"llama-server-{double}"), root / f"build-{double}/bin/llama-server"]
        if backend == "CPU":
            candidates.extend(
                [
                    shutil.which("llama-server"),
                    root / "build/bin/llama-server",
                    root / "build-vulkan/bin/llama-server",
                    root / "build-hip/bin/llama-server",
                ]
            )
        else:
            candidates.append(shutil.which("llama-server"))
        detected[backend] = next(
            (path for candidate in candidates if candidate and (path := executable(str(candidate)))), None
        )
    return detected


@dataclass(frozen=True)
class Capabilities:
    flags: frozenset[str]
    modern_flash: bool
    devices: tuple[str, ...]
    mtp: bool = False
    core_logs_at_trace: bool = False

    def has(self, flag: str) -> bool:
        return flag in self.flags


def inspect_binary(path: Path, backend: str) -> Capabilities:
    def capture(flag: str) -> str:
        try:
            result = subprocess.run(
                [str(path), flag],
                cwd=path.parent,
                env=environment(),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InferenceError(f"Could not inspect {backend} llama-server ({exc}).") from exc
        if result.returncode:
            failure = (result.stderr or result.stdout).strip().splitlines()
            detail = failure[-1] if failure else f"exit code {result.returncode}"
            raise InferenceError(f"{backend} llama-server did not start ({detail}).")
        return result.stdout + result.stderr

    help_text = capture("--help")
    flags = frozenset(re.findall(r"(?<![\w-])--[a-z][\w-]*", help_text))
    needed = {"--api-key-file", "--host", "--port", "--parallel", "--no-context-shift", "--alias"}
    if not needed <= flags:
        raise InferenceError(
            f"{backend} llama-server lacks required options: {', '.join(sorted(needed - flags))}. "
            "Select a compatible build in Settings."
        )
    devices = ()
    if backend != "CPU":
        if "--list-devices" not in flags or "--device" not in flags:
            raise InferenceError("This build cannot verify GPU devices. Choose a newer llama.cpp build.")
        prefix = DEVICE_PREFIX[backend]
        devices = tuple(re.findall(rf"^\s*({prefix}\d+):", capture("--list-devices"), re.MULTILINE))
        if not devices:
            raise InferenceError(
                f"This binary does not report an available {backend} device. "
                "Check the driver and executable in Settings."
            )
    return Capabilities(
        flags, bool(re.search(r"--flash-attn\s+\[on\|off\|auto\]", help_text)), devices,
        bool(re.search(r"(?m)^\s*--spec-type\s+[^\r\n]*\bdraft-mtp\b", help_text)),
        bool(re.search(r"(?m)^\s*-\s*4:\s+trace\b", help_text)),
    )


def command(
    path: Path,
    model: Model,
    settings: InferenceSettings,
    caps: Capabilities,
    plan: MemoryEstimate,
    port: int,
    key_fd: int,
    mtp_args: tuple[str, ...] = (),
    projector: str = "",
) -> list[str]:
    args = [
        str(path),
        "-m",
        model.path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--parallel",
        "1",
        "--ctx-size",
        str(plan.context),
        "--alias",
        "infera-studio",
        "--no-context-shift",
        "--api-key-file",
        f"/proc/self/fd/{key_fd}",
        *mtp_args,
    ]
    ui_flag = "--no-ui" if caps.has("--no-ui") else "--no-webui"
    for flag in (ui_flag, "--no-ui-mcp-proxy", "--jinja", "--no-prefill-assistant", "--metrics"):
        if caps.has(flag):
            args.append(flag)
    if projector:
        args.extend(("--mmproj", projector))
        if (settings.backend == "CPU" or settings.gpu_layers == 0) and caps.has("--no-mmproj-offload"):
            args.append("--no-mmproj-offload")
    elif caps.has("--no-mmproj"):
        args.append("--no-mmproj")
    if caps.has("--log-colors"):
        args.extend(("--log-colors", "off"))
    if caps.core_logs_at_trace and caps.has("--log-verbosity"):
        args.extend(("--log-verbosity", "4"))
    if caps.has("--reasoning-format"):
        args.extend(("--reasoning-format", "deepseek"))
    if settings.backend == "CPU" or settings.gpu_layers == 0:
        args.extend(("--gpu-layers", "0"))
        if caps.has("--device"):
            args.extend(("--device", "none"))
    else:
        args.extend(("--device", ",".join(caps.devices)))
        if settings.gpu_layers >= 0:
            args.extend(("--gpu-layers", str(settings.gpu_layers)))
        elif not caps.has("--fit"):
            if plan.headroom is None or plan.headroom < 512 * MIB:
                raise InferenceError(
                    "This build has no automatic memory fitter. Select CPU or set GPU layers manually."
                )
            args.extend(("--gpu-layers", "999"))
        else:
            args.extend(("--fit", "on"))
    flash = settings.flash
    if (
        flash == "auto"
        and settings.backend == "HIP"
        and plan.gpu
        and plan.gpu.pci_id == "0x73bf"
        and model.architecture in QWEN_HYBRIDS
        and caps.modern_flash
    ):
        # RDNA2 HIP Flash Attention can abort during the first Qwen3.5 decode.
        flash = "off"
        LOG.info("Using compatibility Flash Attention setting for HIP and %s", model.architecture)
    if caps.has("--flash-attn") and (caps.modern_flash or flash == "on"):
        if flash != "auto":
            args.append("--flash-attn")
            if caps.modern_flash:
                args.append(flash)
    elif flash != "auto":
        raise InferenceError("This build does not support the requested Flash Attention setting.")
    for cache, kind in ((settings.cache_k, "k"), (settings.cache_v, "v")):
        if cache == "f16":
            continue
        flag = f"--cache-type-{kind}"
        if not caps.has(flag):
            raise InferenceError(f"This build does not support configurable cache type {kind.upper()}.")
        args.extend((flag, cache))
    return args


@dataclass(frozen=True)
class ServerInfo:
    client: Client
    context: int
    plan: MemoryEstimate
    thinking_supported: bool
    samplers: frozenset[str]
    allocations: tuple[tuple[str, str, float], ...]
    observed_vram: int | None = None
    mtp: str = ""
    mtp_supported: bool = False
    mtp_enabled: bool = False
    vision_supported: bool = False
    kv_allocated_mib: float | None = None
    metrics_supported: bool = False


class ServerManager:
    def __init__(self, startup_timeout: float = 180):
        self.timeout = startup_timeout
        self.child: subprocess.Popen | None = None
        self.info: ServerInfo | None = None
        self.signature: tuple | None = None
        self.lines: deque[str] = deque(maxlen=110)
        self.allocations: dict[tuple[str, str], float] = {}
        self._kv_allocated_mib: float | None = None
        self._caps: dict[tuple[Path, str, int], Capabilities] = {}
        self._reader: threading.Thread | None = None
        self._ready = threading.Event()
        self._spec_failed = threading.Event()

    def stop(self) -> None:
        self.info = self.signature = None
        child, self.child = self.child, None
        if child is None:
            return
        if child.stdin is not None:
            child.stdin.close()
        try:
            child.wait(timeout=7)
        except subprocess.TimeoutExpired:
            # The guardian normally exits after its own four-second stop deadline.
            LOG.error("Inference guardian exceeded its shutdown deadline")
            child.wait(timeout=5)
        if self._reader is not None:
            self._reader.join(timeout=2)
            self._reader = None
        LOG.info("Inference process stopped")

    def diagnostics(self) -> str:
        return "\n".join(self.lines.copy())

    def _read(self, stream) -> None:
        with stream:
            for line in stream:
                clean = line.strip()
                if not clean:
                    continue
                if not self._ready.is_set():
                    # GGUF metadata values can contain arbitrary text. Keep only operational lines.
                    if not any(
                        marker in clean
                        for marker in ("llama_model_loader:", "print_info:", "- kv ", "- type ")
                    ):
                        self.lines.append(clean[:600])
                        if (
                            "speculative decoding not supported by this context" in clean
                            or "failed to initialize speculative decoding context" in clean
                        ):
                            self._spec_failed.set()
                    if match := ALLOC.search(clean):
                        device, buffer, size = match.groups()
                        if size_mib := number(float(size)):
                            self.allocations[(device, buffer)] = size_mib
                    if match := KV_ALLOC.match(clean):
                        if size_mib := number(float(match[1])):
                            self._kv_allocated_mib = (self._kv_allocated_mib or 0.0) + size_mib
                elif re.search(
                    r"out of (?:device )?memory|failed to allocate|segmentation fault|GGML_ASSERT",
                    clean,
                    re.I,
                ):
                    self.lines.append(clean[:600])

    def _detail(self) -> str:
        lines = list(self.lines.copy())
        if not lines:
            return "Waiting for llama.cpp"
        if any("warmup" in line.lower() or "warm up" in line.lower() for line in lines[-8:]):
            return "Warming up the model"
        for line in reversed(lines):
            if match := SPLIT_PROGRESS.search(line):
                return f"Offloaded {match[1]} of {match[2]} model layers"
        if any("load_tensors" in line for line in lines[-12:]):
            return "Loading model weights"
        return "Initializing llama.cpp"

    def load(
        self,
        model: Model,
        settings: InferenceSettings,
        prefs: Preferences,
        cancel: Cancellation,
        status: Callable[[str, str], None],
        *,
        mtp_draft: Model | None = None,
        vision_projector: Model | None = None,
    ) -> ServerInfo:
        path = detect(prefs).get(settings.backend)
        if not model.available:
            self.stop()
            raise InferenceError(model.issue or "Model files are missing. Rescan or choose a replacement.")
        if path is None:
            self.stop()
            raise InferenceError(
                f"No usable {settings.backend} llama-server executable. Open Settings to configure one."
            )
        try:
            stamps = tuple((s.st_mtime_ns, s.st_size) for s in (Path(name).stat() for name in model.shards))
            safe_path = path.resolve()
            mtime = safe_path.stat().st_mtime_ns
        except OSError as exc:
            self.stop()
            raise InferenceError("An inference file moved during loading. Rescan and try again.") from exc
        draft_stamps = ()
        if mtp_draft and mtp_draft.is_mtp_head_for(model):
            try:
                draft_stamps = tuple(
                    (name, stamp.st_mtime_ns, stamp.st_size)
                    for name, stamp in ((name, Path(name).stat()) for name in mtp_draft.shards)
                )
            except OSError:
                mtp_draft = None
        else:
            mtp_draft = None
        projector = ""
        projector_stamp: tuple[int, int] | None = None
        if vision_projector and vision_projector.path == settings.vision_projector and vision_projector.is_vision_projector:
            try:
                stat = Path(vision_projector.path).stat()
                projector, projector_stamp = vision_projector.path, (stat.st_mtime_ns, stat.st_size)
            except OSError:
                pass
        signature = (
            model.path, stamps, safe_path, mtime, draft_stamps, projector_stamp, settings.launch_signature()
        )
        if signature == self.signature and self.info and self.child and self.child.poll() is None:
            return self.info
        self.stop()
        cancel.check()
        status("Loading", "Checking the inference executable")
        self.lines.clear()
        caps_key = (safe_path, settings.backend, mtime)
        caps = self._caps.get(caps_key)
        if caps is None:
            caps = inspect_binary(path, settings.backend)
            self._caps[caps_key] = caps
        if projector and not caps.has("--mmproj"):
            raise InferenceError("This llama-server build does not support vision projectors.")
        cancel.check()
        gpu = max(amd_gpus(), key=lambda device: device.free, default=None)
        plan = recommend(model, settings, gpu)
        if plan.gpu and plan.gpu.free < GIB:
            raise InferenceError("Less than 1 GiB of GPU memory is available. Free GPU memory or select CPU.")
        if (
            settings.backend != "CPU"
            and settings.gpu_layers > 0
            and gpu
            and plan.required
            and plan.required > gpu.free
        ):
            raise InferenceError(
                "Estimated GPU allocation exceeds available memory. Reduce GPU layers or context size."
            )
        if (
            not settings.context
            and settings.backend != "CPU"
            and settings.gpu_layers == -1
            and not caps.has("--fit")
            and gpu is None
        ):
            raise InferenceError("GPU memory capacity is unknown. Set GPU layers manually or select CPU.")
        if settings.backend != "CPU" and settings.gpu_layers == -1 and caps.has("--fit") and gpu:
            if plan.kv is not None and plan.kv + plan.recurrent + GIB > gpu.free:
                raise InferenceError(
                    "The cache estimate exceeds available GPU memory. Choose a smaller context or CPU."
                )
        mtp = ""
        mtp_args: tuple[str, ...] = ()
        draft_flag = next((flag for flag in ("--spec-draft-model", "--model-draft") if caps.has(flag)), None)
        mtp_supported = bool(caps.mtp and (bool(draft_flag) if mtp_draft else model.has_mtp_layers))
        if settings.mtp == "Off":
            mtp = "MTP off"
        elif settings.mtp_draft and not mtp_draft:
            mtp = "MTP draft file unavailable"
        if settings.mtp != "Off" and (model.has_mtp_layers or mtp_draft):
            if not mtp_supported:
                mtp = "MTP unavailable in this llama-server build"
            else:
                mtp_args = ("--spec-type", "draft-mtp")
                if mtp_draft and draft_flag:
                    mtp_args += (draft_flag, mtp_draft.path)
                    mtp = "MTP enabled with the selected draft head"
                else:
                    mtp = "MTP enabled with the embedded head"
        try:
            info = self._launch(path, model, settings, caps, plan, cancel, status, mtp_args, projector)
        except InferenceError as exc:
            cancel.check()
            if not mtp_args:
                raise
            self.lines.append(f"MTP startup failed: {exc}")
            status("Loading", "Retrying without MTP")
            info = self._launch(path, model, settings, caps, plan, cancel, status, (), projector)
            mtp_args = ()
            mtp = "MTP failed to initialize. The server is running with standard decoding."
        self.info = replace(info, mtp=mtp, mtp_supported=mtp_supported, mtp_enabled=bool(mtp_args))
        self.signature = signature
        LOG.info("Inference ready backend=%s context=%d mtp=%s", settings.backend, info.context, mtp)
        return self.info

    def _launch(
        self,
        path: Path,
        model: Model,
        settings: InferenceSettings,
        caps: Capabilities,
        plan: MemoryEstimate,
        cancel: Cancellation,
        status: Callable[[str, str], None],
        mtp_args: tuple[str, ...],
        projector: str,
    ) -> ServerInfo:
        cancel.check()
        self.allocations.clear()
        self._kv_allocated_mib = None
        self._ready.clear()
        self._spec_failed.clear()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        key = secrets.token_urlsafe(36)
        fd = os.memfd_create("infera-studio-api-key", os.MFD_CLOEXEC)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, (key + "\n").encode())
            os.lseek(fd, 0, os.SEEK_SET)
            args = command(path, model, settings, caps, plan, port, fd, mtp_args, projector)
            self.child = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("guardian.py")), str(fd), *args],
                cwd=path.parent,
                env=environment(),
                pass_fds=(fd,),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,
            )
        except OSError as exc:
            raise InferenceError(f"Unable to launch llama-server ({exc}).") from exc
        finally:
            os.close(fd)
        assert self.child.stdout is not None
        self._reader = threading.Thread(target=self._read, args=(self.child.stdout,), daemon=True)
        self._reader.start()
        client = Client(port, key)
        try:
            props = self._ready_props(client, cancel, status)
            if Path(props.get("model_path", "")).resolve() != Path(model.path).resolve():
                raise InferenceError(
                    "The local server reported an unexpected model. Retry to select another port."
                )
            defaults = props.get("default_generation_settings") or {}
            context = defaults.get("n_ctx")
            if not isinstance(context, int) or context <= 0:
                raise InferenceError("llama.cpp did not report the loaded slot context.")
            if mtp_args and self._spec_failed.is_set():
                raise InferenceError("llama.cpp could not initialize speculative decoding.")
            params = defaults.get("params")
            samplers = SAMPLERS & params.keys() if isinstance(params, dict) else SAMPLERS - {"min_p"}
            self._ready.set()
            allocations = tuple(
                (device, kind, value) for (device, kind), value in self.allocations.copy().items()
            )
            template = props.get("chat_template", "")
            thinking_supported = (
                isinstance(template, str) and "enable_thinking" in template
            ) or model.has_thinking_switch
            modalities = props.get("modalities")
            vision_supported = bool(projector and isinstance(modalities, dict) and modalities.get("vision") is True)
            if projector and not vision_supported:
                raise InferenceError("llama-server did not report vision support with the selected projector.")
            measured = next(
                (
                    device.used - plan.gpu.used
                    for device in amd_gpus()
                    if plan.gpu and device.device == plan.gpu.device
                ),
                None,
            )
            return ServerInfo(
                client, context, plan, thinking_supported, frozenset(samplers), allocations, measured,
                vision_supported=vision_supported, kv_allocated_mib=self._kv_allocated_mib,
                metrics_supported=caps.has("--metrics"),
            )
        except BaseException:
            self.stop()
            raise

    def _ready_props(self, client: Client, cancel: Cancellation, status: Callable[[str, str], None]) -> dict:
        assert self.child is not None
        deadline = time.monotonic() + self.timeout
        previous = ""
        while time.monotonic() < deadline:
            cancel.check()
            code = self.child.poll()
            if code is not None:
                raise InferenceError(
                    f"llama-server exited during loading (status {code}). Open Diagnostics for details."
                )
            detail = self._detail()
            if detail != previous:
                status("Loading", detail)
                previous = detail
            try:
                if client.json("/health", cancel=cancel, timeout=1).get("status") == "ok":
                    return client.json("/props", cancel=cancel, timeout=2)
            except HttpError as exc:
                if exc.status not in (503, 502, 504):
                    raise
            except InferenceError:
                cancel.check()
            cancel.wait(0.25)
        raise InferenceError(
            "Timed out waiting for llama-server. Check Diagnostics or use a smaller model or context."
        )

    def health(self) -> str:
        if self.child is None or self.info is None:
            return ""
        if self.child.poll() is not None:
            code = self.child.returncode
            self.stop()
            return f"llama-server exited unexpectedly (status {code}). Select Reload to restart it."
        try:
            if self.info.client.json("/health", timeout=1).get("status") == "ok":
                return ""
        except InferenceError:
            pass
        self.stop()
        return "llama-server stopped responding. Select Reload to restart it."
