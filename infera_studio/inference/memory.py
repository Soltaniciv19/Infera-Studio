"""Estimate llama.cpp memory use and choose a context size."""

from dataclasses import dataclass

from infera_studio.catalog.gguf import Model
from infera_studio.config.settings import CONTEXT_SIZES, InferenceSettings
from infera_studio.host.hardware import GIB, GPU, MIB, system_available

CACHE_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
DENSE_ARCHES = {
    "llama",
    "mistral",
    "qwen2",
    "qwen2moe",
    "qwen3",
    "qwen3moe",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma4",
    "phi3",
    "phi4",
    "gptneox",
}
QWEN_HYBRIDS = {"qwen35", "qwen35moe"}


@dataclass(frozen=True)
class MemoryEstimate:
    context: int
    weights: int
    kv: int | None
    recurrent: int
    runtime_margin: int
    gpu: GPU | None
    offload_fraction: float

    @property
    def required(self) -> int | None:
        if self.kv is None:
            return None
        return self.weights + self.kv + self.recurrent + self.runtime_margin

    @property
    def headroom(self) -> int | None:
        required = self.required
        return self.gpu.free - required if self.gpu and required is not None else None


def estimate(
    model: Model, settings: InferenceSettings, context: int, gpu: GPU | None, parallel: int = 1
) -> MemoryEstimate:
    arch, meta = model.architecture, model.metadata

    def number(key: str, default: int = 0) -> int:
        value = meta.get(f"{arch}.{key}")
        return value if isinstance(value, int) and value > 0 else default

    def layer_values(key: str, count: int, default: int = 0) -> list[int]:
        value = meta.get(f"{arch}.{key}")
        if (
            isinstance(value, list)
            and len(value) >= count
            and all(type(item) is int and item >= 0 for item in value[:count])
        ):
            return value[:count]
        return [number(key, default)] * count

    blocks = min(number("block_count"), 4096)
    gpu_active = settings.backend != "CPU" and settings.gpu_layers != 0
    if not gpu_active:
        fraction = 0.0
    elif settings.gpu_layers == -1 or not blocks:
        fraction = 1.0
    else:
        fraction = min(1.0, settings.gpu_layers / (blocks + 1))
    # If GPU is disabled this is a host estimate. When enabled these are device bytes.
    weights = int(model.size * (fraction if gpu_active else 1.0))
    runtime_margin = max(GIB, int(weights * 0.08)) if gpu_active else GIB
    per_token = 0.0
    recurrent = 0
    known = bool(blocks and arch in DENSE_ARCHES | QWEN_HYBRIDS)
    if known:
        recurrent_flags = meta.get(f"{arch}.attention.recurrent_layers")
        interval = number("full_attention_interval", 4)
        heads = layer_values("attention.head_count", blocks)
        kv_heads = layer_values("attention.head_count_kv", blocks) or heads
        embedding = number("embedding_length")
        for index in range(blocks):
            if gpu_active and fraction < 1 and index < blocks - settings.gpu_layers:
                continue
            is_recurrent = arch in QWEN_HYBRIDS and (
                bool(recurrent_flags[index])
                if isinstance(recurrent_flags, list) and len(recurrent_flags) >= blocks
                else index < blocks - number("nextn_predict_layers") and (index + 1) % interval != 0
            )
            if is_recurrent:
                head_dim = number("ssm.state_size")
                rank = number("ssm.time_step_rank")
                groups = number("ssm.group_count")
                kernel = number("ssm.conv_kernel")
                inner = number("ssm.inner_size")
                if not all((head_dim, rank, groups, kernel, inner)):
                    known = False
                recurrent += 4 * (head_dim * head_dim * rank + (kernel - 1) * (inner + 2 * groups * head_dim))
                continue
            fallback = embedding // heads[index] if heads[index] else 0
            key = number("attention.key_length", fallback)
            value = number("attention.value_length", fallback)
            head_count = kv_heads[index] or heads[index]
            if not key or not value or not head_count:
                known = False
            per_token += head_count * (
                key * CACHE_BYTES[settings.cache_k] + value * CACHE_BYTES[settings.cache_v]
            )
    # Cache padding varies with backend and attention implementation.
    padded = (context + 255) // 256 * 256
    kv = int(per_token * padded * parallel * 1.10) if known else None
    return MemoryEstimate(
        context, weights, kv, recurrent * parallel, runtime_margin, gpu if gpu_active else None, fraction
    )


def recommend(
    model: Model,
    settings: InferenceSettings,
    gpu: GPU | None,
    parallel: int = 1,
    host_available: int | None = None,
) -> MemoryEstimate:
    if settings.context:
        return estimate(model, settings, settings.context, gpu, parallel)
    native = model.context or 4096
    gpu_active = settings.backend != "CPU" and settings.gpu_layers != 0
    if gpu_active and gpu is not None:
        choices = {min(native, size) for size in (*CONTEXT_SIZES, native)}
        available = max(0, gpu.free - max(512 * MIB, gpu.total // 12))
    elif not gpu_active:
        choices = {min(native, size) for size in (4096, 8192, 16384)}
        available = int(0.7 * (system_available() if host_available is None else host_available))
    else:
        choices, available = {min(native, 4096)}, 0
    for size in sorted(choices, reverse=True):
        plan = estimate(model, settings, size, gpu, parallel)
        if plan.required is not None and plan.required <= available:
            return plan
    return estimate(model, settings, min(native, 4096), gpu, parallel)
