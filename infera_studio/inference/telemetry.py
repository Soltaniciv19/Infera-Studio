"""Normalize llama.cpp timings and usage reported during inference."""

import math
from typing import Any

from infera_studio.conversations.metrics import Metrics


def count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**31 - 1 else None


def number(value: Any) -> float | None:
    if type(value) in (int, float) and 0 <= value <= 1e15 and math.isfinite(value):
        return float(value)
    return None


def fields(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def update_metrics(metrics: Metrics, event: dict[str, Any]) -> None:
    progress = fields(event.get("prompt_progress"))
    timings = fields(event.get("timings"))
    usage = fields(event.get("usage"))
    usage_details = fields(usage.get("prompt_tokens_details"))
    prompt = count(usage.get("prompt_tokens"))
    if prompt is None:
        prompt = count(progress.get("total"))
    generated = count(usage.get("completion_tokens"))
    if generated is None:
        generated = count(timings.get("predicted_n"))
    cached = count(usage_details.get("cached_tokens"))
    if cached is None:
        cached = count(timings.get("cache_n"))
    if cached is None:
        cached = count(progress.get("cache"))
    processed = count(timings.get("prompt_n"))
    progress_tokens = count(progress.get("processed"))
    prompt_rate = number(timings.get("prompt_per_second"))
    if prompt is None and processed is not None and cached is not None and not progress:
        prompt = processed + cached
    for key, value in (
        ("prompt_tokens", prompt),
        ("generated_tokens", generated),
        ("cached_prompt_tokens", cached),
        ("processed_prompt_tokens", processed),
        ("prompt_progress_tokens", progress_tokens),
        ("prompt_per_second", prompt_rate),
        ("generation_per_second", number(timings.get("predicted_per_second"))),
    ):
        if value is not None:
            setattr(metrics, key, value)
    prompt_ms = number(timings.get("prompt_ms"))
    if prompt_ms is None:
        prompt_ms = number(progress.get("time_ms"))
    if prompt_ms is not None:
        metrics.prompt_seconds = prompt_ms / 1000
    generation_ms = number(timings.get("predicted_ms"))
    if generation_ms is not None:
        metrics.generation_seconds = generation_ms / 1000
    if prompt_rate is None and progress_tokens is not None and cached is not None and prompt_ms:
        if progress_tokens >= cached:
            rate = number(1000 * (progress_tokens - cached) / prompt_ms)
            if rate is not None:
                metrics.prompt_per_second = rate
