"""Measurements stored with a chat completion."""

from dataclasses import dataclass, field
from typing import NotRequired, TypedDict

MAX_PERFORMANCE_SAMPLES = 96


class PerformanceSample(TypedDict):
    elapsed_seconds: float
    prompt_per_second: float | None
    generation_per_second: float | None
    generated_tokens: int | None
    context_tokens: int | None
    prompt_tokens: NotRequired[int | None]
    processed_prompt_tokens: NotRequired[int | None]
    prompt_progress_tokens: NotRequired[int | None]
    cached_prompt_tokens: NotRequired[int | None]
    kv_allocated_mib: NotRequired[float | None]


@dataclass
class Metrics:
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    prompt_per_second: float | None = None
    generation_per_second: float | None = None
    first_token_seconds: float | None = None
    seconds: float = 0.0
    finish_reason: str | None = None
    prompt_seconds: float | None = None
    generation_seconds: float | None = None
    cached_prompt_tokens: int | None = None
    processed_prompt_tokens: int | None = None
    prompt_progress_tokens: int | None = None
    context: int | None = None
    budget_prompt: int | None = None
    budget_estimated: bool = False
    kv_allocated_mib: float | None = None
    model_seconds: float | None = None
    token_check_seconds: float | None = None
    turn_setup_seconds: float | None = None
    request_seconds: float | None = None
    stage: str = "model"
    samples: list[PerformanceSample] = field(default_factory=list)

    def record_sample(self, elapsed: float) -> None:
        prompt_tokens = self.prompt_tokens
        if prompt_tokens is None and not self.budget_estimated:
            prompt_tokens = self.budget_prompt
        context_tokens = self.prompt_progress_tokens
        if self.generated_tokens is not None and self.prompt_tokens is not None:
            context_tokens = self.prompt_tokens + self.generated_tokens
        if all(value is None for value in (
            self.prompt_per_second, self.generation_per_second, self.generated_tokens, context_tokens,
            prompt_tokens, self.processed_prompt_tokens, self.cached_prompt_tokens, self.kv_allocated_mib,
        )):
            return
        sample: PerformanceSample = {
            "elapsed_seconds": elapsed,
            "prompt_per_second": self.prompt_per_second,
            "generation_per_second": self.generation_per_second,
            "generated_tokens": self.generated_tokens,
            "context_tokens": context_tokens,
            "prompt_tokens": prompt_tokens,
            "processed_prompt_tokens": self.processed_prompt_tokens,
            "prompt_progress_tokens": self.prompt_progress_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "kv_allocated_mib": self.kv_allocated_mib,
        }
        if self.samples and elapsed == self.samples[-1]["elapsed_seconds"]:
            self.samples[-1] = sample
            return
        if len(self.samples) >= MAX_PERFORMANCE_SAMPLES:
            # Thin the earlier trace while preserving the first point and more recent detail.
            halfway = MAX_PERFORMANCE_SAMPLES // 2
            self.samples[1:halfway] = self.samples[1:halfway:2]
        self.samples.append(sample)
