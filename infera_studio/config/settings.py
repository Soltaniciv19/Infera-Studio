"""Validated per-conversation inference settings."""

from dataclasses import dataclass, fields
from typing import Any

BACKENDS = ("Vulkan", "HIP", "CPU")
CACHE_TYPES = ("f16", "q8_0", "q4_0")
CONTEXT_SIZES = (4096, 8192, 16384, 32768, 65536, 131072)


@dataclass(frozen=True)
class InferenceSettings:
    backend: str = "Vulkan"
    context: int = 0
    output: int = 1024
    gpu_layers: int = -1
    flash: str = "auto"
    cache_k: str = "f16"
    cache_v: str = "f16"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    seed: int | None = None
    thinking: str = "Automatic"
    system: str = ""
    mtp: str = "Automatic"
    mtp_draft: str = ""
    vision_projector: str = ""

    def __post_init__(self) -> None:
        def check(name: str, low: float, high: float, kind: type) -> None:
            value = getattr(self, name)
            if value is not None and (
                type(value) not in ((int, float) if kind is float else (int,)) or not low <= value <= high
            ):
                raise ValueError(f"{name} must be between {low:g} and {high:g}.")

        check("context", 0, 2_147_483_647, int)
        check("output", 1, 2_147_483_647, int)
        check("gpu_layers", -1, 100_000, int)
        check("temperature", 0, 10, float)
        check("top_p", 0, 1, float)
        check("top_k", 0, 1_000_000, int)
        check("min_p", 0, 1, float)
        check("repeat_penalty", 0, 10, float)
        check("seed", 0, 4_294_967_295, int)
        if any(type(getattr(self, name)) is not int for name in ("context", "output", "gpu_layers")):
            raise ValueError("Context, output and GPU layers must be integers.")
        if self.backend not in BACKENDS or self.flash not in ("auto", "on", "off"):
            raise ValueError("Invalid backend or Flash Attention setting.")
        if self.cache_k not in CACHE_TYPES or self.cache_v not in CACHE_TYPES:
            raise ValueError("Unsupported cache type.")
        if self.thinking not in ("Automatic", "On", "Off") or not isinstance(self.system, str):
            raise ValueError("Invalid thinking mode or system instructions.")
        if self.mtp not in ("Automatic", "Off") or not isinstance(self.mtp_draft, str):
            raise ValueError("Invalid MTP setting or draft model.")
        if not isinstance(self.vision_projector, str):
            raise ValueError("Invalid vision projector.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceSettings":
        settings = {item.name: data[item.name] for item in fields(cls) if item.name in data}
        if settings.get("thinking") == "Model default":
            settings["thinking"] = "Automatic"
        return cls(**settings)

    def launch_signature(self) -> tuple[Any, ...]:
        return (
            self.backend, self.context, self.gpu_layers, self.flash, self.cache_k, self.cache_v,
            self.mtp, self.mtp_draft, self.vision_projector,
        )
