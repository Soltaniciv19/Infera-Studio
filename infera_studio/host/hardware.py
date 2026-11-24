"""Linux CPU, memory, and GPU measurements."""

import csv
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class GPU:
    device: str
    total: int
    used: int
    pci_id: str = ""

    @property
    def free(self) -> int:
        return max(0, self.total - self.used)


def _read_text(file: Path) -> str:
    try:
        return file.read_text().strip()
    except (OSError, UnicodeError):
        return ""


def _read_integer(file: Path) -> int | None:
    try:
        return int(_read_text(file))
    except ValueError:
        return None


def _amd_devices(root: Path) -> Iterator[Path]:
    for card in sorted(root.glob("card*")):
        if not card.name.removeprefix("card").isdigit():
            continue
        device = card / "device"
        if _read_text(device / "vendor") == "0x1002":
            yield device.resolve()


def _vram(device: Path) -> tuple[int | None, int | None]:
    total = _read_integer(device / "mem_info_vram_total")
    used = _read_integer(device / "mem_info_vram_used")
    return total if total and total > 0 else None, used if used is not None and used >= 0 else None


def amd_gpus(root: Path = Path("/sys/class/drm")) -> tuple[GPU, ...]:
    result = []
    for device in _amd_devices(root):
        total, used = _vram(device)
        if total is not None and used is not None:
            result.append(GPU(device.name, total, used, _read_text(device / "device")))
    return tuple(result)


@dataclass(frozen=True)
class GPUStats:
    device: str
    load: int | None
    vram_used: int | None
    vram_total: int | None
    temperature_c: float | None
    vendor: str = "AMD"
    name: str = ""


def amd_gpu_stats(root: Path) -> Iterator[GPUStats]:
    for device in dict.fromkeys(_amd_devices(root)):
        total, used = _vram(device)
        busy = _read_integer(device / "gpu_busy_percent")
        load = busy if busy is not None and 0 <= busy <= 100 else None
        temperature = next(
            (
                millidegrees / 1000
                for sensor in sorted(device.glob("hwmon/hwmon*/temp1_input"))
                if (millidegrees := _read_integer(sensor)) is not None
            ),
            None,
        )
        if any(value is not None for value in (load, used, total, temperature)):
            yield GPUStats(device.name, load, used, total, temperature)


def gpu_stats(root: Path = Path("/sys/class/drm")) -> GPUStats | None:
    return max(amd_gpu_stats(root), key=lambda reading: reading.vram_total or 0, default=None)


def nvidia_gpu_stats(binary: str) -> Iterator[GPUStats]:
    try:
        result = subprocess.run(
            [binary, "--query-gpu=pci.bus_id,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, errors="replace", timeout=1.5, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return

    def reading(value: str, low: int, high: int) -> int | None:
        try:
            result = int(value)
        except ValueError:
            return None
        return result if low <= result <= high else None

    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            continue
        device, name, busy, used, total, temp = row
        load = reading(busy, 0, 100)
        used_mib = reading(used, 0, 2**24)
        total_mib = reading(total, 1, 2**24)
        temperature = reading(temp, -80, 200)
        if any(value is not None for value in (load, used_mib, total_mib, temperature)):
            yield GPUStats(
                device.strip(), load, used_mib * MIB if used_mib is not None else None,
                total_mib * MIB if total_mib is not None else None, temperature, "NVIDIA", name.strip(),
            )


@dataclass(frozen=True)
class _CPUTimes:
    busy: int
    idle: int

    def load_since(self, previous: "_CPUTimes") -> float | None:
        busy = self.busy - previous.busy
        idle = self.idle - previous.idle
        if busy < 0 or idle < 0 or busy + idle == 0:
            return None
        return 100 * busy / (busy + idle)


def _cpu_times(proc: Path) -> _CPUTimes | None:
    first_line = _read_text(proc / "stat").partition("\n")[0].split()
    if len(first_line) < 9 or first_line[0] != "cpu":
        return None
    try:
        # Guest time is already included in user and nice. Treat I/O wait as idle.
        counters = [int(value) for value in first_line[1:9]]
    except ValueError:
        return None
    idle = sum(counters[3:5])
    return _CPUTimes(sum(counters) - idle, idle)


def cpu_temperature(root: Path = Path("/sys/class/hwmon")) -> float | None:
    values: list[int] = []
    for monitor in root.glob("hwmon*"):
        driver = _read_text(monitor / "name")
        if driver == "k10temp":
            # AMD exposes Tdie in temp2 when it differs from the Tctl reading in temp1.
            value = _read_integer(monitor / "temp2_input")
            if value is None:
                value = _read_integer(monitor / "temp1_input")
        elif driver == "coretemp":
            value = _read_integer(monitor / "temp1_input")
            if value is None:
                value = max(
                    (reading for sensor in monitor.glob("temp*_input")
                     if (reading := _read_integer(sensor)) is not None),
                    default=None,
                )
        else:
            continue
        if value is not None:
            values.append(value)
    return max(values) / 1000 if values else None


def _memory_info(proc: Path) -> dict[str, int]:
    memory: dict[str, int] = {}
    for line in _read_text(proc / "meminfo").splitlines():
        key, _, value = line.partition(":")
        if key not in ("MemTotal", "MemAvailable"):
            continue
        try:
            amount, unit = value.split()
            if unit == "kB" and (kilobytes := int(amount)) >= 0:
                memory[key] = kilobytes * 1024
        except ValueError:
            continue
    return memory


def system_available() -> int:
    return _memory_info(Path("/proc")).get("MemAvailable", 0)


@dataclass(frozen=True)
class SystemStats:
    gpu: GPUStats | None
    cpu_load: float | None
    cpu_temperature_c: float | None
    ram_used: int | None
    ram_total: int | None
    gpus: tuple[GPUStats, ...] = ()


class SystemMonitor:
    def __init__(
        self, proc: Path = Path("/proc"), hwmon: Path = Path("/sys/class/hwmon"), drm: Path = Path("/sys/class/drm"),
    ) -> None:
        self.proc = proc
        self.hwmon = hwmon
        self.drm = drm
        self._nvidia = shutil.which("nvidia-smi")
        self._previous_cpu: _CPUTimes | None = None

    def sample(self) -> SystemStats:
        cpu = _cpu_times(self.proc)
        load = cpu.load_since(self._previous_cpu) if cpu and self._previous_cpu else None
        self._previous_cpu = cpu
        memory = _memory_info(self.proc)
        total = memory.get("MemTotal") or None
        available = memory.get("MemAvailable")
        used = max(0, total - available) if total is not None and available is not None else None
        devices = tuple(amd_gpu_stats(self.drm))
        if self._nvidia:
            devices += tuple(nvidia_gpu_stats(self._nvidia))
        gpu = max(devices, key=lambda device: device.vram_total or 0, default=None)
        return SystemStats(gpu, load, cpu_temperature(self.hwmon), used, total, devices)
