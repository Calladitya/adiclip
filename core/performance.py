"""Shared performance helpers: stable file identities, JSON cache, and encoder selection."""
import hashlib
import json
import os
import subprocess
import uuid
import ctypes
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable
from ctypes import wintypes

from core.config import OUTPUT_DIR, TEMP_DIR, get_ffmpeg_path

CACHE_DIR = TEMP_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def file_identity(path: str) -> Dict[str, Any]:
    file_path = Path(path).resolve()
    stat = file_path.stat()
    return {"path": str(file_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def make_cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(namespace: str, key: str, suffix: str = ".json") -> Path:
    folder = CACHE_DIR / namespace
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{key}{suffix}"


def load_json_cache(namespace: str, key: str) -> Any:
    path = cache_path(namespace, key)
    try:
        with path.open("r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except (OSError, ValueError, TypeError):
        return None


def save_json_cache(namespace: str, key: str, value: Any) -> None:
    path = cache_path(namespace, key)
    temporary = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as cache_file:
            json.dump(value, cache_file, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@lru_cache(maxsize=8)
def _probe_encoder(encoder: str) -> bool:
    """Verify the encoder can actually open a device and write a short test stream."""
    output = OUTPUT_DIR / f".encoder-probe-{encoder}-{uuid.uuid4().hex}.mp4"
    cmd = [
        get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=0.35",
        "-an", "-c:v", encoder,
    ]
    if encoder == "h264_amf":
        cmd += ["-quality", "speed", "-rc", "cqp", "-qp_i", "24", "-qp_p", "24"]
    elif encoder == "h264_nvenc":
        cmd += ["-preset", "p1", "-cq", "24"]
    elif encoder == "h264_qsv":
        cmd += ["-preset", "veryfast", "-global_quality", "24"]
    cmd += [str(output)]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
        return result.returncode == 0 and output.exists() and output.stat().st_size > 0
    except (OSError, subprocess.TimeoutExpired):
        return False
    finally:
        try:
            output.unlink(missing_ok=True)
        except OSError:
            pass


@lru_cache(maxsize=3)
def select_video_encoder(mode: str) -> tuple[str, list[str]]:
    """Choose only a tested encoder; AMF is preferred on this AMD Windows machine."""
    if mode in {"FAST MODE", "BALANCED MODE"}:
        for encoder in ("h264_amf", "h264_nvenc", "h264_qsv"):
            if _probe_encoder(encoder):
                if encoder == "h264_amf":
                    quality = "speed" if mode == "FAST MODE" else "balanced"
                    qp = "25" if mode == "FAST MODE" else "22"
                    return encoder, ["-quality", quality, "-rc", "cqp", "-qp_i", qp, "-qp_p", qp]
                if encoder == "h264_nvenc":
                    preset = "p1" if mode == "FAST MODE" else "p4"
                    return encoder, ["-preset", preset, "-cq", "24"]
                return encoder, ["-preset", "veryfast", "-global_quality", "24"]
    if mode == "FAST MODE":
        return "libx264", ["-preset", "superfast", "-crf", "24"]
    if mode == "HIGH QUALITY MODE":
        return "libx264", ["-preset", "slow", "-crf", "20"]
    return "libx264", ["-preset", "faster", "-crf", "22"]


def face_sample_interval(mode: str) -> float:
    # Sparse seeks are much cheaper than scanning every decoded frame in 4K AV1 sources.
    return {"FAST MODE": 8.0, "BALANCED MODE": 4.0, "HIGH QUALITY MODE": 2.0}.get(mode, 4.0)


def face_inference_width(mode: str) -> int:
    return 480 if mode == "HIGH QUALITY MODE" else 320


def clip_worker_count(mode: str) -> int:
    """Bound parallel encodes to avoid saturating memory and the source disk."""
    if mode not in {"FAST MODE", "BALANCED MODE"} or (os.cpu_count() or 1) < 8:
        return 1
    return 2 if available_memory_mb() >= 4096 else 1


def available_memory_mb() -> int:
    """Return available physical memory on Windows; use a conservative fallback elsewhere."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys / (1024 * 1024))
    return 0


def process_resource_snapshot() -> Dict[str, float]:
    """Return Windows process CPU time and peak working set when available."""
    if os.name != "nt":
        return {}
    class FileTime(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    class MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(FileTime)] * 4
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    creation, exit_time, kernel, user = FileTime(), FileTime(), FileTime(), FileTime()
    cpu_seconds = None
    if kernel32.GetProcessTimes(process, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel), ctypes.byref(user)):
        ticks = (user.high << 32 | user.low) + (kernel.high << 32 | kernel.low)
        cpu_seconds = ticks / 10_000_000
    memory = MemoryCounters()
    memory.cb = ctypes.sizeof(memory)
    peak_mb = None
    if psapi.GetProcessMemoryInfo(process, ctypes.byref(memory), memory.cb):
        peak_mb = memory.PeakWorkingSetSize / (1024 * 1024)
    return {"cpu_seconds": cpu_seconds, "peak_working_set_mb": peak_mb}
