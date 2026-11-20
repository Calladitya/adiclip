"""
Configuration and constants for the AI Podcast Clipping Application.
"""
import os
import shutil
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = BASE_DIR / "core"
TEMP_DIR = BASE_DIR / "temp"
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

for directory in [TEMP_DIR, DOWNLOADS_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Face Detection Model Path (YuNet ONNX)
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

# Video Specifications for 9:16 Vertical Export
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = 30

# Clip Constraints
MIN_CLIP_DURATION = 30.0  # seconds
MAX_CLIP_DURATION = 60.0  # seconds
DEFAULT_NUM_CLIPS = 10

# Whisper Configuration
DEFAULT_WHISPER_MODEL = "base"
AVAILABLE_WHISPER_MODELS = ["tiny", "base", "small"]

# Gemini Configuration
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def get_ffmpeg_path() -> str:
    """
    Returns a usable FFmpeg executable.
    Prefer the system FFmpeg on Linux deployments (installed with apt); on Windows,
    prefer imageio-ffmpeg and fall back to the system PATH.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg and os.name != "nt":
        return system_ffmpeg

    # 1. Check imageio_ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and os.path.exists(ffmpeg_exe):
            return ffmpeg_exe
    except Exception:
        pass

    # 2. Check system PATH
    if system_ffmpeg:
        return system_ffmpeg

    raise RuntimeError(
        "FFmpeg binary not found. Please install imageio-ffmpeg or ensure ffmpeg is in system PATH."
    )
