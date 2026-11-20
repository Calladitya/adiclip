"""
Downloader module for YouTube videos and local video file handling.
"""
import os
import re
import cv2
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import yt_dlp

from core.config import DOWNLOADS_DIR, get_ffmpeg_path


def _youtube_proxy_options() -> Dict[str, str]:
    """Ignore the dead loopback proxy injected by some local environments."""
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(name)
        if not value:
            continue
        parsed = urlparse(value if "://" in value else f"http://{value}")
        if parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port == 9:
            return {"proxy": ""}
    return {}


def sanitize_filename(name: str) -> str:
    """Sanitizes strings for safe filenames."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().replace(" ", "_")
    return name[:60] if len(name) > 60 else name


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extracts video duration, resolution, fps, and frame count using OpenCV.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "path": video_path,
        "filename": os.path.basename(video_path),
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "duration": round(duration, 2),
        "frame_count": frame_count,
        "is_vertical": height > width,
    }


def fetch_youtube_info(url: str) -> Dict[str, Any]:
    """
    Fetches YouTube video metadata without downloading the full video.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        **_youtube_proxy_options(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise ValueError("Could not extract video metadata from URL.")
            return {
                "id": info.get("id", ""),
                "title": info.get("title", "Untitled Video"),
                "duration": info.get("duration", 0),
                "author": info.get("uploader", "Unknown Uploader"),
                "thumbnail": info.get("thumbnail", ""),
                "description": info.get("description", "")[:200] if info.get("description") else "",
            }
        except Exception as e:
            raise RuntimeError(f"Failed to fetch YouTube metadata: {str(e)}")


def download_youtube_video(
    url: str,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Downloads a YouTube video to the output directory with progress updates.
    """
    if output_dir is None:
        output_dir = DOWNLOADS_DIR

    ffmpeg_exe = get_ffmpeg_path()

    # First fetch info
    info = fetch_youtube_info(url)
    clean_title = sanitize_filename(info["title"])
    out_template = str(output_dir / f"{clean_title}_%(id)s.%(ext)s")

    def yt_hook(d):
        if progress_callback and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                pct = min(1.0, downloaded / total)
                progress_callback(pct, f"Downloading YouTube video: {pct * 100:.1f}%")
        elif progress_callback and d.get("status") == "finished":
            progress_callback(1.0, "YouTube download complete. Finalizing file...")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_template,
        # Pass the binary itself: imageio-ffmpeg uses a versioned filename,
        # which yt-dlp cannot discover when given only the containing folder.
        "ffmpeg_location": ffmpeg_exe,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [yt_hook],
        "merge_output_format": "mp4",
        **_youtube_proxy_options(),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        dl_info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(dl_info)
        # Ensure .mp4 extension
        if not filename.endswith(".mp4"):
            base, _ = os.path.splitext(filename)
            mp4_path = base + ".mp4"
            if os.path.exists(mp4_path):
                filename = mp4_path

    metadata = get_video_metadata(filename)
    metadata["title"] = info["title"]
    metadata["thumbnail"] = info.get("thumbnail")
    metadata["author"] = info.get("author")
    return metadata


def save_uploaded_file(uploaded_file, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Saves an uploaded Streamlit file to disk and extracts video metadata.
    """
    if output_dir is None:
        output_dir = DOWNLOADS_DIR

    clean_name = sanitize_filename(uploaded_file.name)
    if not clean_name.lower().endswith(".mp4"):
        clean_name += ".mp4"

    save_path = output_dir / clean_name
    # Avoid overwriting with collision check
    counter = 1
    base_stem = save_path.stem
    while save_path.exists():
        save_path = output_dir / f"{base_stem}_{counter}.mp4"
        counter += 1

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    metadata = get_video_metadata(str(save_path))
    metadata["title"] = Path(uploaded_file.name).stem
    return metadata
