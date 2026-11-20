"""
Downloader module for YouTube videos and local video file handling.
"""
import os
import re
import logging
import shutil
import subprocess
import time
import uuid
import cv2
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import yt_dlp

from core.config import DOWNLOADS_DIR, get_ffmpeg_path

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
DOWNLOAD_TIMEOUT_SECONDS = 2 * 60 * 60
YTDLP_SOCKET_TIMEOUT_SECONDS = 20
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtube-nocookie.com"}


def _validate_youtube_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").lower() not in YOUTUBE_HOSTS:
        raise ValueError("Enter a valid YouTube video URL (youtube.com or youtu.be).")
    return url.strip()


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


def _is_http_403(error: Exception) -> bool:
    message = str(error).lower()
    return "403" in message or "forbidden" in message


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


def _validate_download(path: str) -> None:
    """Require readable audio/video streams; use ffprobe when installed, PyAV otherwise."""
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise RuntimeError("Download produced no video file or an empty file.")
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "stream=codec_type", "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"FFprobe could not read the downloaded video: {result.stderr.strip()[:500]}")
        import json
        probe = json.loads(result.stdout or "{}")
        stream_types = {stream.get("codec_type") for stream in probe.get("streams", [])}
        try:
            duration = float(probe.get("format", {}).get("duration", 0))
        except (TypeError, ValueError):
            duration = 0
        if "video" not in stream_types or "audio" not in stream_types or duration <= 0:
            raise RuntimeError("FFprobe found no valid audio/video streams or duration in the downloaded file.")
        return

    # Windows imageio-ffmpeg bundles ffmpeg without ffprobe. Validate through PyAV,
    # which is already an application dependency, and decode an actual video frame.
    import av
    try:
        with av.open(path) as container:
            if not any(stream.type == "video" for stream in container.streams):
                raise RuntimeError("Downloaded file contains no video stream.")
            if not any(stream.type == "audio" for stream in container.streams):
                raise RuntimeError("Downloaded file contains no audio stream.")
            if not next(container.decode(video=0), None):
                raise RuntimeError("Downloaded video stream could not decode a frame.")
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Downloaded video failed PyAV stream/decode validation: {exc}") from exc


def fetch_youtube_info(url: str, cancel_event=None) -> Dict[str, Any]:
    """
    Fetches YouTube video metadata without downloading the full video.
    """
    url = _validate_youtube_url(url)
    logger.info("[DOWNLOAD] Validating URL and connecting to YouTube")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        "extract_flat": False,
        "socket_timeout": YTDLP_SOCKET_TIMEOUT_SECONDS,
        "retries": 2,
        "extractor_retries": 2,
        "noplaylist": True,
        **_youtube_proxy_options(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("YouTube metadata lookup was cancelled.")
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
        except InterruptedError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to fetch YouTube metadata: {str(e)}")


def download_youtube_video(
    url: str,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    video_info: Optional[Dict[str, Any]] = None,
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Downloads a YouTube video to the output directory with progress updates.
    """
    if output_dir is None:
        output_dir = DOWNLOADS_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = get_ffmpeg_path()
    started = time.monotonic()
    info = video_info or fetch_youtube_info(url, cancel_event=cancel_event)
    clean_title = sanitize_filename(info["title"])
    # Path.mkdir inherits the app directory ACL on Windows. tempfile.mkdtemp can
    # create a directory whose inherited ACL denies the Streamlit process writes
    # in restricted Windows environments.
    job_dir = output_dir / f"podclip_yt_{uuid.uuid4().hex}"
    job_dir.mkdir(parents=False, exist_ok=False)

    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("YouTube download was cancelled.")
        if time.monotonic() - started > DOWNLOAD_TIMEOUT_SECONDS:
            raise TimeoutError("YouTube download exceeded the 2 hour limit.")

    last_logged_percent = -10
    def yt_hook(d):
        nonlocal last_logged_percent
        check_cancelled()
        if progress_callback and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                pct = min(1.0, downloaded / total)
                speed = d.get("speed")
                eta = d.get("eta")
                details = []
                if speed:
                    details.append(f"{speed / (1024 * 1024):.1f} MB/s")
                if eta is not None:
                    details.append(f"ETA {int(eta)}s")
                suffix = f" ({', '.join(details)})" if details else ""
                progress_callback(pct, f"Downloading video: {pct * 100:.1f}% · {downloaded / (1024 * 1024):.1f} MB{suffix}")
                if int(pct * 100) >= last_logged_percent + 10:
                    logger.info("[DOWNLOAD] Progress update: %.1f%%, %.1f MB downloaded", pct * 100, downloaded / (1024 * 1024))
                    last_logged_percent = int(pct * 100)
            elif progress_callback:
                progress_callback(None, f"Downloading YouTube video: {downloaded / (1024 * 1024):.1f} MB received")
        elif progress_callback and d.get("status") == "finished":
            progress_callback(None, "Download data received; finalizing MP4...")

    ydl_opts = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]",
        # Pass the binary itself: imageio-ffmpeg uses a versioned filename,
        # which yt-dlp cannot discover when given only the containing folder.
        "ffmpeg_location": ffmpeg_exe,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [yt_hook],
        "socket_timeout": YTDLP_SOCKET_TIMEOUT_SECONDS,
        "retries": 3,
        "fragment_retries": 3,
        "file_access_retries": 2,
        "noplaylist": True,
        "merge_output_format": "mp4",
        **_youtube_proxy_options(),
    }

    try:
        check_cancelled()
        logger.info("[DOWNLOAD] Initializing; validating URL and connecting to YouTube (video id=%s)", info.get("id", "unknown"))
        logger.info("[DOWNLOAD] Download started (video id=%s)", info.get("id", "unknown"))
        attempts = [
            ("default", dict(ydl_opts)),
            (
                "lower-bandwidth format fallback",
                {
                    **ydl_opts,
                    "format": "best[ext=mp4][height<=720]/best[ext=mp4][height<=480]/best[height<=720]/best",
                },
            ),
        ]
        filename = None
        primary_403_error = None
        for attempt_index, (attempt_name, attempt_options) in enumerate(attempts):
            check_cancelled()
            attempt_dir = job_dir / f"attempt_{attempt_index + 1}"
            attempt_dir.mkdir()
            attempt_options["outtmpl"] = str(attempt_dir / f"{clean_title}_%(id)s.%(ext)s")
            if attempt_index:
                logger.warning("[DOWNLOAD] Retrying with %s after HTTP 403", attempt_name)
                if progress_callback:
                    progress_callback(None, "YouTube refused the first format; retrying with a lower-bandwidth format...")
            try:
                with yt_dlp.YoutubeDL(attempt_options) as ydl:
                    dl_info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(dl_info)
                    if not os.path.isfile(filename):
                        base, _ = os.path.splitext(filename)
                        merged_mp4 = base + ".mp4"
                        if os.path.isfile(merged_mp4):
                            filename = merged_mp4
            except Exception as attempt_error:
                if attempt_index == 0 and _is_http_403(attempt_error):
                    primary_403_error = attempt_error
                    logger.warning("[DOWNLOAD] Primary YouTube format returned HTTP 403; trying compatible fallback")
                    shutil.rmtree(attempt_dir, ignore_errors=True)
                    continue
                if primary_403_error is not None:
                    raise RuntimeError(
                        f"YouTube rejected the initial download with HTTP 403; the lower-bandwidth retry also failed: {attempt_error}"
                    ) from attempt_error
                raise
            break
        if not filename:
            raise RuntimeError("YouTube download did not produce a file.")
        check_cancelled()
        logger.info("[DOWNLOAD] Validating downloaded file")
        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            raise RuntimeError("YouTube download finished without producing a video file.")
        _validate_download(filename)
        if progress_callback:
            progress_callback(1.0, "Download completed and validated.")
        metadata = get_video_metadata(filename)
        extension = Path(filename).suffix.lower() or ".mp4"
        destination = output_dir / f"{clean_title}_{info.get('id') or 'youtube'}_{uuid.uuid4().hex[:8]}{extension}"
        shutil.move(filename, destination)
        metadata["path"] = str(destination)
        metadata["filename"] = destination.name
        metadata["title"] = info["title"]
        metadata["thumbnail"] = info.get("thumbnail")
        metadata["author"] = info.get("author")
        logger.info("[DOWNLOAD] Complete: %s (%.1f MB)", destination.name, destination.stat().st_size / (1024 * 1024))
        return metadata
    except Exception as exc:
        if cancel_event is not None and cancel_event.is_set():
            logger.info("[DOWNLOAD] Cancelled by user; removing this job's partial files")
            raise InterruptedError("YouTube download was cancelled.") from exc
        logger.exception("[DOWNLOAD] Failed for video id=%s", info.get("id", "unknown"))
        message = str(exc)
        if "requested format is not available" in message.lower() and not shutil.which("deno"):
            raise RuntimeError(
                "yt-dlp found no downloadable format because the Deno JavaScript runtime is missing. "
                "Install the updated project requirements (yt-dlp[default,deno]) and restart the app."
            ) from exc
        if "403" in message or "Forbidden" in message:
            raise RuntimeError(
                f"YouTube refused the video download (HTTP 403), including the lower-bandwidth retry. "
                f"This can be a YouTube or hosting-network restriction. Details: {message[:400]} "
                "Try another public video or upload the file."
            ) from exc
        raise
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


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
