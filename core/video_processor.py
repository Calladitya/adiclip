"""
FFmpeg Video Processing Pipeline for cutting clips, intelligent 9:16 vertical formatting,
face auto-framing, burning animated subtitles, and exporting watermark-free MP4s.
"""
import os
import subprocess
import time
import hashlib
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

from core.config import (
    OUTPUT_DIR,
    TEMP_DIR,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    TARGET_FPS,
    get_ffmpeg_path,
)
from core.face_tracker import compute_smart_crop_window, compute_dynamic_reframe_plan
from core.subtitle_generator import generate_ass_subtitles
from core.performance import clip_worker_count, face_inference_width, face_sample_interval, file_identity, make_cache_key, select_video_encoder


def escape_ffmpeg_path(file_path: str) -> str:
    r"""
    Escapes Windows/Unix file paths for FFmpeg filter arguments (subtitles filter).
    E.g. C:\path\file.ass -> C\\:/path/file.ass
    """
    p = str(Path(file_path).resolve()).replace("\\", "/")
    # Escape drive letter colon (C: -> C\\:)
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    # Escape single quotes and brackets
    p = p.replace("'", "'\\''").replace("[", "\\[").replace("]", "\\]")
    return p


def generate_clip_thumbnail(video_path: str, output_thumb_path: str, timestamp: float = 1.0) -> str:
    """
    Extracts a single frame from video to use as thumbnail.
    """
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
        ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imwrite(output_thumb_path, frame)
            cap.release()
            return output_thumb_path
        cap.release()

    # Fallback to FFmpeg
    ffmpeg_exe = get_ffmpeg_path()
    subprocess.run([
        ffmpeg_exe, "-y", "-ss", str(timestamp), "-i", video_path,
        "-vframes", "1", "-q:v", "2", output_thumb_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_thumb_path


def _crop_x_expression(points: List[tuple[float, int]]) -> str:
    """Build an FFmpeg per-frame crop expression with linear moves between samples."""
    if len(points) < 2:
        return str(points[0][1] if points else 0)
    expression = str(points[-1][1])
    for (t0, x0), (t1, x1) in reversed(list(zip(points, points[1:]))):
        span = max(t1 - t0, 0.001)
        segment = f"{x0}+({x1}-{x0})*(t-{t0:.3f})/{span:.3f}"
        expression = f"if(lt(t\\,{t1:.3f})\\,{segment}\\,{expression})"
    return expression


def _split_screen_filter(plan: Dict[str, Any], target_width: int, target_height: int, subtitles: str) -> str:
    """Stack two independently framed interview guests into a vertical split layout."""
    centers = sorted(plan.get("track_centers", []), key=lambda item: item["cx"])
    if len(centers) < 2:
        return ""
    source_h = plan["crop_h"]
    panel_h = target_height // 2
    source_panel_w = min(plan["crop_w"] * 2, int(source_h * target_width / panel_h))
    max_x = max(0, (plan["source_width"] - source_panel_w))
    x1 = int(np.clip(centers[0]["cx"] - source_panel_w / 2, 0, max_x))
    x2 = int(np.clip(centers[-1]["cx"] - source_panel_w / 2, 0, max_x))
    x1 -= x1 % 2
    x2 -= x2 % 2
    graph = (
        f"[0:v]split=2[guest1][guest2];"
        f"[guest1]crop={source_panel_w}:{source_h}:{x1}:0,scale={target_width}:{panel_h}[top];"
        f"[guest2]crop={source_panel_w}:{source_h}:{x2}:0,scale={target_width}:{panel_h}[bottom];"
        f"[top][bottom]vstack=inputs=2[stacked]"
    )
    return graph + (f";[stacked]{subtitles}[outv]" if subtitles else ";[stacked]null[outv]")


def validate_browser_mp4(
    file_path: str,
    expected_width: int = TARGET_WIDTH,
    expected_height: int = TARGET_HEIGHT,
    expect_audio: bool = True,
    expected_duration: Optional[float] = None,
) -> Dict[str, Any]:
    """Probe and fully decode an MP4 before publishing it to the UI."""
    import av

    path = Path(file_path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("Rendered MP4 is missing or empty.")

    with av.open(str(path), mode="r") as container:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not video_streams:
            raise ValueError("Rendered MP4 has no video stream.")
        if expect_audio and not audio_streams:
            raise ValueError("Rendered MP4 is missing the source audio stream.")

        video = video_streams[0]
        if video.codec_context.name != "h264":
            raise ValueError(f"Browser output must use H.264; found {video.codec_context.name}.")
        if (video.codec_context.width, video.codec_context.height) != (expected_width, expected_height):
            raise ValueError(
                f"Unexpected video dimensions: {video.codec_context.width}x{video.codec_context.height}; "
                f"expected {expected_width}x{expected_height}."
            )
        if container.duration is None or container.duration <= 0:
            raise ValueError("Rendered MP4 has no valid duration.")
        duration_seconds = container.duration / av.time_base
        if expected_duration is not None and abs(duration_seconds - expected_duration) > 0.25:
            raise ValueError(
                f"Unexpected MP4 duration: {duration_seconds:.3f}s; expected about {expected_duration:.3f}s."
            )

        audio = audio_streams[0] if audio_streams else None
        if audio is not None:
            if audio.codec_context.name != "aac":
                raise ValueError(f"Browser output must use AAC audio; found {audio.codec_context.name}.")
            if audio.codec_context.sample_rate not in (44100, 48000):
                raise ValueError(f"Unexpected AAC sample rate: {audio.codec_context.sample_rate}.")

        video_start = float(video.start_time * video.time_base) if video.start_time is not None else 0.0
        audio_start = float(audio.start_time * audio.time_base) if audio is not None and audio.start_time is not None else 0.0
        start_skew = abs(video_start - audio_start)
        if audio is not None and start_skew > max(0.04, 1.0 / TARGET_FPS):
            raise ValueError(
                f"Audio/video start timestamps differ by {start_skew:.3f}s; expected at most one frame."
            )

        decoded_video_frames = 0
        decoded_audio_frames = 0
        pixel_format = None
        streams = [video] + ([audio] if audio is not None else [])
        for packet in container.demux(*streams):
            for frame in packet.decode():
                if isinstance(frame, av.VideoFrame):
                    decoded_video_frames += 1
                    if pixel_format is None:
                        pixel_format = frame.format.name
                elif isinstance(frame, av.AudioFrame):
                    decoded_audio_frames += 1

        if decoded_video_frames == 0:
            raise ValueError("Rendered MP4 contains no decodable video frames.")
        if pixel_format != "yuv420p":
            raise ValueError(f"Browser output pixel format is {pixel_format}; expected yuv420p.")
        if audio is not None and decoded_audio_frames == 0:
            raise ValueError("Rendered MP4 contains no decodable audio frames.")

        return {
            "valid": True,
            "size_bytes": path.stat().st_size,
            "duration_seconds": round(duration_seconds, 3),
            "video_codec": video.codec_context.name,
            "pixel_format": pixel_format,
            "width": video.codec_context.width,
            "height": video.codec_context.height,
            "video_start_seconds": round(video_start, 4),
            "audio_start_seconds": round(audio_start, 4) if audio is not None else None,
            "av_start_skew_seconds": round(start_skew, 4) if audio is not None else None,
            "video_frames_decoded": decoded_video_frames,
            "audio_codec": audio.codec_context.name if audio is not None else None,
            "audio_frames_decoded": decoded_audio_frames,
        }


def normalize_clip_timestamps(file_path: str, run_ffmpeg: Callable[[List[str]], tuple[int, str]]) -> Dict[str, Any]:
    """Rebase video PTS and trim AAC seek-preroll while copying the encoded video."""
    import av

    path = Path(file_path)
    with av.open(str(path), mode="r") as container:
        video = next((stream for stream in container.streams if stream.type == "video"), None)
        audio = next((stream for stream in container.streams if stream.type == "audio"), None)
        if video is None:
            raise ValueError("Rendered MP4 has no video stream to timestamp-normalize.")
        video_start = float(video.start_time * video.time_base) if video.start_time is not None else 0.0
        audio_start = float(audio.start_time * audio.time_base) if audio is not None and audio.start_time is not None else 0.0
        skew = video_start - audio_start

    tolerance = max(0.04, 1.0 / TARGET_FPS)
    if abs(skew) <= tolerance:
        return {"applied": False, "video_start_before": video_start, "audio_start_before": audio_start}

    ffmpeg = get_ffmpeg_path()
    normalized_path = str(path.with_name(f".{path.stem}.{uuid.uuid4().hex}.sync.mp4"))
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if video_start > audio_start:
        # Fast input seeking can leave AAC packets from the preceding video keyframe.
        # Keep video packets, but start the audio input at the first retained video PTS.
        command += ["-itsoffset", str(-video_start), "-i", str(path)]
        if audio is not None:
            command += ["-ss", str(video_start), "-i", str(path)]
    else:
        command += ["-itsoffset", str(-video_start), "-i", str(path)]
        if audio is not None:
            command += ["-itsoffset", str(-audio_start), "-i", str(path)]
    command += ["-map", "0:v:0"]
    if audio is not None:
        command += [
            "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-b:a", "192k", "-ar", "44100", "-af", "asetpts=PTS-STARTPTS",
        ]
    else:
        command += ["-c:v", "copy"]
    command += ["-avoid_negative_ts", "make_zero", "-movflags", "+faststart", normalized_path]
    return_code, stderr = run_ffmpeg(command)
    if return_code != 0:
        Path(normalized_path).unlink(missing_ok=True)
        raise RuntimeError(f"Could not normalize output audio/video timestamps:\n{stderr}")
    try:
        os.replace(normalized_path, str(path))
    except Exception:
        Path(normalized_path).unlink(missing_ok=True)
        raise
    return {
        "applied": True,
        "video_start_before": round(video_start, 4),
        "audio_start_before": round(audio_start, 4),
        "start_skew_before": round(skew, 4),
    }


def render_single_clip(
    video_path: str,
    clip_info: Dict[str, Any],
    segments: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    layout_mode: str = "smart_crop",  # "smart_crop" or "blurred_background"
    subtitle_style: str = "Neon Gold",
    burn_subtitles: bool = True,
    caption_font: Optional[str] = None,
    caption_size: Optional[int] = None,
    caption_position: str = "Bottom",
    caption_background: Optional[bool] = None,
    caption_text_color: Optional[str] = None,
    caption_highlight_color: Optional[str] = None,
    caption_animation: bool = True,
    performance_mode: str = "BALANCED MODE",
    reframe_plan: Optional[Dict[str, Any]] = None,
    cancel_event=None,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Processes and exports a single 9:16 vertical clip with speaker centering & animated subtitles.
    """
    start_time = clip_info["start_time"]
    end_time = clip_info["end_time"]
    duration = end_time - start_time
    clip_idx = clip_info.get("clip_index", 1)
    cache_hit = False

    base_output_path = output_path
    if output_path is None:
        safe_title = "".join(c for c in clip_info.get("title", f"clip_{clip_idx}") if c.isalnum() or c in (" ", "_", "-")).strip()
        safe_title = safe_title.replace(" ", "_")[:35]
        output_filename = f"clip_{clip_idx:02d}_{safe_title}.mp4"
        output_path = str(OUTPUT_DIR / output_filename)

    caption_segments = [
        (s.get("start"), s.get("end"), s.get("text"), s.get("words", []))
        for s in segments if burn_subtitles and s.get("end", 0) > start_time and s.get("start", 0) < end_time
    ]
    captions_hash = hashlib.sha256(json.dumps(caption_segments, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    render_key = make_cache_key(
        "av-sync-v1", file_identity(video_path), start_time, end_time, clip_info.get("title"),
        clip_info.get("keywords", []), layout_mode, subtitle_style, burn_subtitles,
        caption_font, caption_size, caption_position, caption_background,
        caption_text_color, caption_highlight_color, caption_animation,
        target_width, target_height, performance_mode, captions_hash,
    )
    if base_output_path is None:
        output_file = Path(output_path)
        output_path = str(output_file.with_name(f"{output_file.stem}_{render_key[:10]}{output_file.suffix}"))
    thumb_path = str(OUTPUT_DIR / f"{Path(output_path).stem}_thumb.jpg")
    if base_output_path is None and os.path.exists(output_path):
        try:
            cached_validation = validate_browser_mp4(
                output_path,
                expected_width=target_width,
                expected_height=target_height,
                expect_audio=False,
                expected_duration=duration,
            )
        except Exception:
            # A stale or truncated cached file is not ready; rerender it below.
            pass
        else:
            if not os.path.exists(thumb_path):
                generate_clip_thumbnail(output_path, thumb_path, timestamp=1.0)
            cache_hit = True
            cached_breakdown = dict(clip_info.get("quality_breakdown") or {})
            if layout_mode != "blurred_background":
                cached_breakdown["visual_quality"] = cached_breakdown.get("visual_quality", 60)
            return {
                "clip_index": clip_idx, "title": clip_info.get("title", f"Clip #{clip_idx}"),
                "output_path": output_path, "filename": os.path.basename(output_path),
                "thumbnail_path": thumb_path, "start_time": start_time, "end_time": end_time,
                "duration": round(duration, 2), "engagement_score": clip_info.get("engagement_score", 90),
                "quality_score": clip_info.get("quality_score"), "quality_breakdown": cached_breakdown,
                "framing_layout": layout_mode, "speaker_confidence": None,
                "hook": clip_info.get("hook", ""), "reason": clip_info.get("reason", ""),
                "keywords": clip_info.get("keywords", []),
                "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2),
                "cache_hit": True, "validation": cached_validation,
                "timings": {"render_seconds": 0.0, "validation_seconds": 0.0},
            }

    ffmpeg_exe = get_ffmpeg_path()

    if progress_callback:
        progress_callback(f"Clip #{clip_idx}: Computing speaker tracking...")

    # 1. Subtitles generation
    subtitles_filter_str = ""
    ass_path = None
    reframe_layout = layout_mode
    reframe_elapsed = 0.0
    if burn_subtitles:
        ass_path = str(TEMP_DIR / f"sub_{clip_idx}_{int(start_time)}.ass")
        generate_ass_subtitles(
            segments=segments,
            clip_start=start_time,
            clip_end=end_time,
            output_path=ass_path,
            style_name=subtitle_style,
            keywords=clip_info.get("keywords", []),
            font_name=caption_font,
            font_size=caption_size,
            position=caption_position,
            background=caption_background,
            text_color=caption_text_color,
            highlight_color=caption_highlight_color,
            animate=caption_animation,
        )
        escaped_ass = escape_ffmpeg_path(ass_path)
        subtitles_filter_str = f"subtitles='{escaped_ass}'"

    # 2. Build Filter Graph for 9:16 conversion
    filter_option = "-vf"
    if layout_mode == "blurred_background":
        if subtitles_filter_str:
            filter_graph = (
                f"[0:v]split=2[bg][fg];"
                f"[bg]scale={target_width}:{target_height}:force_original_aspect_ratio=increase,crop={target_width}:{target_height},boxblur=25:5[bg_blur];"
                f"[fg]scale={target_width}:-2[fg_scaled];"
                f"[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2[merged];"
                f"[merged]{subtitles_filter_str}"
            )
        else:
            filter_graph = (
                f"[0:v]split=2[bg][fg];"
                f"[bg]scale={target_width}:{target_height}:force_original_aspect_ratio=increase,crop={target_width}:{target_height},boxblur=25:5[bg_blur];"
                f"[fg]scale={target_width}:-2[fg_scaled];"
                f"[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2"
            )

    else:
        # Dynamic mode aligns visual mouth movement with Whisper speech intervals.
        reframe_started = time.perf_counter()
        plan = reframe_plan or compute_dynamic_reframe_plan(
            video_path, start_time, end_time,
            speech_segments=[s for s in segments if s.get("end", 0) > start_time and s.get("start", 0) < end_time],
            sample_interval=face_sample_interval(performance_mode),
            inference_width=face_inference_width(performance_mode),
        )
        reframe_elapsed = time.perf_counter() - reframe_started
        use_split = layout_mode == "split_screen" or (layout_mode == "smart_dynamic" and plan["layout"] == "split")
        if use_split:
            split_graph = _split_screen_filter(plan, target_width, target_height, subtitles_filter_str)
        else:
            split_graph = ""
        if split_graph:
            filter_graph = split_graph
            filter_option = "-filter_complex"
            reframe_layout = "split_screen"
        else:
            x_expr = _crop_x_expression(plan["points"])
            filters = [
                f"crop={plan['crop_w']}:{plan['crop_h']}:'{x_expr}':0",
                f"scale={target_width}:{target_height}:flags=lanczos",
            ]
            if subtitles_filter_str:
                filters.append(subtitles_filter_str)
            filter_graph = ",".join(filters)
            reframe_layout = "single_speaker"

    if progress_callback:
        progress_callback(f"Clip #{clip_idx}: Rendering 9:16 vertical video & subtitles...")

    # Execute FFmpeg command
    encoder, encoder_options = select_video_encoder(performance_mode)
    try:
        import av
        with av.open(video_path) as media:
            audio_stream = next((stream for stream in media.streams if stream.type == "audio"), None)
            audio_codec = audio_stream.codec_context.name if audio_stream else None
    except Exception:
        audio_codec = None
    audio_options = ["-c:a", "copy"] if audio_codec == "aac" else ["-c:a", "aac", "-b:a", "192k", "-ar", "44100"]
    render_started = time.perf_counter()
    ffmpeg = get_ffmpeg_path()
    temporary_output_path = str(
        Path(output_path).with_name(f".{Path(output_path).stem}.{uuid.uuid4().hex}.partial.mp4")
    )
    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", video_path,
        filter_option,
        filter_graph,
        "-c:v", encoder,
        *encoder_options,
        "-pix_fmt", "yuv420p",
        "-r", str(TARGET_FPS),
        *audio_options,
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        *( ["-map", "[outv]", "-map", "0:a?"] if filter_option == "-filter_complex" and layout_mode != "blurred_background" else [] ),
        temporary_output_path,
    ]

    def run_ffmpeg(command):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Clip rendering was cancelled.")
        current = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        while current.poll() is None:
            try:
                current.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                if cancel_event is not None and cancel_event.is_set():
                    current.terminate()
                    try:
                        current.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        current.kill()
                    raise InterruptedError("Clip rendering was cancelled.")
        return current.returncode, current.stderr.read() if current.stderr else ""

    validation = None
    validation_elapsed = 0.0
    timestamp_normalization = {"applied": False}
    timestamp_normalization_elapsed = 0.0
    try:
        while True:
            return_code, stderr = run_ffmpeg(cmd)
            if return_code == 0:
                timestamp_started = time.perf_counter()
                try:
                    timestamp_normalization = normalize_clip_timestamps(
                        temporary_output_path, run_ffmpeg=run_ffmpeg
                    )
                except Exception as error:
                    stderr = f"Audio/video timestamp normalization failed: {error}"
                    return_code = 1
                finally:
                    timestamp_normalization_elapsed += time.perf_counter() - timestamp_started

            if return_code == 0:
                validation_started = time.perf_counter()
                try:
                    validation = validate_browser_mp4(
                        temporary_output_path,
                        expected_width=target_width,
                        expected_height=target_height,
                        expect_audio=audio_stream is not None,
                        expected_duration=duration,
                    )
                    validation["timestamp_normalization"] = timestamp_normalization
                except Exception as error:
                    stderr = f"Post-render MP4 validation failed: {error}"
                    return_code = 1
                finally:
                    validation_elapsed += time.perf_counter() - validation_started

            if return_code == 0:
                # Only expose a complete, decodable file to the UI.
                os.replace(temporary_output_path, output_path)
                break

            try:
                Path(temporary_output_path).unlink(missing_ok=True)
            except OSError:
                pass

            if encoder != "libx264":
                # Device, driver, or output-validation failures retry with software x264.
                fallback_options = ["-preset", "superfast", "-crf", "24"] if performance_mode == "FAST MODE" else ["-preset", "faster", "-crf", "22"]
                video_option_index = cmd.index("-c:v")
                pixel_format_index = cmd.index("-pix_fmt")
                cmd = cmd[:video_option_index] + ["-c:v", "libx264", *fallback_options] + cmd[pixel_format_index:]
                encoder = "libx264"
                continue

            raise RuntimeError(f"FFmpeg rendering or MP4 validation failed for clip #{clip_idx}:\n{stderr}")
    except Exception:
        try:
            Path(temporary_output_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise

    render_elapsed = time.perf_counter() - render_started

    # Generate thumbnail
    thumb_path = str(OUTPUT_DIR / f"{Path(output_path).stem}_thumb.jpg")
    generate_clip_thumbnail(output_path, thumb_path, timestamp=1.0)

    # Clean temporary ASS file
    if ass_path and os.path.exists(ass_path):
        try:
            os.remove(ass_path)
        except Exception:
            pass

    quality_breakdown = dict(clip_info.get("quality_breakdown") or {})
    visual_quality = None
    if layout_mode != "blurred_background":
        face_count = len(plan.get("track_centers", []))
        visual_quality = min(100, 45 + face_count * 12 + int(plan.get("confidence", 0) * 30))
        quality_breakdown["visual_quality"] = visual_quality
    quality_score = clip_info.get("quality_score")
    if quality_score is not None and visual_quality is not None:
        quality_score = int(round(quality_score * 0.9 + visual_quality * 0.1))
        quality_breakdown["overall"] = quality_score

    return {
        "clip_index": clip_idx,
        "title": clip_info.get("title", f"Clip #{clip_idx}"),
        "output_path": output_path,
        "filename": os.path.basename(output_path),
        "thumbnail_path": thumb_path,
        "start_time": start_time,
        "end_time": end_time,
        "duration": round(duration, 2),
        "engagement_score": clip_info.get("engagement_score", 90),
        "quality_score": quality_score,
        "quality_breakdown": quality_breakdown,
        "framing_layout": reframe_layout,
        "speaker_confidence": round(float(plan.get("confidence", 0)), 2) if layout_mode != "blurred_background" else None,
        "cache_hit": cache_hit,
        "video_encoder": encoder,
        "validation": validation,
        "timings": {
            "reframe_seconds": round(reframe_elapsed, 3),
            "render_seconds": round(render_elapsed, 3),
            "validation_seconds": round(validation_elapsed, 3),
            "timestamp_normalization_seconds": round(timestamp_normalization_elapsed, 3),
        },
        "hook": clip_info.get("hook", ""),
        "reason": clip_info.get("reason", ""),
        "keywords": clip_info.get("keywords", []),
        "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2),
    }


def process_all_clips(
    video_path: str,
    clips: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    layout_mode: str = "smart_crop",
    subtitle_style: str = "Neon Gold",
    burn_subtitles: bool = True,
    caption_font: Optional[str] = None,
    caption_size: Optional[int] = None,
    caption_position: str = "Bottom",
    caption_background: Optional[bool] = None,
    caption_text_color: Optional[str] = None,
    caption_highlight_color: Optional[str] = None,
    caption_animation: bool = True,
    performance_mode: str = "BALANCED MODE",
    cancel_event=None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Analyze framing once per clip, then render independent clips with bounded workers.
    """
    total_clips = len(clips)
    if total_clips == 0:
        return []
    plans = [None] * total_clips
    for idx, clip in enumerate(clips):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Clip processing was cancelled.")
        if layout_mode != "blurred_background":
            if progress_callback:
                progress_callback(0.05 * idx / total_clips, f"Analyzing framing for clip {idx + 1} of {total_clips}...")
            plans[idx] = compute_dynamic_reframe_plan(
                video_path,
                clip["start_time"],
                clip["end_time"],
                speech_segments=[s for s in segments if s.get("end", 0) > clip["start_time"] and s.get("start", 0) < clip["end_time"]],
                sample_interval=face_sample_interval(performance_mode),
                inference_width=face_inference_width(performance_mode),
            )

    worker_count = min(total_clips, clip_worker_count(performance_mode))
    processed = [None] * total_clips
    futures = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="clip-render") as executor:
        for idx, clip in enumerate(clips):
            futures[executor.submit(
                render_single_clip,
                video_path=video_path,
                clip_info=clip,
                segments=segments,
                layout_mode=layout_mode,
                subtitle_style=subtitle_style,
                burn_subtitles=burn_subtitles,
                caption_font=caption_font,
                caption_size=caption_size,
                caption_position=caption_position,
                caption_background=caption_background,
                caption_text_color=caption_text_color,
                caption_highlight_color=caption_highlight_color,
                caption_animation=caption_animation,
                performance_mode=performance_mode,
                reframe_plan=plans[idx],
                cancel_event=cancel_event,
            )] = idx

        completed = 0
        for future in as_completed(futures):
            if cancel_event is not None and cancel_event.is_set():
                for pending in futures:
                    pending.cancel()
                raise InterruptedError("Clip processing was cancelled.")
            idx = futures[future]
            processed[idx] = future.result()
            completed += 1
            if progress_callback:
                progress_callback(0.05 + 0.95 * completed / total_clips,
                                  f"Finished clip {completed} of {total_clips}: {clips[idx].get('title', '')}")

    processed_clips = [item for item in processed if item is not None]
    if progress_callback:
        progress_callback(1.0, f"All {total_clips} clips rendered successfully!")

    return processed_clips


def create_clips_zip(clip_records: List[Dict[str, Any]], zip_filename: str = "podcast_clips.zip") -> str:
    """
    Creates a ZIP archive containing all rendered MP4 clips.
    """
    import zipfile
    zip_path = str(OUTPUT_DIR / zip_filename)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for c in clip_records:
            fpath = c.get("output_path")
            if fpath and os.path.exists(fpath):
                zipf.write(fpath, arcname=os.path.basename(fpath))
    return zip_path

