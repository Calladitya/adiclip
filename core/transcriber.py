"""
Audio extraction and speech transcription using Whisper (faster-whisper).
"""
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from faster_whisper import WhisperModel

from core.config import DEFAULT_WHISPER_MODEL, get_ffmpeg_path
from core.performance import cache_path, file_identity, load_json_cache, make_cache_key, save_json_cache

# Global cache for the loaded Whisper model
_LOADED_MODELS: Dict[str, WhisperModel] = {}


def get_whisper_model(model_size: str = DEFAULT_WHISPER_MODEL, device: str = "cpu") -> WhisperModel:
    """
    Returns a cached or newly initialized WhisperModel instance.
    """
    cache_key = f"{model_size}_{device}"
    if cache_key not in _LOADED_MODELS:
        # Use int8 compute type on CPU for optimal speed and memory efficiency
        compute_type = "int8" if device == "cpu" else "float16"
        _LOADED_MODELS[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=max(1, min(8, (os.cpu_count() or 4) - 2)),
        )
    return _LOADED_MODELS[cache_key]


def extract_audio(video_path: str, output_wav_path: Optional[str] = None, cancel_event=None) -> str:
    """
    Extracts 16kHz mono WAV audio from a video using FFmpeg.
    """
    if output_wav_path is None:
        identity = file_identity(video_path)
        key = make_cache_key(identity, "audio", 16000, "mono", "pcm_s16le")
        output_wav_path = str(cache_path("audio", key, ".wav"))
        if os.path.exists(output_wav_path):
            return output_wav_path

    ffmpeg_exe = get_ffmpeg_path()
    cmd = [
        ffmpeg_exe,
        "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path,
    ]

    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Audio extraction was cancelled.")
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace")
    while process.poll() is None:
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            if cancel_event is not None and cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                try:
                    Path(output_wav_path).unlink(missing_ok=True)
                except OSError:
                    pass
                raise InterruptedError("Audio extraction was cancelled.")
    stderr = process.stderr.read() if process.stderr else ""
    if process.returncode != 0:
        try:
            Path(output_wav_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{stderr}")

    return output_wav_path


def transcribe_video(
    video_path: str,
    model_size: str = DEFAULT_WHISPER_MODEL,
    device: str = "cpu",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    performance_mode: str = "BALANCED MODE",
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Extracts audio and transcribes the speech with word-level timestamps.
    """
    beam_size = {"FAST MODE": 1, "BALANCED MODE": 2, "HIGH QUALITY MODE": 5}.get(performance_mode, 2)
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Transcription was cancelled.")
    cache_key = make_cache_key(file_identity(video_path), "faster-whisper-v1", model_size, device, beam_size, True, True)
    cached_result = load_json_cache("transcripts", cache_key)
    if cached_result is not None:
        timings = cached_result.setdefault("timings", {})
        timings["cached_saved_audio_extraction_seconds"] = timings.get("audio_extraction_seconds")
        timings["cached_saved_transcription_seconds"] = timings.get("transcription_seconds")
        timings["audio_extraction_seconds"] = 0.0
        timings["transcription_seconds"] = 0.0
        timings["transcription_cache_hit"] = True
        if progress_callback:
            progress_callback(1.0, "Using cached transcript (same source and settings).")
        return cached_result

    extraction_started = time.perf_counter()
    if progress_callback:
        progress_callback(0.1, "Extracting audio track from video...")

    wav_path = extract_audio(video_path, cancel_event=cancel_event)
    extraction_elapsed = time.perf_counter() - extraction_started

    if progress_callback:
        progress_callback(0.3, f"Loading Whisper model ({model_size})...")

    model_load_started = time.perf_counter()
    model = get_whisper_model(model_size=model_size, device=device)
    model_load_elapsed = time.perf_counter() - model_load_started

    if progress_callback:
        progress_callback(0.5, "Transcribing speech and generating word timestamps...")

    transcription_started = time.perf_counter()
    segments_generator, info = model.transcribe(
        wav_path,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,  # Voice activity detection to trim silence
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    parsed_segments: List[Dict[str, Any]] = []
    full_transcript_parts = []
    total_duration = info.duration or 1.0

    for idx, seg in enumerate(segments_generator):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Transcription was cancelled.")
        words = []
        if seg.words:
            for w in seg.words:
                cleaned_word = w.word.strip()
                if cleaned_word:
                    words.append({
                        "word": cleaned_word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3),
                    })

        seg_dict = {
            "id": idx,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "duration": round(seg.end - seg.start, 2),
            "text": seg.text.strip(),
            "words": words,
        }
        parsed_segments.append(seg_dict)
        full_transcript_parts.append(seg.text.strip())

        if progress_callback and total_duration > 0:
            pct = min(0.95, 0.5 + 0.45 * (seg.end / total_duration))
            progress_callback(pct, f"Transcribing... ({seg.end:.1f}s / {total_duration:.1f}s)")

    if progress_callback:
        progress_callback(1.0, "Transcription completed successfully!")

    # Format transcript with timestamps for LLM inspection
    formatted_transcript = ""
    for seg in parsed_segments:
        m_start = int(seg["start"] // 60)
        s_start = int(seg["start"] % 60)
        m_end = int(seg["end"] // 60)
        s_end = int(seg["end"] % 60)
        formatted_transcript += f"[{m_start:02d}:{s_start:02d} - {m_end:02d}:{s_end:02d}] {seg['text']}\n"

    transcription_elapsed = time.perf_counter() - transcription_started
    result = {
        "audio_path": wav_path,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2),
        "full_text": " ".join(full_transcript_parts),
        "formatted_transcript": formatted_transcript,
        "segments": parsed_segments,
        "timings": {
            "audio_extraction_seconds": round(extraction_elapsed, 3),
            "model_load_seconds": round(model_load_elapsed, 3),
            "transcription_seconds": round(transcription_elapsed, 3),
            "transcription_cache_hit": False,
        },
    }
    save_json_cache("transcripts", cache_key, result)
    return result
