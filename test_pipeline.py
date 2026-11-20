"""
Comprehensive integration test for the PodClip AI pipeline components.
"""
import os
import sys
import subprocess
from pathlib import Path

from core.config import get_ffmpeg_path, OUTPUT_DIR, TEMP_DIR
from core.downloader import get_video_metadata
from core.transcriber import extract_audio, get_whisper_model
from core.analyzer import analyze_with_local_heuristics
from core.face_tracker import compute_smart_crop_window
from core.subtitle_generator import generate_ass_subtitles
from core.video_processor import render_single_clip, create_clips_zip


def run_tests():
    print("=" * 60)
    print("🧪 Running PodClip AI Pipeline Integration Tests")
    print("=" * 60)

    # 1. FFmpeg verification
    ffmpeg_exe = get_ffmpeg_path()
    print(f"[1/7] FFmpeg path verified: {ffmpeg_exe}")
    assert os.path.exists(ffmpeg_exe), "FFmpeg executable does not exist!"

    # 2. Create a test video if not exists
    test_vid = "test_pipeline_video.mp4"
    if not os.path.exists(test_vid):
        print("[2/7] Generating synthetic test video with audio...")
        cmd = [
            ffmpeg_exe, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=5",
            "-c:v", "libx264", "-c:a", "aac", test_vid
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"[2/7] Test video ready: {test_vid}")

    # 3. Metadata extraction
    meta = get_video_metadata(test_vid)
    print(f"[3/7] Video metadata extracted: {meta['width']}x{meta['height']}, {meta['duration']}s, {meta['fps']}fps")
    assert meta["width"] == 1280 and meta["height"] == 720

    # 4. Audio extraction
    wav_path = extract_audio(test_vid)
    print(f"[4/7] Audio extracted: {wav_path} ({os.path.getsize(wav_path)} bytes)")
    assert os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000

    # 5. Smart face tracking & cropping
    crop_info = compute_smart_crop_window(test_vid, 0.0, 3.0)
    print(f"[5/7] Smart 9:16 crop window calculated: {crop_info['crop_w']}x{crop_info['crop_h']} at x={crop_info['crop_x']}")
    assert crop_info["crop_w"] > 0 and crop_info["crop_h"] > 0

    # 6. Subtitles & Local AI Analysis
    mock_segments = [
        {
            "id": i,
            "start": round(i * 10.0, 2),
            "end": round((i + 1) * 10.0, 2),
            "text": f"This is segment {i} discussing why failure and crazy secrets lead to success.",
            "words": [
                {"word": "This", "start": round(i * 10.0, 2), "end": round(i * 10.0 + 0.5, 2), "probability": 0.95},
                {"word": "failure", "start": round(i * 10.0 + 1.0, 2), "end": round(i * 10.0 + 1.5, 2), "probability": 0.95},
                {"word": "success", "start": round((i + 1) * 10.0 - 1.0, 2), "end": round((i + 1) * 10.0, 2), "probability": 0.95},
            ]
        }
        for i in range(10)
    ]
    clips = analyze_with_local_heuristics(mock_segments, num_clips=5)
    print(f"[6/7] AI Moment Detection extracted {len(clips)} candidate clips between 30-60s.")
    assert len(clips) > 0
    for c in clips:
        assert 30.0 <= c["duration"] <= 60.0 or c["duration"] == mock_segments[-1]["end"]

    # 7. Render 9:16 clip with subtitles
    print("[7/7] Rendering 9:16 vertical short with animated subtitles...")
    sample_clip = clips[0]
    sample_clip["start_time"] = 0.0
    sample_clip["end_time"] = 3.0
    sample_clip["duration"] = 3.0

    rendered = render_single_clip(
        video_path=test_vid,
        clip_info=sample_clip,
        segments=mock_segments,
        subtitle_style="Neon Gold",
    )
    print(f"       -> Output: {rendered['filename']} ({rendered['file_size_mb']} MB)")
    assert os.path.exists(rendered["output_path"])
    assert os.path.exists(rendered["thumbnail_path"])

    # Clean up test video
    if os.path.exists(test_vid):
        try:
            os.remove(test_vid)
        except Exception:
            pass

    print("\n✅ ALL 7 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
