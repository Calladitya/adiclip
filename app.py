"""
PodClip AI - Streamlit Application
AI-powered Podcast Clipping & Viral Shorts Generator.
"""
import os
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import streamlit as st

from core.config import (
    DEFAULT_WHISPER_MODEL,
    AVAILABLE_WHISPER_MODELS,
    DEFAULT_NUM_CLIPS,
    MIN_CLIP_DURATION,
    MAX_CLIP_DURATION,
    DEFAULT_GEMINI_MODEL,
)
from core.downloader import (
    fetch_youtube_info,
    download_youtube_video,
    save_uploaded_file,
    get_video_metadata,
)
from core.transcriber import transcribe_video
from core.analyzer import identify_engaging_moments
from core.subtitle_generator import SUBTITLE_STYLES, CAPTION_COLORS
from core.video_processor import process_all_clips, create_clips_zip
from core.performance import process_resource_snapshot

_pipeline_slot_lock = threading.Lock()


@st.cache_resource
def _pipeline_executor():
    """One bounded pipeline worker per Streamlit server process."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="podclip-pipeline")


def _run_pipeline_job(source, settings, progress_queue, cancel_event, started_event):
    started_event.set()
    progress_queue.put((0.01, "Pipeline worker started. Preparing source..."))
    total_started = time.perf_counter()
    resource_start = process_resource_snapshot()
    timings = {}
    def report(progress, message):
        progress_queue.put((None if progress is None else max(0.0, min(1.0, float(progress))), str(message)))

    def ensure_not_cancelled():
        if cancel_event.is_set():
            raise InterruptedError("Processing was cancelled by the user.")

    stage_started = time.perf_counter()
    if source["kind"] == "upload":
        report(0.02, "Saving uploaded source video...")
        video_meta = save_uploaded_file(source["file"])
    else:
        report(0.02, "Initializing download and connecting to YouTube...")

        def download_progress(pct, message):
            ensure_not_cancelled()
            report(None if pct is None else 0.02 + min(1.0, pct) * 0.18, message)

        video_meta = download_youtube_video(
            source["url"],
            progress_callback=download_progress,
            video_info=source.get("info"),
            cancel_event=cancel_event,
        )
    timings["ingest_seconds"] = round(time.perf_counter() - stage_started, 3)
    ensure_not_cancelled()

    video_path = video_meta["path"]
    report(0.20, "Transcribing speech...")

    def whisper_progress(pct, message):
        report(0.20 + min(1.0, pct) * 0.36, message)

    transcript_data = transcribe_video(
        video_path=video_path,
        model_size=settings["whisper_model"],
        progress_callback=whisper_progress,
        performance_mode=settings["performance_mode"],
        cancel_event=cancel_event,
    )
    timings.update(transcript_data.get("timings", {}))
    ensure_not_cancelled()
    report(0.58, "Selecting complete, engaging clip moments...")
    analysis_started = time.perf_counter()
    candidate_clips = identify_engaging_moments(
        transcript_data=transcript_data,
        api_key=settings["gemini_key"],
        num_clips=settings["num_clips"],
        gemini_model=settings["gemini_model"],
        use_gemini=settings["use_gemini"],
    )
    timings["clip_selection_seconds"] = round(time.perf_counter() - analysis_started, 3)
    if not candidate_clips:
        raise RuntimeError("No suitable clips were found. Ensure the video contains enough spoken content.")
    ensure_not_cancelled()

    def render_progress(pct, message):
        report(0.62 + min(1.0, pct) * 0.36, message)

    rendering_started = time.perf_counter()
    final_clips = process_all_clips(
        video_path=video_path,
        clips=candidate_clips,
        segments=transcript_data["segments"],
        layout_mode=settings["layout_mode"],
        subtitle_style=settings["subtitle_style"],
        burn_subtitles=settings["burn_subtitles"],
        caption_font=settings["caption_font"],
        caption_size=settings["caption_size"],
        caption_position=settings["caption_position"],
        caption_background=settings["caption_background"],
        caption_text_color=settings["caption_text_color"],
        caption_highlight_color=settings["caption_highlight_color"],
        caption_animation=settings["caption_animation"],
        performance_mode=settings["performance_mode"],
        cancel_event=cancel_event,
        progress_callback=render_progress,
    )
    timings["render_wall_seconds"] = round(time.perf_counter() - rendering_started, 3)
    timings["face_analysis_seconds"] = round(sum(c.get("timings", {}).get("reframe_seconds", 0) for c in final_clips), 3)
    timings["clip_encode_seconds_sum"] = round(sum(c.get("timings", {}).get("render_seconds", 0) for c in final_clips), 3)
    ensure_not_cancelled()
    report(0.99, "Packaging clips for download...")
    zip_started = time.perf_counter()
    zip_file = create_clips_zip(final_clips)
    timings["zip_seconds"] = round(time.perf_counter() - zip_started, 3)
    timings["pipeline_total_seconds"] = round(time.perf_counter() - total_started, 3)
    resource_end = process_resource_snapshot()
    if resource_start.get("cpu_seconds") is not None and resource_end.get("cpu_seconds") is not None:
        cpu_delta = max(0.0, resource_end["cpu_seconds"] - resource_start["cpu_seconds"])
        timings["process_cpu_seconds"] = round(cpu_delta, 3)
        timings["average_process_cpu_percent"] = round(100 * cpu_delta / max(timings["pipeline_total_seconds"], 0.001) / max(os.cpu_count() or 1, 1), 1)
    if resource_end.get("peak_working_set_mb") is not None:
        timings["process_peak_working_set_mb"] = round(resource_end["peak_working_set_mb"], 1)
    report(1.0, "All clips are ready.")
    return {
        "video_metadata": video_meta,
        "transcript_data": transcript_data,
        "candidate_clips": candidate_clips,
        "final_clips": final_clips,
        "zip_path": zip_file,
        "pipeline_timings": timings,
    }


@st.fragment(run_every="1s")
def _pipeline_progress_fragment():
    job = st.session_state.get("pipeline_job")
    if not job:
        return
    while True:
        try:
            progress, message = job["queue"].get_nowait()
            if progress is not None:
                st.session_state.pipeline_progress = progress
            st.session_state.pipeline_message = message
        except queue.Empty:
            break
    if not job["started"].is_set() and not job["future"].done():
        waited = int(time.monotonic() - job["submitted_at"])
        st.session_state.pipeline_message = f"Waiting for the processing worker ({waited}s). Another job is active."
    st.progress(st.session_state.get("pipeline_progress", 0.0))
    st.info(st.session_state.get("pipeline_message", "Processing video..."))
    if not job["future"].done():
        if st.button("Cancel processing", key="cancel_pipeline"):
            job["cancel"].set()
            if job["future"].cancel():
                st.session_state.pipeline_message = "Queued job cancelled."
            else:
                st.session_state.pipeline_message = "Cancellation requested. The active stage will stop at its next safe checkpoint."
            st.warning(st.session_state.pipeline_message)
        return
    try:
        result = job["future"].result()
        for key, value in result.items():
            setattr(st.session_state, key, value)
        st.session_state.pipeline_error = None
    except Exception as error:
        st.session_state.pipeline_error = str(error)
    st.session_state.pipeline_job = None
    st.rerun()

# Page Configuration
st.set_page_config(
    page_title="PodClip AI - Podcast to Viral Shorts",
    page_icon="ðŸŽ™ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Modern CSS
st.markdown("""
<style>
    /* Dark Slate Theme */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Branding */
    .hero-container {
        background: linear-gradient(135deg, #161b22 0%, #1f2937 50%, #111827 100%);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 26px 32px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #8b949e;
        line-height: 1.5;
    }
    .pill-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 10px;
    }

    /* Card styling */
    .clip-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .clip-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
    }
    .clip-badge {
        background-color: #238636;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .score-badge {
        background: linear-gradient(90deg, #ec4899, #8b5cf6);
        color: white;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.78rem;
    }
    .keyword-tag {
        display: inline-block;
        background-color: #21262d;
        color: #58a6ff;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-right: 5px;
        margin-top: 4px;
    }
    
    /* Button enhancements */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #4338ca 0%, #6d28d9 100%);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "video_metadata" not in st.session_state:
    st.session_state.video_metadata = None
if "transcript_data" not in st.session_state:
    st.session_state.transcript_data = None
if "candidate_clips" not in st.session_state:
    st.session_state.candidate_clips = None
if "final_clips" not in st.session_state:
    st.session_state.final_clips = None
if "zip_path" not in st.session_state:
    st.session_state.zip_path = None
if "youtube_info" not in st.session_state:
    st.session_state.youtube_info = None
if "youtube_info_url" not in st.session_state:
    st.session_state.youtube_info_url = None
if "pipeline_job" not in st.session_state:
    st.session_state.pipeline_job = None
if "pipeline_progress" not in st.session_state:
    st.session_state.pipeline_progress = 0.0
if "pipeline_message" not in st.session_state:
    st.session_state.pipeline_message = ""
if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None
if "pipeline_timings" not in st.session_state:
    st.session_state.pipeline_timings = {}

# Sidebar Configuration
st.sidebar.markdown("## âš™ï¸ Pipeline Settings")

# 1. AI Moment Selection Engine
st.sidebar.markdown("### ðŸ§  AI Highlight Engine")
ai_provider = st.sidebar.radio(
    "Choose AI Provider:",
    ["Free Local AI (Offline)", "Google Gemini API"],
    index=0,
    help="Free Local AI analyzes speech pace, questions, and emotional buzzwords without an API key. Gemini API uses an LLM to evaluate viral potential."
)

gemini_key = ""
gemini_model_choice = DEFAULT_GEMINI_MODEL
if ai_provider == "Google Gemini API":
    gemini_key = st.sidebar.text_input(
        "Gemini API Key:",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Get a free Gemini API key from https://aistudio.google.com"
    )
    gemini_model_choice = st.sidebar.selectbox(
        "Gemini Model:",
        ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )
    if not gemini_key:
        st.sidebar.caption("âš ï¸ Enter an API key or switch to Free Local AI.")

# 2. Clips & Duration
num_clips = st.sidebar.selectbox("Number of clips to generate:", [5, 10, 20], index=1)
st.sidebar.info(f"â±ï¸ Clip duration: **{int(MIN_CLIP_DURATION)}s - {int(MAX_CLIP_DURATION)}s** (Standard for Shorts & Reels)")

performance_mode = st.sidebar.selectbox(
    "Performance mode:",
    ["FAST MODE", "BALANCED MODE", "HIGH QUALITY MODE"],
    index=1,
    help="Fast mode uses lower Whisper effort, wider face sampling, and a tested hardware encoder when available. High quality uses slower software encoding.",
)

# 3. Whisper Model
whisper_model = st.sidebar.selectbox(
    "Whisper Model Size:",
    AVAILABLE_WHISPER_MODELS,
    index=AVAILABLE_WHISPER_MODELS.index({"FAST MODE": "tiny", "BALANCED MODE": DEFAULT_WHISPER_MODEL, "HIGH QUALITY MODE": "small"}[performance_mode]),
    help="'base' provides strong accuracy on CPU. 'tiny' is ultra fast."
)

# 4. Framing & Subtitles
st.sidebar.markdown("### ðŸŽ¬ Video & Subtitles")
layout_choice = st.sidebar.selectbox(
    "Vertical Format Layout (9:16):",
    [
        "🧠 Smart Dynamic (auto single / split)",
        "👤 Single Speaker Follow",
        "👥 Two-Person Split Screen",
        "ðŸŽ¨ Blurred Background + Centered Video",
    ],
    index=0,
    help="Smart Dynamic follows the visually active face during Whisper speech and switches to a two-person split when both speakers are detected. Ambiguous moments hold the previous framing."
)
layout_mode = (
    "single_speaker" if "Single Speaker" in layout_choice else
    "split_screen" if "Split Screen" in layout_choice else
    "blurred_background" if "Blurred Background" in layout_choice else
    "smart_dynamic"
)

burn_subtitles = st.sidebar.checkbox("Add Synchronized Karaoke Subtitles", value=True)
subtitle_style = "Keyword Focus"
caption_font = "Arial Black"
caption_size = 58
caption_position = "Bottom"
caption_background = None
caption_text_color = "White"
caption_highlight_color = "Gold"
caption_animation = True
if burn_subtitles:
    subtitle_style = st.sidebar.selectbox(
        "Caption preset:", list(SUBTITLE_STYLES.keys()),
        index=list(SUBTITLE_STYLES.keys()).index("Keyword Focus"),
    )
    with st.sidebar.expander("Caption customization"):
        caption_font = st.selectbox("Font", ["Arial Black", "Arial", "Trebuchet MS", "DejaVu Sans"], index=0)
        caption_size = st.slider("Font size", min_value=38, max_value=84, value=SUBTITLE_STYLES[subtitle_style].get("size", 58))
        caption_position = st.selectbox("Position", ["Top", "Middle", "Bottom"], index=2)
        caption_background = st.checkbox("Caption background box", value=SUBTITLE_STYLES[subtitle_style].get("background", False))
        caption_text_color = st.selectbox("Text color", list(CAPTION_COLORS), index=0)
        caption_highlight_color = st.selectbox("Highlight color", list(CAPTION_COLORS), index=1)
        caption_animation = st.checkbox("Active-word pop animation", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size: 0.8rem; color: #8b949e;'>
    <b>PodClip AI v1.0</b><br>
    â€¢ No paid API required<br>
    â€¢ Watermark-free MP4 exports<br>
    â€¢ FFmpeg + Whisper + OpenCV YuNet
</div>
""", unsafe_allow_html=True)


# Main Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">ðŸŽ™ï¸ PodClip AI</div>
    <div class="hero-sub">
        Turn long-form podcasts into engaging 9:16 vertical clips (30-60s) with 
        intelligent speaker face tracking and animated karaoke subtitles.
    </div>
    <div>
        <span class="pill-badge">ðŸ¤– Whisper Transcription</span>
        <span class="pill-badge">ðŸ‘¤ YuNet Face Auto-Framing</span>
        <span class="pill-badge">âœ¨ Word-Sync Subtitles</span>
        <span class="pill-badge">ðŸš« 100% Watermark-Free</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Step 1: Video Input
st.markdown("### ðŸ“¥ Step 1: Select Input Video")
input_tab1, input_tab2 = st.tabs(["ðŸ”— YouTube URL", "ðŸ“ Upload MP4 File"])

video_file_to_process = None
video_display_title = ""

with input_tab1:
    yt_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
    if yt_url:
        col_meta, col_btn = st.columns([3, 1])
        with col_btn:
            fetch_btn = st.button("ðŸ” Check Video", key="yt_check")

        if fetch_btn:
            try:
                with st.spinner("Fetching YouTube video details..."):
                    yt_info = fetch_youtube_info(yt_url)
                st.session_state.youtube_info = yt_info
                st.session_state.youtube_info_url = yt_url.strip()
            except Exception as e:
                st.session_state.youtube_info = None
                st.session_state.youtube_info_url = None
                st.error(f"Error fetching YouTube info: {str(e)}")

        yt_info = st.session_state.youtube_info
        if yt_info and st.session_state.youtube_info_url == yt_url.strip():
            c_thumb, c_details = st.columns([1, 2])
            with c_thumb:
                if yt_info.get("thumbnail"):
                    st.image(yt_info["thumbnail"], use_container_width=True)
            with c_details:
                st.markdown(f"**Title:** {yt_info['title']}")
                st.markdown(f"**Author:** {yt_info['author']}")
                dur_min = int(yt_info['duration'] // 60)
                dur_sec = int(yt_info['duration'] % 60)
                st.markdown(f"**Duration:** {dur_min}m {dur_sec}s")

with input_tab2:
    uploaded_file = st.file_uploader(
        "Upload local podcast video (MP4, MOV, MKV, AVI):",
        type=["mp4", "mov", "mkv", "avi"],
        help="Upload any local podcast video file without YouTube limits."
    )
    if uploaded_file:
        st.success(f"Selected file: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.1f} MB)")


# Processing Action Trigger
st.markdown("---")
st.markdown("### ðŸš€ Step 2: Generate Viral Clips")

can_process = bool(yt_url.strip() or uploaded_file is not None)
if not can_process:
    st.info("ðŸ‘† Please enter a YouTube URL or upload an MP4 file above to begin.")

col_start, col_dummy = st.columns([1, 2])
with col_start:
    active_job = st.session_state.pipeline_job is not None
    start_button = st.button(
        "âš¡ Start AI Clipping Pipeline",
        disabled=not can_process or active_job,
        use_container_width=True,
    )

if start_button:
    if not _pipeline_slot_lock.acquire(blocking=False):
        st.session_state.pipeline_error = "Another processing job is already active on this server. Wait for it to finish before starting another."
    else:
        st.session_state.final_clips = None
        st.session_state.zip_path = None
        st.session_state.candidate_clips = None
        st.session_state.pipeline_timings = {}
        st.session_state.pipeline_error = None
        source = {"kind": "upload", "file": uploaded_file} if uploaded_file is not None else {"kind": "youtube", "url": yt_url.strip()}
        if source["kind"] == "youtube" and st.session_state.youtube_info_url == source["url"]:
            source["info"] = st.session_state.youtube_info
        settings = {
            "whisper_model": whisper_model,
            "performance_mode": performance_mode,
            "num_clips": num_clips,
            "gemini_key": gemini_key,
            "gemini_model": gemini_model_choice,
            "use_gemini": ai_provider == "Google Gemini API",
            "layout_mode": layout_mode,
            "subtitle_style": subtitle_style,
            "burn_subtitles": burn_subtitles,
            "caption_font": caption_font,
            "caption_size": caption_size,
            "caption_position": caption_position,
            "caption_background": caption_background,
            "caption_text_color": caption_text_color,
            "caption_highlight_color": caption_highlight_color,
            "caption_animation": caption_animation,
        }
        event_queue = queue.Queue()
        cancel_signal = threading.Event()
        started_event = threading.Event()
        submitted_at = time.monotonic()
        future = _pipeline_executor().submit(_run_pipeline_job, source, settings, event_queue, cancel_signal, started_event)
        future.add_done_callback(lambda _future: _pipeline_slot_lock.release())
        st.session_state.pipeline_job = {
            "future": future, "queue": event_queue, "cancel": cancel_signal,
            "started": started_event, "submitted_at": submitted_at,
        }
        st.session_state.pipeline_progress = 0.0
        st.session_state.pipeline_message = "Queued for the processing worker..."

if st.session_state.pipeline_job is not None:
    _pipeline_progress_fragment()

if st.session_state.pipeline_error:
    st.error(f"Processing failed: {st.session_state.pipeline_error}")
if st.session_state.pipeline_timings:
    with st.expander("Measured pipeline performance"):
        st.json(st.session_state.pipeline_timings)


# Step 3: Showcase Results
if st.session_state.final_clips:
    clips = st.session_state.final_clips
    st.markdown("---")
    
    col_hdr, col_dl_all = st.columns([3, 1])
    with col_hdr:
        st.markdown(f"## ðŸ† Generated Shorts ({len(clips)} Clips)")
    with col_dl_all:
        if st.session_state.zip_path and os.path.exists(st.session_state.zip_path):
            with open(st.session_state.zip_path, "rb") as zf:
                st.download_button(
                    label="ðŸ“¦ Download All (.ZIP)",
                    data=zf,
                    file_name=os.path.basename(st.session_state.zip_path),
                    mime="application/zip",
                    use_container_width=True,
                )

    # Display clips in a 2-column responsive grid
    cols = st.columns(2)
    for idx, clip in enumerate(clips):
        c = cols[idx % 2]
        with c:
            st.markdown(f"""
            <div class="clip-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="clip-badge">CLIP #{clip['clip_index']:02d}</span>
                    <span class="score-badge">Quality: {clip.get('quality_score', clip['engagement_score'])}/100</span>
                </div>
                <h4 style="margin: 4px 0 8px 0; color: #f0f6fc;">{clip['title']}</h4>
                <div style="color: #8b949e; font-size: 0.85rem; margin-bottom: 12px;">
                    â±ï¸ <b>Duration:</b> {clip['duration']}s ({clip['start_time']}s - {clip['end_time']}s) &nbsp;â€¢&nbsp; 
                    ðŸ’¾ <b>Size:</b> {clip['file_size_mb']} MB
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Video Player
            if os.path.exists(clip["output_path"]):
                if os.path.getsize(clip["output_path"]) <= 0:
                    st.error(f"Clip file is empty: {clip['filename']}. Run processing again to regenerate it.")
                else:
                    st.video(clip["output_path"])

                    # Download individual clip button
                    with open(clip["output_path"], "rb") as vid_file:
                        st.download_button(
                            label=f"â¬‡ï¸ Download {clip['filename']}",
                            data=vid_file,
                            file_name=clip["filename"],
                            mime="video/mp4",
                            key=f"dl_btn_{idx}",
                            use_container_width=True,
                        )
            else:
                st.error(f"Clip file is missing: {clip['filename']}. Run processing again to regenerate it.")

            # Metadata Expander
            with st.expander(f"â„¹ï¸ Hook & Keywords details", expanded=False):
                if clip.get("hook"):
                    st.markdown(f"**Hook:** {clip['hook']}")
                if clip.get("reason"):
                    st.markdown(f"**Why this works:** {clip['reason']}")
                if clip.get("quality_breakdown"):
                    quality = clip["quality_breakdown"]
                    st.markdown(
                        "**Quality:** "
                        f"Hook {quality.get('hook_strength', '—')} · "
                        f"Engagement {quality.get('engagement', '—')} · "
                        f"Context {quality.get('context_completeness', '—')} · "
                        f"Audio {quality.get('audio_quality', '—')} · "
                        f"Framing {quality.get('visual_quality') if quality.get('visual_quality') is not None else 'not assessed'} · "
                        f"Duration {quality.get('duration_suitability', '—')}"
                    )
                if clip.get("framing_layout"):
                    confidence = clip.get("speaker_confidence")
                    confidence_text = f" · speaker cue confidence {round(confidence * 100)}%" if confidence is not None else ""
                    st.caption(f"Framing: {clip['framing_layout'].replace('_', ' ')}{confidence_text}")
                if clip.get("keywords"):
                    st.markdown("**Highlighted Keywords:** " + " ".join([f"`{k}`" for k in clip['keywords']]))

            st.markdown("<br>", unsafe_allow_html=True)
