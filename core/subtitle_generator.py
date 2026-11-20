"""
Animated and synchronized subtitle generator using Advanced SubStation Alpha (.ass).
Creates modern, high-engagement TikTok/Reels style karaoke-like subtitles with keyword highlights.
"""
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.config import TEMP_DIR, TARGET_WIDTH, TARGET_HEIGHT

# Palette options for subtitle styling (ASS uses &HAABBGGRR color format)
SUBTITLE_STYLES = {
    "Bold Pop": {
        "primary": "&H00FFFFFF", "highlight": "&H0000D7FF",
        "keyword": "&H0000FFFF", "outline": "&H00000000",
        "font": "Arial Black", "size": 64, "outline_width": 5, "shadow": 2,
        "background": False,
    },
    "Minimal Podcast": {
        "primary": "&H00FFFFFF", "highlight": "&H00D8B15C",
        "keyword": "&H00D8B15C", "outline": "&H90000000",
        "font": "Arial", "size": 48, "outline_width": 2, "shadow": 0,
        "background": False,
    },
    "Keyword Focus": {
        "primary": "&H00FFFFFF", "highlight": "&H0000D7FF",
        "keyword": "&H0000FF66", "outline": "&H00000000",
        "font": "Arial Black", "size": 58, "outline_width": 4, "shadow": 2,
        "background": False,
    },
    "Interview Clean": {
        "primary": "&H00FFFFFF", "highlight": "&H00E6B85C",
        "keyword": "&H00E6B85C", "outline": "&H90000000",
        "font": "Arial", "size": 52, "outline_width": 2, "shadow": 0,
        "background": True,
    },
    "Neon Gold": {
        "primary": "&H00FFFFFF",      # Crisp White
        "highlight": "&H0000D7FF",    # Vivid Gold / Yellow
        "keyword": "&H0000FFFF",      # Bright Yellow
        "outline": "&H00000000",      # Deep Black
    },
    "Cyan Glow": {
        "primary": "&H00FFFFFF",
        "highlight": "&H00FFFF00",    # Electric Cyan
        "keyword": "&H00FFB200",      # Bright Teal
        "outline": "&H00000000",
    },
    "Emerald Green": {
        "primary": "&H00FFFFFF",
        "highlight": "&H0032FF00",    # Vibrant Lime Green
        "keyword": "&H0000FF66",      # Bright Mint Green
        "outline": "&H00000000",
    },
    "Electric Magenta": {
        "primary": "&H00FFFFFF",
        "highlight": "&H00FF00EA",    # Neon Pink / Magenta
        "keyword": "&H00FF50D0",      # Vivid Rose
        "outline": "&H00000000",
    },
}

CAPTION_COLORS = {
    "White": "&H00FFFFFF",
    "Gold": "&H0000D7FF",
    "Cyan": "&H00FFFF00",
    "Lime": "&H0000FF66",
    "Pink": "&H00FF50D0",
}


def _format_ass_time(seconds: float) -> str:
    """Formats seconds into ASS timestamp: H:MM:SS.cs (centiseconds)."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_ass_subtitles(
    segments: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    output_path: Optional[str] = None,
    style_name: str = "Neon Gold",
    keywords: Optional[List[str]] = None,
    words_per_phrase: int = 4,
    font_name: Optional[str] = None,
    font_size: Optional[int] = None,
    position: str = "Bottom",
    background: Optional[bool] = None,
    text_color: Optional[str] = None,
    highlight_color: Optional[str] = None,
    animate: bool = True,
) -> str:
    """
    Generates an ASS subtitle file for the specific clip with synchronized,
    word-by-word highlighted animation.
    """
    if output_path is None:
        output_path = str(TEMP_DIR / f"subtitles_{int(clip_start)}_{int(clip_end)}.ass")

    style_cfg = SUBTITLE_STYLES.get(style_name, SUBTITLE_STYLES["Neon Gold"])
    color_primary = CAPTION_COLORS.get(text_color, style_cfg["primary"])
    color_highlight = CAPTION_COLORS.get(highlight_color, style_cfg["highlight"])
    color_keyword = color_highlight if highlight_color else style_cfg["keyword"]
    color_outline = style_cfg["outline"]
    font_name = font_name or style_cfg.get("font", "Arial Black")
    font_size = int(font_size or style_cfg.get("size", 58))
    use_background = style_cfg.get("background", False) if background is None else background
    alignment, margin_v = {"Top": (8, 150), "Middle": (5, 80), "Bottom": (2, 300)}.get(position, (2, 300))
    border_style = 3 if use_background else 1

    kw_set = {k.lower().strip(".,!?") for k in (keywords or [])}

    # Extract all words falling inside [clip_start, clip_end]
    clip_words = []
    for s in segments:
        for w in s.get("words", []):
            w_start = w.get("start", 0.0)
            w_end = w.get("end", w_start + 0.5)
            if w_end > clip_start and w_start < clip_end:
                # Adjust timestamps relative to clip start (t=0 at clip_start)
                rel_start = max(0.0, w_start - clip_start)
                rel_end = max(rel_start + 0.1, min(clip_end - clip_start, w_end - clip_start))
                clip_words.append({
                    "word": w.get("word", "").strip(),
                    "start": rel_start,
                    "end": rel_end,
                })

    # If no word-level timestamps, construct from segment texts
    if not clip_words:
        for s in segments:
            s_start = s.get("start", 0.0)
            s_end = s.get("end", s_start + 3.0)
            if s_end > clip_start and s_start < clip_end:
                text_words = s.get("text", "").split()
                if not text_words:
                    continue
                s_rel_start = max(0.0, s_start - clip_start)
                s_rel_end = max(s_rel_start + 0.5, min(clip_end - clip_start, s_end - clip_start))
                w_dur = (s_rel_end - s_rel_start) / len(text_words)
                for i, tw in enumerate(text_words):
                    clip_words.append({
                        "word": tw.strip(),
                        "start": s_rel_start + i * w_dur,
                        "end": s_rel_start + (i + 1) * w_dur,
                    })

    # Group words into short phrases and wrap each into at most two readable lines.
    phrases = []
    for i in range(0, len(clip_words), words_per_phrase):
        phrase_group = clip_words[i : i + words_per_phrase]
        if phrase_group:
            lines = [[]]
            line_chars = 0
            for item in phrase_group:
                token_len = len(item["word"]) + (1 if lines[-1] else 0)
                if lines[-1] and line_chars + token_len > 22:
                    lines.append([])
                    line_chars = 0
                lines[-1].append(item)
                line_chars += token_len
            phrases.append(([word for line in lines[:2] for word in line], len(lines[0])))

    # Build ASS content
    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {TARGET_WIDTH}
PlayResY: {TARGET_HEIGHT}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{color_primary},&H000000FF,{color_outline},&H90000000,-1,0,0,0,100,100,0,0,{border_style},{style_cfg.get('outline_width', 4)},{style_cfg.get('shadow', 2)},{alignment},72,72,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    for phrase, line_break in phrases:
        phrase_start = phrase[0]["start"]
        phrase_end = phrase[-1]["end"]

        # For each word in the phrase, create an event during its duration where it is highlighted
        for word_idx, active_word in enumerate(phrase):
            w_start = active_word["start"]
            w_end = active_word["end"]

            ass_start = _format_ass_time(w_start)
            ass_end = _format_ass_time(w_end)

            # Assemble phrase text with active word highlighted
            line_parts = []
            for idx, w in enumerate(phrase):
                clean_text = w["word"].upper()
                is_kw = clean_text.lower().strip(".,!?") in kw_set

                if idx == word_idx:
                    # Active spoken word - highlighted!
                    active_color = color_keyword if is_kw else color_highlight
                    # Subtle pop scaling for active word
                    if animate:
                        line_parts.append(r"{\c" + active_color + r"\fscx108\fscy108}" + clean_text + r"{\fscx100\fscy100\c" + color_primary + r"}")
                    else:
                        line_parts.append(r"{\c" + active_color + "}" + clean_text + r"{\c" + color_primary + "}")
                else:
                    # Inactive word in phrase
                    line_parts.append(clean_text)

            dialogue_text = " ".join(line_parts[:line_break])
            if line_break and line_break < len(line_parts):
                dialogue_text += r"\N" + " ".join(line_parts[line_break:])
            events.append(f"Dialogue: 0,{ass_start},{ass_end},Default,,0,0,0,,{dialogue_text}")

    full_ass_content = ass_header + "\n".join(events) + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_ass_content)

    return output_path
