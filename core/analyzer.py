"""
AI Highlight & Engagement Analyzer for identifying top podcast moments.
Supports Google Gemini API with intelligent Local NLP Heuristic fallback.
"""
import os
import re
import json
import hashlib
from typing import Dict, Any, List, Optional
from core.config import MIN_CLIP_DURATION, MAX_CLIP_DURATION, DEFAULT_NUM_CLIPS, DEFAULT_GEMINI_MODEL
from core.performance import load_json_cache, make_cache_key, save_json_cache

# Buzzwords and engagement markers for scoring
ENGAGING_HOOK_TERMS = {
    # Questions & Curiosities
    "why", "how", "what if", "did you know", "reason", "because", "question",
    # Intense & Shocking
    "crazy", "insane", "shocking", "unbelievable", "huge", "worst", "best", "biggest",
    "secret", "truth", "hidden", "lie", "exposed", "dangerous", "never", "always",
    # Value & Wisdom
    "mistake", "lesson", "advice", "rule", "framework", "mindset", "strategy", "failure",
    "success", "money", "future", "life", "career", "business", "rich", "freedom",
    # Emotional & Conversational
    "feel", "believe", "hate", "love", "regret", "changed", "moment", "realize", "actually",
    "nobody", "everyone", "stop", "remember", "game changer", "brutal", "painful"
}


def _snap_to_word_boundary(
    start_time: float,
    end_time: float,
    segments: List[Dict[str, Any]],
    min_dur: float = MIN_CLIP_DURATION,
    max_dur: float = MAX_CLIP_DURATION,
) -> tuple[float, float]:
    """
    Snaps raw timestamps to exact Whisper word or segment boundaries so words are not cut off.
    Enforces min_dur and max_dur limits.
    """
    all_words = []
    for s in segments:
        for w in s.get("words", []):
            all_words.append(w)

    if not all_words:
        # Fallback to segment start/end
        dur = max(min_dur, min(max_dur, end_time - start_time))
        return round(start_time, 2), round(start_time + dur, 2)

    # Find closest word start to start_time
    best_start = min(all_words, key=lambda w: abs(w["start"] - start_time))["start"]

    # Target end time based on desired duration
    target_end = best_start + (end_time - start_time)
    # Clamp target duration between min_dur and max_dur
    if target_end - best_start < min_dur:
        target_end = best_start + min_dur
    elif target_end - best_start > max_dur:
        target_end = best_start + max_dur

    # Find closest word end to target_end
    best_end_word = min(all_words, key=lambda w: abs(w["end"] - target_end))
    best_end = best_end_word["end"]

    # Final clamp check
    actual_dur = best_end - best_start
    if actual_dur < min_dur:
        best_end = round(best_start + min_dur, 2)
    elif actual_dur > max_dur:
        best_end = round(best_start + max_dur, 2)

    return round(best_start, 2), round(best_end, 2)


def _extract_keywords_from_text(text: str) -> List[str]:
    """Extracts high-impact keywords for dynamic subtitle highlighting."""
    words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())
    found_keywords = [w for w in words if w in ENGAGING_HOOK_TERMS]
    # Add frequent nouns/verbs if too few
    if len(found_keywords) < 3:
        common_words = {"this", "that", "with", "from", "have", "were", "they", "there", "what", "when", "your"}
        unique_long_words = [w for w in set(words) if w not in common_words]
        found_keywords.extend(unique_long_words[:5])

    # Return top 4 unique keywords
    return list(dict.fromkeys(found_keywords))[:5]


def _generate_smart_title(text: str, fallback_index: int) -> str:
    """Generates a click-worthy punchy title from text content."""
    sentences = re.split(r"[.!?]", text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]

    for s in valid_sentences:
        lower_s = s.lower()
        if any(w in lower_s for w in ["why", "how", "what", "never", "secret", "truth", "mistake"]):
            words = s.split()[:7]
            title = " ".join(words).title()
            return re.sub(r'[^a-zA-Z0-9\s]', '', title)

    if valid_sentences:
        words = valid_sentences[0].split()[:6]
        return " ".join(words).title()

    return f"Engaging Moment #{fallback_index}"


def _quality_breakdown(text: str, duration: float, covered_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate transcript-side quality; visual quality is added after framing."""
    words = re.findall(r"[a-z']+", text.lower())
    hook = 55 + min(25, sum(1 for word in words if word in ENGAGING_HOOK_TERMS) * 6)
    if "?" in text or any(text.lower().startswith(prefix) for prefix in ("why ", "how ", "what ")):
        hook += 12
    engagement = min(100, 45 + min(40, len(set(words) & ENGAGING_HOOK_TERMS) * 7) + (10 if "?" in text else 0))
    first = covered_segments[0].get("text", "").strip() if covered_segments else ""
    last = covered_segments[-1].get("text", "").strip() if covered_segments else ""
    context = 82 + (10 if first[:1].isupper() else 0) + (8 if last.endswith((".", "!", "?")) else -12)
    duration_score = max(35, 100 - int(abs(duration - 45) * 2.2))
    probabilities = [w.get("probability") for s in covered_segments for w in s.get("words", []) if isinstance(w.get("probability"), (int, float))]
    audio = int(round(sum(probabilities) / len(probabilities) * 100)) if probabilities else 70
    return {
        "hook_strength": min(100, hook),
        "engagement": engagement,
        "context_completeness": max(0, min(100, context)),
        "audio_quality": max(0, min(100, audio)),
        "duration_suitability": duration_score,
        "visual_quality": None,
        "overall": int(round(min(100, hook) * 0.25 + engagement * 0.25 + max(0, min(100, context)) * 0.20 + audio * 0.15 + duration_score * 0.15)),
    }


def analyze_with_local_heuristics(
    segments: List[Dict[str, Any]],
    num_clips: int = DEFAULT_NUM_CLIPS,
    min_dur: float = MIN_CLIP_DURATION,
    max_dur: float = MAX_CLIP_DURATION,
) -> List[Dict[str, Any]]:
    """
    Local AI/NLP Heuristic engine for identifying top podcast clips without requiring external API keys.
    """
    if not segments:
        return []

    total_video_duration = segments[-1]["end"]
    if total_video_duration < min_dur:
        # Video is shorter than minimum clip duration; return single full clip
        short_text = " ".join(s.get("text", "") for s in segments)
        short_quality = _quality_breakdown(short_text, total_video_duration, segments)
        return [{
            "clip_index": 1,
            "title": "Complete Highlights",
            "start_time": 0.0,
            "end_time": round(total_video_duration, 2),
            "duration": round(total_video_duration, 2),
            "engagement_score": 95,
            "quality_score": short_quality["overall"],
            "quality_breakdown": short_quality,
            "reason": "Full video selection",
            "hook": segments[0]["text"][:60],
            "keywords": _extract_keywords_from_text(segments[0]["text"]),
        }]

    # Build candidate windows
    candidates = []
    n_seg = len(segments)

    for i in range(n_seg):
        start_time = segments[i]["start"]
        accumulated_text = []

        for j in range(i, n_seg):
            accumulated_text.append(segments[j]["text"])
            end_time = segments[j]["end"]
            dur = end_time - start_time

            if dur < min_dur:
                continue
            if dur > max_dur:
                break

            window_text = " ".join(accumulated_text)
            words = window_text.lower().split()
            word_count = len(words)

            if word_count < 10:
                continue

            # Calculate engagement score
            score = 60.0  # Base score

            # 1. Buzzword density
            buzz_count = sum(1 for w in words if w.strip(".,!?") in ENGAGING_HOOK_TERMS)
            score += min(25.0, buzz_count * 4.0)

            # 2. Question / Hook presence
            if any(q in window_text.lower() for q in ["?", "why", "how", "what if"]):
                score += 8.0

            # 3. Speech Pace / Dynamism (target ~130-170 words per minute)
            wpm = (word_count / dur) * 60
            if 120 <= wpm <= 180:
                score += 7.0
            elif wpm < 80:
                score -= 10.0

            candidates.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration": round(dur, 2),
                "text": window_text,
                "score": round(min(99.0, max(50.0, score)), 1),
                "quality": _quality_breakdown(window_text, dur, segments[i:j + 1]),
            })

    # Sort candidates by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Non-maximum suppression (NMS) to avoid overlapping clips
    selected_clips = []
    for cand in candidates:
        if len(selected_clips) >= num_clips:
            break

        # Check overlap with already selected clips (allow at most 10s overlap)
        overlaps = False
        for s in selected_clips:
            overlap_start = max(cand["start_time"], s["start_time"])
            overlap_end = min(cand["end_time"], s["end_time"])
            if overlap_end > overlap_start + 10.0:
                overlaps = True
                break

        if not overlaps:
            snap_s, snap_e = _snap_to_word_boundary(
                cand["start_time"], cand["end_time"], segments, min_dur, max_dur
            )
            title = _generate_smart_title(cand["text"], len(selected_clips) + 1)
            keywords = _extract_keywords_from_text(cand["text"])

            selected_clips.append({
                "clip_index": len(selected_clips) + 1,
                "title": title,
                "start_time": snap_s,
                "end_time": snap_e,
                "duration": round(snap_e - snap_s, 2),
                "engagement_score": int(cand["score"]),
                "quality_score": cand["quality"]["overall"],
                "quality_breakdown": cand["quality"],
                "reason": f"High engagement speech containing key terms: {', '.join(keywords[:3])}",
                "hook": (re.split(r"(?<=[.!?])\s+", cand["text"].strip(), maxsplit=1)[0] or cand["text"][:100])[:140],
                "keywords": keywords,
            })

    # Sort final clips chronologically for pleasant review
    selected_clips.sort(key=lambda c: c["start_time"])
    for idx, c in enumerate(selected_clips):
        c["clip_index"] = idx + 1

    return selected_clips


def analyze_with_gemini(
    formatted_transcript: str,
    segments: List[Dict[str, Any]],
    api_key: str,
    num_clips: int = DEFAULT_NUM_CLIPS,
    model_name: str = DEFAULT_GEMINI_MODEL,
    min_dur: float = MIN_CLIP_DURATION,
    max_dur: float = MAX_CLIP_DURATION,
) -> List[Dict[str, Any]]:
    """
    Uses Google Gemini API to analyze the transcript and return the most viral 10 clips.
    """
    prompt = f"""You are a world-class viral video editor and podcast clipping expert (specializing in TikTok, YouTube Shorts, and Instagram Reels).

Analyze this timestamped podcast transcript and identify the top {num_clips} most engaging, viral moments.

CRITICAL RULES:
1. Every clip's duration MUST be between {int(min_dur)} and {int(max_dur)} seconds (end_time - start_time must be >= {int(min_dur)} and <= {int(max_dur)}).
2. Choose moments with high emotion, punchy insights, controversy, actionable advice, relatable humor, or shocking revelations.
3. Each clip MUST have a complete thought (starts cleanly at the beginning of an idea, ends on a punchline or conclusion).
4. Clips must not significantly overlap with each other.
5. Create a catchy, click-worthy title for each clip (5-8 words).
6. List 3 to 5 emphasized keywords that should be visually highlighted on screen.

Respond ONLY with a valid JSON array of objects conforming to this schema:
[
  {{
    "clip_index": 1,
    "title": "The Brutal Truth About Success",
    "start_time": 45.0,
    "end_time": 85.0,
    "engagement_score": 96,
    "hook": "Why 99% of people quit right before succeeding",
    "reason": "Intense emotional peak and counterintuitive life advice",
    "keywords": ["truth", "success", "mistake", "quit"]
  }}
]

TRANSCRIPT:
{formatted_transcript}
"""

    raw_response = ""
    # Try using google.genai first
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        raw_response = response.text or ""
    except Exception:
        # Fallback to google.generativeai if available
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=api_key)
            gmodel = legacy_genai.GenerativeModel(model_name)
            resp = gmodel.generate_content(prompt)
            raw_response = resp.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini API request failed: {str(e)}")

    # Extract JSON from markdown fences if present
    json_match = re.search(r"\[.*\]", raw_response, re.DOTALL)
    if not json_match:
        raise ValueError(f"Gemini response did not contain valid JSON array: {raw_response[:200]}")

    clips_data = json.loads(json_match.group(0))

    validated_clips = []
    for idx, item in enumerate(clips_data):
        raw_start = float(item.get("start_time", 0.0))
        raw_end = float(item.get("end_time", raw_start + min_dur))

        # Enforce boundary snapping and duration limits
        snap_s, snap_e = _snap_to_word_boundary(raw_start, raw_end, segments, min_dur, max_dur)

        keywords = item.get("keywords") or _extract_keywords_from_text(item.get("hook", ""))
        covered = [s for s in segments if s.get("end", 0) > snap_s and s.get("start", 0) < snap_e]
        clip_text = item.get("hook", "") or " ".join(s.get("text", "") for s in covered)
        quality = _quality_breakdown(clip_text, snap_e - snap_s, covered)

        validated_clips.append({
            "clip_index": idx + 1,
            "title": item.get("title", f"Clip #{idx + 1}"),
            "start_time": snap_s,
            "end_time": snap_e,
            "duration": round(snap_e - snap_s, 2),
            "engagement_score": int(item.get("engagement_score", 85)),
            "quality_score": quality["overall"],
            "quality_breakdown": quality,
            "reason": item.get("reason", "Identified by Gemini as high viral potential"),
            "hook": item.get("hook", ""),
            "keywords": keywords,
        })

    # Sort chronologically
    validated_clips.sort(key=lambda c: c["start_time"])
    for i, c in enumerate(validated_clips):
        c["clip_index"] = i + 1

    return validated_clips[:num_clips]


def identify_engaging_moments(
    transcript_data: Dict[str, Any],
    api_key: Optional[str] = None,
    num_clips: int = DEFAULT_NUM_CLIPS,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    use_gemini: bool = False,
) -> List[Dict[str, Any]]:
    """
    Main entry point for identifying top engaging podcast moments.
    Uses Gemini API if enabled and key provided, otherwise seamlessly uses Local Heuristic AI.
    """
    segments = transcript_data.get("segments", [])
    formatted_transcript = transcript_data.get("formatted_transcript", "")
    transcript_hash = hashlib.sha256(formatted_transcript.encode("utf-8")).hexdigest()
    key_fingerprint = hashlib.sha256((api_key or "").encode("utf-8")).hexdigest() if use_gemini else "local"
    cache_key = make_cache_key(transcript_hash, num_clips, gemini_model, use_gemini, key_fingerprint)
    cached = load_json_cache("clip_analysis", cache_key)
    if cached is not None:
        return cached

    if use_gemini and api_key and api_key.strip():
        try:
            result = analyze_with_gemini(
                formatted_transcript=formatted_transcript,
                segments=segments,
                api_key=api_key.strip(),
                num_clips=num_clips,
                model_name=gemini_model,
            )
            save_json_cache("clip_analysis", cache_key, result)
            return result
        except Exception as e:
            print(f"[Warning] Gemini API analysis failed: {e}. Falling back to Local AI.")

    # Local Heuristic AI analysis
    result = analyze_with_local_heuristics(
        segments=segments,
        num_clips=num_clips,
    )
    save_json_cache("clip_analysis", cache_key, result)
    return result
