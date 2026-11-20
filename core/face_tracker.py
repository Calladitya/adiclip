"""
Intelligent speaker tracking and auto-framing using OpenCV YuNet DNN face detection.
Ensures the speaker is kept centered when converting 16:9 videos into 9:16 vertical shorts.
"""
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from core.config import YUNET_MODEL_PATH
from core.performance import file_identity, load_json_cache, make_cache_key, save_json_cache

# Keep OpenCV from oversubscribing the CPU and allocating per-core scratch buffers.
cv2.setNumThreads(2)

# Global detector cache
_YUNET_DETECTOR: Optional[cv2.FaceDetectorYN] = None


def get_face_detector(input_size: Tuple[int, int] = (320, 320)) -> cv2.FaceDetectorYN:
    """
    Loads and caches OpenCV's YuNet neural face detector.
    """
    global _YUNET_DETECTOR
    if _YUNET_DETECTOR is None:
        if not os.path.exists(YUNET_MODEL_PATH):
            raise FileNotFoundError(f"YuNet model file not found at: {YUNET_MODEL_PATH}")

        _YUNET_DETECTOR = cv2.FaceDetectorYN_create(
            str(YUNET_MODEL_PATH),
            "",
            input_size,
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=500,
        )
    else:
        _YUNET_DETECTOR.setInputSize(input_size)

    return _YUNET_DETECTOR


def detect_dominant_face(
    frame: np.ndarray,
    detector: Optional[cv2.FaceDetectorYN] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detects the dominant face (largest face by area) in a single BGR frame.
    Returns (x, y, w, h) in original frame coordinates, or None if no face detected.
    """
    h_orig, w_orig = frame.shape[:2]
    # Resize frame for ultra-fast inference while maintaining aspect ratio
    infer_w = 320
    infer_h = int(h_orig * (infer_w / w_orig))
    infer_h = infer_h - (infer_h % 2)  # ensure even

    resized = cv2.resize(frame, (infer_w, infer_h))

    if detector is None:
        detector = get_face_detector(input_size=(infer_w, infer_h))
    else:
        detector.setInputSize((infer_w, infer_h))

    _, faces = detector.detect(resized)
    if faces is None or len(faces) == 0:
        return None

    # Scale factor from resized back to original
    scale_x = w_orig / infer_w
    scale_y = h_orig / infer_h

    # Find face with largest area
    best_face = None
    max_area = 0.0

    for f in faces:
        fx, fy, fw, fh = f[0], f[1], f[2], f[3]
        area = fw * fh
        if area > max_area and f[14] > 0.5:  # confidence score check
            max_area = area
            best_face = (
                int(fx * scale_x),
                int(fy * scale_y),
                int(fw * scale_x),
                int(fh * scale_y),
            )

    return best_face


def detect_faces(frame: np.ndarray, detector=None, inference_width: int = 320) -> List[Dict[str, Any]]:
    """Detect every confident face and return boxes plus an approximate mouth ROI."""
    h_orig, w_orig = frame.shape[:2]
    infer_w = max(160, int(inference_width))
    infer_h = max(2, int(h_orig * (infer_w / max(1, w_orig))))
    infer_h -= infer_h % 2
    resized = cv2.resize(frame, (infer_w, infer_h))
    detector = detector or get_face_detector(input_size=(infer_w, infer_h))
    detector.setInputSize((infer_w, infer_h))
    _, faces = detector.detect(resized)
    if faces is None:
        return []

    sx, sy = w_orig / infer_w, h_orig / infer_h
    found = []
    for face in faces:
        x, y, w, h = [int(round(v)) for v in (face[0] * sx, face[1] * sy, face[2] * sx, face[3] * sy)]
        if face[14] < 0.55 or w < 12 or h < 12:
            continue
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_orig - x), min(h, h_orig - y)
        # YuNet provides mouth-corner landmarks; use their ROI when valid.
        mx1, my1 = face[10] * sx, face[11] * sy
        mx2, my2 = face[12] * sx, face[13] * sy
        mouth_w = abs(mx2 - mx1)
        if mouth_w >= 5:
            mx = int(max(0, min(mx1, mx2) - mouth_w * 0.25))
            my = int(max(0, min(my1, my2) - mouth_w * 0.2))
            mw = int(mouth_w * 1.5)
            mh = int(max(12, mouth_w * 0.8))
        else:
            mx, my, mw, mh = x + int(w * 0.2), y + int(h * 0.58), int(w * 0.6), int(h * 0.3)
        found.append({"x": x, "y": y, "w": w, "h": h,
                      "cx": x + w / 2, "cy": y + h / 2,
                      "mouth": (mx, my, mw, mh), "confidence": float(face[14])})
    return found


def compute_dynamic_reframe_plan(
    video_path: str,
    start_time: float,
    end_time: float,
    speech_segments: Optional[List[Dict[str, Any]]] = None,
    sample_interval: float = 0.75,
    inference_width: int = 320,
) -> Dict[str, Any]:
    """Track faces, use mouth motion during recognized speech as a soft speaker cue,
    and return temporally smoothed crop positions. Ambiguous samples hold the last
    confident target, so the frame does not randomly jump between faces.
    """
    speech_signature = [(round(float(s.get("start", 0)), 2), round(float(s.get("end", 0)), 2)) for s in (speech_segments or [])]
    cache_key = make_cache_key(file_identity(video_path), "face-plan-v3", start_time, end_time, sample_interval, inference_width, speech_signature)
    cached_plan = load_json_cache("face_plans", cache_key)
    if cached_plan is not None:
        return cached_plan

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for face tracking: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    crop_w = min(width, int(height * 9 / 16) // 2 * 2)
    crop_h = height - height % 2
    if width <= crop_w:
        cap.release()
        plan = {"crop_w": width, "crop_h": crop_h, "source_width": width, "points": [(0.0, 0)], "track_centers": [], "layout": "single", "confidence": 0.0}
        save_json_cache("face_plans", cache_key, plan)
        return plan

    detector = get_face_detector(input_size=(inference_width, max(2, int(height * inference_width / width) // 2 * 2)))
    tracks: Dict[int, Dict[str, Any]] = {}
    last_mouth: Dict[int, np.ndarray] = {}
    next_id = 0
    observations = []
    timestamps = np.arange(start_time, end_time + 0.001, sample_interval).tolist()
    if not timestamps or timestamps[-1] < end_time:
        timestamps.append(end_time)

    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            observations.append({"time": ts, "faces": []})
            continue
        faces = detect_faces(frame, detector, inference_width=inference_width)
        used = set()
        scored = []
        for face in sorted(faces, key=lambda item: item["cx"]):
            mx, my, mw, mh = face["mouth"]
            mouth = frame[max(0, my):min(height, my + mh), max(0, mx):min(width, mx + mw)]
            motion = 0.0
            if mouth.size:
                mouth = cv2.resize(cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY), (32, 16))
            scale_x, scale_y = width / frame.shape[1], height / frame.shape[0]
            face["cx"] *= scale_x
            face["cy"] *= scale_y
            face["x"] = int(face["x"] * scale_x)
            face["y"] = int(face["y"] * scale_y)
            face["w"] = int(face["w"] * scale_x)
            face["h"] = int(face["h"] * scale_y)
            candidates = [(abs(face["cx"] - data["cx"]), tid) for tid, data in tracks.items() if tid not in used]
            distance, track_id = min(candidates, default=(float("inf"), -1))
            if distance > width * 0.20:
                track_id = next_id
                next_id += 1
            used.add(track_id)
            previous = last_mouth.get(track_id)
            if mouth.size and previous is not None:
                motion = float(cv2.absdiff(mouth, previous).mean()) / 255.0
            if mouth.size:
                last_mouth[track_id] = mouth
            tracks[track_id] = {"cx": face["cx"], "hits": tracks.get(track_id, {}).get("hits", 0) + 1}
            face.update({"track_id": track_id, "motion": motion})
            scored.append(face)
        observations.append({"time": ts, "faces": scored})
    cap.release()

    speech_segments = speech_segments or []
    active_ids = []
    confidence_values = []
    current_id = None
    for observation in observations:
        ts = observation["time"]
        speaking = any(float(s.get("start", 0)) <= ts <= float(s.get("end", 0)) for s in speech_segments)
        ranked = sorted(observation["faces"], key=lambda f: f["motion"], reverse=True)
        if speaking and ranked and ranked[0]["motion"] >= 0.012:
            top = ranked[0]
            second = ranked[1]["motion"] if len(ranked) > 1 else 0.0
            confidence = min(1.0, max(0.0, (top["motion"] - second) / max(top["motion"], 0.01)))
            if len(ranked) == 1 or confidence >= 0.18:
                current_id = top["track_id"]
                confidence_values.append(confidence)
        face = next((f for f in observation["faces"] if f["track_id"] == current_id), None)
        # Safe fallback: keep previous speaker; use a detected face only at clip start.
        if face is None and current_id is None and observation["faces"]:
            face = max(observation["faces"], key=lambda f: f["w"] * f["h"])
        if face is not None:
            active_ids.append(current_id if current_id is not None else face["track_id"])
            target = int(np.clip(face["cx"] - crop_w / 2, 0, width - crop_w))
        else:
            active_ids.append(current_id)
            target = int(width / 2 - crop_w / 2)
        observation["target_x"] = target

    # Exponential smoothing and a movement-rate limit suppress camera jumps.
    points = []
    previous_x = int(width / 2 - crop_w / 2)
    max_step = max(2, int(width * 0.055))
    for index, observation in enumerate(observations):
        desired = observation["target_x"]
        delta = int(np.clip(desired - previous_x, -max_step, max_step))
        previous_x = int(np.clip(previous_x + delta * 0.65, 0, width - crop_w))
        points.append((round(observation["time"] - start_time, 3), previous_x - previous_x % 2))

    present = [track_id for track_id, data in tracks.items() if data["hits"] >= 3]
    track_centers = []
    for track_id in present:
        centers = [f["cx"] for o in observations for f in o["faces"] if f["track_id"] == track_id]
        if centers:
            track_centers.append({"track_id": track_id, "cx": float(np.median(centers))})
    confident_speakers = {track_id for track_id in active_ids if track_id is not None}
    layout = "split" if len(present) >= 2 and len(confident_speakers) >= 2 else "single"
    plan = {"crop_w": crop_w, "crop_h": crop_h, "source_width": width, "points": points,
            "track_centers": track_centers, "layout": layout,
            "confidence": round(float(np.mean(confidence_values)), 2) if confidence_values else 0.0}
    save_json_cache("face_plans", cache_key, plan)
    return plan


def compute_smart_crop_window(
    video_path: str,
    start_time: float,
    end_time: float,
    sample_interval: float = 0.5,
) -> Dict[str, Any]:
    """
    Samples frames across [start_time, end_time], tracks the speaker's face,
    and calculates optimal smooth 9:16 crop window coordinates.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for face tracking: {video_path}")

    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Target 9:16 crop dimensions
    # For a vertical video, crop_w = int(h_orig * 9 / 16)
    target_crop_w = int(h_orig * 9 / 16)
    target_crop_w = target_crop_w - (target_crop_w % 2)  # Even dimensions for FFmpeg
    target_crop_h = h_orig - (h_orig % 2)

    # If video is already vertical or narrower than 9:16
    if w_orig <= target_crop_w:
        cap.release()
        return {
            "crop_w": w_orig,
            "crop_h": h_orig,
            "crop_x": 0,
            "crop_y": 0,
            "face_detected": False,
            "mode": "already_vertical",
        }

    # Sample timestamps
    timestamps = []
    t = start_time
    while t <= end_time:
        timestamps.append(t)
        t += sample_interval

    face_centers_x = []
    detector = get_face_detector()

    for ts in timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        face = detect_dominant_face(frame, detector=detector)
        if face:
            fx, fy, fw, fh = face
            center_x = fx + (fw / 2.0)
            face_centers_x.append(center_x)

    cap.release()

    default_center_x = w_orig / 2.0

    if face_centers_x:
        # Use robust median to ignore temporary false turns or quick hand gestures
        chosen_center_x = float(np.median(face_centers_x))
        face_detected = True
    else:
        chosen_center_x = default_center_x
        face_detected = False

    # Calculate crop_x such that chosen_center_x is centered in target_crop_w
    ideal_crop_x = int(chosen_center_x - (target_crop_w / 2.0))

    # Clamp crop_x within frame boundaries [0, w_orig - target_crop_w]
    max_crop_x = w_orig - target_crop_w
    clamped_crop_x = max(0, min(ideal_crop_x, max_crop_x))
    clamped_crop_x = clamped_crop_x - (clamped_crop_x % 2)  # FFmpeg prefers even coordinates

    return {
        "crop_w": target_crop_w,
        "crop_h": target_crop_h,
        "crop_x": clamped_crop_x,
        "crop_y": 0,
        "face_detected": face_detected,
        "speaker_center_x": round(chosen_center_x, 1),
        "total_face_detections": len(face_centers_x),
        "samples_checked": len(timestamps),
        "mode": "smart_face_centered" if face_detected else "center_crop",
    }
