import logging
import time
from datetime import datetime, timezone

import httpx

from attention import detect_attention
from config import CAMERA_INDEX, BACKEND_URL, AUTH_TOKEN

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def calculate_productivity_metrics(attention_metrics, session_duration):
    """Calculate productivity metrics based on attention tracking"""
    
    attention_lost = attention_metrics['total_attention_lost']
    work_time = session_duration - attention_lost
    work_time = max(0, work_time)
    
    focus_score = (work_time / session_duration * 100) if session_duration > 0 else 0
    attention_score = focus_score  # Same as focus score now
    
    if focus_score >= 90:
        quality = "Excellent"
    elif focus_score >= 75:
        quality = "Good"
    elif focus_score >= 60:
        quality = "Fair"
    else:
        quality = "Poor"
    
    return {
        'session_duration': session_duration,
        'work_time': work_time,
        'distracted_time': attention_lost,
        'attention_lost': attention_lost,
        'focus_score': focus_score,
        'attention_score': attention_score,
        'quality': quality
    }

def post_session(payload: dict, backend_url: str | None, auth_token: str | None) -> None:
    if not auth_token:
        logger.warning("AUTH_TOKEN is not set — session will not be saved. Add AUTH_TOKEN to .env.")
        return
    if not backend_url:
        logger.warning("BACKEND_URL is not set — session will not be saved. Add BACKEND_URL to .env.")
        return

    url = f"{backend_url.rstrip('/')}/api/v1/sessions/"
    headers = {"Authorization": f"Bearer {auth_token}"}

    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=10)
    except httpx.RequestError as exc:
        logger.error("Network error while saving session: %s", exc)
        print(f"Could not reach backend ({url}): {exc}")
        return

    if response.status_code == 201:
        session_id = response.json().get("id", "<unknown>")
        print(f"Session saved. ID: {session_id}")
    elif response.status_code == 401:
        print("Session not saved: unauthenticated. Check AUTH_TOKEN in .env.")
    elif response.status_code == 422:
        try:
            detail = response.json().get("detail", response.text[:200])
        except Exception:
            detail = response.text[:200]
        print(f"Session not saved: validation error — {detail}")
    else:
        content_type = response.headers.get("content-type", "")
        snippet = response.text[:200] if content_type.startswith("text") else "<binary body>"
        print(f"Session not saved: HTTP {response.status_code} — {snippet}")


if __name__ == "__main__":
    # Record session start time
    session_start_time = time.time()
    logger.info("Starting productivity tracking session...")
    logger.info("Camera index: %s", CAMERA_INDEX)
    logger.info("Backend URL: %s", BACKEND_URL)
    logger.info("Auth token: %s", "set" if AUTH_TOKEN else "not set")
    logger.info("Press 'q' in the camera window to stop tracking")

    attention_metrics = detect_attention()
    
    session_end_time = time.time()
    session_duration = session_end_time - session_start_time
    
    productivity = calculate_productivity_metrics(attention_metrics, session_duration)
    
    print("\n========= PRODUCTIVITY METRICS =========")
    print(f"Total session time: {productivity['session_duration']:.1f} seconds ({productivity['session_duration']/60:.1f} minutes)")
    print(f"Focused time: {productivity['work_time']:.1f} seconds ({productivity['focus_score']:.1f}%)")
    print(f"Distracted time: {productivity['distracted_time']:.1f} seconds ({100-productivity['focus_score']:.1f}%)")
    
    print("\nDetailed Breakdown:")
    attention_percentage = (productivity['attention_lost'] / session_duration * 100) if session_duration > 0 else 0
    print(f"- Attention lost: {productivity['attention_lost']:.1f} seconds ({attention_percentage:.1f}%)")
    
    print("\nScores:")
    print(f"Focus Score: {productivity['focus_score']:.1f}%")
    print(f"Attention Score: {productivity['attention_score']:.1f}%")
    print(f"Session Quality: {productivity['quality']}")
    
    print("\n========= SESSION COMPLETE =========")

    if int(session_duration) >= 1:
        session_payload = {
            "started_at": datetime.fromtimestamp(session_start_time, tz=timezone.utc).isoformat(),
            "ended_at": datetime.fromtimestamp(session_end_time, tz=timezone.utc).isoformat(),
            "duration_seconds": int(round(session_duration)),
            "eyes_closed_time": attention_metrics["eyes_closed_time"],
            "face_missing_time": attention_metrics["face_missing_time"],
            "head_pose_off_time": attention_metrics["head_pose_off_time"],
            "total_attention_lost": attention_metrics["total_attention_lost"],
            "notes": None,
        }
        post_session(session_payload, BACKEND_URL, AUTH_TOKEN)
    else:
        logger.info("Session too short to save (< 1 second).")
