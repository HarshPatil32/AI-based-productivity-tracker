from fastapi import APIRouter, HTTPException, status, Depends, Request, Query
from typing import Optional, List
from uuid import UUID
from datetime import date
import logging

from backend.models.session import SessionCreate, SessionResponse, SessionSummary
from backend.middleware.auth import require_auth, require_same_user
from backend.services.database import get_supabase_client, get_supabase_admin_client
from backend.utils.auth import TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])



def _compute_quality(focus_score: float) -> str:
    if focus_score >= 85:
        return "Excellent"
    elif focus_score >= 70:
        return "Good"
    elif focus_score >= 50:
        return "Fair"
    return "Poor"


def _build_session_payload(data: SessionCreate, user_id: str) -> dict:
    """Map SessionCreate fields onto the study_sessions schema columns."""
    duration = data.duration_seconds if data.duration_seconds > 0 else 1  # guard divide-by-zero

    distracted_time = (
        data.eyes_closed_time
        + data.face_missing_time
        + data.head_pose_off_time
    )
    focused_time = max(duration - int(distracted_time), 0)

    attention_score = round(
        max(0.0, 1.0 - (data.total_attention_lost / duration)) * 100, 2
    )
    focus_score = round((focused_time / duration) * 100, 2)
    quality = _compute_quality(focus_score)

    session_date = (
        data.started_at.date() if data.started_at else date.today()
    )

    return {
        "user_id": user_id,
        "title": data.notes or f"Study Session – {session_date}",
        "description": data.notes,
        "session_duration": duration,
        "focused_time": focused_time,
        "distracted_time": int(distracted_time),
        "eyes_closed_time": int(data.eyes_closed_time),
        "face_missing_time": int(data.face_missing_time),
        "head_pose_off_time": int(data.head_pose_off_time),
        "attention_lost": int(data.total_attention_lost),
        "focus_score": focus_score,
        "attention_score": attention_score,
        "quality": quality,
        "session_date": session_date.isoformat(),
        "session_start_time": data.started_at.isoformat() if data.started_at else None,
        "session_end_time": data.ended_at.isoformat() if data.ended_at else None,
        "is_public": True,
    }


def _to_session_response(row: dict) -> dict:
    """Normalise a DB row back into a SessionResponse-compatible dict."""
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "started_at": row.get("session_start_time"),
        "ended_at": row.get("session_end_time"),
        "duration_seconds": row["session_duration"],
        "eyes_closed_time": row.get("eyes_closed_time", 0),
        "face_missing_time": row.get("face_missing_time", 0),
        "head_pose_off_time": row.get("head_pose_off_time", 0),
        "total_attention_lost": row.get("attention_lost", 0),
        "notes": row.get("description"),
        "attention_score": row.get("attention_score", 0.0),
        "created_at": row.get("created_at"),
    }



@router.post(
    "/",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a completed study session",
)
async def create_session(
    session_data: SessionCreate,
    current_user: TokenData = Depends(require_auth),
):
    """Persist a completed tracker session for the authenticated user."""
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)

    payload = _build_session_payload(session_data, user_id)

    try:
        result = client.table("study_sessions").insert(payload).execute()
    except Exception as e:
        logger.error(f"Failed to insert session for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save session",
        )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session insert returned no data",
        )

    return _to_session_response(result.data[0])


# GET /sessions/me  – past user's sessions

@router.get(
    "/me",
    response_model=List[SessionResponse],
    summary="Get authenticated user's sessions",
)
async def get_my_sessions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_client()
    user_id = str(current_user.user_id)

    try:
        result = (
            client.table("study_sessions")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch sessions for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve sessions",
        )

    return [_to_session_response(row) for row in result.data]


# GET /sessions/me/summary  – aggregated stats

@router.get(
    "/me/summary",
    response_model=SessionSummary,
    summary="Get aggregated stats for the authenticated user",
)
async def get_my_summary(
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_client()
    user_id = str(current_user.user_id)

    try:
        result = (
            client.table("study_sessions")
            .select(
                "session_duration, attention_score, eyes_closed_time, "
                "face_missing_time, head_pose_off_time, attention_lost"
            )
            .eq("user_id", user_id)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch summary for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve summary",
        )

    rows = result.data
    if not rows:
        return SessionSummary(
            total_sessions=0,
            total_study_seconds=0,
            avg_attention_score=0.0,
            avg_eyes_closed_time=0.0,
            avg_face_missing_time=0.0,
            avg_head_pose_off_time=0.0,
            total_attention_lost=0.0,
        )

    n = len(rows)
    return SessionSummary(
        total_sessions=n,
        total_study_seconds=sum(r["session_duration"] for r in rows),
        avg_attention_score=round(sum(r.get("attention_score", 0) for r in rows) / n, 2),
        avg_eyes_closed_time=round(sum(r.get("eyes_closed_time", 0) for r in rows) / n, 2),
        avg_face_missing_time=round(sum(r.get("face_missing_time", 0) for r in rows) / n, 2),
        avg_head_pose_off_time=round(sum(r.get("head_pose_off_time", 0) for r in rows) / n, 2),
        total_attention_lost=round(sum(r.get("attention_lost", 0) for r in rows), 2),
    )


# GET /sessions/{session_id}  – single session

@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get a specific session by ID",
)
async def get_session(
    session_id: UUID,
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_client()
    user_id = str(current_user.user_id)

    try:
        result = (
            client.table("study_sessions")
            .select("*")
            .eq("id", str(session_id))
            .single()
            .execute()
        )
    except Exception as e:
        logger.error(f"Session {session_id} fetch failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    row = result.data
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # Only the owner or a public session can be viewed
    if row["user_id"] != user_id and not row.get("is_public", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This session is private",
        )

    return _to_session_response(row)


# DELETE /sessions/{session_id}

@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session",
)
async def delete_session(
    session_id: UUID,
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)

    # Verify ownership first
    try:
        check = (
            client.table("study_sessions")
            .select("user_id")
            .eq("id", str(session_id))
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if not check.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if check.data["user_id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own sessions",
        )

    try:
        client.table("study_sessions").delete().eq("id", str(session_id)).execute()
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete session",
        )


# GET /sessions/user/{user_id}  – another user's public sessions

@router.get(
    "/user/{user_id}",
    response_model=List[SessionResponse],
    summary="Get public sessions for a specific user",
)
async def get_user_sessions(
    user_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_client()
    target_id = str(user_id)

    query = (
        client.table("study_sessions")
        .select("*")
        .eq("user_id", target_id)
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )

    # If not the owner, restrict to public sessions only
    if str(current_user.user_id) != target_id:
        query = query.eq("is_public", True)

    try:
        result = query.execute()
    except Exception as e:
        logger.error(f"Failed to fetch sessions for user {target_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve sessions",
        )

    return [_to_session_response(row) for row in result.data]
