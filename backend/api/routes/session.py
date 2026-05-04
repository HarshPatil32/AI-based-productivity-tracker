from fastapi import APIRouter, HTTPException, status, Depends, Request, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
from uuid import UUID
from datetime import date
import logging

from backend.models.session import SessionCreate, SessionResponse, SessionSummary, LikeEntry, CommentCreate, CommentResponse
from backend.middleware.auth import require_auth, require_same_user
from backend.services.database import get_supabase_client, get_supabase_admin_client
from backend.services.notifications import create_notification
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
        "focus_score": row.get("focus_score", 0.0),
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
    client = get_supabase_admin_client()
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
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)

    try:
        result = (
            client.table("study_sessions")
            .select(
                "session_duration, attention_score, focus_score, eyes_closed_time, "
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
            avg_focus_score=0.0,
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
        avg_focus_score=round(sum(r.get("focus_score", 0) for r in rows) / n, 2),
        avg_eyes_closed_time=round(sum(r.get("eyes_closed_time", 0) for r in rows) / n, 2),
        avg_face_missing_time=round(sum(r.get("face_missing_time", 0) for r in rows) / n, 2),
        avg_head_pose_off_time=round(sum(r.get("head_pose_off_time", 0) for r in rows) / n, 2),
        total_attention_lost=round(sum(r.get("attention_lost", 0) for r in rows), 2),
    )


# GET /sessions/user/{user_id}  – another user's sessions, respecting session_visibility

@router.get(
    "/user/{user_id}",
    response_model=List[SessionResponse],
    summary="Get a user's sessions",
)
async def get_user_sessions(
    user_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    target_id = str(user_id)
    caller_id = str(current_user.user_id)

    # Non-owners must pass the session_visibility gate
    if caller_id != target_id:
        try:
            settings_result = (
                client.table("user_settings")
                .select("session_visibility")
                .eq("user_id", target_id)
                .single()
                .execute()
            )
            session_visibility = (settings_result.data or {}).get("session_visibility", "public")
        except Exception:
            session_visibility = "public"

        if session_visibility == "private":
            return []

        if session_visibility == "friends":
            follow_check = (
                client.table("user_relationships")
                .select("id")
                .eq("follower_id", caller_id)
                .eq("following_id", target_id)
                .execute()
            )
            if not follow_check.data:
                return []

    try:
        query = (
            client.table("study_sessions")
            .select("*")
            .eq("user_id", target_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
        )
        if caller_id != target_id:
            query = query.eq("is_public", True)
        result = query.execute()
    except Exception as e:
        logger.error(f"Failed to fetch sessions for user {target_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve sessions",
        )

    return [_to_session_response(row) for row in result.data]


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
    client = get_supabase_admin_client()
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


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/like
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/like",
    status_code=status.HTTP_201_CREATED,
    summary="Like a session (idempotent)",
)
async def like_session(
    session_id: UUID,
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)
    sid = str(session_id)

    # Verify session exists and is visible to the caller
    try:
        check = (
            client.table("study_sessions")
            .select("id, is_public, user_id")
            .eq("id", sid)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if not check.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if not check.data.get("is_public") and check.data["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This session is private")

    # Idempotent: if already liked, return 200 (not 201) — nothing was created
    existing = (
        client.table("session_likes")
        .select("id")
        .eq("session_id", sid)
        .eq("user_id", user_id)
        .execute()
    )
    if existing.data:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"detail": "Already liked"})

    try:
        client.table("session_likes").insert(
            {"session_id": sid, "user_id": user_id}
        ).execute()
    except Exception as e:
        logger.error(f"Failed to like session {sid} for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not like session",
        )

    create_notification(
        client,
        user_id=check.data["user_id"],
        actor_id=user_id,
        type="like",
        entity_id=sid,
    )

    return {"detail": "Session liked"}


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}/like
# ---------------------------------------------------------------------------

@router.delete(
    "/{session_id}/like",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unlike a session (idempotent)",
)
async def unlike_session(
    session_id: UUID,
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)
    sid = str(session_id)

    try:
        client.table("session_likes").delete().eq("session_id", sid).eq(
            "user_id", user_id
        ).execute()
    except Exception as e:
        logger.error(f"Failed to unlike session {sid} for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not unlike session",
        )


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}/likes
# ---------------------------------------------------------------------------

@router.get(
    "/{session_id}/likes",
    response_model=List[LikeEntry],
    summary="List users who liked a session",
)
async def get_session_likes(
    session_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    sid = str(session_id)

    # Verify session exists and is visible
    try:
        check = (
            client.table("study_sessions")
            .select("id, is_public, user_id")
            .eq("id", sid)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if not check.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if not check.data.get("is_public") and check.data["user_id"] != str(current_user.user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This session is private")

    try:
        likes_result = (
            client.table("session_likes")
            .select("user_id, created_at")
            .eq("session_id", sid)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch likes for session {sid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve likes",
        )

    if not likes_result.data:
        return []

    user_ids = [row["user_id"] for row in likes_result.data]
    liked_at_map = {row["user_id"]: row["created_at"] for row in likes_result.data}

    try:
        profiles_result = (
            client.table("profiles")
            .select("id, username, full_name, avatar_url")
            .in_("id", user_ids)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch profiles for likes on session {sid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve liker profiles",
        )

    return [
        LikeEntry(
            user_id=p["id"],
            username=p["username"],
            full_name=p.get("full_name"),
            avatar_url=p.get("avatar_url"),
            liked_at=liked_at_map[p["id"]],
        )
        for p in (profiles_result.data or [])
    ]


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/comments
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/comments",
    response_model=CommentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Post a comment on a session",
)
async def create_comment(
    session_id: UUID,
    body: CommentCreate,
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)
    sid = str(session_id)

    # Verify session exists and is visible
    try:
        check = (
            client.table("study_sessions")
            .select("id, is_public, user_id")
            .eq("id", sid)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if not check.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if not check.data.get("is_public") and check.data["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This session is private")

    # Validate parent comment: must belong to this session and must be top-level
    # (max nesting depth of 1 — replies to replies are not allowed)
    parent_comment_author_id: Optional[str] = None

    if body.parent_comment_id is not None:
        try:
            parent_check = (
                client.table("session_comments")
                .select("id, session_id, user_id, parent_comment_id")
                .eq("id", str(body.parent_comment_id))
                .single()
                .execute()
            )
        except Exception:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Parent comment not found")

        if not parent_check.data or parent_check.data["session_id"] != sid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parent comment does not belong to this session",
            )

        if parent_check.data.get("parent_comment_id") is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Replies to replies are not supported",
            )

        parent_comment_author_id = parent_check.data["user_id"]

    payload = {
        "session_id": sid,
        "user_id": user_id,
        "content": body.content,
        "parent_comment_id": str(body.parent_comment_id) if body.parent_comment_id else None,
    }

    try:
        result = client.table("session_comments").insert(payload).execute()
    except Exception as e:
        logger.error(f"Failed to create comment on session {sid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create comment",
        )

    row = result.data[0]

    # Fetch author profile for the response
    try:
        profile = (
            client.table("profiles")
            .select("username, full_name, avatar_url")
            .eq("id", user_id)
            .single()
            .execute()
        )
    except Exception:
        profile = None

    p = profile.data if profile and profile.data else {}

    # Notify the session owner (not the commenter themselves)
    create_notification(
        client,
        user_id=check.data["user_id"],
        actor_id=user_id,
        type="comment",
        entity_id=sid,
    )

    # Notify the parent comment author on a reply, if they are not the session owner
    # (avoids sending a duplicate notification to the same person)
    if parent_comment_author_id and parent_comment_author_id != check.data["user_id"]:
        create_notification(
            client,
            user_id=parent_comment_author_id,
            actor_id=user_id,
            type="comment",
            entity_id=sid,
        )

    return CommentResponse(
        id=row["id"],
        session_id=row["session_id"],
        user_id=row["user_id"],
        username=p.get("username", ""),
        full_name=p.get("full_name"),
        avatar_url=p.get("avatar_url"),
        content=row["content"],
        parent_comment_id=row.get("parent_comment_id"),
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}/comments
# ---------------------------------------------------------------------------

@router.get(
    "/{session_id}/comments",
    response_model=List[CommentResponse],
    summary="List comments on a session (paginated)",
)
async def list_comments(
    session_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    parent_comment_id: Optional[UUID] = Query(None, description="Filter replies to a specific comment"),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)
    sid = str(session_id)

    # Verify session exists and is visible
    try:
        check = (
            client.table("study_sessions")
            .select("id, is_public, user_id")
            .eq("id", sid)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if not check.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if not check.data.get("is_public") and check.data["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This session is private")

    try:
        query = (
            client.table("session_comments")
            .select("id, session_id, user_id, parent_comment_id, content, created_at")
            .eq("session_id", sid)
            .order("created_at", desc=False)
            .range(offset, offset + limit - 1)
        )
        if parent_comment_id is not None:
            query = query.eq("parent_comment_id", str(parent_comment_id))
        else:
            query = query.is_("parent_comment_id", "null")

        comments_result = query.execute()
    except Exception as e:
        logger.error(f"Failed to fetch comments for session {sid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve comments",
        )

    rows = comments_result.data or []
    if not rows:
        return []

    # Fetch author profiles in bulk
    author_ids = list({row["user_id"] for row in rows})
    try:
        profiles_result = (
            client.table("profiles")
            .select("id, username, full_name, avatar_url")
            .in_("id", author_ids)
            .execute()
        )
        profiles_map = {p["id"]: p for p in (profiles_result.data or [])}
    except Exception:
        profiles_map = {}

    return [
        CommentResponse(
            id=row["id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            username=profiles_map.get(row["user_id"], {}).get("username", ""),
            full_name=profiles_map.get(row["user_id"], {}).get("full_name"),
            avatar_url=profiles_map.get(row["user_id"], {}).get("avatar_url"),
            content=row["content"],
            parent_comment_id=row.get("parent_comment_id"),
            created_at=row["created_at"],
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}/comments/{comment_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/{session_id}/comments/{comment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a comment (owner only)",
)
async def delete_comment(
    session_id: UUID,
    comment_id: UUID,
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)
    sid = str(session_id)
    cid = str(comment_id)

    # Verify comment exists and belongs to this session
    try:
        check = (
            client.table("session_comments")
            .select("id, session_id, user_id")
            .eq("id", cid)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comment not found")

    if not check.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comment not found")

    if check.data["session_id"] != sid:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comment not found")

    if check.data["user_id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own comments",
        )

    # Deleting a parent comment cascades to all its replies via the FK in schema.sql.
    # This is intentional — replies are meaningless without their parent context.
    try:
        client.table("session_comments").delete().eq("id", cid).execute()
    except Exception as e:
        logger.error(f"Failed to delete comment {cid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete comment",
        )
