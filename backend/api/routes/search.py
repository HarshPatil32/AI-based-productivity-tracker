from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Literal, Optional

from uuid import UUID
from datetime import datetime
import logging

from pydantic import BaseModel, ConfigDict

from backend.middleware.auth import require_auth
from backend.services.database import get_supabase_client
from backend.utils.auth import TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])

_MAX_QUERY_LEN = 100


# --------------- Response models ---------------

class UserSearchResult(BaseModel):
    id: UUID
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    total_study_time: int = 0
    followers_count: Optional[int] = None
    following_count: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class SessionSearchResult(BaseModel):
    id: UUID
    user_id: UUID
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    title: str
    description: Optional[str] = None
    session_duration: int
    focus_score: float
    attention_score: float
    quality: str
    session_date: Optional[str] = None
    session_start_time: Optional[datetime] = None
    likes_count: int = 0
    comments_count: int = 0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SearchResponse(BaseModel):
    users: List[UserSearchResult] = []
    users_total: int = 0
    sessions: List[SessionSearchResult] = []
    sessions_total: int = 0


# --------------- Helpers ---------------

def _escape_like(value: str) -> str:
    """Escape LIKE metacharacters and PostgREST filter special characters."""
    return (
        value.replace("\\", "\\\\")
             .replace("%", "\\%")
             .replace("_", "\\_")
             .replace(",", "")   # commas would break the .or_() filter string
             .replace(".", "")   # dots would break PostgREST field references
    )


# --------------- Endpoint ---------------

@router.get(
    "/",
    response_model=SearchResponse,
    summary="Search users and/or sessions",
    description=(
        "Case-insensitive partial-match search across users (username, full_name) "
        "and public sessions (title, description). Use `type` to narrow the search. "
        "When type=all, limit and offset apply independently to users and sessions. "
        "Only public sessions are returned."
    ),
)
async def search(
    q: str = Query(..., min_length=1, max_length=_MAX_QUERY_LEN, description="Search term"),
    search_type: Literal["users", "sessions", "all"] = Query("all", alias="type", description="Entity type to search"),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(require_auth),
):
    q = q.strip()
    if not q:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Search query must not be blank",
        )

    client = get_supabase_client()
    safe_q = _escape_like(q)
    pattern = f"%{safe_q}%"

    users: List[UserSearchResult] = []
    users_total: int = 0
    sessions: List[SessionSearchResult] = []
    sessions_total: int = 0

    if search_type in ("users", "all"):
        try:
            result = (
                client.table("user_profile_summary")
                .select(
                    "id, username, full_name, avatar_url, bio, "
                    "total_study_time, followers_count, following_count",
                    count="exact",
                )
                .or_(f"username.ilike.{pattern},full_name.ilike.{pattern}")
                .range(offset, offset + limit - 1)
                .execute()
            )
            users = [UserSearchResult(**row) for row in (result.data or [])]
            users_total = result.count or 0
        except Exception as e:
            logger.error(f"User search failed (q={q!r}): {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not complete user search",
            )

    if search_type in ("sessions", "all"):
        try:
            result = (
                client.table("feed_sessions")
                .select(
                    "id, user_id, username, full_name, avatar_url, "
                    "title, description, session_duration, focus_score, "
                    "attention_score, quality, session_date, session_start_time, "
                    "likes_count, comments_count, created_at",
                    count="exact",
                )
                .or_(f"title.ilike.{pattern},description.ilike.{pattern}")
                .range(offset, offset + limit - 1)
                .execute()
            )
            sessions = [SessionSearchResult(**row) for row in (result.data or [])]
            sessions_total = result.count or 0
        except Exception as e:
            logger.error(f"Session search failed (q={q!r}): {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not complete session search",
            )

    return SearchResponse(
        users=users,
        users_total=users_total,
        sessions=sessions,
        sessions_total=sessions_total,
    )
