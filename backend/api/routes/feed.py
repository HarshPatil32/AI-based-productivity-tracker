from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional
from uuid import UUID
from datetime import datetime
import logging

from backend.middleware.auth import require_auth
from backend.services.database import get_supabase_client, get_supabase_admin_client
from backend.utils.auth import TokenData
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feed", tags=["Feed"])


# --------------- Response model ---------------

class FeedSessionResponse(BaseModel):
    """A session card as it appears in the social feed, enriched with author info."""
    id: UUID
    user_id: UUID
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None

    title: str
    description: Optional[str] = None
    session_duration: int
    focused_time: int
    focus_score: float
    attention_score: float
    quality: str
    session_date: str
    session_start_time: Optional[datetime] = None
    session_end_time: Optional[datetime] = None
    likes_count: int = 0
    comments_count: int = 0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --------------- Endpoints ---------------

@router.get(
    "/",
    response_model=List[FeedSessionResponse],
    summary="Get the social feed for the authenticated user",
    description=(
        "Returns public study sessions from users that the authenticated user follows, "
        "ordered by most recent. Falls back to a global public feed if the user doesn't "
        "follow anyone yet."
    ),
)
async def get_feed(
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)

    # Resolve who the current user follows
    try:
        follows_result = (
            client.table("user_relationships")
            .select("following_id")
            .eq("follower_id", user_id)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch follow list for feed (user {user_id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not build feed",
        )

    original_following_ids = [row["following_id"] for row in (follows_result.data or [])]
    following_ids = original_following_ids

    # Filter out followed users who have set session_visibility=private.
    # (friends: current user already follows them, so their sessions are visible;
    #  private: no one except the owner sees them)
    if following_ids:
        try:
            vis_result = (
                client.table("user_settings")
                .select("user_id, session_visibility")
                .in_("user_id", following_ids)
                .execute()
            )
            private_ids = {
                row["user_id"]
                for row in (vis_result.data or [])
                if row.get("session_visibility") == "private"
            }
            following_ids = [uid for uid in following_ids if uid not in private_ids]
        except Exception as e:
            logger.error(f"Failed to fetch visibility settings for feed: {e}")

    try:
        query = client.table("feed_sessions").select("*")

        if original_following_ids:
            # User follows people: show allowed followed users + their own sessions.
            # If all followed users are private, feed_ids collapses to just [user_id].
            feed_ids = following_ids + [user_id]
            query = query.in_("user_id", feed_ids)
        # If truly following nobody, fall through with no filter → global public feed

        result = (
            query
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch feed sessions for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve feed",
        )

    return [FeedSessionResponse(**row) for row in (result.data or [])]


@router.get(
    "/global",
    response_model=List[FeedSessionResponse],
    summary="Get the global public feed (all public sessions)",
)
async def get_global_feed(
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_client()

    try:
        result = (
            client.table("feed_sessions")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch global feed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve global feed",
        )

    return [FeedSessionResponse(**row) for row in (result.data or [])]
