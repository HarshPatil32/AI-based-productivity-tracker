from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List
from uuid import UUID
import logging

from backend.models.users import FollowerEntry
from backend.middleware.auth import require_auth
from backend.services.database import get_supabase_client, get_supabase_admin_client
from backend.utils.auth import TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Follows"])


# --------------- Helpers ---------------

def _row_to_follower_entry(row: dict) -> FollowerEntry:
    """Convert a profiles row to a slim FollowerEntry."""
    return FollowerEntry(
        id=row["id"],
        username=row["username"],
        full_name=row.get("full_name"),
        avatar_url=row.get("avatar_url"),
        total_study_time=row.get("total_study_time", 0),
    )


def _resolve_profile_ids(client, relationship_rows: list, id_field: str) -> List[FollowerEntry]:
    """Bulk-fetch profiles for a list of relationship rows."""
    ids = [row[id_field] for row in relationship_rows]
    if not ids:
        return []
    profiles = (
        client.table("profiles")
        .select("id, username, full_name, avatar_url, total_study_time")
        .in_("id", ids)
        .execute()
    )
    return [_row_to_follower_entry(p) for p in (profiles.data or [])]


# --------------- Follow / Unfollow ---------------

@router.post(
    "/{user_id}/follow",
    status_code=status.HTTP_201_CREATED,
    summary="Follow a user",
)
async def follow_user(
    user_id: UUID,
    current_user: TokenData = Depends(require_auth),
):
    follower_id = str(current_user.user_id)
    following_id = str(user_id)

    if follower_id == following_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot follow yourself",
        )

    client = get_supabase_admin_client()

    # Ensure the target user exists
    target = client.table("profiles").select("id").eq("id", following_id).execute()
    if not target.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Idempotent: ignore if already following
    existing = (
        client.table("user_relationships")
        .select("id")
        .eq("follower_id", follower_id)
        .eq("following_id", following_id)
        .execute()
    )
    if existing.data:
        return {"detail": "Already following"}

    try:
        client.table("user_relationships").insert(
            {"follower_id": follower_id, "following_id": following_id}
        ).execute()
    except Exception as e:
        logger.error(f"Failed to follow user {following_id} for {follower_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not follow user",
        )

    return {"detail": "Now following"}


@router.delete(
    "/{user_id}/follow",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unfollow a user",
)
async def unfollow_user(
    user_id: UUID,
    current_user: TokenData = Depends(require_auth),
):
    follower_id = str(current_user.user_id)
    following_id = str(user_id)

    client = get_supabase_admin_client()

    try:
        client.table("user_relationships").delete().eq("follower_id", follower_id).eq(
            "following_id", following_id
        ).execute()
    except Exception as e:
        logger.error(f"Failed to unfollow user {following_id} for {follower_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not unfollow user",
        )


# --------------- Convenience: my own lists (must be before /{user_id}/... routes) ---------------

@router.get(
    "/me/followers",
    response_model=List[FollowerEntry],
    summary="Get the authenticated user's followers",
)
async def get_my_followers(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(require_auth),
):
    return await get_followers(
        user_id=current_user.user_id,
        limit=limit,
        offset=offset,
        current_user=current_user,
    )


@router.get(
    "/me/following",
    response_model=List[FollowerEntry],
    summary="Get users the authenticated user is following",
)
async def get_my_following(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(require_auth),
):
    return await get_following(
        user_id=current_user.user_id,
        limit=limit,
        offset=offset,
        current_user=current_user,
    )


# --------------- Follower / Following lists ---------------

@router.get(
    "/{user_id}/followers",
    response_model=List[FollowerEntry],
    summary="Get a user's followers",
)
async def get_followers(
    user_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_client()
    target_id = str(user_id)

    try:
        result = (
            client.table("user_relationships")
            .select("follower_id")
            .eq("following_id", target_id)
            .range(offset, offset + limit - 1)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch followers for {target_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve followers",
        )

    return _resolve_profile_ids(client, result.data or [], "follower_id")


@router.get(
    "/{user_id}/following",
    response_model=List[FollowerEntry],
    summary="Get users that a user is following",
)
async def get_following(
    user_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_client()
    target_id = str(user_id)

    try:
        result = (
            client.table("user_relationships")
            .select("following_id")
            .eq("follower_id", target_id)
            .range(offset, offset + limit - 1)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch following for {target_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve following list",
        )

    return _resolve_profile_ids(client, result.data or [], "following_id")
