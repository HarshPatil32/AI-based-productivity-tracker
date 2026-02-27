from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List
from uuid import UUID
import logging

from backend.models.users import (
    ProfileResponse,
    ProfileUpdate,
    UserSettingsResponse,
    UserSettingsUpdate,
)
from backend.middleware.auth import require_auth
from backend.services.database import get_supabase_client, get_supabase_admin_client
from backend.utils.auth import TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Users"])


# --------------- Helpers ---------------

def _row_to_profile(row: dict) -> ProfileResponse:
    return ProfileResponse(
        id=row["id"],
        username=row["username"],
        full_name=row.get("full_name"),
        avatar_url=row.get("avatar_url"),
        bio=row.get("bio"),
        total_study_time=row.get("total_study_time", 0),
        created_at=row["created_at"],
        total_sessions=row.get("total_sessions"),
        avg_focus_score=row.get("avg_focus_score"),
        followers_count=row.get("followers_count"),
        following_count=row.get("following_count"),
    )


# --------------- My profile ---------------

@router.get(
    "/me",
    response_model=ProfileResponse,
    summary="Get the authenticated user's profile",
)
async def get_my_profile(
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_client()
    user_id = str(current_user.user_id)

    try:
        result = (
            client.table("user_profile_summary")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch profile for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve profile",
        )

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    return _row_to_profile(result.data)


@router.patch(
    "/me",
    response_model=ProfileResponse,
    summary="Update the authenticated user's profile",
)
async def update_my_profile(
    updates: ProfileUpdate,
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)

    # Only send fields that were actually provided
    payload = updates.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No update fields provided",
        )

    try:
        result = (
            client.table("profiles")
            .update(payload)
            .eq("id", user_id)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to update profile for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update profile",
        )

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    # Return enriched profile from the view
    view_result = (
        client.table("user_profile_summary")
        .select("*")
        .eq("id", user_id)
        .single()
        .execute()
    )
    return _row_to_profile(view_result.data)


# --------------- Public profile lookup ---------------

@router.get(
    "/{username}",
    response_model=ProfileResponse,
    summary="Get a user's public profile by username",
)
async def get_profile_by_username(
    username: str,
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_client()

    try:
        result = (
            client.table("user_profile_summary")
            .select("*")
            .eq("username", username)
            .single()
            .execute()
        )
    except Exception as e:
        logger.error(f"Profile lookup failed for username '{username}': {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return _row_to_profile(result.data)


# --------------- Settings ---------------

@router.get(
    "/me/settings",
    response_model=UserSettingsResponse,
    summary="Get the authenticated user's settings",
)
async def get_my_settings(
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)

    try:
        # Upsert ensures the row exists with defaults, then fetch it
        client.table("user_settings").upsert(
            {"user_id": user_id}, on_conflict="user_id"
        ).execute()
        result = (
            client.table("user_settings")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch settings for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve settings",
        )

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Settings not found")

    return UserSettingsResponse(**result.data)


@router.patch(
    "/me/settings",
    response_model=UserSettingsResponse,
    summary="Update the authenticated user's settings",
)
async def update_my_settings(
    updates: UserSettingsUpdate,
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)

    payload = updates.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No update fields provided",
        )

    payload["user_id"] = user_id
    try:
        client.table("user_settings").upsert(payload, on_conflict="user_id").execute()
        result = (
            client.table("user_settings")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to update settings for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update settings",
        )

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Settings not found")

    return UserSettingsResponse(**result.data)
