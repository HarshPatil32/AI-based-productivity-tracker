from fastapi import APIRouter, HTTPException, status, Depends, Query, UploadFile, File
from typing import List
from uuid import UUID
import logging

from backend.models.users import (
    AvatarUploadResponse,
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


# --------------- Avatar upload ---------------

_ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
_MAX_AVATAR_BYTES = 5 * 1024 * 1024  # 5 MB
_CHUNK_SIZE = 64 * 1024  # 64 KB

# Leading magic bytes used to verify actual file content against the declared content type
_MAGIC_SIGNATURES: dict[str, bytes] = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/webp": b"RIFF",  # first 4 bytes; bytes 8-12 must also be b"WEBP"
}


def _verify_magic_bytes(content_type: str, data: bytes) -> bool:
    sig = _MAGIC_SIGNATURES.get(content_type)
    if sig is None or len(data) < len(sig):
        return False
    if content_type == "image/webp":
        return data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP"
    return data[: len(sig)] == sig


@router.post(
    "/me/avatar",
    response_model=AvatarUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload an avatar image for the authenticated user",
)
async def upload_my_avatar(
    file: UploadFile = File(...),
    current_user: TokenData = Depends(require_auth),
):
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported file type. Allowed: JPEG, PNG, WebP",
        )

    # Stream-read in chunks so oversized uploads are rejected before fully buffering
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_CHUNK_SIZE):
        total += len(chunk)
        if total > _MAX_AVATAR_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File exceeds the 5 MB size limit",
            )
        chunks.append(chunk)
    file_bytes = b"".join(chunks)

    # Verify actual file content matches the declared content type
    if not _verify_magic_bytes(content_type, file_bytes):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File content does not match the declared content type",
        )

    user_id = str(current_user.user_id)
    # Fixed path without extension so re-uploads with a different type never leave orphaned files
    storage_path = f"{user_id}/avatar"

    admin = get_supabase_admin_client()
    try:
        admin.storage.from_("avatars").upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": content_type, "upsert": "true"},
        )
    except Exception as e:
        logger.error(f"Avatar upload to storage failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not upload avatar",
        )

    public_url: str = admin.storage.from_("avatars").get_public_url(storage_path)

    if not public_url:
        logger.error(f"get_public_url returned empty for user {user_id}, path {storage_path}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate public avatar URL",
        )

    try:
        admin.table("profiles").update({"avatar_url": public_url}).eq("id", user_id).execute()
    except Exception as e:
        logger.error(f"Failed to update avatar_url for user {user_id}: {e}")
        # Roll back the storage upload so the file and DB stay in sync
        try:
            admin.storage.from_("avatars").remove([storage_path])
            logger.info(f"Rolled back avatar storage for user {user_id} after DB failure")
        except Exception as cleanup_err:
            logger.error(f"Storage rollback also failed for user {user_id}: {cleanup_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update profile with avatar URL",
        )

    return AvatarUploadResponse(avatar_url=public_url)


# --------------- User search ---------------

@router.get(
    "/search",
    response_model=List[ProfileResponse],
    summary="Search users by username or full name",
)
async def search_users(
    q: str = Query("", min_length=0, max_length=100),
    limit: int = Query(10, ge=1, le=50),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_client()
    pattern = f"%{q}%"

    try:
        result = (
            client.table("user_profile_summary")
            .select("*")
            .or_(f"username.ilike.{pattern},full_name.ilike.{pattern}")
            .neq("id", str(current_user.user_id))
            .limit(limit)
            .execute()
        )
    except Exception as e:
        logger.error(f"User search failed for query '{q}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed",
        )

    return [_row_to_profile(row) for row in (result.data or [])]



# --------------- Suggested users ---------------

@router.get(
    "/suggested",
    response_model=List[ProfileResponse],
    summary="Get suggested users to follow",
)
async def get_suggested_users(
    limit: int = Query(5, ge=1, le=20),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_client()
    user_id = str(current_user.user_id)

    try:
        follows_result = (
            client.table("user_relationships")
            .select("following_id")
            .eq("follower_id", user_id)
            .execute()
        )
        following_ids = [row["following_id"] for row in (follows_result.data or [])] + [user_id]
        # Supabase Python client expects a string like '(id1,id2,...)' for 'in' operator
        in_str = f"({','.join(following_ids)})" if following_ids else "()"
        query = client.table("user_profile_summary").select("*")
        query = query.filter("id", "not.in", in_str)
        result = (
            query
            .order("followers_count", desc=True)
            .limit(limit)
            .execute()
        )
        return [_row_to_profile(row) for row in (result.data or [])]
    except Exception as e:
        logger.error(f"/users/suggested failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not get suggested users: {e}",
        )

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
    client = get_supabase_admin_client()
    caller_id = str(current_user.user_id)

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

    target_id = str(result.data["id"])

    # Owner always sees their own profile
    if caller_id != target_id:
        try:
            settings_result = (
                client.table("user_settings")
                .select("profile_visibility")
                .eq("user_id", target_id)
                .single()
                .execute()
            )
            profile_visibility = (settings_result.data or {}).get("profile_visibility", "public")
        except Exception:
            profile_visibility = "public"

        if profile_visibility == "private":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This profile is private")

        if profile_visibility == "friends":
            follow_check = (
                client.table("user_relationships")
                .select("id")
                .eq("follower_id", caller_id)
                .eq("following_id", target_id)
                .execute()
            )
            if not follow_check.data:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This profile is only visible to followers")

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
