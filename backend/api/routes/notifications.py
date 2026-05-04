from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional
from uuid import UUID
import logging

from backend.models.notifications import NotificationResponse
from backend.middleware.auth import require_auth
from backend.services.database import get_supabase_admin_client
from backend.utils.auth import TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notifications", tags=["Notifications"])


# ---------------------------------------------------------------------------
# GET /notifications
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=List[NotificationResponse],
    summary="Get notifications for the authenticated user",
    description=(
        "Returns notifications for the current user, sorted by most recent. "
        "Optionally filter to only unread notifications via `is_read=false`."
    ),
)
async def get_notifications(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    is_read: Optional[bool] = Query(None, description="Filter by read status"),
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)

    try:
        query = (
            client.table("notifications")
            .select("*")
            .eq("user_id", user_id)
        )

        if is_read is not None:
            query = query.eq("is_read", is_read)

        result = (
            query
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch notifications for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve notifications",
        )

    return [NotificationResponse(**row) for row in (result.data or [])]


# ---------------------------------------------------------------------------
# POST /notifications/read-all  (must be before /{notification_id}/read)
# ---------------------------------------------------------------------------

@router.post(
    "/read-all",
    status_code=status.HTTP_200_OK,
    summary="Mark all notifications as read",
)
async def mark_all_notifications_read(
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)

    try:
        client.table("notifications").update({"is_read": True}).eq("user_id", user_id).eq(
            "is_read", False
        ).execute()
    except Exception as e:
        logger.error(f"Failed to bulk-mark notifications as read for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update notifications",
        )

    return {"detail": "All notifications marked as read"}


# ---------------------------------------------------------------------------
# PATCH /notifications/{id}/read
# ---------------------------------------------------------------------------

@router.patch(
    "/{notification_id}/read",
    response_model=NotificationResponse,
    summary="Mark a single notification as read",
)
async def mark_notification_read(
    notification_id: UUID,
    current_user: TokenData = Depends(require_auth),
):
    client = get_supabase_admin_client()
    user_id = str(current_user.user_id)
    nid = str(notification_id)

    # Single UPDATE filtered by both id and user_id — no separate SELECT needed
    # in the happy path. On the rare miss, a secondary read distinguishes 404/403.
    try:
        result = (
            client.table("notifications")
            .update({"is_read": True})
            .eq("id", nid)
            .eq("user_id", user_id)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to mark notification {nid} as read for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update notification",
        )

    if result.data:
        return NotificationResponse(**result.data[0])

    # No rows matched — determine whether this is a 404 or 403
    try:
        check = (
            client.table("notifications")
            .select("user_id")
            .eq("id", nid)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")

    if not check.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You can only update your own notifications",
    )
