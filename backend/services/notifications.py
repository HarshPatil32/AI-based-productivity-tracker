import logging
from typing import Literal, Optional

logger = logging.getLogger(__name__)

NotificationType = Literal["follow", "like", "comment"]


def create_notification(
    client,
    *,
    user_id: str,
    actor_id: str,
    type: NotificationType,
    entity_id: Optional[str] = None,
) -> None:
    """
    Insert a notification row for `user_id`.

    Failures are logged but never propagated — a notification error must never
    break the primary action that triggered it.
    """
    if user_id == actor_id:
        # Never notify someone about their own actions.
        return

    payload: dict = {
        "user_id": user_id,
        "actor_id": actor_id,
        "type": type,
        "is_read": False,
    }
    if entity_id is not None:
        payload["entity_id"] = entity_id

    try:
        client.table("notifications").insert(payload).execute()
    except Exception as e:
        logger.error(
            f"Failed to create '{type}' notification for user {user_id} "
            f"(actor={actor_id}, entity={entity_id}): {e}"
        )
