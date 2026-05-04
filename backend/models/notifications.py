from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime
from uuid import UUID


class NotificationResponse(BaseModel):
    id: UUID
    user_id: UUID
    actor_id: Optional[UUID] = None
    type: str
    entity_id: Optional[UUID] = None
    is_read: bool = False
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="ignore")
