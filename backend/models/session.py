from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime
from uuid import UUID

class SessionCreate(BaseModel):
    started_at: datetime
    ended_at: datetime
    duration_seconds: int = Field(..., ge = 0, description = "Total session length in seconds")
    eyes_closed_time: float = Field(..., ge = 0, description = "Total seconds eyes were closed")
    face_missing_time: float = Field(..., ge = 0, description = "Total seconds face was not detected")
    head_pose_off_time: float = Field(..., ge = 0, description = "Total seconds head pose was off")
    total_attention_lost: float = Field(..., ge = 0, description = "Sum of all distraction time")

    notes: Optional[str] = Field(None, max_length = 500)


class SessionResponse(SessionCreate):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    attention_score: float = Field(..., description = "Derived score: 1 - (attention_lost / duration)")
    created_at: datetime


class SessionSummary(BaseModel):
    total_sessions: int
    total_study_seconds: int
    avg_attention_score: float
    avg_eyes_closed_time: float
    avg_face_missing_time: float
    avg_head_pose_off_time: float
    total_attention_lost: float


