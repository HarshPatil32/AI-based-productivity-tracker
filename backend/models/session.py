from pydantic import BaseModel, ConfigDict, Field, field_validator
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
    focus_score: float = Field(..., description = "Derived score: focused_time / duration")
    created_at: datetime


class SessionSummary(BaseModel):
    total_sessions: int
    total_study_seconds: int
    avg_attention_score: float
    avg_focus_score: float
    avg_eyes_closed_time: float
    avg_face_missing_time: float
    avg_head_pose_off_time: float
    total_attention_lost: float


class LikeEntry(BaseModel):
    user_id: UUID
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    liked_at: datetime


class CommentCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000, description="Comment text")
    parent_comment_id: Optional[UUID] = Field(None, description="ID of parent comment for replies")

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("content must not be blank or whitespace only")
        return stripped


class CommentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    user_id: UUID
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    content: str
    parent_comment_id: Optional[UUID] = None
    created_at: datetime


