from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


# --------------- Profile models ---------------

class ProfileResponse(BaseModel):
    id: UUID
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    total_study_time: int = 0          # seconds
    created_at: datetime

    # Enriched fields (from user_profile_summary view)
    total_sessions: Optional[int] = None
    avg_focus_score: Optional[float] = None
    followers_count: Optional[int] = None
    following_count: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(None, max_length=200)
    avatar_url: Optional[str] = Field(None)
    bio: Optional[str] = Field(None, max_length=500)


# --------------- Follow models ---------------

class FollowResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    follower_id: UUID
    following_id: UUID
    created_at: datetime


class FollowerEntry(BaseModel):
    """A slim profile card used in follower / following lists."""
    id: UUID
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    total_study_time: int = 0


# --------------- User settings models ---------------

class UserSettingsResponse(BaseModel):
    user_id: UUID

    # Privacy
    profile_visibility: str = "public"
    session_visibility: str = "public"
    show_study_time: bool = True
    show_focus_scores: bool = True

    # Notifications
    email_notifications: bool = True
    email_on_like: bool = True
    email_on_comment: bool = True
    email_on_follow: bool = True

    # Display
    theme: str = "light"
    language: str = "en"
    timezone: str = "UTC"

    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class UserSettingsUpdate(BaseModel):
    profile_visibility: Optional[str] = Field(None, pattern="^(public|friends|private)$")
    session_visibility: Optional[str] = Field(None, pattern="^(public|friends|private)$")
    show_study_time: Optional[bool] = None
    show_focus_scores: Optional[bool] = None

    email_notifications: Optional[bool] = None
    email_on_like: Optional[bool] = None
    email_on_comment: Optional[bool] = None
    email_on_follow: Optional[bool] = None

    theme: Optional[str] = Field(None, pattern="^(light|dark|system)$")
    language: Optional[str] = Field(None, max_length=10)
    timezone: Optional[str] = Field(None, max_length=50)
