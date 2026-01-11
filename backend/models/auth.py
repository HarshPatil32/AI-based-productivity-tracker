from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from uuid import UUID

class UserRegisterModel(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PasswordReset(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length = 8)

class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length = 8)

class TokenRefresh(BaseModel):
    refresh_token: str

# RESPONSE MODELS

class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # in seconds
    user: UserResponse

class AuthResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None


class TokenData(BaseModel):
    user_id: Optional[UUID] = None
    email: Optional[str] = None
    exp: Optional[datetime] = None  

class UserInDB(UserResponse):
    hashed_password: str