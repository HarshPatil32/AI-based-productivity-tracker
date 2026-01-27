from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID
import jwt
from passlib.context import CryptoContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.config.settings import get_settings
from backend.models.auth import TokenData, UserInDB

settings = get_settings()

pwd_context = CryptoContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def hash_password(password: str) -> str:
    return pwd_context.hash(password) # Hash a pw using bcrypt

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password) # Verify a pw against a hash

def create_access_token(user_id: UUID, email: str, expires_delta: Optional[timedelta] = None):
    # Creates JWT access token
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
                }
    
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

def create_refresh_token(user_id, UUID, email: str, expires_delta: Optional[timedelta] = None):
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta

    else:
        expire = datetime.now(timezone.utc) + timedelta(days = 30) # refresh token lasts 30 days


def decode_token(token: str):
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id = payload.get("sub")
        email=payload.get("email")
        exp = payload.get("exp")

        if user_id is None:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
                headers = {"WWW-Authenticate": "Bearer"}
            )
        
        return TokenData(
            user_id = UUID(user_id),
            email=email,
            exp = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None
        )
    
    except jwt.ExpireSignatureError:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            details = "Expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    


def verify_refresh_token(token: str):
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms = [settings.JWT_ALGORITHM]
        )

        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail = "Invalid token type: expected refresh token"
            )
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code =status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        return UUID(user_id)
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code =status.HTTP_401_UNAUTHORIZED,
            detail="Expired refresh token"
        )
    
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code =status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # This is a FastAPI dependency to get the authenticated user from the JWT token

    token = credentials.credentials
    return decode_token(token)

async def get_current_user_id(current_user: TokenData = Depends(get_current_user)):
    # FastAPI dependency to get only the current user's ID

    return current_user.user_id

def validate_password_strength(password):
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any (c.islower() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True


def generate_password_reset_token(email: str):
    expire = datetime.now(timezone.utc) + timedelta(hours = 1)
    payload = {
        "sub": email,
        "exp": expire,
        "type": "password_reset"
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm = settings.JWT_ALGORITHM
    )

def verify_password_reset_token(token: str):
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        if payload.get("type") != "password_reset":
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail="Invalid token type"
            )
        
        email = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code =status.HTTP_400_BAD_REQUEST,
                detail="Invalid password reset token"
            )
        
        return email
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password reset token has expired"
        )

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid password reset token"
        )
    



