from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.config.settings import get_settings
from backend.models.auth import TokenData, UserInDB

settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def hash_password(password: str) -> str:
    return pwd_context.hash(password) # Hash a pw using bcrypt

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password) # Verify a pw against a hash

def create_access_token(user_id: UUID, email: str, expires_delta: Optional[timedelta] = None):
    # Creates JWT access token
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "iat": now,
        "type": "access"
    }

    import logging
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    logging.warning(f"[DEBUG] Creating access token: now={now}, exp={expire}, payload={payload}, token={token}")
    return token

def create_refresh_token(user_id: UUID, email: str, expires_delta: Optional[timedelta] = None):
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=30)

    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "iat": now,
        "type": "refresh"
    }

    import logging
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    logging.warning(f"[DEBUG] Creating refresh token: now={now}, exp={expire}, payload={payload}, token={token}")
    return token


def decode_token(token: str):
    import logging
    try:
        logging.warning(f"[DEBUG] Decoding token string: {token}")
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id = payload.get("sub")
        email = payload.get("email")
        exp = payload.get("exp")
        now = datetime.now(timezone.utc)
        logging.warning(f"[DEBUG] Decoding token: now={now}, exp={exp}, payload={payload}")

        if user_id is None:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
                headers = {"WWW-Authenticate": "Bearer"}
            )
        return TokenData(
            user_id = UUID(user_id),
            email = email,
            exp = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None
        )
    except jwt.ExpiredSignatureError:
        logging.warning("[DEBUG] Token has expired!")
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError:
        logging.warning("[DEBUG] Invalid token!")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
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
    



