from fastapi import APIRouter, HTTPException, status, Depends, Request
from typing import Optional
from uuid import UUID
import logging

from backend.models.auth import (
    UserRegisterModel,
    UserLogin,
    PasswordReset,
    PasswordUpdate,
    TokenRefresh,
    UserResponse,
    TokenResponse,
    AuthResponse,
    TokenData
)

from backend.utils.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_refresh_token,
    validate_password_strength,
    generate_password_reset_token,
    verify_password_reset_token
)

from backend.middleware.auth import require_auth, get_current_user_from_request
from backend.services.database import get_supabase_client, get_supabase_admin_client
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model = AuthResponse, status_code = status.HTTP_201_CREATED)
async def register (user_data: UserRegisterModel):
    # Register a new user with email, password and username
    client = get_supabase_admin_client()

    try:
        if not validate_password_strength(user_data.password):
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters with uppercase, lowercase, and a number"
            )
        
        existing_username = client.table("profiles").select("id").eq("username", user_data.username).execute()
        if existing_username.data:
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        auth_response = client.auth.admin.create_user(
            {
                "email": user_data.email,
                "password": user_data.password,
                "email_confirm": True,
                "user_metadata": {
                    "username": user_data.username,
                    "full_name": user_data.full_name
                }
            }
        )

        if not auth_response.user:
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail="Failed to create the user"
            )
        
        user_id = auth_response.user.id

        profile_data = {
            "id": str(user_id),
            "username": user_data.username,
            "full_name": user_data.full_name,
            "avatar_url": None,
            "bio": None,
            "total_study_time": 0
        }

        profile_response = client.table("profiles").insert(profile_data).execute()
        # Delete the auth user if it doesn't succesfully create the profile
        if not profile_response.data:
            client.auth.admin.delete_user(str(user_id))
            raise HTTPException(
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create user profile"
            )
        
        # Tokens
        access_token = create_access_token(user_id = UUID(str(user_id)), email = user_data.email)
        refresh_token = create_refresh_token(user_id = UUID(str(user_id)), email = user_data.email)

        logger.info(f"User registered: {user_data.email}")

        return AuthResponse (
            success = True,
            message = "Registration succesful",
            data = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "user": {
                    "id": str(user_id),
                    "email": user_data.email,
                    "username": user_data.username,
                    "full_name": user_data.full_name
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error during registration"
            )
    


@router.post("/login", response_model=AuthResponse)
async def login (credentials: UserLogin):
    # Authenticate the user with email and password
    client = get_supabase_client()

    try:
        auth_response = client.auth.sign_in_with_password({
            "email": credentials.email,
            "password": credentials.password
        })

        if not auth_response.user:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        user = auth_response.user
        user_id = user.id

        profile_response = client.table("profiles").select("*").eq("id", str(user_id)).single().execute()
        profile = profile_response.data if profile_response.data else {}

        # Generate tokens
        access_token = create_access_token(user_id=UUID(str(user_id)), email=credentials.email)
        refresh_token = create_refresh_token(user_id = UUID(str(user_id)), email = credentials.email)

        logger.info(f"User logged in: {credentials.email}")

        return AuthResponse(
            success=True,
            message = "Login successful",
            data = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "user": {
                    "id": str(user_id),
                    "email": credentials.email,
                    "username": profile.get("username"),
                    "full_name": profile.get("full_name"),
                    "avatar_url": profile.get("avatar_url"),
                    "created_at": profile.get("created_at")
                }
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
    

@router.post("/logout", response_model=AuthResponse)
async def logout(request: Request, current_user: TokenData = Depends(require_auth)):
    # Logs out the user, invalidates the session on the server side

    try:
        client = get_supabase_client()
        client.auth.sign_out()

        logger.info(f"User logged out: {current_user.email}")

        return AuthResponse(
            success = True,
            message = "Logged out succesfully",
            data = None
        )
    
    except Exception as e:
        logger.error(f"Logout error: {e}")
        #Shuld still return success as the client should clear tokens
        return AuthResponse (
            success = True,
            message = "Logged out succesfully",
            data = None
        )
    


@router.post("/refresh", response_model = AuthResponse)
async def refresh_token(token_data: TokenRefresh):
    # Refresh access token using a valid refresh token
    try:
        user_id = verify_refresh_token(token_data.refresh_token)

        client = get_supabase_admin_client()
        user_response = client.auth.admin.get_user_by_id(str(user_id))

        if not user_response.user:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        user = user_response.user
        email = user.email

        new_access_token = create_access_token(user_id = user_id, email = email)
        new_refresh_token = create_refresh_token(user_id = user_id, email = email)

        logger.info(f"Token refreshed for user: {email}")
        return AuthResponse (
            success = True,
            message = "Token refreshed succesfully",
            data = {
                "access_token": new_access_token,
                "refresh_token": new_refresh_token,
                "token_type": "bearer",
                "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )



@router.post("/forgot-password", response_model = AuthResponse)
async def forgot_password(email: str):
    # Forgot password process

    try:
        client = get_supabase_client()

        # I want to send the passward reset email through supabase
        client.auth.reset_password_email(email)

        logger.info(f"Password reset requested for: {email}")

        return AuthResponse(
            success = True,
            message = " If an account with that email exists, the password reset link was sent",
            data = None
        )
    
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        # Just return success anyways to prevent weird stuff
        return AuthResponse(
            success = True,
            message="If an account with that email exists, the password reset link was sent",
            data = None
        )
    

@router.post("/reset-password", response_model = AuthResponse)
async def reset_password(token: str, new_password: str):
    # Reset password with a valid token

    try:
        if not validate_password_strength(new_password):
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters with uppercase, lowercase, and a number"
            )
        
        email = verify_password_reset_token(token)
        client = get_supabase_admin_client()

        users_response = client.auth.admin.list_users()
        user = None
        for u in users_response:
            if u.email == email:
                user = u
                break

        if not user:
            raise HTTPException(
                status_code = status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        client.auth.admin.update_user_by_id(str(user.id), {"password": new_password})

        logger.info(f"Password reset successful for: {email}")

        return AuthResponse(
            success = True,
            message = "Password has been reset succesfully",
            data = None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail="Could not reset password"
            )
    



@router.put("/update-password", response_model = AuthResponse)
async def update_password(
    password_data: PasswordUpdate,
    current_user: TokenData = Depends(require_auth)
):
    # Update the password if user is authenticated

    try:
        if not validate_password_strength(password_data.new_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters with uppercase, lowercase, and a number"
            )
        
        client = get_supabase_client()
        # Verify the curret password by trying to sign in
        try:
            auth_response = client.auth.sign_in_with_password({
                "email": current_user.email,
                "password": password_data.current_password
            })
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
        
        admin_client = get_supabase_admin_client()
        admin_client.auth.admin.update_user_by_id(
            str(current_user.user_id),
            {"password": password_data.new_password}
        )

        logger.info(f"Password updated for user: {current_user.email}")

        return AuthResponse(
            success = True,
            message = "Password updated succesfully",
            data = None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password update error: {e}")
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )
        

@router.get("/me", response_model = AuthResponse)
async def get_current_user(current_user: TokenData = Depends(require_auth)):
    # Get the authenticated user's profile info

    try:
        client = get_supabase_client()
        profile_response = client.table("profiles").select("*").eq("id", str(current_user.user_id)).single().execute()

        if not profile_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        profile = profile_response.data

        return AuthResponse(
            success = True,
            message = "User retrieved succesfully",
            data = {
                "user": {
                    "id": str(current_user.user_id),
                    "email": current_user.email,
                    "username": profile.get("username"),
                    "full_name": profile.get("full_name"),
                    "avatar_url": profile.get("avatar_url"),
                    "bio": profile.get("bio"),
                    "total_study_time": profile.get("total_study_time", 0),
                    "created_at": profile.get("created_at"),
                    "updated_at": profile.get("updated_at")
                }
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get current user error: {e}")
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve user information"
            )
    


@router.get("/verify", response_model = AuthResponse)
async def verify_token(current_user: TokenData = Depends(require_auth)):
    # Check if the current access token is valid
    return AuthResponse(
        success = True,
        message = "Token is valid",
        data = {
            "user_id": str(current_user.user_id),
            "email": current_user.email,
            "expires_at": current_user.exp.isoformat() if current_user.exp else None
        }
    )

@router.delete("/account", response_model = AuthResponse)
async def delete_account(current_user: TokenData = Depends(require_auth)):
    # Delete the user's account
    try:
        admin_client = get_supabase_admin_client()
        admin_client.auth.admin.delete_user(str(current_user.user_id))

        logger.info(f"Account deleted for user: {current_user.email}")

        return AuthResponse(
            success = True,
            message="Account deleted successfully",
            data = None
        )
    
    except Exception as e:
        logger.error(f"Account deletion error: {e}")
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete account"
            )
            
