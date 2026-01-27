from typing import Optional, List
from uuid import UUID

from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from backend.config.settings import get_settings
from backend.utils.auth import decode_token, TokenData
from backend.services.database import SupabaseClient

settings = get_settings()
security = HTTPBearer(auto_error = False)

class AuthMiddleware (BaseHTTPMiddleware):
    PUBLIC_PATHS = [
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        f"/api/{settings.API_VERSION}/auth/login",
        f"/api/{settings.API_VERSION}/auth/register",
        f"/api/{settings.API_VERSION}/auth/refresh",
        f"/api/{settings.API_VERSION}/auth/forgot-password",
        f"/api/{settings.API_VERSION}/auth/reset-password",
        f"/api/{settings.API_VERSION}/leaderboard"
    ]

    async def dispatch (self, request, call_next):
        request.state.user = None
        request.state.user_id = None
        request.state.is_authenticated = False

        # Skip auth for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                token_data = decode_token(token)
                request.state.user = token_data
                request.state.user_id = token_data.user_id
                request.state.is_authenticated = True
            except HTTPException: # Route handlers will enforce auth if needed
                pass

        return await call_next(request)

    def _is_public_path(self, path):
        if path in self.PUBLIC_PATHS:
            return True
        for public_path in self.PUBLIC_PATHS:
            if public_path.endswith("/") and path.startswith(public_path):
                return True
        return False

class AuthenticationRequired:
    def __init__(self, auto_error: bool = True):
        self.auto_error = auto_error

    async def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Optional[TokenData]:
        if hasattr(request.state, "is_authenticated") and request.state.is_authenticated:
            return request.state.user
        
        if credentials is None:
            if self.auto_error:
                raise HTTPException(
                    status_code = status.HTTP_401_UNAUTHORIZED,
                    detail = "Authentication required",
                    headers = {"WWW-Authenticate": "Bearer"}
                )
            return None
        
        try:
            token_data = decode_token(credentials.credentials)
            request.state.user = token_data
            request.state.user_id = token_data.user_id
            request.state.is_authenticated = True
            return token_data
        except HTTPException:
            if self.auto_error:
                raise
            return None
        
require_auth = AuthenticationRequired()
optional_auth = AuthenticationRequired(auto_error = False)

async def get_current_user_from_request(request):
    if hasattr(request.state, "user"):
        return request.state.user
    return None

async def get_current_user_id_from_request(request):
    if hasattr(request.state, "user_id"):
        return request.state.user_id
    
    return None

def require_same_user(user_id_param: str = "user_id"):
    """Ensure the authenticated user matches the path parameter."""
    async def dependency(
        request: Request,
        current_user: TokenData = Depends(require_auth)
    ) -> TokenData:
        target_user_id = request.path_params.get(user_id_param)
        if target_user_id is None:
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail = f"Missing path parameter: {user_id_param}"
            )
        
        if str(current_user.user_id) != str(target_user_id):
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail = "You can only access your own stuff"
            )
        
        return current_user
    
    return dependency

class PrivacyCheck:
    # Checks privacy settings before accessing user content

    def __init__(self, resource_type: str = "profile"):
        self.resource_type = resource_type

    async def __call__(self, request, target_user_id, current_user):
        # Returns True if access allowed, HTTPException if denied
        if current_user and str(current_user.user_id) == str(target_user_id):
            return True
        
        db = SupabaseClient()
        try:
            result = db.client.table("user_settings").select("profile_visibility, session_visibility").eq("user_id", str(target_user_id)).single().execute()
            if not result.data:
                return True # Default to public if no settings
            visibility_field = f"{self.resource_type}_visibility"
            visibility = result.data.get(visibility_field, "public")

            if visibility == "public":
                return True
            
            if visibility == "private":
                raise HTTPException(
                    status_code = status.HTTP_403_FORBIDDEN,
                    detail = f"This {self.resource_type} is private"
                )
            
            if visibility == "friends":
                if not current_user:
                    raise HTTPException(
                        status_code = status.HTTP_403_FORBIDDEN,
                        detail=f"This {self.resource_type} is only visible to followers"
                    )
                
                follow_result = db.client.table("user_relationships").select("id").eq("follower_id", str(current_user.user_id)).eq("following_id", str(target_user_id)).execute()
                if not follow_result.data:
                    raise HTTPException(
                        status_code = status.HTTP_403_FORBIDDEN,
                        detail = f"This {self.resource_type} is only visible to followers"
                    )
                
            return True
        except HTTPException:
            raise
        except Exception as e:
            return True # Just log the error but allow access
        

# Convenience instances for privacy checks
check_profile_privacy = PrivacyCheck(resource_type = "profile")
check_session_privacy = PrivacyCheck(resource_type="session")

async def verify_session_access(session_id, current_user):
    # Verify to acces a specific study session

    db = SupabaseClient()

    try:
        result = db.client.table("study_sessions").select("*, profiles(id, username)").eq("id", str(session_id)).single().execute()

        if not result.data:
            raise HTTPException(
                status_code = status.HTTP_404_NOT_FOUND,
                detail = "Session not found"
            )
        
        session = result.data
        owner_id = session.get("user_id")

        if current_user and str(current_user.user_id) == str(owner_id):
            return session
        
        if not session.get("is_public", True):
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail = "This session is private"
            )
        
        settings_result = db.client.table("user_settings").select("session_visibility").eq("user_id", str(owner_id)).single().execute()

        if settings_result.data:
            visibility = settings_result.data.get("session_visibility", "public")

            if visibility == "private":
                raise HTTPException(
                    status_code = status.HTTP_403_FORBIDDEN,
                    detail = "This users sessions are private"
                )
            
            if visibility == "friends":
                if not current_user:
                    raise HTTPException(
                        status_code = status.HTTP_403_FORBIDDEN,
                        detail = "This session can only be seen by followers"
                    )
                
                follow_check = db.client.table("user_relationships").select("id").eq("follower_id", str(current_user.user_id)).eq("following_id", str(owner_id)).execute()

                if not follow_check.data:
                    raise HTTPException(
                        status_code = status.HTTP_403_FORBIDDEN,
                        detail = "This session is only visible to followers"
                    )
        return session
    
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = "Error verifying session access"
        )
    

async def verify_resource_owner(
    resource_table: str,
    resource_id: UUID,
    current_user: TokenData = Depends(require_auth)
) -> bool:
    """Verify the current user owns a resource (for edit/delete operations)."""
    db = SupabaseClient()
    result = db.client.table(resource_table).select("user_id").eq("id", str(resource_id)).single().execute()

    if not result.data:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "Resource not found"
        )
    
    if str(result.data.get("user_id")) != str(current_user.user_id):
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "You don't have permission to modify this resource"
        )
    
    return True

class RateLimitMiddleWare(BaseHTTPMiddleware):
    # Rate limiting per user/IP
    # Redis based rate limiting?

    def __init__(self, app, requests_per_minute: int = None):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute or settings.RATE_LIMIT_PER_MINUTE
        self._request_counts: dict = {}

    async def dispatch (self, request, call_next):
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        identifier = self._get_identifier(request)

        if not self._check_rate_limit(identifier):
            return JSONResponse(
                status_code = status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded, try again later"}
            )
        
        return await call_next(request)
    
    def _get_identifier(self, request):
        # Get the unique identifier for rate limiting
        if hasattr(request.state, "user_id") and request.state.user_id:
            return f"user:{request.state.user_id}"
        
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host if request.client else 'unknown'}"
    
    def _check_rate_limit(self, identifier):
        # Check if the identifier is within the rate limit
        import time
        current_minute = int(time.time() / 60)
        key = f"{identifier}:{current_minute}"

        count = self._request_counts.get(key, 0)

        if count >= self.requests_per_minute:
            return False
        
        self._request_counts[key] = count + 1

        old_keys = [k for k in self._request_counts if not k.endswith(f":{current_minute}")]
        for k in old_keys:
            del self._request_counts[k]

        return True

