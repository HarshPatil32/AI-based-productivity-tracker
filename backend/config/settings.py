from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    App settings, using my env vars. Pydantic settings loads them from .env
    """

    APP_NAME: str = "Productivity Tracker"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_VERSION: str = "v1"

    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    MAX_SESSION_DURATION_HOURS: int = 12
    MIN_ATTENTION_SCORE: float = 0.0
    MAX_ATTENTION_SCORE: float = 100.0

    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_PER_MINUTE: int = 60

    LOG_LEVEL: str = "INFO"
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @property
    def is_production(self):
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self):
        return self.ENVIRONMENT.lower() == "development"
    
    def get_cors_origins(self):
        # Get CORS origins in production to only allow specific domains
        if self.is_production:
            return [
                "my frontend domain when deployed",
                "my backend domain when deployed"
            ]
        return self.CORS_ORIGINS
    
settings = Settings()

def validate_settings():
    errors = []

    if not settings.SUPABASE_URL:
        errors.append("SUPABASE_URL is required")
    if not settings.SUPABASE_ANON_KEY:
        errors.append("SUPABASE_ANON_KEY is required")
    if not settings.SUPABASE_SERVICE_ROLE_KEY:
        errors.append("SUPABASE_SERVICE_ROLE_KEY is required")

    if len(settings.JWT_SECRET_KEY) < 32:
        errors.append("JWT_SECRET_KEY must be at least 32 char long")
    if errors:
        raise ValueError(f"Config errors:\n" + "\n".join(f" - {err}" for err in errors))
    return True
