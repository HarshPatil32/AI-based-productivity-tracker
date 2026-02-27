from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from backend.config.settings import get_settings
from backend.middleware.auth import AuthMiddleware, RateLimitMiddleWare
from backend.api.routes.auth import router as auth_router
from backend.api.routes.session import router as session_router
from backend.api.routes.users import router as users_router
from backend.api.routes.follows import router as follows_router
from backend.api.routes.feed import router as feed_router

settings = get_settings()
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=settings.DEBUG,
    )

    # --------------- Middleware (order matters: outermost first) ---------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleWare)
    app.add_middleware(AuthMiddleware)

    # --------------- Routers ---------------
    prefix = f"/api/{settings.API_VERSION}"
    app.include_router(auth_router,    prefix=prefix)
    app.include_router(session_router, prefix=prefix)
    app.include_router(users_router,   prefix=prefix)
    app.include_router(follows_router, prefix=prefix)
    app.include_router(feed_router,    prefix=prefix)

    # --------------- Health check ---------------
    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "version": settings.API_VERSION}

    @app.get("/", tags=["Health"])
    async def root():
        return {"message": f"Welcome to {settings.APP_NAME} API"}

    logger.info(f"{settings.APP_NAME} started in {settings.ENVIRONMENT} mode")
    return app


app = create_app()
