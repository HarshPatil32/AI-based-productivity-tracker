"""
Shared fixtures for the entire test suite.

NOTE: These are integration tests against a real Supabase instance.
Ensure your .env is populated and the schema (schema.sql) has been applied
before running. Run with:

    pytest tests/ -v
"""

import uuid
import pytest
from fastapi.testclient import TestClient
import sys
from unittest.mock import MagicMock

from backend.main import app

# --------------- Unique credentials ---------------
# A random suffix keeps tests isolated across multiple runs.
_RUN_ID = uuid.uuid4().hex[:8]


@pytest.fixture(scope="session")
def client():
    """Synchronous TestClient wrapping the FastAPI app (one per session)."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# --------------- User 1 (primary test user) ---------------

@pytest.fixture(scope="session")
def user1_creds():
    return {
        "email": f"testuser1_{_RUN_ID}@example.com",
        "password": "TestPass1!",
        "username": f"testuser1_{_RUN_ID}",
        "full_name": "Test User One",
    }


@pytest.fixture(scope="session")
def user1_tokens(client, user1_creds):
    """Register user1 and return the token payload. Deleted at session end."""
    resp = client.post("/api/v1/auth/register", json=user1_creds)
    assert resp.status_code == 201, f"user1 register failed: {resp.text}"
    data = resp.json()["data"]
    yield data
    # Cleanup — delete the account so Supabase stays clean.
    headers = {"Authorization": f"Bearer {data['access_token']}"}
    client.delete("/api/v1/auth/account", headers=headers)


@pytest.fixture(scope="session")
def user1_headers(user1_tokens):
    return {"Authorization": f"Bearer {user1_tokens['access_token']}"}


@pytest.fixture(scope="session")
def user1_id(user1_tokens):
    return user1_tokens["user"]["id"]


# --------------- User 2 (second actor for follows/feed tests) ---------------

@pytest.fixture(scope="session")
def user2_creds():
    return {
        "email": f"testuser2_{_RUN_ID}@example.com",
        "password": "TestPass2!",
        "username": f"testuser2_{_RUN_ID}",
        "full_name": "Test User Two",
    }


@pytest.fixture(scope="session")
def user2_tokens(client, user2_creds):
    resp = client.post("/api/v1/auth/register", json=user2_creds)
    assert resp.status_code == 201, f"user2 register failed: {resp.text}"
    data = resp.json()["data"]
    yield data
    headers = {"Authorization": f"Bearer {data['access_token']}"}
    client.delete("/api/v1/auth/account", headers=headers)


@pytest.fixture(scope="session")
def user2_headers(user2_tokens):
    return {"Authorization": f"Bearer {user2_tokens['access_token']}"}


@pytest.fixture(scope="session")
def user2_id(user2_tokens):
    return user2_tokens["user"]["id"]


# --------------- Reusable session payload ---------------

@pytest.fixture(scope="session")
def sample_session_payload():
    from datetime import datetime, timezone, timedelta
    start = datetime(2026, 2, 26, 9, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    return {
        "started_at": start.isoformat(),
        "ended_at": end.isoformat(),
        "duration_seconds": 3600,
        "eyes_closed_time": 120.0,
        "face_missing_time": 60.0,
        "head_pose_off_time": 90.0,
        "total_attention_lost": 270.0,
        "notes": "Pytest integration test session",
    }


# Ensure dlib is always mocked for all tests that import face_utils
@pytest.fixture(autouse=True, scope="session")
def _mock_dlib():
    sys.modules["dlib"] = MagicMock()
    yield
    sys.modules.pop("dlib", None)
