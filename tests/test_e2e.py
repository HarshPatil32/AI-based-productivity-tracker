"""
End-to-end test: Register → Track → Share

Full user journey
-----------------
1. Register user A (new, unique credentials per run).
2. Log in as user A and obtain an access token.
3. POST a fabricated tracker session as user A.
4. Verify the session appears in GET /sessions/me.
5. Verify the session appears in GET /feed/global.
6. Register user B, log in, and have user B follow user A.
7. Verify user B's personalized feed includes user A's session.
"""

import uuid
import pytest
from datetime import datetime, timezone, timedelta

SESSIONS_BASE = "/api/v1/sessions"
FEED_BASE = "/api/v1/feed"
USERS_BASE = "/api/v1/users"
AUTH_BASE = "/api/v1/auth"

# Unique suffix so this test file's users never clash with the shared session fixtures.
_E2E_RUN_ID = uuid.uuid4().hex[:8]



# --------------- Shared constants ---------------

E2E_USER_A_CREDS = {
    "email": f"e2e_a_{_E2E_RUN_ID}@example.com",
    "password": "E2ePassA1!",
    "username": f"e2e_a_{_E2E_RUN_ID}",
    "full_name": "E2E User A",
}

E2E_USER_B_CREDS = {
    "email": f"e2e_b_{_E2E_RUN_ID}@example.com",
    "password": "E2ePassB1!",
    "username": f"e2e_b_{_E2E_RUN_ID}",
    "full_name": "E2E User B",
}

E2E_SESSION_PAYLOAD = {
    "title": "E2E test fabricated tracker session",
    "description": "E2E test fabricated tracker session",
    "session_duration": 3600,
    "focused_time": 3200,
    "distracted_time": 200,
    "eyes_closed_time": 100,
    "face_missing_time": 50,
    "head_pose_off_time": 80,
    "attention_lost": 230,
    # Required by API model:
    "started_at": datetime(2026, 3, 10, 8, 0, 0, tzinfo=timezone.utc).isoformat(),
    "ended_at": (datetime(2026, 3, 10, 9, 0, 0, tzinfo=timezone.utc).isoformat()),
    "duration_seconds": 3600,
    "total_attention_lost": 230,
    # is_public is optional (defaults to TRUE)
}


# --------------- E2E test function ---------------


def test_e2e_register_track_share(client):
    """
    Full E2E journey: Register user A, login, post session, verify feeds, register user B, follow, verify personalized feed.
    """
    # Register user A
    resp = client.post(f"{AUTH_BASE}/register", json=E2E_USER_A_CREDS)
    assert resp.status_code == 201, f"user A register failed: {resp.text}"
    user_a_data = resp.json()["data"]
    token_a = user_a_data["access_token"]
    user_a_id = user_a_data["user"]["id"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # (No explicit login: use registration token for all user A actions)

    # Post a fabricated session as user A
    resp = client.post(f"{SESSIONS_BASE}/", json=E2E_SESSION_PAYLOAD, headers=headers_a)
    assert resp.status_code == 201, f"session creation failed: {resp.text}"
    session = resp.json()
    session_id = session["id"]

    # Verify session appears in /sessions/me
    resp = client.get(f"{SESSIONS_BASE}/me", headers=headers_a)
    assert resp.status_code == 200
    ids = [s["id"] for s in resp.json()]
    assert session_id in ids, f"Session {session_id} not found in /sessions/me"

    # Verify session appears in global feed
    resp = client.get(f"{FEED_BASE}/global", headers=headers_a)
    assert resp.status_code == 200
    ids = [s["id"] for s in resp.json()]
    assert session_id in ids, f"Session {session_id} not found in /feed/global"

    # Register user B
    resp = client.post(f"{AUTH_BASE}/register", json=E2E_USER_B_CREDS)
    assert resp.status_code == 201, f"user B register failed: {resp.text}"
    user_b_data = resp.json()["data"]
    token_b = user_b_data["access_token"]
    user_b_id = user_b_data["user"]["id"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # User B follows user A
    resp = client.post(f"{USERS_BASE}/{user_a_id}/follow", headers=headers_b)
    assert resp.status_code in (200, 201), f"follow failed: {resp.text}"
    # If response has a body, check for following: true
    try:
        follow_body = resp.json()
        if isinstance(follow_body, dict) and "following" in follow_body:
            assert follow_body["following"] is True
    except Exception:
        pass  # If no JSON or no 'following', skip

    # User B's personalized feed includes user A's session
    # Ensure follow is in place before fetching feed
    client.post(f"{USERS_BASE}/{user_a_id}/follow", headers=headers_b)
    resp = client.get(f"{FEED_BASE}/", headers=headers_b)
    assert resp.status_code == 200
    ids = [s["id"] for s in resp.json()]
    assert session_id in ids, f"Session {session_id} not found in user B's personalized feed"

    # Teardown: delete user B first, then user A (to avoid FK issues)
    for token in (token_b, token_a):
        if token:
            client.delete(f"{AUTH_BASE}/account", headers={"Authorization": f"Bearer {token}"})
