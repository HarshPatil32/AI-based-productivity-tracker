"""Tests for /api/v1/feed/* endpoints."""

import pytest


FEED_BASE = "/api/v1/feed"
SESSIONS_BASE = "/api/v1/sessions"
USERS_BASE = "/api/v1/users"


@pytest.fixture(scope="module")
def user2_session(client, user2_headers, sample_session_payload):
    """Create a public session for user2 for feed visibility tests."""
    resp = client.post(
        f"{SESSIONS_BASE}/", json=sample_session_payload, headers=user2_headers
    )
    assert resp.status_code == 201, f"user2 session creation failed: {resp.text}"
    return resp.json()


class TestPersonalizedFeed:
    def test_feed_returns_list(self, client, user1_headers):
        resp = client.get(f"{FEED_BASE}/", headers=user1_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_feed_no_auth(self, client):
        resp = client.get(f"{FEED_BASE}/")
        assert resp.status_code == 401

    def test_feed_shows_followed_user_sessions(
        self, client, user1_headers, user2_headers, user2_id, user2_session
    ):
        """After user1 follows user2, user2's session should appear in user1's feed."""
        # Follow user2
        client.post(f"{USERS_BASE}/{user2_id}/follow", headers=user1_headers)

        resp = client.get(f"{FEED_BASE}/", headers=user1_headers)
        assert resp.status_code == 200
        ids = [s["id"] for s in resp.json()]
        assert user2_session["id"] in ids

    def test_feed_session_has_author_fields(
        self, client, user1_headers, user2_id
    ):
        """Feed items must include username/full_name/avatar_url from the join."""
        client.post(f"{USERS_BASE}/{user2_id}/follow", headers=user1_headers)
        resp = client.get(f"{FEED_BASE}/", headers=user1_headers)
        assert resp.status_code == 200
        sessions = resp.json()
        assert len(sessions) > 0
        first = sessions[0]
        assert "username" in first
        assert "focus_score" in first
        assert "quality" in first
        assert "likes_count" in first
        assert "comments_count" in first

    def test_feed_pagination(self, client, user1_headers):
        resp = client.get(f"{FEED_BASE}/?limit=1", headers=user1_headers)
        assert resp.status_code == 200
        assert len(resp.json()) <= 1

    def test_feed_limit_max_enforced(self, client, user1_headers):
        # limit > 50 should be rejected (Query max is 50)
        resp = client.get(f"{FEED_BASE}/?limit=100", headers=user1_headers)
        assert resp.status_code == 422

    def test_feed_falls_back_to_global_when_no_follows(
        self, client, user2_headers
    ):
        """user2 follows nobody — should still get a non-empty global fallback feed."""
        resp = client.get(f"{FEED_BASE}/", headers=user2_headers)
        assert resp.status_code == 200
        # At minimum user2's own session should appear
        assert len(resp.json()) >= 1


class TestGlobalFeed:
    def test_global_feed_returns_list(self, client, user1_headers):
        resp = client.get(f"{FEED_BASE}/global", headers=user1_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_global_feed_no_auth(self, client):
        resp = client.get(f"{FEED_BASE}/global")
        assert resp.status_code == 401

    def test_global_feed_contains_public_session(
        self, client, user1_headers, user2_session
    ):
        resp = client.get(f"{FEED_BASE}/global", headers=user1_headers)
        ids = [s["id"] for s in resp.json()]
        assert user2_session["id"] in ids

    def test_global_feed_pagination(self, client, user1_headers):
        resp = client.get(f"{FEED_BASE}/global?limit=2&offset=0", headers=user1_headers)
        assert resp.status_code == 200
        assert len(resp.json()) <= 2

    def test_global_feed_offset(self, client, user1_headers):
        page1 = client.get(f"{FEED_BASE}/global?limit=1&offset=0", headers=user1_headers).json()
        page2 = client.get(f"{FEED_BASE}/global?limit=1&offset=1", headers=user1_headers).json()
        if len(page1) > 0 and len(page2) > 0:
            assert page1[0]["id"] != page2[0]["id"]
