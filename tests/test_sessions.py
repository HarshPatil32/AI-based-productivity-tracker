"""Tests for /api/v1/sessions/* endpoints."""

import pytest


BASE = "/api/v1/sessions"


@pytest.fixture(scope="module")
def created_session(client, user1_headers, sample_session_payload):
    """Create one session for user1 and return the response body."""
    resp = client.post(f"{BASE}/", json=sample_session_payload, headers=user1_headers)
    assert resp.status_code == 201, f"Session creation failed: {resp.text}"
    return resp.json()


class TestCreateSession:
    def test_create_session_success(self, created_session):
        assert "id" in created_session
        assert "attention_score" in created_session
        assert created_session["duration_seconds"] == 3600

    def test_create_session_no_auth(self, client, sample_session_payload):
        resp = client.post(f"{BASE}/", json=sample_session_payload)
        assert resp.status_code == 401

    def test_create_session_missing_fields(self, client, user1_headers):
        resp = client.post(
            f"{BASE}/", json={"duration_seconds": 3600}, headers=user1_headers
        )
        assert resp.status_code == 422

    def test_attention_score_computed(self, created_session, sample_session_payload):
        """Attention score should equal 1 - (attention_lost / duration) * 100."""
        duration = sample_session_payload["duration_seconds"]
        lost = sample_session_payload["total_attention_lost"]
        expected = round((1 - lost / duration) * 100, 2)
        assert abs(created_session["attention_score"] - expected) < 0.1


class TestGetMySessions:
    def test_get_my_sessions_success(self, client, user1_headers, created_session):
        resp = client.get(f"{BASE}/me", headers=user1_headers)
        assert resp.status_code == 200
        sessions = resp.json()
        assert isinstance(sessions, list)
        ids = [s["id"] for s in sessions]
        assert created_session["id"] in ids

    def test_get_my_sessions_no_auth(self, client):
        resp = client.get(f"{BASE}/me")
        assert resp.status_code == 401

    def test_pagination_limit(self, client, user1_headers):
        resp = client.get(f"{BASE}/me?limit=1", headers=user1_headers)
        assert resp.status_code == 200
        assert len(resp.json()) <= 1

    def test_pagination_invalid_limit(self, client, user1_headers):
        resp = client.get(f"{BASE}/me?limit=999", headers=user1_headers)
        assert resp.status_code == 422


class TestGetSessionSummary:
    def test_summary_success(self, client, user1_headers, created_session):
        resp = client.get(f"{BASE}/me/summary", headers=user1_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_sessions"] >= 1
        assert body["total_study_seconds"] >= created_session["duration_seconds"]
        assert "avg_attention_score" in body

    def test_summary_no_auth(self, client):
        resp = client.get(f"{BASE}/me/summary")
        assert resp.status_code == 401


class TestGetSessionById:
    def test_get_by_id_success(self, client, user1_headers, created_session):
        sid = created_session["id"]
        resp = client.get(f"{BASE}/{sid}", headers=user1_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == sid

    def test_get_by_id_not_found(self, client, user1_headers):
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = client.get(f"{BASE}/{fake_id}", headers=user1_headers)
        assert resp.status_code == 404

    def test_get_by_id_no_auth(self, client, created_session):
        sid = created_session["id"]
        resp = client.get(f"{BASE}/{sid}")
        assert resp.status_code == 401


class TestGetUserSessions:
    def test_get_public_sessions_for_user(self, client, user1_headers, user1_id, created_session):
        resp = client.get(f"{BASE}/user/{user1_id}", headers=user1_headers)
        assert resp.status_code == 200
        ids = [s["id"] for s in resp.json()]
        assert created_session["id"] in ids

    def test_get_sessions_unknown_user(self, client, user1_headers):
        fake_id = "00000000-0000-0000-0000-000000000001"
        resp = client.get(f"{BASE}/user/{fake_id}", headers=user1_headers)
        assert resp.status_code == 200  # Returns empty list, not 404
        assert resp.json() == []


class TestDeleteSession:
    def test_delete_other_users_session_forbidden(
        self, client, user2_headers, created_session
    ):
        sid = created_session["id"]
        resp = client.delete(f"{BASE}/{sid}", headers=user2_headers)
        assert resp.status_code == 403

    def test_delete_own_session(self, client, user1_headers, sample_session_payload):
        # Create a throwaway session and delete it
        create_resp = client.post(
            f"{BASE}/", json=sample_session_payload, headers=user1_headers
        )
        assert create_resp.status_code == 201
        sid = create_resp.json()["id"]

        del_resp = client.delete(f"{BASE}/{sid}", headers=user1_headers)
        assert del_resp.status_code == 204

        # Confirm gone
        get_resp = client.get(f"{BASE}/{sid}", headers=user1_headers)
        assert get_resp.status_code == 404

    def test_delete_nonexistent_session(self, client, user1_headers):
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = client.delete(f"{BASE}/{fake_id}", headers=user1_headers)
        assert resp.status_code == 404



# --------------- Privacy helpers ---------------

def _follow_user(client, follower_headers, followee_id):
    resp = client.post(f"/api/v1/users/{followee_id}/follow", headers=follower_headers)
    assert resp.status_code in (200, 201)

def _unfollow_user(client, follower_headers, followee_id):
    resp = client.delete(f"/api/v1/users/{followee_id}/follow", headers=follower_headers)
    # 204 = success, 404 = not following
    assert resp.status_code in (204, 404)

@pytest.fixture()
def user2_privacy_session(client, user2_headers, sample_session_payload):
    """A session owned by user2, used to verify visibility enforcement."""
    resp = client.post(f"{BASE}/", json=sample_session_payload, headers=user2_headers)
    assert resp.status_code == 201, f"user2 session creation failed: {resp.text}"
    session = resp.json()
    yield session
    client.delete(f"{BASE}/{session['id']}", headers=user2_headers)


# --------------- Privacy tests ---------------

SETTINGS_BASE = "/api/v1/users"


class TestSessionPrivacy:
    """
    Verify that session_visibility in user_settings gates what other users
    and the feed can see. Three values are tested: public (baseline), private,
    and friends (follow-only).

    Expected behaviour when NOT yet enforced by the backend: these tests will
    fail, surfacing the gap in privacy implementation.
    """

    def _set_session_visibility(self, client, headers, value):
        resp = client.patch(
            f"{SETTINGS_BASE}/me/settings",
            json={"session_visibility": value},
            headers=headers,
        )
        assert resp.status_code == 200, (
            f"Failed to set session_visibility={value}: {resp.text}"
        )

    def test_private_sessions_hidden_from_other_user(
        self, client, user2_headers, user2_id, user1_headers, user2_privacy_session
    ):
        """user1 must not see user2's sessions when session_visibility=private."""
        self._set_session_visibility(client, user2_headers, "private")
        try:
            resp = client.get(f"{BASE}/user/{user2_id}", headers=user1_headers)
            assert resp.status_code == 200
            sessions = resp.json()
            assert all(s["user_id"] != user2_id for s in sessions), "Private sessions must not be visible to other users"
        finally:
            self._set_session_visibility(client, user2_headers, "public")

    def test_private_sessions_visible_to_owner(
        self, client, user2_headers, user2_privacy_session
    ):
        """Owner must always access their own sessions even when visibility=private."""
        self._set_session_visibility(client, user2_headers, "private")
        try:
            resp = client.get(f"{BASE}/me", headers=user2_headers)
            assert resp.status_code == 200
            ids = [s["id"] for s in resp.json()]
            assert user2_privacy_session["id"] in ids
        finally:
            self._set_session_visibility(client, user2_headers, "public")

    def test_private_sessions_not_in_feed(
        self, client, user2_headers, user2_id, user1_headers, user2_privacy_session
    ):
        """user2's sessions must not appear in user1's feed when visibility=private."""
        _follow_user(client, user1_headers, user2_id)
        self._set_session_visibility(client, user2_headers, "private")
        try:
            resp = client.get("/api/v1/feed/", headers=user1_headers)
            assert resp.status_code == 200
            user2_session_ids = {
                s["id"] for s in resp.json() if s["user_id"] == user2_id
            }
            assert user2_session_ids == set(), (
                "Private sessions must not appear in another user's feed"
            )
        finally:
            self._set_session_visibility(client, user2_headers, "public")
            _unfollow_user(client, user1_headers, user2_id)

    def test_friends_sessions_hidden_from_stranger(
        self, client, user2_headers, user2_id, user1_headers, user2_privacy_session
    ):
        """Non-follower must not see user2's sessions when session_visibility=friends."""
        _unfollow_user(client, user1_headers, user2_id)
        self._set_session_visibility(client, user2_headers, "friends")
        try:
            resp = client.get(f"{BASE}/user/{user2_id}", headers=user1_headers)
            assert resp.status_code == 200
            sessions = resp.json()
            assert all(s["user_id"] != user2_id for s in sessions), (
                "Strangers must not see sessions when visibility=friends"
            )
        finally:
            self._set_session_visibility(client, user2_headers, "public")

    def test_friends_sessions_visible_to_follower(
        self, client, user2_headers, user2_id, user1_headers, user2_privacy_session
    ):
        """Follower must see user2's sessions when session_visibility=friends."""
        _follow_user(client, user1_headers, user2_id)
        self._set_session_visibility(client, user2_headers, "friends")
        try:
            resp = client.get(f"{BASE}/user/{user2_id}", headers=user1_headers)
            assert resp.status_code == 200
            ids = [s["id"] for s in resp.json()]
            assert user2_privacy_session["id"] in ids, (
                "Followers must see sessions when session_visibility=friends"
            )
        finally:
            self._set_session_visibility(client, user2_headers, "public")
            _unfollow_user(client, user1_headers, user2_id)

    def test_private_sessions_hidden_from_unauthenticated(self, client, user2_headers, user2_id, user2_privacy_session):
        """Unauthenticated user must not see user2's sessions when session_visibility=private."""
        self._set_session_visibility(client, user2_headers, "private")
        try:
            resp = client.get(f"{BASE}/user/{user2_id}")
            # Should be 401 (unauthenticated)
            assert resp.status_code == 401
        finally:
            self._set_session_visibility(client, user2_headers, "public")
