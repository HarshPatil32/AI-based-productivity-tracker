"""Tests for /api/v1/users/* (profile + settings) endpoints."""

# --------------- Privacy helpers ---------------


    resp = client.post(f"/api/v1/users/{followee_id}/follow", headers=follower_headers)
    assert resp.status_code in (200, 201)

class TestGetMySettings:
    resp = client.delete(f"/api/v1/users/{followee_id}/follow", headers=follower_headers)
    assert resp.status_code in (204, 404)


class TestProfilePrivacy:
    """
    Verify that profile_visibility in user_settings gates who can view a
    user's profile via GET /api/v1/users/{username}. Three values are tested:
    public (baseline), private, and friends (follow-only).

    Expected behaviour when NOT yet enforced by the backend: these tests will
    fail, surfacing the gap in privacy implementation.
    """

    def _set_profile_visibility(self, client, headers, value):
        resp = client.patch(
            f"{BASE}/me/settings",
            json={"profile_visibility": value},
            headers=headers,
        )
        assert resp.status_code == 200, (
            f"Failed to set profile_visibility={value}: {resp.text}"
        )

    def test_private_profile_hidden_from_stranger(
        self, client, user2_headers, user2_creds, user1_headers, user2_id
    ):
        """Non-follower must get 403 or 404 when viewing a private profile."""
        _unfollow_user(client, user1_headers, user2_id)
        self._set_profile_visibility(client, user2_headers, "private")
        try:
            resp = client.get(f"{BASE}/{user2_creds['username']}", headers=user1_headers)
            assert resp.status_code in (403, 404), (
                f"Expected 403 or 404 for private profile accessed by a stranger, "
                f"got {resp.status_code}"
            )
            if resp.status_code in (403, 404):
                # Optionally check error message
                body = resp.json()
                assert "detail" in body
        finally:
            self._set_profile_visibility(client, user2_headers, "public")

    def test_private_profile_visible_to_self(
        self, client, user2_headers
    ):
        """Owner must always be able to view their own profile regardless of setting."""
        self._set_profile_visibility(client, user2_headers, "private")
        try:
            resp = client.get(f"{BASE}/me", headers=user2_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "username" in data
        finally:
            self._set_profile_visibility(client, user2_headers, "public")

    def test_friends_profile_hidden_from_stranger(
        self, client, user2_headers, user2_creds, user1_headers, user2_id
    ):
        """Non-follower must get 403 or 404 for a friends-only profile."""
        _unfollow_user(client, user1_headers, user2_id)
        self._set_profile_visibility(client, user2_headers, "friends")
        try:
            resp = client.get(f"{BASE}/{user2_creds['username']}", headers=user1_headers)
            assert resp.status_code in (403, 404), (
                f"Expected 403 or 404 for friends-only profile accessed by a stranger, "
                f"got {resp.status_code}"
            )
            if resp.status_code in (403, 404):
                body = resp.json()
                assert "detail" in body
        finally:
            self._set_profile_visibility(client, user2_headers, "public")

    def test_friends_profile_visible_to_follower(
        self, client, user2_headers, user2_creds, user1_headers, user2_id
    ):
        """Follower must be able to view a profile with visibility=friends."""
        _follow_user(client, user1_headers, user2_id)
        self._set_profile_visibility(client, user2_headers, "friends")
        try:
            resp = client.get(f"{BASE}/{user2_creds['username']}", headers=user1_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["username"] == user2_creds["username"]
        finally:
            self._set_profile_visibility(client, user2_headers, "public")
            _unfollow_user(client, user1_headers, user2_id)

    def test_private_profile_hidden_from_unauthenticated(self, client, user2_headers, user2_creds):
        """Unauthenticated user must not see user2's profile when profile_visibility=private."""
        self._set_profile_visibility(client, user2_headers, "private")
        try:
            resp = client.get(f"{BASE}/{user2_creds['username']}")
            assert resp.status_code == 401
        finally:
            self._set_profile_visibility(client, user2_headers, "public")
    def test_get_settings_success(self, client, user1_headers):
        resp = client.get(f"{BASE}/me/settings", headers=user1_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "profile_visibility" in data
        assert "session_visibility" in data
        assert "email_notifications" in data
        assert "theme" in data

    def test_get_settings_no_auth(self, client):
        resp = client.get(f"{BASE}/me/settings")
        assert resp.status_code == 401


class TestUpdateMySettings:
    def test_patch_theme(self, client, user1_headers):
        resp = client.patch(
            f"{BASE}/me/settings", json={"theme": "dark"}, headers=user1_headers
        )
        assert resp.status_code == 200
        assert resp.json()["theme"] == "dark"

    def test_patch_visibility(self, client, user1_headers):
        resp = client.patch(
            f"{BASE}/me/settings",
            json={"profile_visibility": "friends"},
            headers=user1_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["profile_visibility"] == "friends"
        # Reset to public
        client.patch(
            f"{BASE}/me/settings",
            json={"profile_visibility": "public"},
            headers=user1_headers,
        )

    def test_patch_invalid_visibility_value(self, client, user1_headers):
        resp = client.patch(
            f"{BASE}/me/settings",
            json={"profile_visibility": "everyone"},
            headers=user1_headers,
        )
        assert resp.status_code == 422

    def test_patch_notification_toggle(self, client, user1_headers):
        resp = client.patch(
            f"{BASE}/me/settings",
            json={"email_on_like": False, "email_on_comment": False},
            headers=user1_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["email_on_like"] is False
        assert data["email_on_comment"] is False

    def test_patch_settings_empty_body_rejected(self, client, user1_headers):
        resp = client.patch(f"{BASE}/me/settings", json={}, headers=user1_headers)
        assert resp.status_code == 400

    def test_patch_settings_no_auth(self, client):
        resp = client.patch(f"{BASE}/me/settings", json={"theme": "dark"})
        assert resp.status_code == 401


class TestGetSuggestedUsers:
    def test_get_suggested_users_success(self, client, user1_headers, user1_creds, user2_creds):
        resp = client.get(f"{BASE}/suggested", headers=user1_headers)
        assert resp.status_code == 200
        data = resp.json()
        # Should not include self
        assert all(u["username"] != user1_creds["username"] for u in data)
        # Should not include already-followed users
        followed_usernames = [user2_creds["username"]]
        assert all(u["username"] not in followed_usernames for u in data)

    def test_get_suggested_users_no_auth(self, client):
        resp = client.get(f"{BASE}/suggested")
        assert resp.status_code == 401

    def test_get_suggested_users_limit(self, client, user1_headers):
        resp = client.get(f"{BASE}/suggested?limit=2", headers=user1_headers)
        assert resp.status_code == 200
        assert len(resp.json()) <= 2


# --------------- Privacy tests ---------------


class TestProfilePrivacy:
    """
    Verify that profile_visibility in user_settings gates who can view a
    user's profile via GET /api/v1/users/{username}. Three values are tested:
    public (baseline), private, and friends (follow-only).

    Expected behaviour when NOT yet enforced by the backend: these tests will
    fail, surfacing the gap in privacy implementation.
    """

    def _set_profile_visibility(self, client, headers, value):
        resp = client.patch(
            f"{BASE}/me/settings",
            json={"profile_visibility": value},
            headers=headers,
        )
        assert resp.status_code == 200, (
            f"Failed to set profile_visibility={value}: {resp.text}"
        )

    def test_private_profile_hidden_from_stranger(
        self, client, user2_headers, user2_creds, user1_headers, user2_id
    ):
        """Non-follower must get 403 or 404 when viewing a private profile."""
        client.delete(f"{BASE}/{user2_id}/follow", headers=user1_headers)
        self._set_profile_visibility(client, user2_headers, "private")
        try:
            resp = client.get(f"{BASE}/{user2_creds['username']}", headers=user1_headers)
            assert resp.status_code in (403, 404), (
                f"Expected 403 or 404 for private profile accessed by a stranger, "
                f"got {resp.status_code}"
            )
        finally:
            self._set_profile_visibility(client, user2_headers, "public")

    def test_private_profile_visible_to_self(
        self, client, user2_headers
    ):
        """Owner must always be able to view their own profile regardless of setting."""
        self._set_profile_visibility(client, user2_headers, "private")
        try:
            resp = client.get(f"{BASE}/me", headers=user2_headers)
            assert resp.status_code == 200
        finally:
            self._set_profile_visibility(client, user2_headers, "public")

    def test_friends_profile_hidden_from_stranger(
        self, client, user2_headers, user2_creds, user1_headers, user2_id
    ):
        """Non-follower must get 403 or 404 for a friends-only profile."""
        client.delete(f"{BASE}/{user2_id}/follow", headers=user1_headers)
        self._set_profile_visibility(client, user2_headers, "friends")
        try:
            resp = client.get(f"{BASE}/{user2_creds['username']}", headers=user1_headers)
            assert resp.status_code in (403, 404), (
                f"Expected 403 or 404 for friends-only profile accessed by a stranger, "
                f"got {resp.status_code}"
            )
        finally:
            self._set_profile_visibility(client, user2_headers, "public")

    def test_friends_profile_visible_to_follower(
        self, client, user2_headers, user2_creds, user1_headers, user2_id
    ):
        """Follower must be able to view a profile with visibility=friends."""
        client.post(f"{BASE}/{user2_id}/follow", headers=user1_headers)
        self._set_profile_visibility(client, user2_headers, "friends")
        try:
            resp = client.get(f"{BASE}/{user2_creds['username']}", headers=user1_headers)
            assert resp.status_code == 200
            assert resp.json()["username"] == user2_creds["username"]
        finally:
            self._set_profile_visibility(client, user2_headers, "public")
            client.delete(f"{BASE}/{user2_id}/follow", headers=user1_headers)
