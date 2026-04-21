"""Tests for /api/v1/users/* (profile + settings) endpoints."""

import pytest


BASE = "/api/v1/users"


class TestGetMyProfile:
    def test_get_my_profile_success(self, client, user1_headers, user1_creds):
        resp = client.get(f"{BASE}/me", headers=user1_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == user1_creds["username"]
        assert data["full_name"] == user1_creds["full_name"]
        assert "id" in data
        assert "total_study_time" in data

    def test_get_my_profile_no_auth(self, client):
        resp = client.get(f"{BASE}/me")
        assert resp.status_code == 401

    def test_profile_includes_social_stats(self, client, user1_headers):
        resp = client.get(f"{BASE}/me", headers=user1_headers)
        assert resp.status_code == 200
        data = resp.json()
        # These fields come from the user_profile_summary view
        assert "followers_count" in data
        assert "following_count" in data
        assert "total_sessions" in data


class TestUpdateMyProfile:
    def test_patch_bio(self, client, user1_headers):
        resp = client.patch(
            f"{BASE}/me",
            json={"bio": "Updated bio from pytest"},
            headers=user1_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["bio"] == "Updated bio from pytest"

    def test_patch_full_name(self, client, user1_headers):
        resp = client.patch(
            f"{BASE}/me",
            json={"full_name": "Updated Name"},
            headers=user1_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["full_name"] == "Updated Name"

    def test_patch_empty_body_rejected(self, client, user1_headers):
        resp = client.patch(f"{BASE}/me", json={}, headers=user1_headers)
        assert resp.status_code == 400

    def test_patch_profile_no_auth(self, client):
        resp = client.patch(f"{BASE}/me", json={"bio": "should fail"})
        assert resp.status_code == 401


class TestGetProfileByUsername:
    def test_get_profile_by_username_success(self, client, user1_headers, user1_creds):
        username = user1_creds["username"]
        resp = client.get(f"{BASE}/{username}", headers=user1_headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == username

    def test_get_profile_not_found(self, client, user1_headers):
        resp = client.get(f"{BASE}/nonexistent__xyz", headers=user1_headers)
        assert resp.status_code == 404

    def test_get_profile_no_auth(self, client, user1_creds):
        resp = client.get(f"{BASE}/{user1_creds['username']}")
        assert resp.status_code == 401


class TestGetMySettings:
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
