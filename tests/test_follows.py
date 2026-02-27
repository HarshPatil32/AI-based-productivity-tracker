"""Tests for /api/v1/users/{id}/follow* endpoints."""

import pytest


BASE = "/api/v1/users"


class TestFollowUser:
    def test_follow_user2(self, client, user1_headers, user2_id):
        resp = client.post(f"{BASE}/{user2_id}/follow", headers=user1_headers)
        assert resp.status_code in (201, 200)  # 200 = already following (idempotent)

    def test_follow_idempotent(self, client, user1_headers, user2_id):
        """Following someone already followed should not error."""
        resp = client.post(f"{BASE}/{user2_id}/follow", headers=user1_headers)
        assert resp.status_code in (200, 201)
        assert "already following" in resp.json().get("detail", "").lower() or resp.status_code == 201

    def test_cannot_follow_self(self, client, user1_headers, user1_id):
        resp = client.post(f"{BASE}/{user1_id}/follow", headers=user1_headers)
        assert resp.status_code == 400

    def test_follow_nonexistent_user(self, client, user1_headers):
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = client.post(f"{BASE}/{fake_id}/follow", headers=user1_headers)
        assert resp.status_code == 404

    def test_follow_no_auth(self, client, user2_id):
        resp = client.post(f"{BASE}/{user2_id}/follow")
        assert resp.status_code == 401


class TestGetFollowers:
    def test_user2_has_follower(self, client, user1_headers, user2_id, user1_id):
        """After user1 followed user2, user2's followers list should include user1."""
        resp = client.get(f"{BASE}/{user2_id}/followers", headers=user1_headers)
        assert resp.status_code == 200
        follower_ids = [f["id"] for f in resp.json()]
        assert user1_id in follower_ids

    def test_followers_pagination(self, client, user1_headers, user2_id):
        resp = client.get(f"{BASE}/{user2_id}/followers?limit=1", headers=user1_headers)
        assert resp.status_code == 200
        assert len(resp.json()) <= 1

    def test_followers_no_auth(self, client, user2_id):
        resp = client.get(f"{BASE}/{user2_id}/followers")
        assert resp.status_code == 401


class TestGetFollowing:
    def test_user1_is_following_user2(self, client, user1_headers, user1_id, user2_id):
        resp = client.get(f"{BASE}/{user1_id}/following", headers=user1_headers)
        assert resp.status_code == 200
        following_ids = [f["id"] for f in resp.json()]
        assert user2_id in following_ids

    def test_following_no_auth(self, client, user1_id):
        resp = client.get(f"{BASE}/{user1_id}/following")
        assert resp.status_code == 401


class TestMyFollowerConvenience:
    def test_get_my_followers(self, client, user2_headers, user1_id):
        """user2's /me/followers should include user1."""
        resp = client.get(f"{BASE}/me/followers", headers=user2_headers)
        assert resp.status_code == 200
        ids = [f["id"] for f in resp.json()]
        assert user1_id in ids

    def test_get_my_following(self, client, user1_headers, user2_id):
        """user1's /me/following should include user2."""
        resp = client.get(f"{BASE}/me/following", headers=user1_headers)
        assert resp.status_code == 200
        ids = [f["id"] for f in resp.json()]
        assert user2_id in ids

    def test_my_followers_no_auth(self, client):
        resp = client.get(f"{BASE}/me/followers")
        assert resp.status_code == 401


class TestUnfollowUser:
    def test_unfollow_user2(self, client, user1_headers, user2_id, user1_id):
        # First make sure we're following
        client.post(f"{BASE}/{user2_id}/follow", headers=user1_headers)
        resp = client.delete(f"{BASE}/{user2_id}/follow", headers=user1_headers)
        assert resp.status_code == 204

        # Confirm removed from follower list
        check = client.get(f"{BASE}/{user2_id}/followers", headers=user1_headers)
        ids = [f["id"] for f in check.json()]
        assert user1_id not in ids

    def test_unfollow_no_auth(self, client, user2_id):
        resp = client.delete(f"{BASE}/{user2_id}/follow")
        assert resp.status_code == 401
