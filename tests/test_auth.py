"""Tests for /api/v1/auth/* endpoints."""

import pytest


BASE = "/api/v1/auth"


class TestRegister:
    def test_register_success(self, user1_tokens):
        """Fixture already registers user1; just assert the token shape."""
        assert "access_token" in user1_tokens
        assert "refresh_token" in user1_tokens
        assert user1_tokens["token_type"] == "bearer"
        assert "user" in user1_tokens

    def test_register_duplicate_email(self, client, user1_creds):
        resp = client.post(f"{BASE}/register", json=user1_creds)
        assert resp.status_code in (400, 422)

    def test_register_weak_password(self, client):
        resp = client.post(
            f"{BASE}/register",
            json={
                "email": "weak@example.com",
                "password": "abc",
                "username": "weakuser",
            },
        )
        assert resp.status_code in (400, 422)

    def test_register_missing_fields(self, client):
        resp = client.post(f"{BASE}/register", json={"email": "nopass@example.com"})
        assert resp.status_code == 422


class TestLogin:
    def test_login_success(self, client, user1_creds):
        resp = client.post(
            f"{BASE}/login",
            json={"email": user1_creds["email"], "password": user1_creds["password"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "access_token" in data["data"]

    def test_login_wrong_password(self, client, user1_creds):
        resp = client.post(
            f"{BASE}/login",
            json={"email": user1_creds["email"], "password": "WrongPassword9!"},
        )
        assert resp.status_code == 401

    def test_login_nonexistent_email(self, client):
        resp = client.post(
            f"{BASE}/login",
            json={"email": "nobody@nowhere.com", "password": "TestPass1!"},
        )
        assert resp.status_code == 401


class TestTokenRefresh:
    def test_refresh_success(self, client, user1_tokens):
        resp = client.post(
            f"{BASE}/refresh",
            json={"refresh_token": user1_tokens["refresh_token"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "access_token" in data["data"]

    def test_refresh_invalid_token(self, client):
        resp = client.post(
            f"{BASE}/refresh", json={"refresh_token": "this.is.not.valid"}
        )
        assert resp.status_code == 401


class TestAuthMe:
    def test_get_current_user_authenticated(self, client, user1_headers, user1_creds):
        resp = client.get(f"{BASE}/me", headers=user1_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["user"]["username"] == user1_creds["username"]

    def test_get_current_user_no_token(self, client):
        resp = client.get(f"{BASE}/me")
        assert resp.status_code == 401


class TestVerifyToken:
    def test_verify_valid_token(self, client, user1_headers):
        resp = client.get(f"{BASE}/verify", headers=user1_headers)
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_verify_no_token(self, client):
        resp = client.get(f"{BASE}/verify")
        assert resp.status_code == 401

    def test_verify_malformed_token(self, client):
        resp = client.get(
            f"{BASE}/verify", headers={"Authorization": "Bearer badtoken"}
        )
        assert resp.status_code == 401


class TestLogout:
    def test_logout_authenticated(self, client, user1_headers):
        resp = client.post(f"{BASE}/logout", headers=user1_headers)
        assert resp.status_code == 200
        assert resp.json()["success"] is True
