"""Tests for GET /api/v1/search — users and sessions search endpoint."""

import pytest

BASE = "/api/v1/search"


class TestSearchAuth:
    def test_search_requires_auth(self, client):
        resp = client.get(f"{BASE}/", params={"q": "test", "type": "all"})
        assert resp.status_code == 401

    def test_search_with_valid_auth(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "test", "type": "all"}, headers=user1_headers)
        assert resp.status_code == 200


class TestSearchResponseShape:
    def test_response_has_users_and_sessions_keys(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "test", "type": "all"}, headers=user1_headers)
        data = resp.json()
        assert "users" in data
        assert "sessions" in data
        assert "users_total" in data
        assert "sessions_total" in data

    def test_type_users_returns_empty_sessions(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "test", "type": "users"}, headers=user1_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "users" in data
        assert "sessions" in data
        assert data["sessions"] == []
        assert data["sessions_total"] == 0

    def test_type_sessions_returns_empty_users(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "test", "type": "sessions"}, headers=user1_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "users" in data
        assert "sessions" in data
        assert data["users"] == []
        assert data["users_total"] == 0

    def test_totals_are_integers(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "test", "type": "all"}, headers=user1_headers)
        data = resp.json()
        assert isinstance(data["users_total"], int)
        assert isinstance(data["sessions_total"], int)


class TestSearchQueryValidation:
    def test_missing_q_returns_422(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"type": "all"}, headers=user1_headers)
        assert resp.status_code == 422

    def test_blank_q_returns_422(self, client, user1_headers):
        # min_length=1 on the Query param catches this
        resp = client.get(f"{BASE}/", params={"q": "", "type": "all"}, headers=user1_headers)
        assert resp.status_code == 422

    def test_whitespace_only_q_returns_422(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "   ", "type": "all"}, headers=user1_headers)
        assert resp.status_code == 422

    def test_invalid_type_returns_422(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "test", "type": "invalid"}, headers=user1_headers)
        assert resp.status_code == 422

    def test_q_exceeds_max_length_returns_422(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "x" * 101, "type": "all"}, headers=user1_headers)
        assert resp.status_code == 422


class TestSearchPagination:
    def test_pagination_params_respected(self, client, user1_headers):
        resp = client.get(
            f"{BASE}/",
            params={"q": "test", "type": "all", "limit": 1, "offset": 0},
            headers=user1_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["users"]) <= 1
        assert len(data["sessions"]) <= 1

    def test_limit_too_large_returns_422(self, client, user1_headers):
        resp = client.get(
            f"{BASE}/",
            params={"q": "test", "type": "all", "limit": 51},
            headers=user1_headers,
        )
        assert resp.status_code == 422

    def test_negative_offset_returns_422(self, client, user1_headers):
        resp = client.get(
            f"{BASE}/",
            params={"q": "test", "type": "all", "offset": -1},
            headers=user1_headers,
        )
        assert resp.status_code == 422


class TestSearchUserResults:
    def test_search_finds_registered_user_by_username(self, client, user1_headers, user1_creds):
        # Use a substring of the unique username
        q = user1_creds["username"][:10]
        resp = client.get(f"{BASE}/", params={"q": q, "type": "users"}, headers=user1_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert any(u["username"] == user1_creds["username"] for u in data["users"])

    def test_search_finds_registered_user_by_full_name(self, client, user1_headers, user1_creds):
        q = user1_creds["full_name"].split()[0]  # first word of full name
        resp = client.get(f"{BASE}/", params={"q": q, "type": "users"}, headers=user1_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert any(u["username"] == user1_creds["username"] for u in data["users"])

    def test_user_result_has_expected_fields(self, client, user1_headers, user1_creds):
        q = user1_creds["username"][:10]
        resp = client.get(f"{BASE}/", params={"q": q, "type": "users"}, headers=user1_headers)
        data = resp.json()
        assert data["users"], "Expected at least one user result"
        user = data["users"][0]
        for field in ("id", "username", "total_study_time"):
            assert field in user

    def test_no_match_returns_empty_list(self, client, user1_headers):
        resp = client.get(
            f"{BASE}/",
            params={"q": "zzznomatchzzz9999", "type": "users"},
            headers=user1_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["users"] == []
        assert data["users_total"] == 0


class TestSearchSpecialCharacters:
    def test_percent_in_query_does_not_error(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "a%b", "type": "users"}, headers=user1_headers)
        assert resp.status_code == 200

    def test_underscore_in_query_does_not_error(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "a_b", "type": "users"}, headers=user1_headers)
        assert resp.status_code == 200

    def test_comma_in_query_does_not_error(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "a,b", "type": "all"}, headers=user1_headers)
        assert resp.status_code == 200

    def test_dot_in_query_does_not_error(self, client, user1_headers):
        resp = client.get(f"{BASE}/", params={"q": "a.b", "type": "all"}, headers=user1_headers)
        assert resp.status_code == 200
