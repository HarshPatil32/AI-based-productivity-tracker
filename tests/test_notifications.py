import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

@pytest.fixture
def auth_headers():
    # This should be replaced with a real token for a seeded user in your test DB
    return {"Authorization": "Bearer test-token"}

def test_notifications_route_order():
    # Smoke test: route order should not 422
    resp = client.post("/api/v1/notifications/read-all", headers={})
    assert resp.status_code in (200, 401, 403)  # 401 if not logged in


def test_notification_type_literal():
    # Ensure only allowed types are accepted by type checker (static test)
    from backend.services.notifications import NotificationType
    def accepts_type(t: NotificationType):
        return t
    assert accepts_type("follow") == "follow"
    assert accepts_type("like") == "like"
    assert accepts_type("comment") == "comment"
    # The following should fail type checking (mypy/pyright), not at runtime
    # accepts_type("invalid")


def test_notification_response_extra_ignore():
    from backend.models.notifications import NotificationResponse
    # Should not raise if extra fields are present
    data = {
        "id": "00000000-0000-0000-0000-000000000000",
        "user_id": "00000000-0000-0000-0000-000000000000",
        "actor_id": "00000000-0000-0000-0000-000000000000",
        "type": "follow",
        "entity_id": None,
        "is_read": False,
        "created_at": "2024-01-01T00:00:00Z",
        "extra_field": "should be ignored"
    }
    NotificationResponse(**data)
