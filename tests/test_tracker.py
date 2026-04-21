"""
Unit tests for tracker head pose threshold logic.

These tests do not require a camera or dlib model — they cover the pure
classification function and the config validation in isolation.
"""

import importlib
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# Ensure TRACKER_SKIP_DOTENV is set before any tracker import so that
# the .env file does not override the env vars we set in tests.
os.environ.setdefault("TRACKER_SKIP_DOTENV", "1")

# Stub out modules that have heavy C-extension or hardware dependencies so
# that attention.py can be imported without a camera or dlib model present.
# attention.py uses bare (non-package-relative) imports that only resolve
# when the file is run from inside the tracker/ directory, so we satisfy
# them here by inserting lightweight stand-ins into sys.modules.
_config_stub = types.ModuleType("config")
_config_stub.EAR_THRESHOLD = 0.2
_config_stub.HEAD_YAW_THRESHOLD = 35.0
_config_stub.HEAD_PITCH_THRESHOLD = 45.0
_config_stub.YAW_OFFSET = 145.0
_config_stub.SHOW_OVERLAY = False
_config_stub.model_points = None
_config_stub.CAMERA_INDEX = 0
_config_stub.BACKEND_URL = "http://localhost:8000"
_config_stub.AUTH_TOKEN = None

_attention_stub = types.ModuleType("attention")
_attention_stub.detect_attention = MagicMock(return_value={
    "eyes_closed_time": 0,
    "face_missing_time": 0,
    "head_pose_off_time": 0,
    "total_attention_lost": 0,
})

for _name in ("face_utils", "camera", "config", "attention"):
    sys.modules.setdefault(
        _name,
        _config_stub if _name == "config"
        else _attention_stub if _name == "attention"
        else MagicMock(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_config(env_overrides: dict) -> object:
    """Reload tracker.config with the given env var overrides."""
    original = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)
    # Force a clean reimport so module-level constants are recalculated.
    if "tracker.config" in sys.modules:
        del sys.modules["tracker.config"]
    try:
        return importlib.import_module("tracker.config")
    finally:
        # Restore original env state.
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if "tracker.config" in sys.modules:
            del sys.modules["tracker.config"]


# ---------------------------------------------------------------------------
# Config default value
# ---------------------------------------------------------------------------

def test_yaw_threshold_default_is_sensible():
    from tracker.config import HEAD_YAW_THRESHOLD
    assert 25 <= HEAD_YAW_THRESHOLD <= 50, (
        f"HEAD_YAW_THRESHOLD={HEAD_YAW_THRESHOLD} is outside the expected 25–50° range"
    )


# ---------------------------------------------------------------------------
# is_head_pose_distracted — classification logic
# ---------------------------------------------------------------------------

from tracker.attention import is_head_pose_distracted

# Use the same values as the stub so there is only one place to update
# when config.py gains or changes a constant.
YAW_T = _config_stub.HEAD_YAW_THRESHOLD
PITCH_T = _config_stub.HEAD_PITCH_THRESHOLD


def test_both_within_threshold_not_distracted():
    assert not is_head_pose_distracted(yaw=10, pitch=10, yaw_threshold=YAW_T, pitch_threshold=PITCH_T)


def test_yaw_below_threshold_not_distracted():
    assert not is_head_pose_distracted(yaw=20, pitch=0, yaw_threshold=YAW_T, pitch_threshold=PITCH_T)


def test_yaw_at_threshold_not_distracted():
    # Equal to threshold is not beyond it — boundary should not trigger.
    assert not is_head_pose_distracted(yaw=35, pitch=0, yaw_threshold=YAW_T, pitch_threshold=PITCH_T)


def test_yaw_above_threshold_is_distracted():
    assert is_head_pose_distracted(yaw=40, pitch=0, yaw_threshold=YAW_T, pitch_threshold=PITCH_T)


def test_yaw_negative_above_threshold_is_distracted():
    # Left-turn: negative yaw beyond threshold must also trigger.
    assert is_head_pose_distracted(yaw=-40, pitch=0, yaw_threshold=YAW_T, pitch_threshold=PITCH_T)


def test_pitch_above_threshold_is_distracted():
    assert is_head_pose_distracted(yaw=0, pitch=50, yaw_threshold=YAW_T, pitch_threshold=PITCH_T)


def test_pitch_negative_above_threshold_is_distracted():
    assert is_head_pose_distracted(yaw=0, pitch=-50, yaw_threshold=YAW_T, pitch_threshold=PITCH_T)


# ---------------------------------------------------------------------------
# Config validation — env var guard rails
# ---------------------------------------------------------------------------

def test_config_rejects_zero_yaw_threshold():
    with pytest.raises(ValueError, match="HEAD_YAW_THRESHOLD"):
        _reload_config({
            "TRACKER_SKIP_DOTENV": "1",
            "HEAD_YAW_THRESHOLD": "0",
        })


def test_config_rejects_excessive_yaw_threshold():
    with pytest.raises(ValueError, match="HEAD_YAW_THRESHOLD"):
        _reload_config({
            "TRACKER_SKIP_DOTENV": "1",
            "HEAD_YAW_THRESHOLD": "160",
        })


def test_config_accepts_valid_yaw_threshold():
    cfg = _reload_config({
        "TRACKER_SKIP_DOTENV": "1",
        "HEAD_YAW_THRESHOLD": "35",
    })
    assert cfg.HEAD_YAW_THRESHOLD == 35.0


def test_config_yaw_offset_default():
    cfg = _reload_config({"TRACKER_SKIP_DOTENV": "1"})
    assert cfg.YAW_OFFSET == 145.0


def test_config_rejects_invalid_yaw_offset():
    with pytest.raises(ValueError, match="YAW_OFFSET"):
        _reload_config({"TRACKER_SKIP_DOTENV": "1", "YAW_OFFSET": "not_a_number"})


# Yaw normalisation — guards the `yaw = yaw - YAW_OFFSET` step

def _normalised_yaw(raw: float, offset: float = 145.0) -> float:
    """Mirror the normalisation applied in attention.py."""
    return raw - offset


def test_raw_yaw_at_offset_is_zero_after_normalisation():
    # Raw value of 145 should normalise to 0 (straight ahead) — must not distract.
    normalised = _normalised_yaw(145.0)
    assert not is_head_pose_distracted(normalised, 0, YAW_T, PITCH_T)


def test_raw_yaw_well_above_offset_triggers_distraction():
    # Raw 185 → normalised +40, beyond 35° threshold.
    normalised = _normalised_yaw(185.0)
    assert is_head_pose_distracted(normalised, 0, YAW_T, PITCH_T)


def test_raw_yaw_well_below_offset_triggers_distraction():
    # Raw 105 → normalised -40, beyond threshold in negative direction.
    normalised = _normalised_yaw(105.0)
    assert is_head_pose_distracted(normalised, 0, YAW_T, PITCH_T)


def test_raw_yaw_left_at_threshold_boundary_not_distracted():
    # Symmetry check for the left-turn side: raw 110 → normalised -35,
    # exactly at threshold — should NOT trigger (mirrors the right-side
    # boundary test above).
    normalised = _normalised_yaw(110.0)
    assert not is_head_pose_distracted(normalised, 0, YAW_T, PITCH_T)


def test_raw_yaw_left_just_beyond_threshold_is_distracted():
    # Raw 108 → normalised -37, just past the -35° boundary.
    normalised = _normalised_yaw(108.0)
    assert is_head_pose_distracted(normalised, 0, YAW_T, PITCH_T)


def test_without_normalisation_raw_yaw_would_always_distract():
    # Sanity check: raw 145 compared WITHOUT normalisation would falsely
    # exceed a 35° threshold — confirms normalisation is necessary.
    assert is_head_pose_distracted(145.0, 0, YAW_T, PITCH_T)


# Structural guard — prevent re-introduction of deleted duplicates

def test_track_face_does_not_exist():
    tracker_dir = os.path.join(os.path.dirname(__file__), "..", "tracker")
    assert not os.path.exists(os.path.join(tracker_dir, "track_face.py")), (
        "track_face.py was re-introduced; remove it and use attention.py instead"
    )


# ---------------------------------------------------------------------------
# post_session — unit tests (no real HTTP calls)
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock as _MagicMock
from tracker.main import post_session, calculate_productivity_metrics

_SAMPLE_PAYLOAD = {
    "started_at": "2026-04-15T10:00:00+00:00",
    "ended_at": "2026-04-15T10:30:00+00:00",
    "duration_seconds": 1800,
    "eyes_closed_time": 10.0,
    "face_missing_time": 5.0,
    "head_pose_off_time": 8.0,
    "total_attention_lost": 23.0,
    "notes": None,
}


def _mock_response(status_code: int, json_data: dict | None = None, text: str = "", content_type: str = "application/json"):
    resp = _MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data or {}
    resp.headers = {"content-type": content_type}
    return resp


def test_post_session_skips_when_no_auth_token(capsys):
    post_session(_SAMPLE_PAYLOAD, "http://localhost:8000", None)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_post_session_skips_when_empty_auth_token(capsys):
    post_session(_SAMPLE_PAYLOAD, "http://localhost:8000", "")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_post_session_skips_when_no_backend_url(capsys):
    post_session(_SAMPLE_PAYLOAD, None, "sometoken")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_post_session_skips_when_empty_backend_url(capsys):
    post_session(_SAMPLE_PAYLOAD, "", "sometoken")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_post_session_prints_id_on_201(capsys):
    mock_resp = _mock_response(201, json_data={"id": "abc-123"})
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        post_session(_SAMPLE_PAYLOAD, "http://localhost:8000", "token")
    captured = capsys.readouterr()
    assert "abc-123" in captured.out
    mock_post.assert_called_once()


def test_post_session_uses_bearer_auth():
    mock_resp = _mock_response(201, json_data={"id": "x"})
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        post_session(_SAMPLE_PAYLOAD, "http://localhost:8000", "mytoken")
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer mytoken"


def test_post_session_strips_trailing_slash_from_url():
    mock_resp = _mock_response(201, json_data={"id": "x"})
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        post_session(_SAMPLE_PAYLOAD, "http://localhost:8000/", "token")
    url = mock_post.call_args[0][0]
    assert not url.startswith("http://localhost:8000//")


def test_post_session_prints_unauth_message_on_401(capsys):
    mock_resp = _mock_response(401, text="Unauthorized")
    with patch("httpx.post", return_value=mock_resp):
        post_session(_SAMPLE_PAYLOAD, "http://localhost:8000", "badtoken")
    captured = capsys.readouterr()
    assert "unauthenticated" in captured.out.lower() or "AUTH_TOKEN" in captured.out


def test_post_session_prints_detail_on_422_json(capsys):
    mock_resp = _mock_response(422, json_data={"detail": "duration_seconds must be >= 0"})
    with patch("httpx.post", return_value=mock_resp):
        post_session(_SAMPLE_PAYLOAD, "http://localhost:8000", "token")
    captured = capsys.readouterr()
    assert "duration_seconds must be >= 0" in captured.out


def test_post_session_handles_422_non_json_body(capsys):
    mock_resp = _mock_response(422, text="<html>Bad Request</html>", content_type="text/html")
    mock_resp.json.side_effect = ValueError("No JSON")
    with patch("httpx.post", return_value=mock_resp):
        post_session(_SAMPLE_PAYLOAD, "http://localhost:8000", "token")
    captured = capsys.readouterr()
    # Must not raise; must print something about the error
    assert "422" in captured.out or "validation" in captured.out.lower() or "Bad Request" in captured.out


def test_post_session_handles_network_error(capsys):
    import httpx as _httpx
    with patch("httpx.post", side_effect=_httpx.RequestError("Connection refused")):
        post_session(_SAMPLE_PAYLOAD, "http://localhost:8000", "token")
    captured = capsys.readouterr()
    assert "Connection refused" in captured.out or "Could not reach" in captured.out


# ---------------------------------------------------------------------------
# eye_aspect_ratio — pure geometry, no dlib model required
# ---------------------------------------------------------------------------

# Each eye is a list of 6 (x, y) tuples: [p0, p1, p2, p3, p4, p5]
# Formula: EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)

def test_ear_symmetric_open_eye():
    from tracker.face_utils import eye_aspect_ratio
    # Perfectly symmetric eye: vertical distances equal the horizontal width.
    # A = dist(p1, p5) = 2, B = dist(p2, p4) = 2, C = dist(p0, p3) = 2
    # EAR = (2+2)/(2*2) = 1.0
    eye = [(0, 0), (0, 1), (1, 1), (2, 0), (1, -1), (0, -1)]
    assert eye_aspect_ratio(eye) == pytest.approx(1.0)


def test_ear_fully_closed_eye():
    from tracker.face_utils import eye_aspect_ratio
    # All landmarks on the horizontal axis — zero vertical distances.
    # EAR = 0.0
    eye = [(0, 0), (1, 0), (2, 0), (4, 0), (2, 0), (1, 0)]
    assert eye_aspect_ratio(eye) == pytest.approx(0.0)


def test_ear_known_ratio():
    from tracker.face_utils import eye_aspect_ratio
    # A = dist((1,2),(1,-2)) = 4, B = dist((2,2),(2,-2)) = 4, C = dist((0,0),(6,0)) = 6
    # EAR = (4+4)/(2*6) = 8/12 = 2/3
    eye = [(0, 0), (1, 2), (2, 2), (6, 0), (2, -2), (1, -2)]
    assert eye_aspect_ratio(eye) == pytest.approx(2 / 3)


def test_ear_below_default_threshold_for_near_closed_eye():
    from tracker.face_utils import eye_aspect_ratio
    # Tiny vertical gaps relative to horizontal width → EAR well below 0.2.
    eye = [(0, 0), (1, 0.1), (2, 0.1), (4, 0), (2, -0.1), (1, -0.1)]
    assert eye_aspect_ratio(eye) < 0.2


def test_ear_above_default_threshold_for_open_eye():
    from tracker.face_utils import eye_aspect_ratio
    # Clearly open eye geometry → EAR well above 0.2.
    eye = [(0, 0), (0, 1), (1, 1), (2, 0), (1, -1), (0, -1)]
    assert eye_aspect_ratio(eye) > 0.2


def test_ear_division_by_zero():
    from tracker.face_utils import eye_aspect_ratio
    # Horizontal width is zero (p0 == p3), should not raise ZeroDivisionError.
    eye = [(1, 0), (1, 1), (1, 2), (1, 0), (1, -1), (1, -2)]
    try:
        result = eye_aspect_ratio(eye)
    except ZeroDivisionError:
        pytest.fail("eye_aspect_ratio() raised ZeroDivisionError on zero horizontal width")
    # Accept 0.0, np.nan, or np.inf, but must not crash
    import math
    assert (
        result == 0.0
        or (hasattr(result, "__eq__") and result != result)  # np.nan != np.nan
        or math.isinf(result)
    )


# ---------------------------------------------------------------------------
# calculate_productivity_metrics — scores and quality tier boundaries
# ---------------------------------------------------------------------------

def _metrics(duration, attention_lost):
    """Convenience wrapper: build the attention_metrics dict and call the function."""
    return calculate_productivity_metrics(
        {"total_attention_lost": attention_lost},
        duration,
    )


# -- focus_score and attention_score --

def test_metrics_no_distraction_scores_100():
    result = _metrics(duration=100, attention_lost=0)
    assert result["focus_score"] == pytest.approx(100.0)
    assert result["attention_score"] == pytest.approx(100.0)


def test_metrics_attention_score_equals_focus_score():
    result = _metrics(duration=200, attention_lost=50)
    assert result["focus_score"] == result["attention_score"]


def test_metrics_work_time_is_duration_minus_attention_lost():
    result = _metrics(duration=3600, attention_lost=360)
    assert result["work_time"] == pytest.approx(3240.0)


# -- quality tier boundaries --

def test_metrics_exactly_90_percent_is_excellent():
    # focus_score = (90/100)*100 = 90.0 → "Excellent"
    result = _metrics(duration=100, attention_lost=10)
    assert result["focus_score"] == pytest.approx(90.0)
    assert result["quality"] == "Excellent"


def test_metrics_below_90_percent_is_good():
    # focus_score = 89.0 → "Good"
    result = _metrics(duration=100, attention_lost=11)
    assert result["focus_score"] == pytest.approx(89.0)
    assert result["quality"] == "Good"


def test_metrics_exactly_75_percent_is_good():
    # focus_score = 75.0 → "Good"
    result = _metrics(duration=100, attention_lost=25)
    assert result["focus_score"] == pytest.approx(75.0)
    assert result["quality"] == "Good"


def test_metrics_below_75_percent_is_fair():
    # focus_score = 74.0 → "Fair"
    result = _metrics(duration=100, attention_lost=26)
    assert result["focus_score"] == pytest.approx(74.0)
    assert result["quality"] == "Fair"


def test_metrics_exactly_60_percent_is_fair():
    # focus_score = 60.0 → "Fair"
    result = _metrics(duration=100, attention_lost=40)
    assert result["focus_score"] == pytest.approx(60.0)
    assert result["quality"] == "Fair"


def test_metrics_below_60_percent_is_poor():
    # focus_score = 59.0 → "Poor"
    result = _metrics(duration=100, attention_lost=41)
    assert result["focus_score"] == pytest.approx(59.0)
    assert result["quality"] == "Poor"


def test_metrics_100_percent_distraction_is_poor():
    result = _metrics(duration=100, attention_lost=100)
    assert result["focus_score"] == pytest.approx(0.0)
    assert result["quality"] == "Poor"


# -- edge cases --

def test_metrics_zero_duration_returns_zero_focus_score():
    # Guard against division by zero.
    result = _metrics(duration=0, attention_lost=0)
    assert result["focus_score"] == pytest.approx(0.0)
    assert result["quality"] == "Poor"


def test_metrics_zero_duration_returns_zero_work_time():
    result = _metrics(duration=0, attention_lost=0)
    assert result["work_time"] == pytest.approx(0.0)


def test_metrics_attention_lost_exceeds_duration_clamps_work_time_to_zero():
    # When distraction time is recorded longer than the session (e.g. clock drift),
    # work_time must not go negative.
    result = _metrics(duration=60, attention_lost=100)
    assert result["work_time"] == pytest.approx(0.0)
    assert result["focus_score"] == pytest.approx(0.0)


def test_metrics_payload_keys_are_complete():
    result = _metrics(duration=100, attention_lost=20)
    expected_keys = {
        "session_duration", "work_time", "distracted_time",
        "attention_lost", "focus_score", "attention_score", "quality",
    }
    assert expected_keys.issubset(result.keys())
