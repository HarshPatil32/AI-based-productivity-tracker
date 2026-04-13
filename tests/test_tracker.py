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
for _name in ("face_utils", "camera", "config"):
    sys.modules.setdefault(_name, _config_stub if _name == "config" else MagicMock())


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
