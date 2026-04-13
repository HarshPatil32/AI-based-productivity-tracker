import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

if os.getenv("TRACKER_SKIP_DOTENV") != "1":
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


def _get_env(key: str, default: str, cast):
    raw = os.getenv(key, default)
    try:
        return cast(raw)
    except (ValueError, TypeError):
        raise ValueError(
            f"Config error: {key}={raw!r} cannot be parsed as {cast.__name__}. "
            f"Default would have been {default!r}."
        )


CAMERA_INDEX = _get_env("CAMERA_INDEX", "0", int)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
AUTH_TOKEN: str | None = os.getenv("AUTH_TOKEN", None)

EAR_THRESHOLD = _get_env("EAR_THRESHOLD", "0.2", float)
# Empirical offset to centre the raw yaw value returned by cv2.decomposeProjectionMatrix around 0 (facing forward).
# 145 is the observed resting value when using solvePnP with dlib's 68-point model on a standard webcam. If yaw readings seem inverted
# or consistently off, adjust this value via the YAW_OFFSET env var.
YAW_OFFSET = _get_env("YAW_OFFSET", "145", float)
# 35 degrees: at this angle a user has clearly turned away from the screen
HEAD_YAW_THRESHOLD = _get_env("HEAD_YAW_THRESHOLD", "35", float)
# pitch is used as-is (no offset applied). If solvePnP produces a systematic pitch offset on a different camera, introduce a PITCH_OFFSET
# constant here following the same pattern as YAW_OFFSET.
HEAD_PITCH_THRESHOLD = _get_env("HEAD_PITCH_THRESHOLD", "45", float)
# Set SHOW_OVERLAY=1 to render the live Yaw/Pitch values on the camera
# window. Off by default to avoid visual noise in production sessions.
SHOW_OVERLAY = _get_env("SHOW_OVERLAY", "0", int) != 0

if not (0.0 < EAR_THRESHOLD < 1.0):
    raise ValueError(
        f"EAR_THRESHOLD={EAR_THRESHOLD} is out of range. Must be between 0.0 and 1.0."
    )
if not (0 < HEAD_YAW_THRESHOLD <= 90):
    raise ValueError(
        f"HEAD_YAW_THRESHOLD={HEAD_YAW_THRESHOLD} is out of range. Must be between 0 and 90 degrees."
    )
if HEAD_PITCH_THRESHOLD <= 0:
    raise ValueError(
        "HEAD_PITCH_THRESHOLD must be positive."
    )

model_points = np.array([
    (0.0, 0.0, 0.0),             
    (0.0, -330.0, -65.0),        
    (-225.0, 170.0, -135.0),     
    (225.0, 170.0, -135.0),      
    (-150.0, -150.0, -125.0),    
    (150.0, -150.0, -125.0)      
], dtype=np.float64)
