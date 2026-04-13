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
# 35 degrees: at this angle a user has clearly turned away from the screen
# this threshold assumes yaw is expressed on a 0-centred scale where
# 0 = facing forward. track_face.py applies `yaw = yaw - 145` to normalise
# raw cv2.decomposeProjectionMatrix output; attention.py must do the same
# before comparing against this threshold.
HEAD_YAW_THRESHOLD = _get_env("HEAD_YAW_THRESHOLD", "35", float)
HEAD_PITCH_THRESHOLD = _get_env("HEAD_PITCH_THRESHOLD", "45", float)

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
