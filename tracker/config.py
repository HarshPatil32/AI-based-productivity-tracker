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
            f"Invalid value for {key}={raw!r}. Expected {cast.__name__}."
        )


CAMERA_INDEX = _get_env("CAMERA_INDEX", "0", int)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
AUTH_TOKEN: str | None = os.getenv("AUTH_TOKEN", None)

EAR_THRESHOLD = _get_env("EAR_THRESHOLD", "0.2", float)
HEAD_YAW_THRESHOLD = _get_env("HEAD_YAW_THRESHOLD", "160", float)
HEAD_PITCH_THRESHOLD = _get_env("HEAD_PITCH_THRESHOLD", "45", float)

if not (0.0 < EAR_THRESHOLD < 1.0):
    raise ValueError(
        f"EAR_THRESHOLD={EAR_THRESHOLD} is out of range. Must be between 0.0 and 1.0."
    )
if HEAD_YAW_THRESHOLD <= 0 or HEAD_PITCH_THRESHOLD <= 0:
    raise ValueError(
        "HEAD_YAW_THRESHOLD and HEAD_PITCH_THRESHOLD must be positive."
    )

model_points = np.array([
    (0.0, 0.0, 0.0),             
    (0.0, -330.0, -65.0),        
    (-225.0, 170.0, -135.0),     
    (225.0, 170.0, -135.0),      
    (-150.0, -150.0, -125.0),    
    (150.0, -150.0, -125.0)      
], dtype=np.float64)
