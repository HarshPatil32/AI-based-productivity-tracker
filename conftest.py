import sys
import os

# Ensure the project root is on sys.path so `tracker` and `backend` are
# importable as packages. This is a fallback for pytest < 7.0 where the
# `pythonpath` key in pytest.ini is not supported.
sys.path.insert(0, os.path.dirname(__file__))
