"""Make the repo root importable so `from src...` works under pytest."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
