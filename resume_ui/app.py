from pathlib import Path
import os
import sys

# Reuse the existing app code by importing from project root path if needed
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from app import *  # noqa: F401,F403 - reuse existing Streamlit page


