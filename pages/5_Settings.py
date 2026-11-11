from pathlib import Path
import runpy
import sys

try:
    from app import _ensure_backend_running
    _ensure_backend_running()
except Exception:
    pass


_TARGET = Path(__file__).resolve().parent.parent / "frontend" / "pages" / "5_Settings.py"
runpy.run_path(str(_TARGET), run_name="__main__")
