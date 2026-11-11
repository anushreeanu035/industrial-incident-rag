from pathlib import Path
import runpy
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from frontend.components.backend_service import ensure_backend_running
ensure_backend_running()

_TARGET = _PROJECT_ROOT / "frontend" / "pages" / "1_Investigation.py"
runpy.run_path(str(_TARGET), run_name="__main__")
