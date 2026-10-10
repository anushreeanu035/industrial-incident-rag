from pathlib import Path
import runpy
import sys

_TARGET = Path(__file__).resolve().parent.parent / "frontend" / "pages" / "1_Investigation.py"
runpy.run_path(str(_TARGET), run_name="__main__")
