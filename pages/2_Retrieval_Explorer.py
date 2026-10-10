from pathlib import Path
import runpy
import sys

_TARGET = Path(__file__).resolve().parent.parent / "frontend" / "pages" / "2_Retrieval_Explorer.py"
runpy.run_path(str(_TARGET), run_name="__main__")
