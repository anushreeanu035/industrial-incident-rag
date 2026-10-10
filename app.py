"""Main launcher CLI for Industrial Incident Investigator."""

import argparse
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

# If launched directly via `streamlit run app.py`, delegate immediately to frontend/app.py
if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def _get_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    project_root = os.path.abspath(".")
    src_dir = os.path.abspath("src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    paths = [project_root, src_dir]
    if existing_pythonpath:
        paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _wait_for_backend(url="http://127.0.0.1:8000/health", max_retries=30, delay=1.0):
    """Polls backend health check until it responds 200 OK."""
    for i in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def _ensure_backend_running(host="127.0.0.1", port=8000):
    """Starts FastAPI in background if not already active (essential for single-container deploys like Streamlit Cloud)."""
    health_url = f"http://{host}:{port}/health"
    if not _wait_for_backend(health_url, max_retries=1, delay=0.1):
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "incident_rag.api:app",
            "--host",
            host,
            "--port",
            str(port),
            "--app-dir",
            "src",
        ]
        subprocess.Popen(cmd, env=_get_env())
        _wait_for_backend(health_url, max_retries=30, delay=0.8)


# If launched directly via `streamlit run app.py`, ensure backend is up and delegate immediately to frontend/app.py
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        _ensure_backend_running()
        import runpy
        _frontend_entry = os.path.abspath(os.path.join(os.path.dirname(__file__), "frontend", "app.py"))
        runpy.run_path(_frontend_entry, run_name="__main__")
        sys.exit(0)
except Exception:
    pass


def run_backend(host="127.0.0.1", port=8000, reload=True):
    """Starts the FastAPI backend using uvicorn."""
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "incident_rag.api:app",
        "--host",
        host,
        "--port",
        str(port),
        "--app-dir",
        "src",
    ]
    if reload:
        cmd.append("--reload")
    print(f"Starting FastAPI backend on http://{host}:{port} ...")
    subprocess.run(cmd, env=_get_env())


def run_frontend(headless=False, port=8501):
    """Starts the Streamlit frontend."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/app.py",
        "--server.port",
        str(port),
    ]
    if headless:
        cmd.extend(["--server.headless", "true"])
    print(f"Starting Streamlit frontend on http://127.0.0.1:{port} ...")
    subprocess.run(cmd, env=_get_env())


def run_all(host="127.0.0.1", api_port=8000, front_port=8501):
    """Starts both the FastAPI backend and Streamlit frontend concurrently."""
    print("=" * 65)
    print(" 🚀 Launching Industrial Incident Investigator (Full Stack) ")
    print("=" * 65)
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "incident_rag.api:app",
        "--host",
        host,
        "--port",
        str(api_port),
        "--app-dir",
        "src",
    ]
    print(f"[1/2] Starting FastAPI backend on http://{host}:{api_port} ...")
    backend_proc = subprocess.Popen(backend_cmd, env=_get_env())

    try:
        health_url = f"http://{host}:{api_port}/health"
        print(f"      Waiting for backend to initialize at {health_url}...")
        if not _wait_for_backend(health_url, max_retries=25, delay=0.8):
            print("⚠️ Warning: Backend health check timed out. Proceeding to frontend...")
        else:
            print("✅ Backend is online and operational!")

        print(f"[2/2] Starting Streamlit frontend on http://127.0.0.1:{front_port} ...")
        frontend_cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "frontend/app.py",
            "--server.port",
            str(front_port),
        ]
        frontend_proc = subprocess.Popen(frontend_cmd, env=_get_env())

        print("\n" + "=" * 65)
        print(f"🎉 System Active!")
        print(f"   • Frontend Dashboard: http://localhost:{front_port}")
        print(f"   • Backend API Docs:   http://{host}:{api_port}/docs")
        print(f"   • Health Check:       http://{host}:{api_port}/health")
        print("   Press Ctrl+C in this terminal to stop both servers.")
        print("=" * 65 + "\n")

        # Keep running until user terminates
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("Backend terminated unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend terminated.")
                break

    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        print("Shutting down processes...")
        try:
            frontend_proc.terminate()
            frontend_proc.wait(timeout=3)
        except Exception:
            pass
        try:
            backend_proc.terminate()
            backend_proc.wait(timeout=3)
        except Exception:
            pass
        print("All services stopped.")


def run_tests(quiet=False):
    """Executes the test suite with pytest."""
    cmd = [sys.executable, "-m", "pytest"]
    if quiet:
        cmd.append("-q")
    else:
        cmd.append("-v")
    print("Executing test suite...")
    subprocess.run(cmd, env=_get_env())


def run_benchmark():
    """Executes deterministic retrieval benchmark via python runner."""
    cmd = [
        sys.executable,
        "-c",
        "from incident_rag.dependencies import get_benchmark_runner; from incident_rag.storage.experiment_store import ExperimentStore; r = get_benchmark_runner().run_benchmark(); ExperimentStore().record_run(r); print(f'Benchmark completed: {r.run_id}')",
    ]
    print("Running benchmark evaluation...")
    subprocess.run(cmd, env=_get_env())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Incident Investigator Application Manager")
    parser.add_argument(
        "command",
        choices=["all", "backend", "frontend", "test", "benchmark"],
        nargs="?",
        default="all",
        help="Command to run (all | backend | frontend | test | benchmark) [default: all]",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Backend host")
    parser.add_argument("--port", type=int, default=8000, help="Backend port")
    parser.add_argument("--front-port", type=int, default=8501, help="Frontend port")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload for backend")
    parser.add_argument("--headless", action="store_true", help="Run frontend in headless mode")

    args = parser.parse_args()

    if args.command == "all":
        run_all(host=args.host, api_port=args.port, front_port=args.front_port)
    elif args.command == "backend":
        run_backend(host=args.host, port=args.port, reload=not args.no_reload)
    elif args.command == "frontend":
        run_frontend(headless=args.headless, port=args.front_port)
    elif args.command == "test":
        run_tests()
    elif args.command == "benchmark":
        run_benchmark()
