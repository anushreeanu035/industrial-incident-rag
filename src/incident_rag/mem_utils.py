"""Memory management, telemetry, and OS-level memory trimming utilities for cloud deployments."""

import gc
import os
import sys
import threading
from typing import Optional

# Process-wide execution locks for memory-intensive operations
# These ensure that multiple concurrent requests or tasks never execute
# overlapping ingestion or heavy batch embedding operations in parallel.
_INGESTION_EXECUTION_LOCK = threading.Lock()
_EMBEDDING_EXECUTION_LOCK = threading.Lock()


def get_ingestion_execution_lock() -> threading.Lock:
    """Returns the process-wide ingestion execution lock."""
    return _INGESTION_EXECUTION_LOCK


def get_embedding_execution_lock() -> threading.Lock:
    """Returns the process-wide embedding execution lock."""
    return _EMBEDDING_EXECUTION_LOCK


class UploadConcurrencyLimiter:
    """Atomic, non-blocking concurrency limiter protecting low-memory cloud environments.

    Uses an atomic BoundedSemaphore to eliminate check-then-acquire race conditions.
    Two simultaneous requests can never both pass the acquisition check.
    """

    def __init__(self, max_concurrent: int = 1):
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._max_concurrent = max_concurrent

    def try_acquire(self) -> bool:
        """Atomically attempts to acquire an upload execution slot without blocking.

        Returns:
            True if slot was acquired, False if the concurrency limit is reached.
        """
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        """Safely releases the acquired upload slot."""
        try:
            self._semaphore.release()
        except ValueError:
            # Prevent over-release errors
            pass

    @property
    def is_locked(self) -> bool:
        """Returns whether all available concurrency slots are currently active."""
        acquired = self._semaphore.acquire(blocking=False)
        if acquired:
            self._semaphore.release()
            return False
        return True


_UPLOAD_LIMITER: Optional[UploadConcurrencyLimiter] = None
_UPLOAD_LIMITER_LOCK = threading.Lock()


def get_upload_limiter(max_concurrent: Optional[int] = None) -> UploadConcurrencyLimiter:
    """Returns the process-wide upload concurrency limiter singleton."""
    global _UPLOAD_LIMITER
    if _UPLOAD_LIMITER is None:
        with _UPLOAD_LIMITER_LOCK:
            if _UPLOAD_LIMITER is None:
                from incident_rag.config import get_settings
                limit = max_concurrent if max_concurrent is not None else getattr(get_settings(), "MAX_CONCURRENT_UPLOADS", 1)
                _UPLOAD_LIMITER = UploadConcurrencyLimiter(max_concurrent=limit)
    return _UPLOAD_LIMITER


# Pre-configure environment limits before heavy libraries initialize
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")


def get_process_memory_mb() -> float:
    """Returns the current process resident set size (RSS) memory in megabytes (MB)."""
    # 1. Linux (/proc/self/status - exact cgroup and RSS metrics)
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Format: VmRSS:   12345 kB
                    parts = line.split()
                    if len(parts) >= 2:
                        return round(float(parts[1]) / 1024.0, 2)
    except Exception:
        pass

    # 2. Windows (K32GetProcessMemoryInfo)
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        fn.restype = wintypes.BOOL

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if fn(handle, ctypes.byref(pmc), pmc.cb):
            return round(pmc.WorkingSetSize / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    return 0.0


def trim_memory() -> float:
    """Executes Python garbage collection and triggers OS glibc heap trimming.

    Important note:
        This cleans up unreferenced temporary Python objects and instructs the
        Linux glibc allocator (malloc_trim) to return free heap memory pages to the OS.
        It CANNOT and DOES NOT free memory occupied by models that remain loaded
        in memory (such as PyTorch model weights).

    Returns:
        Process RSS memory in MB after trimming.
    """
    gc.collect()

    # On Linux glibc (used in Render containers), malloc_trim forces the runtime
    # to release freed heap memory pages back to the kernel immediately,
    # reducing memory fragmentation and preventing premature cgroup OOM kills.
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

    return get_process_memory_mb()


def configure_runtime_memory_limits() -> None:
    """Applies runtime optimizations to torch."""
    try:
        import torch
        torch.set_grad_enabled(False)
    except ImportError:
        pass
