"""Benchmark dataset loader and validator with SHA-256 integrity verification."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger

logger = get_logger(__name__)


class BenchmarkTestCase(NamedTuple):
    query_id: str
    query: str
    target_filename: str
    machine_id: Optional[str] = None
    incident_type: Optional[str] = None
    key_identifiers: List[str] = []
    reference_answer: str = ""
    ground_truth_passages: List[str] = []


class LoadedBenchmarkDataset(NamedTuple):
    version: str
    dataset_name: str
    description: str
    dataset_hash: str
    test_cases: List[Dict[str, Any]]


def load_benchmark_dataset(dataset_path: Optional[Path] = None) -> LoadedBenchmarkDataset:
    """Loads and hashes the benchmark ground truth dataset."""
    settings = get_settings()
    path = dataset_path or settings.resolve_path(settings.BENCHMARK_DATASET_PATH)

    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at: {path}")

    raw_bytes = path.read_bytes()
    dataset_hash = hashlib.sha256(raw_bytes).hexdigest()

    data = json.loads(raw_bytes.decode("utf-8"))

    return LoadedBenchmarkDataset(
        version=data.get("version", "1.0.0"),
        dataset_name=data.get("dataset_name", "industrial_incident_benchmark"),
        description=data.get("description", ""),
        dataset_hash=dataset_hash,
        test_cases=data.get("test_cases", []),
    )
