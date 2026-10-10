"""Storage manager for benchmark evaluation experiments."""

from pathlib import Path
from typing import List, Optional
from incident_rag.config import get_settings
from incident_rag.evaluation.reports import get_benchmark_report, list_benchmark_reports, save_benchmark_report
from incident_rag.schemas import BenchmarkReport


class ExperimentStore:
    """Manages benchmark run persistence and history lookup."""

    def __init__(self, reports_dir: Optional[Path] = None):
        settings = get_settings()
        self.reports_dir = reports_dir or settings.resolve_path(settings.REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def record_run(self, report: BenchmarkReport) -> None:
        save_benchmark_report(report, self.reports_dir)

    def list_runs(self) -> List[BenchmarkReport]:
        return list_benchmark_reports(self.reports_dir)

    def get_run(self, run_id: str) -> Optional[BenchmarkReport]:
        return get_benchmark_report(run_id, self.reports_dir)
