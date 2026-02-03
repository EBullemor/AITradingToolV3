"""
Metrics Collector Module

Collects and stores operational metrics for monitoring and dashboards.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class PipelineMetrics:
    """Metrics from a pipeline run."""
    run_date: datetime
    status: str
    duration_seconds: float
    
    # Stage timings
    stage_timings: Dict[str, float] = field(default_factory=dict)
    
    # Data metrics
    instruments_loaded: int = 0
    data_quality_issues: int = 0
    
    # Signal metrics
    signals_generated: int = 0
    signals_aggregated: int = 0
    conflicts_detected: int = 0
    
    # Output metrics
    recommendations_count: int = 0
    notion_pushed: bool = False
    slack_sent: bool = False
    
    # Errors
    error_count: int = 0
    warning_count: int = 0


@dataclass
class DailyMetrics:
    """Aggregated daily metrics."""
    date: str
    runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    
    # Signals
    total_signals: int = 0
    long_signals: int = 0
    short_signals: int = 0
    
    # Recommendations
    total_recommendations: int = 0
    avg_confidence: float = 0.0
    
    # Performance
    avg_duration_seconds: float = 0.0
    max_duration_seconds: float = 0.0


class MetricsCollector:
    """
    Collects and persists operational metrics.
    
    Stores metrics in JSON files for simplicity.
    Can be extended to use a database or time-series store.
    """
    
    def __init__(
        self,
        metrics_dir: Path = Path("data/metrics"),
        retention_days: int = 30
    ):
        """
        Initialize metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics
            retention_days: Days to retain metrics
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        
        # In-memory buffer
        self.current_run: Optional[PipelineMetrics] = None
        self.buffer: List[Dict] = []
    
    def start_run(self, run_date: Optional[datetime] = None):
        """Start tracking a new pipeline run."""
        self.current_run = PipelineMetrics(
            run_date=run_date or datetime.now(),
            status="RUNNING",
            duration_seconds=0,
        )
        logger.debug("Metrics collection started")
    
    def record_stage_timing(self, stage: str, duration: float):
        """Record timing for a pipeline stage."""
        if self.current_run:
            self.current_run.stage_timings[stage] = duration
    
    def record_data_metrics(
        self,
        instruments_loaded: int,
        quality_issues: int = 0
    ):
        """Record data ingestion metrics."""
        if self.current_run:
            self.current_run.instruments_loaded = instruments_loaded
            self.current_run.data_quality_issues = quality_issues
    
    def record_signal_metrics(
        self,
        signals_generated: int,
        signals_aggregated: int,
        conflicts: int = 0
    ):
        """Record signal generation metrics."""
        if self.current_run:
            self.current_run.signals_generated = signals_generated
            self.current_run.signals_aggregated = signals_aggregated
            self.current_run.conflicts_detected = conflicts
    
    def record_output_metrics(
        self,
        recommendations: int,
        notion_pushed: bool = False,
        slack_sent: bool = False
    ):
        """Record output metrics."""
        if self.current_run:
            self.current_run.recommendations_count = recommendations
            self.current_run.notion_pushed = notion_pushed
            self.current_run.slack_sent = slack_sent
    
    def record_error(self):
        """Record an error."""
        if self.current_run:
            self.current_run.error_count += 1
    
    def record_warning(self):
        """Record a warning."""
        if self.current_run:
            self.current_run.warning_count += 1
    
    def finish_run(self, status: str, duration: float):
        """Finish tracking current run."""
        if self.current_run:
            self.current_run.status = status
            self.current_run.duration_seconds = duration
            
            # Save to file
            self._save_metrics(self.current_run)
            
            logger.info(
                f"Metrics recorded: {status}, "
                f"{self.current_run.recommendations_count} recs, "
                f"{duration:.1f}s"
            )
    
    def _get_metrics_file(self, date: datetime) -> Path:
        """Get metrics file path for date."""
        return self.metrics_dir / f"metrics_{date.strftime('%Y%m%d')}.jsonl"
    
    def _save_metrics(self, metrics: PipelineMetrics):
        """Save metrics to file."""
        filepath = self._get_metrics_file(metrics.run_date)
        
        data = {
            "run_date": metrics.run_date.isoformat(),
            "status": metrics.status,
            "duration_seconds": metrics.duration_seconds,
            "stage_timings": metrics.stage_timings,
            "instruments_loaded": metrics.instruments_loaded,
            "data_quality_issues": metrics.data_quality_issues,
            "signals_generated": metrics.signals_generated,
            "signals_aggregated": metrics.signals_aggregated,
            "conflicts_detected": metrics.conflicts_detected,
            "recommendations_count": metrics.recommendations_count,
            "notion_pushed": metrics.notion_pushed,
            "slack_sent": metrics.slack_sent,
            "error_count": metrics.error_count,
            "warning_count": metrics.warning_count,
        }
        
        with open(filepath, "a") as f:
            f.write(json.dumps(data) + "\n")
    
    def load_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Load metrics for date range.
        
        Args:
            start_date: Start of range (default: 7 days ago)
            end_date: End of range (default: today)
        
        Returns:
            List of metrics dicts
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)
        
        metrics = []
        current = start_date
        
        while current <= end_date:
            filepath = self._get_metrics_file(current)
            
            if filepath.exists():
                with open(filepath) as f:
                    for line in f:
                        try:
                            metrics.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            
            current += timedelta(days=1)
        
        return metrics
    
    def get_daily_summary(self, date: datetime) -> DailyMetrics:
        """Get aggregated metrics for a day."""
        metrics = self.load_metrics(date, date)
        
        if not metrics:
            return DailyMetrics(date=date.strftime("%Y-%m-%d"))
        
        runs = len(metrics)
        successful = sum(1 for m in metrics if m["status"] == "SUCCESS")
        failed = sum(1 for m in metrics if m["status"] == "FAILED")
        
        total_signals = sum(m.get("signals_generated", 0) for m in metrics)
        total_recs = sum(m.get("recommendations_count", 0) for m in metrics)
        
        durations = [m.get("duration_seconds", 0) for m in metrics]
        
        return DailyMetrics(
            date=date.strftime("%Y-%m-%d"),
            runs=runs,
            successful_runs=successful,
            failed_runs=failed,
            total_signals=total_signals,
            total_recommendations=total_recs,
            avg_duration_seconds=sum(durations) / len(durations) if durations else 0,
            max_duration_seconds=max(durations) if durations else 0,
        )
    
    def get_trend_data(self, days: int = 7) -> List[DailyMetrics]:
        """Get daily metrics for trend analysis."""
        end_date = datetime.now()
        trend = []
        
        for i in range(days):
            date = end_date - timedelta(days=i)
            daily = self.get_daily_summary(date)
            trend.append(daily)
        
        return list(reversed(trend))
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        
        for filepath in self.metrics_dir.glob("metrics_*.jsonl"):
            try:
                date_str = filepath.stem.replace("metrics_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date < cutoff:
                    filepath.unlink()
                    logger.debug(f"Removed old metrics: {filepath.name}")
            except (ValueError, OSError):
                continue


# Global metrics collector instance
_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
