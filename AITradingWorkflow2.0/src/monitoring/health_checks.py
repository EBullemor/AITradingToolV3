"""
Health Checks Module

Individual health check functions for monitoring system status:
- Pipeline completion
- Data freshness
- Signal quality
- Output delivery
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY
    
    @property
    def needs_alert(self) -> bool:
        return self.status in [HealthStatus.WARNING, HealthStatus.ERROR, HealthStatus.CRITICAL]
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HealthReport:
    """Consolidated health report."""
    checks: List[HealthCheckResult]
    overall_status: HealthStatus
    generated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def has_errors(self) -> bool:
        return any(c.status in [HealthStatus.ERROR, HealthStatus.CRITICAL] for c in self.checks)
    
    @property
    def has_warnings(self) -> bool:
        return any(c.status == HealthStatus.WARNING for c in self.checks)
    
    @property
    def healthy_count(self) -> int:
        return sum(1 for c in self.checks if c.is_healthy)
    
    def to_dict(self) -> Dict:
        return {
            "overall_status": self.overall_status.value,
            "healthy_count": self.healthy_count,
            "total_checks": len(self.checks),
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "checks": [c.to_dict() for c in self.checks],
            "generated_at": self.generated_at.isoformat(),
        }
    
    def format_summary(self) -> str:
        """Format as human-readable summary."""
        emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.ERROR: "❌",
            HealthStatus.CRITICAL: "🚨",
            HealthStatus.UNKNOWN: "❓",
        }
        
        lines = [
            f"Health Report - {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            f"Overall: {emoji[self.overall_status]} {self.overall_status.value.upper()}",
            f"Checks: {self.healthy_count}/{len(self.checks)} healthy",
            "",
        ]
        
        for check in self.checks:
            lines.append(f"{emoji[check.status]} {check.name}: {check.message}")
        
        return "\n".join(lines)


# =============================================================================
# Health Check Functions
# =============================================================================

def check_pipeline_health(
    run_result: Optional[Dict] = None,
    max_runtime_minutes: int = 30,
    required_stages: Optional[List[str]] = None
) -> HealthCheckResult:
    """
    Check if pipeline completed successfully.
    
    Args:
        run_result: Pipeline run result dict
        max_runtime_minutes: Maximum acceptable runtime
        required_stages: List of required stages
    """
    if run_result is None:
        return HealthCheckResult(
            name="Pipeline Health",
            status=HealthStatus.UNKNOWN,
            message="No run result provided",
        )
    
    status = run_result.get("status", "UNKNOWN")
    duration = run_result.get("duration_seconds", 0)
    errors = run_result.get("errors", [])
    
    # Check status
    if status == "FAILED":
        return HealthCheckResult(
            name="Pipeline Health",
            status=HealthStatus.ERROR,
            message=f"Pipeline failed: {errors[0] if errors else 'Unknown error'}",
            details={"errors": errors, "duration": duration},
        )
    
    if status == "PARTIAL":
        return HealthCheckResult(
            name="Pipeline Health",
            status=HealthStatus.WARNING,
            message=f"Pipeline completed with warnings",
            details={"errors": errors, "duration": duration},
        )
    
    # Check runtime
    if duration > max_runtime_minutes * 60:
        return HealthCheckResult(
            name="Pipeline Health",
            status=HealthStatus.WARNING,
            message=f"Pipeline slow: {duration/60:.1f} min (max: {max_runtime_minutes})",
            details={"duration": duration},
        )
    
    return HealthCheckResult(
        name="Pipeline Health",
        status=HealthStatus.HEALTHY,
        message=f"Pipeline completed in {duration:.1f}s",
        details={"duration": duration},
    )


def check_data_freshness(
    data_dir: Path,
    max_stale_hours: int = 24,
    required_files: Optional[List[str]] = None
) -> HealthCheckResult:
    """
    Check if data files are fresh.
    
    Args:
        data_dir: Directory containing data files
        max_stale_hours: Maximum age of data files
        required_files: List of required filenames
    """
    if not data_dir.exists():
        return HealthCheckResult(
            name="Data Freshness",
            status=HealthStatus.ERROR,
            message=f"Data directory not found: {data_dir}",
        )
    
    stale_files = []
    missing_files = []
    now = datetime.now()
    max_age = timedelta(hours=max_stale_hours)
    
    # Check for data files
    csv_files = list(data_dir.glob("**/*.csv"))
    
    if not csv_files:
        return HealthCheckResult(
            name="Data Freshness",
            status=HealthStatus.WARNING,
            message="No data files found",
            details={"directory": str(data_dir)},
        )
    
    for file_path in csv_files:
        file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
        if file_age > max_age:
            stale_files.append({
                "file": file_path.name,
                "age_hours": file_age.total_seconds() / 3600,
            })
    
    # Check required files
    if required_files:
        existing = {f.name for f in csv_files}
        for req in required_files:
            if req not in existing:
                missing_files.append(req)
    
    if missing_files:
        return HealthCheckResult(
            name="Data Freshness",
            status=HealthStatus.ERROR,
            message=f"Missing required files: {', '.join(missing_files)}",
            details={"missing": missing_files},
        )
    
    if stale_files:
        return HealthCheckResult(
            name="Data Freshness",
            status=HealthStatus.WARNING,
            message=f"{len(stale_files)} stale files (>{max_stale_hours}h old)",
            details={"stale_files": stale_files},
        )
    
    return HealthCheckResult(
        name="Data Freshness",
        status=HealthStatus.HEALTHY,
        message=f"{len(csv_files)} data files are fresh",
        details={"file_count": len(csv_files)},
    )


def check_signal_distribution(
    signals: List[Dict],
    imbalance_threshold: float = 0.8,
    min_signals: int = 3
) -> HealthCheckResult:
    """
    Check if signals are suspiciously one-directional.
    
    Args:
        signals: List of signal dicts with 'direction' key
        imbalance_threshold: Ratio above which we flag imbalance
        min_signals: Minimum signals to analyze
    """
    if not signals:
        return HealthCheckResult(
            name="Signal Distribution",
            status=HealthStatus.HEALTHY,
            message="No signals to analyze",
        )
    
    if len(signals) < min_signals:
        return HealthCheckResult(
            name="Signal Distribution",
            status=HealthStatus.HEALTHY,
            message=f"Only {len(signals)} signals (min: {min_signals} for analysis)",
        )
    
    long_count = sum(1 for s in signals if s.get("direction") == "LONG")
    short_count = len(signals) - long_count
    
    total = len(signals)
    long_ratio = long_count / total
    short_ratio = short_count / total
    
    max_ratio = max(long_ratio, short_ratio)
    direction = "LONG" if long_ratio > short_ratio else "SHORT"
    
    if max_ratio >= imbalance_threshold:
        return HealthCheckResult(
            name="Signal Distribution",
            status=HealthStatus.WARNING,
            message=f"Signals heavily skewed {direction}: {max_ratio:.0%}",
            details={
                "long_count": long_count,
                "short_count": short_count,
                "long_ratio": long_ratio,
                "short_ratio": short_ratio,
            },
        )
    
    return HealthCheckResult(
        name="Signal Distribution",
        status=HealthStatus.HEALTHY,
        message=f"Signals balanced: {long_count} long, {short_count} short",
        details={
            "long_count": long_count,
            "short_count": short_count,
        },
    )


def check_recommendation_output(
    recommendations: List[Dict],
    min_per_day: int = 0,
    max_per_day: int = 15
) -> HealthCheckResult:
    """
    Check recommendation output is reasonable.
    
    Args:
        recommendations: List of recommendations generated
        min_per_day: Minimum expected
        max_per_day: Maximum expected
    """
    count = len(recommendations)
    
    if count < min_per_day:
        return HealthCheckResult(
            name="Recommendation Output",
            status=HealthStatus.WARNING,
            message=f"Low recommendations: {count} (min: {min_per_day})",
            details={"count": count},
        )
    
    if count > max_per_day:
        return HealthCheckResult(
            name="Recommendation Output",
            status=HealthStatus.WARNING,
            message=f"High recommendations: {count} (max: {max_per_day})",
            details={"count": count},
        )
    
    return HealthCheckResult(
        name="Recommendation Output",
        status=HealthStatus.HEALTHY,
        message=f"{count} recommendations generated",
        details={"count": count},
    )


def check_api_health(
    api_name: str,
    last_success: Optional[datetime] = None,
    error_count: int = 0,
    max_errors: int = 3,
    max_stale_hours: int = 1
) -> HealthCheckResult:
    """
    Check external API health.
    
    Args:
        api_name: Name of API (e.g., "Notion", "Claude")
        last_success: Last successful API call
        error_count: Recent error count
        max_errors: Maximum tolerable errors
        max_stale_hours: Maximum time since last success
    """
    if error_count >= max_errors:
        return HealthCheckResult(
            name=f"{api_name} API",
            status=HealthStatus.ERROR,
            message=f"{error_count} consecutive errors",
            details={"error_count": error_count},
        )
    
    if last_success:
        age = datetime.now() - last_success
        if age > timedelta(hours=max_stale_hours):
            return HealthCheckResult(
                name=f"{api_name} API",
                status=HealthStatus.WARNING,
                message=f"No successful call in {age.total_seconds()/3600:.1f}h",
                details={"last_success": last_success.isoformat()},
            )
    
    return HealthCheckResult(
        name=f"{api_name} API",
        status=HealthStatus.HEALTHY,
        message="API responding normally",
        details={"error_count": error_count},
    )


def check_disk_space(
    path: Path = Path("."),
    min_free_gb: float = 1.0
) -> HealthCheckResult:
    """
    Check available disk space.
    
    Args:
        path: Path to check
        min_free_gb: Minimum required free space
    """
    import shutil
    
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_pct = (usage.used / usage.total) * 100
        
        if free_gb < min_free_gb:
            return HealthCheckResult(
                name="Disk Space",
                status=HealthStatus.WARNING,
                message=f"Low disk space: {free_gb:.1f}GB free",
                details={"free_gb": free_gb, "used_pct": used_pct},
            )
        
        return HealthCheckResult(
            name="Disk Space",
            status=HealthStatus.HEALTHY,
            message=f"{free_gb:.1f}GB free ({used_pct:.0f}% used)",
            details={"free_gb": free_gb, "total_gb": total_gb},
        )
    
    except Exception as e:
        return HealthCheckResult(
            name="Disk Space",
            status=HealthStatus.UNKNOWN,
            message=f"Could not check disk: {e}",
        )


# =============================================================================
# Consolidated Health Check
# =============================================================================

def run_all_health_checks(
    pipeline_result: Optional[Dict] = None,
    data_dir: Optional[Path] = None,
    signals: Optional[List[Dict]] = None,
    recommendations: Optional[List[Dict]] = None,
) -> HealthReport:
    """
    Run all health checks and generate report.
    
    Args:
        pipeline_result: Result from daily pipeline
        data_dir: Path to data directory
        signals: List of generated signals
        recommendations: List of recommendations
    
    Returns:
        Consolidated HealthReport
    """
    checks = []
    
    # Pipeline health
    checks.append(check_pipeline_health(pipeline_result))
    
    # Data freshness
    if data_dir:
        checks.append(check_data_freshness(data_dir))
    
    # Signal distribution
    if signals:
        checks.append(check_signal_distribution(signals))
    
    # Recommendation output
    if recommendations is not None:
        checks.append(check_recommendation_output(recommendations))
    
    # Disk space
    checks.append(check_disk_space())
    
    # Determine overall status
    if any(c.status == HealthStatus.CRITICAL for c in checks):
        overall = HealthStatus.CRITICAL
    elif any(c.status == HealthStatus.ERROR for c in checks):
        overall = HealthStatus.ERROR
    elif any(c.status == HealthStatus.WARNING for c in checks):
        overall = HealthStatus.WARNING
    elif all(c.status == HealthStatus.HEALTHY for c in checks):
        overall = HealthStatus.HEALTHY
    else:
        overall = HealthStatus.UNKNOWN
    
    return HealthReport(
        checks=checks,
        overall_status=overall,
    )
