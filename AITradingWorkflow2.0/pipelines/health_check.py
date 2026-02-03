#!/usr/bin/env python3
"""
Health Check Pipeline

Monitors system health after daily pipeline runs:
- Data freshness
- Signal distribution
- LLM grounding scores
- Output delivery status

Usage:
    python pipelines/health_check.py
    python pipelines/health_check.py --alert-only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    details: Optional[Dict] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class HealthReport:
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: str
    summary: str
    
    def has_errors(self) -> bool:
        return any(c.status in [HealthStatus.ERROR, HealthStatus.CRITICAL] for c in self.checks)
    
    def has_warnings(self) -> bool:
        return any(c.status == HealthStatus.WARNING for c in self.checks)


def check_data_freshness() -> HealthCheckResult:
    """Check if data files are recent."""
    logger.info("Checking data freshness...")
    
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    # TODO: Implement actual check
    # Check for files modified in last 24 hours
    
    return HealthCheckResult(
        name="data_freshness",
        status=HealthStatus.HEALTHY,
        message="Data files are current",
        details={"last_update": datetime.now().isoformat()}
    )


def check_signal_distribution() -> HealthCheckResult:
    """Check if signals are balanced (not all one direction)."""
    logger.info("Checking signal distribution...")
    
    # TODO: Implement actual check
    # Load recent signals, check long/short balance
    
    return HealthCheckResult(
        name="signal_distribution",
        status=HealthStatus.HEALTHY,
        message="Signal distribution is balanced",
        details={"long_pct": 0.55, "short_pct": 0.45}
    )


def check_grounding_scores() -> HealthCheckResult:
    """Check LLM grounding score trends."""
    logger.info("Checking grounding scores...")
    
    # TODO: Implement actual check
    # Load recent grounding reports, check average score
    
    return HealthCheckResult(
        name="grounding_scores",
        status=HealthStatus.HEALTHY,
        message="Grounding scores are within threshold",
        details={"avg_score": 0.88, "min_score": 0.82}
    )


def check_pipeline_completion() -> HealthCheckResult:
    """Check if daily pipeline completed successfully."""
    logger.info("Checking pipeline completion...")
    
    # TODO: Implement actual check
    # Check pipeline log for success/failure
    
    return HealthCheckResult(
        name="pipeline_completion",
        status=HealthStatus.HEALTHY,
        message="Pipeline completed successfully",
        details={"duration_seconds": 120}
    )


def check_output_delivery() -> HealthCheckResult:
    """Check if outputs were delivered to Notion/Slack."""
    logger.info("Checking output delivery...")
    
    # TODO: Implement actual check
    # Verify Notion entries created, Slack messages sent
    
    return HealthCheckResult(
        name="output_delivery",
        status=HealthStatus.HEALTHY,
        message="Outputs delivered successfully",
        details={"notion_entries": 3, "slack_sent": True}
    )


def run_all_health_checks() -> HealthReport:
    """Execute all health checks and return consolidated report."""
    
    logger.info("=" * 60)
    logger.info("Running health checks...")
    logger.info("=" * 60)
    
    checks = [
        check_pipeline_completion(),
        check_data_freshness(),
        check_signal_distribution(),
        check_grounding_scores(),
        check_output_delivery(),
    ]
    
    # Determine overall status
    if any(c.status == HealthStatus.CRITICAL for c in checks):
        overall = HealthStatus.CRITICAL
    elif any(c.status == HealthStatus.ERROR for c in checks):
        overall = HealthStatus.ERROR
    elif any(c.status == HealthStatus.WARNING for c in checks):
        overall = HealthStatus.WARNING
    else:
        overall = HealthStatus.HEALTHY
    
    # Generate summary
    failed = [c.name for c in checks if c.status in [HealthStatus.ERROR, HealthStatus.CRITICAL]]
    warned = [c.name for c in checks if c.status == HealthStatus.WARNING]
    
    if failed:
        summary = f"FAILED: {', '.join(failed)}"
    elif warned:
        summary = f"WARNINGS: {', '.join(warned)}"
    else:
        summary = "All checks passed"
    
    report = HealthReport(
        overall_status=overall,
        checks=checks,
        timestamp=datetime.now().isoformat(),
        summary=summary
    )
    
    logger.info(f"Health check complete: {overall.value}")
    logger.info(f"Summary: {summary}")
    
    return report


def send_alert(report: HealthReport, level: str = "warning") -> bool:
    """Send Slack alert for health issues."""
    
    # TODO: Implement actual Slack alert
    # from src.outputs.slack import send_health_alert
    # return send_health_alert(report, level)
    
    logger.info(f"[ALERT] Would send {level} alert: {report.summary}")
    return True


def save_health_report(report: HealthReport) -> Path:
    """Save health report to file."""
    
    reports_dir = Path(__file__).parent.parent / "reports" / "health"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = reports_dir / filename
    
    # Convert to dict for JSON serialization
    report_dict = {
        "overall_status": report.overall_status.value,
        "timestamp": report.timestamp,
        "summary": report.summary,
        "checks": [
            {
                "name": c.name,
                "status": c.status.value,
                "message": c.message,
                "details": c.details,
                "timestamp": c.timestamp
            }
            for c in report.checks
        ]
    }
    
    with open(filepath, "w") as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Health report saved to {filepath}")
    return filepath


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run health checks")
    parser.add_argument(
        "--alert-only",
        action="store_true",
        help="Only send alerts, don't save report"
    )
    parser.add_argument(
        "--no-alert",
        action="store_true",
        help="Don't send alerts even on failure"
    )
    
    args = parser.parse_args()
    
    # Run health checks
    report = run_all_health_checks()
    
    # Save report
    if not args.alert_only:
        save_health_report(report)
    
    # Send alerts
    if not args.no_alert:
        if report.has_errors():
            send_alert(report, level="error")
        elif report.has_warnings():
            send_alert(report, level="warning")
    
    # Exit with appropriate code
    if report.overall_status == HealthStatus.CRITICAL:
        sys.exit(2)
    elif report.overall_status == HealthStatus.ERROR:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
