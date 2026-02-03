"""
Alerter Module

Sends alerts to Slack and other channels when issues are detected.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .health_checks import HealthReport, HealthStatus, AlertLevel


@dataclass
class AlertConfig:
    """Alert configuration."""
    slack_webhook_url: Optional[str] = None
    alert_levels: List[str] = None  # Levels to alert on
    quiet_hours_start: time = time(22, 0)  # 10 PM
    quiet_hours_end: time = time(6, 0)     # 6 AM
    
    def __post_init__(self):
        if self.alert_levels is None:
            self.alert_levels = ["warning", "error", "critical"]
        
        # Try environment variable if not set
        if not self.slack_webhook_url:
            self.slack_webhook_url = os.environ.get("SLACK_MONITORING_WEBHOOK")


class SlackAlerter:
    """
    Sends alerts to Slack.
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize alerter.
        
        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()
        
        if not self.config.slack_webhook_url:
            logger.warning("No Slack webhook configured - alerts will be logged only")
    
    def _is_quiet_hours(self) -> bool:
        """Check if current time is in quiet hours."""
        now = datetime.now().time()
        start = self.config.quiet_hours_start
        end = self.config.quiet_hours_end
        
        # Handle overnight quiet hours (e.g., 22:00 to 06:00)
        if start > end:
            return now >= start or now <= end
        else:
            return start <= now <= end
    
    def _get_status_emoji(self, status: HealthStatus) -> str:
        """Get emoji for health status."""
        return {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.ERROR: "❌",
            HealthStatus.CRITICAL: "🚨",
            HealthStatus.UNKNOWN: "❓",
        }.get(status, "❓")
    
    def _get_alert_color(self, status: HealthStatus) -> str:
        """Get Slack attachment color for status."""
        return {
            HealthStatus.HEALTHY: "good",
            HealthStatus.WARNING: "warning",
            HealthStatus.ERROR: "danger",
            HealthStatus.CRITICAL: "danger",
        }.get(status, "#808080")
    
    def _build_health_alert(self, report: HealthReport) -> Dict:
        """Build Slack message for health report."""
        emoji = self._get_status_emoji(report.overall_status)
        color = self._get_alert_color(report.overall_status)
        
        # Build failed checks text
        failed_checks = [c for c in report.checks if not c.is_healthy]
        checks_text = ""
        
        if failed_checks:
            for check in failed_checks:
                check_emoji = self._get_status_emoji(check.status)
                checks_text += f"{check_emoji} *{check.name}*: {check.message}\n"
        
        # Build message
        message = {
            "text": f"{emoji} Trading Pipeline Alert",
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{emoji} Trading Pipeline Alert"
                            }
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Status:*\n{report.overall_status.value.upper()}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Checks:*\n{report.healthy_count}/{len(report.checks)} healthy"
                                }
                            ]
                        },
                    ]
                }
            ]
        }
        
        # Add failed checks if any
        if checks_text:
            message["attachments"][0]["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Issues Detected:*\n{checks_text}"
                }
            })
        
        # Add timestamp
        message["attachments"][0]["blocks"].append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}"
                }
            ]
        })
        
        return message
    
    def _build_simple_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel,
        details: Optional[Dict] = None
    ) -> Dict:
        """Build simple alert message."""
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }.get(level, "ℹ️")
        
        color = {
            AlertLevel.INFO: "good",
            AlertLevel.WARNING: "warning",
            AlertLevel.ERROR: "danger",
            AlertLevel.CRITICAL: "danger",
        }.get(level, "#808080")
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            },
        ]
        
        if details:
            detail_text = "\n".join(f"• {k}: {v}" for k, v in details.items())
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Details:*\n{detail_text}"
                }
            })
        
        return {
            "text": f"{emoji} {title}",
            "attachments": [{"color": color, "blocks": blocks}]
        }
    
    def send_health_alert(
        self,
        report: HealthReport,
        force: bool = False
    ) -> bool:
        """
        Send health report alert to Slack.
        
        Args:
            report: Health report
            force: Send even during quiet hours
        
        Returns:
            True if sent successfully
        """
        # Check if we should alert for this status
        if report.overall_status.value not in self.config.alert_levels:
            logger.debug(f"Not alerting for status: {report.overall_status.value}")
            return False
        
        # Check quiet hours
        if not force and self._is_quiet_hours():
            logger.info("Skipping alert during quiet hours")
            return False
        
        # Build and send message
        message = self._build_health_alert(report)
        return self._send_to_slack(message)
    
    def send_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel = AlertLevel.WARNING,
        details: Optional[Dict] = None,
        force: bool = False
    ) -> bool:
        """
        Send simple alert to Slack.
        
        Args:
            title: Alert title
            message: Alert message
            level: Alert level
            details: Additional details
            force: Send even during quiet hours
        
        Returns:
            True if sent successfully
        """
        # Check if we should alert for this level
        if level.value not in self.config.alert_levels:
            return False
        
        # Check quiet hours
        if not force and self._is_quiet_hours():
            logger.info("Skipping alert during quiet hours")
            return False
        
        payload = self._build_simple_alert(title, message, level, details)
        return self._send_to_slack(payload)
    
    def _send_to_slack(self, payload: Dict) -> bool:
        """Send payload to Slack webhook."""
        if not self.config.slack_webhook_url:
            logger.info(f"Alert (no webhook): {payload.get('text', 'Unknown')}")
            return False
        
        if not HAS_REQUESTS:
            logger.error("requests library not installed")
            return False
        
        try:
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Alert sent to Slack")
                return True
            else:
                logger.error(f"Slack API error: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class MockAlerter:
    """Mock alerter for testing."""
    
    def __init__(self):
        self.alerts_sent: List[Dict] = []
    
    def send_health_alert(self, report: HealthReport, force: bool = False) -> bool:
        self.alerts_sent.append({
            "type": "health",
            "status": report.overall_status.value,
            "timestamp": datetime.now().isoformat(),
        })
        return True
    
    def send_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel = AlertLevel.WARNING,
        details: Optional[Dict] = None,
        force: bool = False
    ) -> bool:
        self.alerts_sent.append({
            "type": "simple",
            "title": title,
            "message": message,
            "level": level.value,
            "timestamp": datetime.now().isoformat(),
        })
        return True


def create_alerter(mock: bool = False) -> SlackAlerter:
    """Create alerter instance."""
    if mock:
        return MockAlerter()
    return SlackAlerter()
