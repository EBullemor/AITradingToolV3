"""
Slack Poster Module

Posts trading recommendations and alerts to Slack channels.
Supports both webhook and Bot API methods.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class SlackConfig:
    """Configuration for Slack integration."""
    webhook_url: Optional[str] = None
    channel: str = "#trading-ideas"
    username: str = "Trading Bot"
    icon_emoji: str = ":chart_with_upwards_trend:"
    
    @classmethod
    def from_env(cls) -> "SlackConfig":
        """Load configuration from environment variables."""
        return cls(
            webhook_url=os.environ.get("SLACK_WEBHOOK_URL"),
            channel=os.environ.get("SLACK_CHANNEL", "#trading-ideas"),
        )


class SlackPoster:
    """
    Posts messages to Slack.
    
    Uses incoming webhooks for simplicity.
    """
    
    def __init__(self, config: Optional[SlackConfig] = None):
        """
        Initialize Slack poster.
        
        Args:
            config: Slack configuration (loads from env if not provided)
        """
        self.config = config or SlackConfig.from_env()
        
        if not self.config.webhook_url:
            logger.warning("No Slack webhook URL configured")
    
    def _format_recommendation_text(self, rec: Dict) -> str:
        """Format a single recommendation as text."""
        direction = rec.get("direction", "UNKNOWN")
        instrument = rec.get("instrument", "???")
        confidence = rec.get("confidence", 0)
        confidence_pct = int(confidence * 100) if confidence <= 1 else confidence
        
        # Direction emoji
        emoji = "🟢" if direction == "LONG" else "🔴"
        
        # Confidence label
        if confidence_pct >= 70:
            conf_label = "HIGH"
        elif confidence_pct >= 50:
            conf_label = "MEDIUM"
        else:
            conf_label = "LOW"
        
        # Build line
        line = f"{emoji} *{direction} {instrument}*"
        
        # Add price levels
        entry = rec.get("entry_price")
        stop = rec.get("stop_loss")
        target = rec.get("take_profit_1")
        
        if entry:
            line += f" @ {entry:.5f}" if entry < 10 else f" @ {entry:.2f}"
        if stop:
            line += f", Stop: {stop:.5f}" if stop < 10 else f", Stop: {stop:.2f}"
        if target:
            line += f", Target: {target:.5f}" if target < 10 else f", Target: {target:.2f}"
        
        line += f" ({conf_label} confidence)"
        
        # Add rationale
        rationale = rec.get("rationale")
        if rationale:
            # Truncate long rationales
            if len(rationale) > 100:
                rationale = rationale[:97] + "..."
            line += f"\n   💡 _{rationale}_"
        
        return line
    
    def _build_recommendations_message(
        self,
        recommendations: List[Dict],
        notion_url: Optional[str] = None,
        run_date: Optional[datetime] = None
    ) -> Dict:
        """Build Slack message payload for recommendations."""
        date_str = (run_date or datetime.now()).strftime("%Y-%m-%d")
        
        # Header
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚀 Daily Recommendations ({date_str})",
                    "emoji": True
                }
            },
            {
                "type": "divider"
            }
        ]
        
        # Recommendations
        if not recommendations:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "_No recommendations generated today._"
                }
            })
        else:
            for i, rec in enumerate(recommendations[:10], 1):  # Limit to 10
                rec_text = self._format_recommendation_text(rec)
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{i}.* {rec_text}"
                    }
                })
        
        # Summary stats
        if recommendations:
            long_count = sum(1 for r in recommendations if r.get("direction") == "LONG")
            short_count = len(recommendations) - long_count
            
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📊 {len(recommendations)} total | 🟢 {long_count} long | 🔴 {short_count} short"
                    }
                ]
            })
        
        # Notion link
        if notion_url:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"👉 <{notion_url}|Review in Notion>"
                }
            })
        
        return {
            "channel": self.config.channel,
            "username": self.config.username,
            "icon_emoji": self.config.icon_emoji,
            "blocks": blocks,
        }
    
    def _build_simple_text_message(
        self,
        recommendations: List[Dict],
        notion_url: Optional[str] = None,
        run_date: Optional[datetime] = None
    ) -> str:
        """Build simple text message (fallback for webhooks that don't support blocks)."""
        date_str = (run_date or datetime.now()).strftime("%Y-%m-%d")
        
        lines = [
            f"🚀 *Daily Recommendations ({date_str})*",
            "",
        ]
        
        if not recommendations:
            lines.append("_No recommendations generated today._")
        else:
            for i, rec in enumerate(recommendations[:10], 1):
                rec_text = self._format_recommendation_text(rec)
                lines.append(f"{i}. {rec_text}")
                lines.append("")
        
        if notion_url:
            lines.append(f"👉 Review in Notion: {notion_url}")
        
        return "\n".join(lines)
    
    def post_recommendations(
        self,
        recommendations: List[Dict],
        notion_url: Optional[str] = None,
        run_date: Optional[datetime] = None,
        use_blocks: bool = True
    ) -> bool:
        """
        Post recommendations to Slack.
        
        Args:
            recommendations: List of recommendation dicts
            notion_url: Optional link to Notion page
            run_date: Date of recommendations
            use_blocks: Use Block Kit formatting
        
        Returns:
            True if successful
        """
        if not self.config.webhook_url:
            logger.info(f"Slack (no webhook): {len(recommendations)} recommendations")
            return False
        
        if not HAS_REQUESTS:
            logger.error("requests library not installed")
            return False
        
        try:
            if use_blocks:
                payload = self._build_recommendations_message(
                    recommendations, notion_url, run_date
                )
            else:
                text = self._build_simple_text_message(
                    recommendations, notion_url, run_date
                )
                payload = {"text": text}
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Posted {len(recommendations)} recommendations to Slack")
                return True
            else:
                logger.error(f"Slack post failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to post to Slack: {e}")
            return False
    
    def post_message(
        self,
        text: str,
        channel: Optional[str] = None
    ) -> bool:
        """
        Post simple text message to Slack.
        
        Args:
            text: Message text
            channel: Override channel
        
        Returns:
            True if successful
        """
        if not self.config.webhook_url:
            logger.info(f"Slack (no webhook): {text[:100]}")
            return False
        
        if not HAS_REQUESTS:
            return False
        
        try:
            payload = {
                "text": text,
                "channel": channel or self.config.channel,
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
        
        except Exception as e:
            logger.error(f"Failed to post to Slack: {e}")
            return False
    
    def post_alert(
        self,
        title: str,
        message: str,
        level: str = "warning",
        details: Optional[Dict] = None
    ) -> bool:
        """
        Post alert to Slack.
        
        Args:
            title: Alert title
            message: Alert message
            level: Alert level (info, warning, error, critical)
            details: Additional details
        
        Returns:
            True if successful
        """
        emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨",
        }.get(level, "ℹ️")
        
        color = {
            "info": "good",
            "warning": "warning",
            "error": "danger",
            "critical": "danger",
        }.get(level, "#808080")
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            }
        ]
        
        if details:
            detail_text = "\n".join(f"• *{k}:* {v}" for k, v in details.items())
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Details:*\n{detail_text}"
                }
            })
        
        payload = {
            "channel": self.config.channel,
            "username": self.config.username,
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks
                }
            ]
        }
        
        if not self.config.webhook_url:
            logger.info(f"Slack alert (no webhook): {title}")
            return False
        
        try:
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to post alert to Slack: {e}")
            return False


class MockSlackPoster:
    """Mock Slack poster for testing."""
    
    def __init__(self):
        self.messages_sent: List[Dict] = []
    
    def post_recommendations(
        self,
        recommendations: List[Dict],
        notion_url: Optional[str] = None,
        run_date: Optional[datetime] = None,
        use_blocks: bool = True
    ) -> bool:
        self.messages_sent.append({
            "type": "recommendations",
            "count": len(recommendations),
            "notion_url": notion_url,
            "timestamp": datetime.now().isoformat(),
        })
        logger.info(f"Mock Slack: Posted {len(recommendations)} recommendations")
        return True
    
    def post_message(self, text: str, channel: Optional[str] = None) -> bool:
        self.messages_sent.append({
            "type": "message",
            "text": text,
            "channel": channel,
        })
        return True
    
    def post_alert(
        self,
        title: str,
        message: str,
        level: str = "warning",
        details: Optional[Dict] = None
    ) -> bool:
        self.messages_sent.append({
            "type": "alert",
            "title": title,
            "message": message,
            "level": level,
        })
        return True


def create_slack_poster(mock: bool = False) -> SlackPoster:
    """Factory function to create Slack poster."""
    if mock:
        return MockSlackPoster()
    return SlackPoster()


def post_to_slack(
    recommendations: List[Dict],
    notion_ids: Optional[List[str]] = None,
    notion_base_url: str = "https://notion.so"
) -> bool:
    """
    Convenience function to post recommendations to Slack.
    
    Args:
        recommendations: List of recommendation dicts
        notion_ids: Notion page IDs created
        notion_base_url: Base URL for Notion links
    
    Returns:
        True if successful
    """
    poster = SlackPoster()
    
    # Build Notion URL if IDs provided
    notion_url = None
    if notion_ids and len(notion_ids) > 0:
        notion_url = f"{notion_base_url}/{notion_ids[0]}"
    
    return poster.post_recommendations(recommendations, notion_url)
