"""
Outputs Module

Handles all output integrations:
- Notion database integration
- Slack notifications
- Trade card formatting
"""

from .notion_client import (
    NotionClient,
    NotionConfig,
    create_notion_client,
)
from .formatter import (
    TradeCard,
    RecommendationFormatter,
    format_recommendations_report,
)
from .slack_poster import (
    SlackConfig,
    SlackPoster,
    MockSlackPoster,
    create_slack_poster,
    post_to_slack,
)

__all__ = [
    # Notion
    "NotionClient",
    "NotionConfig",
    "create_notion_client",
    
    # Formatter
    "TradeCard",
    "RecommendationFormatter",
    "format_recommendations_report",
    
    # Slack
    "SlackConfig",
    "SlackPoster",
    "MockSlackPoster",
    "create_slack_poster",
    "post_to_slack",
]
