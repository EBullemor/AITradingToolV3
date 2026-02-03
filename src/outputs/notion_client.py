"""
Notion Client Module

Handles pushing trading recommendations to Notion databases.
Supports creating recommendations, updating status, and logging trades.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from loguru import logger

try:
    from notion_client import Client
    from notion_client.errors import APIResponseError
    HAS_NOTION = True
except ImportError:
    HAS_NOTION = False
    logger.warning("notion-client not installed. Install with: pip install notion-client")


class RecommendationStatus(Enum):
    """Status values for recommendations."""
    NEW = "New"
    REVIEWING = "Reviewing"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    EXECUTED = "Executed"
    EXPIRED = "Expired"
    CLOSED = "Closed"


@dataclass
class NotionConfig:
    """Configuration for Notion integration."""
    api_key: str
    recommendations_db_id: str
    trades_db_id: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "NotionConfig":
        """Load configuration from environment variables."""
        api_key = os.environ.get("NOTION_API_KEY", "")
        recommendations_db = os.environ.get("NOTION_RECOMMENDATIONS_DB", "")
        trades_db = os.environ.get("NOTION_TRADES_DB", "")
        
        if not api_key:
            raise ValueError("NOTION_API_KEY environment variable not set")
        if not recommendations_db:
            raise ValueError("NOTION_RECOMMENDATIONS_DB environment variable not set")
        
        return cls(
            api_key=api_key,
            recommendations_db_id=recommendations_db,
            trades_db_id=trades_db if trades_db else None,
        )


class NotionRecommendationClient:
    """
    Client for pushing trading recommendations to Notion.
    
    Requires:
    - Notion integration with API key
    - Recommendations database shared with integration
    
    Database Schema (expected):
    - Title (title): Instrument name
    - Direction (select): LONG/SHORT
    - Confidence (number): 0-100
    - Entry (number): Entry price
    - Stop Loss (number): Stop loss price
    - Target (number): Take profit price
    - Status (select): New/Reviewing/Approved/Rejected/Executed/Expired
    - Strategy (multi_select): Contributing strategies
    - Rationale (rich_text): Trade rationale
    - Generated (date): When recommendation was created
    - Signal ID (rich_text): Unique identifier
    """
    
    def __init__(self, config: Optional[NotionConfig] = None):
        """
        Initialize Notion client.
        
        Args:
            config: Notion configuration (loads from env if not provided)
        """
        if not HAS_NOTION:
            raise ImportError(
                "notion-client required. Install with: pip install notion-client"
            )
        
        self.config = config or NotionConfig.from_env()
        self.client = Client(auth=self.config.api_key)
        
        logger.info("Notion client initialized")
    
    def _build_recommendation_properties(
        self,
        recommendation: Dict
    ) -> Dict[str, Any]:
        """
        Build Notion properties from recommendation dict.
        
        Args:
            recommendation: Recommendation data (from AggregatedSignal.to_dict())
        
        Returns:
            Notion-formatted properties dict
        """
        properties = {
            # Title - Instrument
            "Title": {
                "title": [
                    {
                        "text": {
                            "content": recommendation.get("instrument", "Unknown")
                        }
                    }
                ]
            },
            
            # Direction - Select
            "Direction": {
                "select": {
                    "name": recommendation.get("direction", "LONG")
                }
            },
            
            # Confidence - Number (as percentage)
            "Confidence": {
                "number": round(recommendation.get("confidence", 0) * 100, 1)
            },
            
            # Status - Select (default to New)
            "Status": {
                "select": {
                    "name": RecommendationStatus.NEW.value
                }
            },
            
            # Signal ID - Rich text
            "Signal ID": {
                "rich_text": [
                    {
                        "text": {
                            "content": recommendation.get("signal_id", "")
                        }
                    }
                ]
            },
        }
        
        # Optional: Entry price
        if recommendation.get("entry_price"):
            properties["Entry"] = {
                "number": recommendation["entry_price"]
            }
        
        # Optional: Stop loss
        if recommendation.get("stop_loss"):
            properties["Stop Loss"] = {
                "number": recommendation["stop_loss"]
            }
        
        # Optional: Take profit
        if recommendation.get("take_profit_1"):
            properties["Target"] = {
                "number": recommendation["take_profit_1"]
            }
        
        # Optional: Risk/Reward
        if recommendation.get("risk_reward_ratio"):
            properties["R:R"] = {
                "number": round(recommendation["risk_reward_ratio"], 2)
            }
        
        # Optional: Contributing strategies
        if recommendation.get("contributing_pods"):
            properties["Strategy"] = {
                "multi_select": [
                    {"name": pod} for pod in recommendation["contributing_pods"]
                ]
            }
        
        # Optional: Rationale
        if recommendation.get("rationale"):
            rationale = recommendation["rationale"][:2000]  # Notion limit
            properties["Rationale"] = {
                "rich_text": [
                    {
                        "text": {
                            "content": rationale
                        }
                    }
                ]
            }
        
        # Optional: Generated date
        if recommendation.get("aggregated_at"):
            # Parse ISO format string
            date_str = recommendation["aggregated_at"]
            if isinstance(date_str, str):
                properties["Generated"] = {
                    "date": {
                        "start": date_str[:10]  # Just the date part
                    }
                }
        
        # Optional: Conflict flag
        if recommendation.get("conflict_flag"):
            properties["Conflicted"] = {
                "checkbox": recommendation["conflict_flag"]
            }
        
        return properties
    
    def create_recommendation(
        self,
        recommendation: Dict
    ) -> str:
        """
        Create a recommendation entry in Notion.
        
        Args:
            recommendation: Recommendation data (from AggregatedSignal.to_dict())
        
        Returns:
            Notion page ID of created entry
        """
        properties = self._build_recommendation_properties(recommendation)
        
        try:
            response = self.client.pages.create(
                parent={"database_id": self.config.recommendations_db_id},
                properties=properties,
            )
            
            page_id = response["id"]
            logger.info(
                f"Created recommendation in Notion: "
                f"{recommendation.get('instrument')} - {page_id}"
            )
            
            return page_id
            
        except APIResponseError as e:
            logger.error(f"Failed to create recommendation: {e}")
            raise
    
    def create_recommendations_batch(
        self,
        recommendations: List[Dict]
    ) -> List[str]:
        """
        Create multiple recommendations in Notion.
        
        Args:
            recommendations: List of recommendation dicts
        
        Returns:
            List of created page IDs
        """
        page_ids = []
        
        for rec in recommendations:
            try:
                page_id = self.create_recommendation(rec)
                page_ids.append(page_id)
            except Exception as e:
                logger.error(
                    f"Failed to create recommendation for "
                    f"{rec.get('instrument')}: {e}"
                )
                page_ids.append(None)
        
        logger.info(
            f"Created {len([p for p in page_ids if p])} of "
            f"{len(recommendations)} recommendations"
        )
        
        return page_ids
    
    def update_recommendation_status(
        self,
        page_id: str,
        status: RecommendationStatus,
        notes: Optional[str] = None
    ) -> bool:
        """
        Update the status of a recommendation.
        
        Args:
            page_id: Notion page ID
            status: New status
            notes: Optional notes to append
        
        Returns:
            True if successful
        """
        properties = {
            "Status": {
                "select": {
                    "name": status.value
                }
            }
        }
        
        if notes:
            properties["Notes"] = {
                "rich_text": [
                    {
                        "text": {
                            "content": notes[:2000]
                        }
                    }
                ]
            }
        
        try:
            self.client.pages.update(
                page_id=page_id,
                properties=properties,
            )
            
            logger.info(f"Updated recommendation status: {page_id} -> {status.value}")
            return True
            
        except APIResponseError as e:
            logger.error(f"Failed to update recommendation: {e}")
            return False
    
    def get_recommendation(self, page_id: str) -> Optional[Dict]:
        """
        Retrieve a recommendation from Notion.
        
        Args:
            page_id: Notion page ID
        
        Returns:
            Recommendation data or None
        """
        try:
            response = self.client.pages.retrieve(page_id=page_id)
            return response
        except APIResponseError as e:
            logger.error(f"Failed to retrieve recommendation: {e}")
            return None
    
    def query_recommendations(
        self,
        status: Optional[RecommendationStatus] = None,
        instrument: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Query recommendations from Notion.
        
        Args:
            status: Filter by status
            instrument: Filter by instrument
            limit: Maximum results
        
        Returns:
            List of recommendation pages
        """
        filters = []
        
        if status:
            filters.append({
                "property": "Status",
                "select": {
                    "equals": status.value
                }
            })
        
        if instrument:
            filters.append({
                "property": "Title",
                "title": {
                    "contains": instrument
                }
            })
        
        filter_obj = None
        if len(filters) == 1:
            filter_obj = filters[0]
        elif len(filters) > 1:
            filter_obj = {"and": filters}
        
        try:
            kwargs = {
                "database_id": self.config.recommendations_db_id,
                "page_size": min(limit, 100),
            }
            if filter_obj:
                kwargs["filter"] = filter_obj
            
            response = self.client.databases.query(**kwargs)
            return response.get("results", [])
            
        except APIResponseError as e:
            logger.error(f"Failed to query recommendations: {e}")
            return []


class NotionTradeLogger:
    """
    Client for logging executed trades to Notion.
    
    Database Schema (expected):
    - Title (title): Trade description
    - Instrument (select): Trading instrument
    - Direction (select): LONG/SHORT
    - Entry Price (number): Actual entry
    - Exit Price (number): Actual exit
    - P&L (number): Profit/loss amount
    - P&L % (number): Percentage return
    - Entry Date (date): When entered
    - Exit Date (date): When exited
    - Recommendation (relation): Link to recommendation
    - Notes (rich_text): Trade notes
    """
    
    def __init__(self, config: Optional[NotionConfig] = None):
        """Initialize trade logger."""
        if not HAS_NOTION:
            raise ImportError("notion-client required")
        
        self.config = config or NotionConfig.from_env()
        
        if not self.config.trades_db_id:
            raise ValueError("NOTION_TRADES_DB environment variable not set")
        
        self.client = Client(auth=self.config.api_key)
        logger.info("Notion trade logger initialized")
    
    def log_trade(
        self,
        instrument: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        entry_date: datetime,
        exit_date: datetime,
        recommendation_page_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> str:
        """
        Log a completed trade.
        
        Returns:
            Notion page ID of trade log entry
        """
        # Calculate P&L
        if direction.upper() == "LONG":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        
        title = f"{direction} {instrument} {entry_date.strftime('%Y-%m-%d')}"
        
        properties = {
            "Title": {
                "title": [{"text": {"content": title}}]
            },
            "Instrument": {
                "select": {"name": instrument}
            },
            "Direction": {
                "select": {"name": direction.upper()}
            },
            "Entry Price": {
                "number": entry_price
            },
            "Exit Price": {
                "number": exit_price
            },
            "P&L %": {
                "number": round(pnl_pct, 2)
            },
            "Entry Date": {
                "date": {"start": entry_date.strftime("%Y-%m-%d")}
            },
            "Exit Date": {
                "date": {"start": exit_date.strftime("%Y-%m-%d")}
            },
        }
        
        if recommendation_page_id:
            properties["Recommendation"] = {
                "relation": [{"id": recommendation_page_id}]
            }
        
        if notes:
            properties["Notes"] = {
                "rich_text": [{"text": {"content": notes[:2000]}}]
            }
        
        try:
            response = self.client.pages.create(
                parent={"database_id": self.config.trades_db_id},
                properties=properties,
            )
            
            page_id = response["id"]
            logger.info(f"Logged trade: {title} - {page_id}")
            
            return page_id
            
        except APIResponseError as e:
            logger.error(f"Failed to log trade: {e}")
            raise


class MockNotionClient:
    """Mock Notion client for testing without API calls."""
    
    def __init__(self):
        self.created_pages = []
        self.updated_pages = []
    
    def create_recommendation(self, recommendation: Dict) -> str:
        """Create mock recommendation."""
        page_id = f"mock_{len(self.created_pages)}"
        self.created_pages.append({
            "id": page_id,
            "data": recommendation,
        })
        logger.info(f"Mock: Created recommendation {page_id}")
        return page_id
    
    def create_recommendations_batch(self, recommendations: List[Dict]) -> List[str]:
        """Create mock recommendations batch."""
        return [self.create_recommendation(r) for r in recommendations]
    
    def update_recommendation_status(
        self,
        page_id: str,
        status: RecommendationStatus,
        notes: Optional[str] = None
    ) -> bool:
        """Update mock recommendation."""
        self.updated_pages.append({
            "id": page_id,
            "status": status.value,
            "notes": notes,
        })
        logger.info(f"Mock: Updated {page_id} -> {status.value}")
        return True


def create_notion_client(
    mock: bool = False,
    config: Optional[NotionConfig] = None
) -> NotionRecommendationClient:
    """
    Factory function to create Notion client.
    
    Args:
        mock: Use mock client for testing
        config: Optional configuration
    
    Returns:
        Notion client instance
    """
    if mock:
        return MockNotionClient()
    
    return NotionRecommendationClient(config=config)
