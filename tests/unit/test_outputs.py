"""
Unit Tests for Outputs Module

Tests Notion client, formatter, and trade cards.
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies.base import SignalDirection
from src.aggregator import AggregatedSignal
from src.outputs import (
    RecommendationFormatter,
    TradeCard,
    format_recommendations_report,
    MockNotionClient,
    RecommendationStatus,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_signal():
    """Create a sample aggregated signal."""
    return AggregatedSignal(
        instrument="EURUSD",
        direction=SignalDirection.LONG,
        confidence=0.72,
        contributing_pods=["fx_carry_momentum", "cross_asset_risk"],
        contributing_signals=[],
        entry_price=1.1050,
        stop_loss=1.1000,
        take_profit_1=1.1150,
        risk_reward_ratio=2.0,
        rationale="Strong carry signal with momentum confirmation in low vol regime.",
        key_factors=["Carry score: 2.1", "Momentum: positive", "Regime: LOW_VOL"],
        conflict_flag=False,
        regime="LOW_VOL",
        aggregated_at=datetime(2026, 2, 1, 10, 30),
        signal_id="AGG_EURUSD_20260201103000",
    )


@pytest.fixture
def conflicted_signal():
    """Create a conflicted signal."""
    return AggregatedSignal(
        instrument="GBPUSD",
        direction=SignalDirection.SHORT,
        confidence=0.55,
        contributing_pods=["cross_asset_risk"],
        contributing_signals=[],
        entry_price=1.2700,
        stop_loss=1.2750,
        take_profit_1=1.2600,
        conflict_flag=True,
        conflict_details="LONG signals from fx_carry_momentum vs SHORT from cross_asset_risk",
        aggregated_at=datetime(2026, 2, 1, 10, 30),
    )


@pytest.fixture
def formatter():
    """Create formatter instance."""
    return RecommendationFormatter()


# =============================================================================
# TradeCard Tests
# =============================================================================

class TestTradeCard:
    """Tests for TradeCard dataclass."""
    
    def test_trade_card_creation(self):
        """Test creating a trade card."""
        card = TradeCard(
            instrument="EURUSD",
            direction="LONG",
            direction_emoji="🟢",
            confidence_pct=72,
            confidence_label="STRONG",
            confidence_bar="███████░░░",
        )
        
        assert card.instrument == "EURUSD"
        assert card.direction == "LONG"
        assert card.confidence_pct == 72
    
    def test_trade_card_optional_fields(self):
        """Test trade card with optional fields."""
        card = TradeCard(
            instrument="USDJPY",
            direction="SHORT",
            direction_emoji="🔴",
            confidence_pct=65,
            confidence_label="STRONG",
            confidence_bar="██████░░░░",
            entry_price="148.500",
            stop_loss="149.000",
            take_profit="147.500",
            risk_reward="2.0:1",
        )
        
        assert card.entry_price == "148.500"
        assert card.risk_reward == "2.0:1"


# =============================================================================
# RecommendationFormatter Tests
# =============================================================================

class TestRecommendationFormatter:
    """Tests for RecommendationFormatter."""
    
    def test_format_trade_card(self, formatter, sample_signal):
        """Test converting signal to trade card."""
        card = formatter.format_trade_card(sample_signal)
        
        assert card.instrument == "EURUSD"
        assert card.direction == "LONG"
        assert card.direction_emoji == "🟢"
        assert card.confidence_pct == 72
        assert card.confidence_label == "STRONG"
    
    def test_format_markdown(self, formatter, sample_signal):
        """Test markdown formatting."""
        md = formatter.format_markdown(sample_signal)
        
        assert "## 🟢 LONG EURUSD" in md
        assert "72%" in md
        assert "STRONG" in md
        assert "Entry" in md
        assert "fx_carry_momentum" in md
    
    def test_format_markdown_conflicted(self, formatter, conflicted_signal):
        """Test markdown with conflict warning."""
        md = formatter.format_markdown(conflicted_signal)
        
        assert "CONFLICTED" in md
        assert "⚠️" in md
        assert "Conflict Warning" in md
    
    def test_format_slack(self, formatter, sample_signal):
        """Test Slack block kit formatting."""
        slack = formatter.format_slack(sample_signal)
        
        assert "blocks" in slack
        blocks = slack["blocks"]
        
        # Should have header block
        header = blocks[0]
        assert header["type"] == "header"
        assert "LONG EURUSD" in header["text"]["text"]
    
    def test_format_text(self, formatter, sample_signal):
        """Test plain text formatting."""
        text = formatter.format_text(sample_signal)
        
        assert "🟢 LONG EURUSD" in text
        assert "72%" in text
        assert "Entry:" in text
    
    def test_confidence_bar(self, formatter):
        """Test confidence bar generation."""
        bar_70 = formatter._get_confidence_bar(0.70)
        bar_30 = formatter._get_confidence_bar(0.30)
        
        # 70% should have more filled blocks
        assert bar_70.count("█") > bar_30.count("█")
        assert len(bar_70) == 10  # Default width
    
    def test_price_formatting(self, formatter):
        """Test price formatting."""
        # Standard FX pair
        eur_price = formatter._format_price(1.10503, "EURUSD")
        assert eur_price == "1.10503"
        
        # JPY pair (3 decimals)
        jpy_price = formatter._format_price(148.567, "USDJPY")
        assert jpy_price == "148.567"


# =============================================================================
# Report Generation Tests
# =============================================================================

class TestReportGeneration:
    """Tests for report generation."""
    
    def test_format_recommendations_report(self, sample_signal, conflicted_signal):
        """Test full report generation."""
        recommendations = [sample_signal, conflicted_signal]
        report = format_recommendations_report(recommendations)
        
        assert "# Trading Recommendations" in report
        assert "Summary" in report
        assert "Total Recommendations:** 2" in report
        assert "EURUSD" in report
        assert "GBPUSD" in report
    
    def test_report_without_summary(self, sample_signal):
        """Test report without summary section."""
        report = format_recommendations_report(
            [sample_signal],
            include_summary=False
        )
        
        assert "Summary" not in report
        assert "EURUSD" in report
    
    def test_report_custom_title(self, sample_signal):
        """Test report with custom title."""
        report = format_recommendations_report(
            [sample_signal],
            title="Custom Report Title"
        )
        
        assert "# Custom Report Title" in report


# =============================================================================
# MockNotionClient Tests
# =============================================================================

class TestMockNotionClient:
    """Tests for MockNotionClient."""
    
    def test_create_recommendation(self):
        """Test creating mock recommendation."""
        client = MockNotionClient()
        
        rec_dict = {
            "instrument": "EURUSD",
            "direction": "LONG",
            "confidence": 0.72,
        }
        
        page_id = client.create_recommendation(rec_dict)
        
        assert page_id.startswith("mock_")
        assert len(client.created_pages) == 1
    
    def test_create_batch(self):
        """Test creating batch of mock recommendations."""
        client = MockNotionClient()
        
        recs = [
            {"instrument": "EURUSD", "direction": "LONG"},
            {"instrument": "USDJPY", "direction": "SHORT"},
        ]
        
        page_ids = client.create_recommendations_batch(recs)
        
        assert len(page_ids) == 2
        assert len(client.created_pages) == 2
    
    def test_update_status(self):
        """Test updating mock recommendation status."""
        client = MockNotionClient()
        
        result = client.update_recommendation_status(
            "mock_0",
            RecommendationStatus.APPROVED,
            notes="Looks good"
        )
        
        assert result is True
        assert len(client.updated_pages) == 1
        assert client.updated_pages[0]["status"] == "Approved"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
