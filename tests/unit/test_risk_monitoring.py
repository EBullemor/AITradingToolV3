"""
Unit Tests for Risk and Monitoring Modules
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.risk import (
    PositionSizer,
    PositionSizeResult,
    SizingMethod,
    PortfolioRiskManager,
    Position,
    RiskLevel,
    calculate_position_size,
)

from src.monitoring import (
    HealthStatus,
    HealthCheckResult,
    HealthReport,
    check_pipeline_health,
    check_signal_distribution,
    check_recommendation_output,
    run_all_health_checks,
    MetricsCollector,
)


# =============================================================================
# Position Sizer Tests
# =============================================================================

class TestPositionSizer:
    """Tests for PositionSizer."""
    
    @pytest.fixture
    def sizer(self):
        return PositionSizer(default_risk_pct=0.01)
    
    def test_basic_position_size(self, sizer):
        """Test basic fixed risk position sizing."""
        result = sizer.calculate_position_size(
            portfolio_value=100000,
            entry_price=1.1000,
            stop_loss=1.0950,
            direction="LONG",
            instrument="EURUSD"
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.risk_pct == 0.01
        assert result.risk_amount == 1000  # 1% of 100k
        
        # Position size = 1000 / (1.1000 - 1.0950) = 1000 / 0.005 = 200,000 units
        assert result.position_size == 200000
    
    def test_position_capped_at_max(self, sizer):
        """Test that position is capped at max size."""
        result = sizer.calculate_position_size(
            portfolio_value=100000,
            entry_price=1.1000,
            stop_loss=1.0999,  # Very tight stop = large position
            direction="LONG",
        )
        
        # Should be capped at 10% of portfolio
        assert result.position_pct <= 0.10
        assert result.was_capped == True
    
    def test_stop_loss_calculation(self, sizer):
        """Test ATR-based stop loss calculation."""
        stop, stop_pct = sizer.calculate_stop_loss(
            entry_price=1.1000,
            direction="LONG",
            atr=0.0050,  # 50 pips
            atr_multiplier=2.0
        )
        
        # Stop should be 2 * 0.0050 = 0.01 below entry
        assert stop == pytest.approx(1.0900, rel=0.001)
        assert stop_pct == pytest.approx(0.0091, rel=0.01)
    
    def test_short_position(self, sizer):
        """Test short position sizing."""
        result = sizer.calculate_position_size(
            portfolio_value=100000,
            entry_price=1.1000,
            stop_loss=1.1050,  # Stop above for short
            direction="SHORT",
        )
        
        assert result.direction == "SHORT"
        assert result.stop_loss > result.entry_price  # Stop above entry


class TestSimplePositionSize:
    """Tests for simple position size function."""
    
    def test_calculate_position_size(self):
        """Test simple calculation."""
        size = calculate_position_size(
            portfolio_value=100000,
            entry_price=1.1000,
            stop_loss=1.0950,
            risk_pct=0.01
        )
        
        # 1000 / 0.005 = 200,000
        assert size == 200000
    
    def test_zero_risk(self):
        """Test with zero risk distance."""
        size = calculate_position_size(
            portfolio_value=100000,
            entry_price=1.1000,
            stop_loss=1.1000,  # Same as entry
            risk_pct=0.01
        )
        
        assert size == 0


# =============================================================================
# Portfolio Risk Tests
# =============================================================================

class TestPortfolioRiskManager:
    """Tests for PortfolioRiskManager."""
    
    @pytest.fixture
    def risk_mgr(self):
        mgr = PortfolioRiskManager(
            max_gross_exposure=1.0,
            max_net_exposure=0.5,
            max_position_pct=0.10,
        )
        mgr.set_portfolio_value(100000, is_daily_start=True)
        return mgr
    
    def test_can_add_position(self, risk_mgr):
        """Test position addition check."""
        can_add, reason = risk_mgr.can_add_position(
            instrument="EURUSD",
            direction="LONG",
            position_value=5000  # 5% of portfolio
        )
        
        assert can_add == True
    
    def test_position_too_large(self, risk_mgr):
        """Test rejection of oversized position."""
        can_add, reason = risk_mgr.can_add_position(
            instrument="EURUSD",
            direction="LONG",
            position_value=15000  # 15% > 10% max
        )
        
        assert can_add == False
        assert "exceeds max" in reason.lower()
    
    def test_gross_exposure_limit(self, risk_mgr):
        """Test gross exposure enforcement."""
        # Add positions up to limit
        for i, pair in enumerate(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]):
            risk_mgr.add_position(Position(
                instrument=pair,
                direction="LONG",
                size=20000,
                entry_price=1.0,
                current_price=1.0,
                stop_loss=0.95,
            ))
        
        # Now at 80% gross exposure, try to add more
        can_add, reason = risk_mgr.can_add_position(
            instrument="USDCHF",
            direction="LONG",
            position_value=25000  # Would push to 105%
        )
        
        assert can_add == False
        assert "gross exposure" in reason.lower()
    
    def test_risk_report(self, risk_mgr):
        """Test risk report generation."""
        risk_mgr.add_position(Position(
            instrument="EURUSD",
            direction="LONG",
            size=10000,
            entry_price=1.1000,
            current_price=1.1050,
            stop_loss=1.0950,
        ))
        
        report = risk_mgr.calculate_risk_report()
        
        assert report.position_count == 1
        assert report.long_count == 1
        assert report.short_count == 0
        assert report.total_pnl > 0  # Price went up


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthChecks:
    """Tests for health check functions."""
    
    def test_pipeline_health_success(self):
        """Test pipeline health check with success."""
        result = check_pipeline_health({
            "status": "SUCCESS",
            "duration_seconds": 60,
            "errors": []
        })
        
        assert result.status == HealthStatus.HEALTHY
    
    def test_pipeline_health_failure(self):
        """Test pipeline health check with failure."""
        result = check_pipeline_health({
            "status": "FAILED",
            "duration_seconds": 60,
            "errors": ["Data ingestion failed"]
        })
        
        assert result.status == HealthStatus.ERROR
    
    def test_signal_distribution_balanced(self):
        """Test signal distribution check with balanced signals."""
        signals = [
            {"direction": "LONG"},
            {"direction": "LONG"},
            {"direction": "SHORT"},
            {"direction": "SHORT"},
            {"direction": "LONG"},
        ]
        
        result = check_signal_distribution(signals)
        
        assert result.status == HealthStatus.HEALTHY
    
    def test_signal_distribution_imbalanced(self):
        """Test signal distribution check with imbalanced signals."""
        signals = [
            {"direction": "LONG"},
            {"direction": "LONG"},
            {"direction": "LONG"},
            {"direction": "LONG"},
            {"direction": "SHORT"},
        ]
        
        result = check_signal_distribution(signals, imbalance_threshold=0.7)
        
        assert result.status == HealthStatus.WARNING
    
    def test_recommendation_output(self):
        """Test recommendation output check."""
        result = check_recommendation_output(
            recommendations=[{"instrument": "EURUSD"}],
            min_per_day=0,
            max_per_day=10
        )
        
        assert result.status == HealthStatus.HEALTHY


class TestHealthReport:
    """Tests for HealthReport."""
    
    def test_overall_healthy(self):
        """Test overall status when all healthy."""
        checks = [
            HealthCheckResult("Check 1", HealthStatus.HEALTHY, "OK"),
            HealthCheckResult("Check 2", HealthStatus.HEALTHY, "OK"),
        ]
        
        report = HealthReport(checks=checks, overall_status=HealthStatus.HEALTHY)
        
        assert report.overall_status == HealthStatus.HEALTHY
        assert report.has_errors == False
        assert report.has_warnings == False
    
    def test_overall_with_warning(self):
        """Test overall status with warning."""
        checks = [
            HealthCheckResult("Check 1", HealthStatus.HEALTHY, "OK"),
            HealthCheckResult("Check 2", HealthStatus.WARNING, "Issue"),
        ]
        
        report = HealthReport(checks=checks, overall_status=HealthStatus.WARNING)
        
        assert report.has_warnings == True
        assert report.has_errors == False


# =============================================================================
# Metrics Collector Tests
# =============================================================================

class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    @pytest.fixture
    def collector(self, tmp_path):
        return MetricsCollector(metrics_dir=tmp_path / "metrics")
    
    def test_start_and_finish_run(self, collector):
        """Test recording a pipeline run."""
        collector.start_run()
        collector.record_data_metrics(instruments_loaded=6)
        collector.record_signal_metrics(
            signals_generated=10,
            signals_aggregated=5,
            conflicts=1
        )
        collector.finish_run("SUCCESS", 45.5)
        
        # Metrics should be saved
        metrics = collector.load_metrics()
        assert len(metrics) == 1
        assert metrics[0]["status"] == "SUCCESS"
        assert metrics[0]["signals_generated"] == 10


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
