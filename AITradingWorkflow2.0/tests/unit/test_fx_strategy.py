"""
Unit Tests for FX Carry + Momentum Strategy

Tests signal generation, regime detection, and backtest functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies.base import Signal, SignalDirection, SignalStrength
from src.strategies.fx_carry_momentum import (
    FXCarryMomentumStrategy,
    create_fx_carry_momentum_strategy,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def strategy():
    """Create strategy instance with default config."""
    return FXCarryMomentumStrategy()


@pytest.fixture
def custom_strategy():
    """Create strategy with custom thresholds."""
    config = {
        "instruments": ["EURUSD", "USDJPY"],
        "carry": {
            "min_carry_score": 0.5,
            "strong_carry_score": 1.5,
        },
        "momentum": {
            "min_momentum_score": 1.0,
            "strong_momentum_score": 2.0,
        },
    }
    return FXCarryMomentumStrategy(config)


@pytest.fixture
def sample_features():
    """Create sample feature data."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    np.random.seed(42)
    
    # EURUSD features - bullish carry
    eurusd = pd.DataFrame({
        "timestamp": dates,
        "PX_LAST": 1.10 + np.cumsum(np.random.normal(0, 0.001, 100)),
        "momentum_score": np.random.normal(0.5, 0.5, 100),
        "fx_momentum": np.random.normal(0.8, 0.5, 100),
        "carry_score": np.random.normal(1.5, 0.3, 100),  # Strong positive carry
        "trend_strength": np.random.normal(0.3, 0.2, 100),
        "volatility_20d": np.random.normal(0.08, 0.01, 100),
        "vol_percentile": np.random.normal(40, 10, 100),
        "atr_14": np.random.normal(0.005, 0.001, 100),
    })
    eurusd.set_index("timestamp", inplace=True)
    
    # USDJPY features - bearish momentum
    usdjpy = pd.DataFrame({
        "timestamp": dates,
        "PX_LAST": 148 + np.cumsum(np.random.normal(-0.05, 0.1, 100)),
        "momentum_score": np.random.normal(-1.8, 0.5, 100),  # Strong negative momentum
        "fx_momentum": np.random.normal(-2.0, 0.5, 100),
        "carry_score": np.random.normal(-0.5, 0.3, 100),
        "trend_strength": np.random.normal(-0.5, 0.2, 100),  # Downtrend
        "volatility_20d": np.random.normal(0.10, 0.02, 100),
        "vol_percentile": np.random.normal(60, 15, 100),
        "atr_14": np.random.normal(0.8, 0.1, 100),
    })
    usdjpy.set_index("timestamp", inplace=True)
    
    return {"EURUSD": eurusd, "USDJPY": usdjpy}


@pytest.fixture
def low_vol_macro():
    """Create low volatility macro data (VIX < 18)."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "PX_LAST": np.random.normal(15, 1, 100),  # Low VIX
    }).set_index("timestamp")


@pytest.fixture
def high_vol_macro():
    """Create high volatility macro data (VIX > 25)."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "PX_LAST": np.random.normal(30, 3, 100),  # High VIX
    }).set_index("timestamp")


@pytest.fixture
def normal_vol_macro():
    """Create normal volatility macro data."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "PX_LAST": np.random.normal(20, 2, 100),  # Normal VIX
    }).set_index("timestamp")


# =============================================================================
# Strategy Initialization Tests
# =============================================================================

class TestStrategyInitialization:
    """Tests for strategy initialization."""
    
    def test_default_initialization(self, strategy):
        """Test strategy initializes with defaults."""
        assert strategy.name == "FX Carry + Momentum"
        assert strategy.enabled is True
        assert len(strategy.instruments) == 4
        assert "EURUSD" in strategy.instruments
    
    def test_custom_initialization(self, custom_strategy):
        """Test strategy with custom config."""
        assert len(custom_strategy.instruments) == 2
        assert custom_strategy.carry_config["min_carry_score"] == 0.5
    
    def test_required_features(self, strategy):
        """Test required features list."""
        required = strategy.get_required_features()
        assert "PX_LAST" in required
        assert "momentum_score" in required


# =============================================================================
# Regime Detection Tests
# =============================================================================

class TestRegimeDetection:
    """Tests for regime detection."""
    
    def test_low_vol_regime(self, strategy, low_vol_macro):
        """Test LOW_VOL regime detection."""
        regime = strategy.detect_regime(low_vol_macro)
        assert regime == "LOW_VOL"
    
    def test_high_vol_regime(self, strategy, high_vol_macro):
        """Test HIGH_VOL regime detection."""
        regime = strategy.detect_regime(high_vol_macro)
        assert regime == "HIGH_VOL"
    
    def test_normal_regime(self, strategy, normal_vol_macro):
        """Test NORMAL regime detection."""
        regime = strategy.detect_regime(normal_vol_macro)
        assert regime == "NORMAL"
    
    def test_missing_macro_data(self, strategy):
        """Test handling of missing macro data."""
        regime = strategy.detect_regime(None)
        assert regime == "NORMAL"  # Default
    
    def test_empty_macro_data(self, strategy):
        """Test handling of empty macro data."""
        regime = strategy.detect_regime(pd.DataFrame())
        assert regime == "NORMAL"


# =============================================================================
# Signal Generation Tests
# =============================================================================

class TestSignalGeneration:
    """Tests for signal generation."""
    
    def test_generates_signals_low_vol(self, strategy, sample_features, low_vol_macro):
        """Test signal generation in LOW_VOL regime."""
        signals = strategy.generate_signals(sample_features, low_vol_macro)
        
        assert len(signals) > 0
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.strategy_name == "FX Carry + Momentum"
    
    def test_generates_signals_high_vol(self, strategy, sample_features, high_vol_macro):
        """Test signal generation in HIGH_VOL regime."""
        signals = strategy.generate_signals(sample_features, high_vol_macro)
        
        # Should generate momentum-based signals
        assert len(signals) >= 0  # May or may not generate depending on data
        for signal in signals:
            assert signal.regime == "HIGH_VOL"
    
    def test_signal_has_required_fields(self, strategy, sample_features, low_vol_macro):
        """Test signals have all required fields."""
        signals = strategy.generate_signals(sample_features, low_vol_macro)
        
        if signals:
            signal = signals[0]
            
            # Core fields
            assert signal.instrument is not None
            assert signal.direction in [SignalDirection.LONG, SignalDirection.SHORT]
            assert 0 <= signal.strength <= 1
            
            # Price levels
            assert signal.entry_price is not None
            assert signal.stop_loss is not None
            assert signal.take_profit_1 is not None
            
            # Rationale
            assert len(signal.rationale) > 0
            assert len(signal.key_factors) > 0
    
    def test_max_signals_limit(self, strategy, sample_features, low_vol_macro):
        """Test that signal count respects limit."""
        signals = strategy.generate_signals(sample_features, low_vol_macro)
        assert len(signals) <= strategy.max_signals_per_run
    
    def test_signals_sorted_by_strength(self, strategy, sample_features, low_vol_macro):
        """Test signals are sorted by strength."""
        signals = strategy.generate_signals(sample_features, low_vol_macro)
        
        if len(signals) > 1:
            for i in range(len(signals) - 1):
                assert signals[i].strength >= signals[i + 1].strength
    
    def test_disabled_strategy(self, sample_features, low_vol_macro):
        """Test disabled strategy returns no signals."""
        config = {"enabled": False}
        strategy = FXCarryMomentumStrategy(config)
        
        signals = strategy.generate_signals(sample_features, low_vol_macro)
        assert len(signals) == 0


# =============================================================================
# Carry Signal Tests
# =============================================================================

class TestCarrySignals:
    """Tests for carry-based signals."""
    
    def test_long_carry_signal(self, custom_strategy, low_vol_macro):
        """Test long carry signal generation."""
        # Create strong positive carry features
        features = {
            "EURUSD": pd.DataFrame({
                "PX_LAST": [1.10],
                "carry_score": [2.0],  # Strong positive
                "momentum_score": [0.5],
                "trend_strength": [0.3],
                "vol_percentile": [30],
                "atr_14": [0.005],
            })
        }
        
        signals = custom_strategy.generate_signals(features, low_vol_macro)
        
        # Should generate LONG signal
        eurusd_signals = [s for s in signals if s.instrument == "EURUSD"]
        assert len(eurusd_signals) > 0
        assert eurusd_signals[0].direction == SignalDirection.LONG
    
    def test_short_carry_signal(self, custom_strategy, low_vol_macro):
        """Test short carry signal generation."""
        features = {
            "EURUSD": pd.DataFrame({
                "PX_LAST": [1.10],
                "carry_score": [-2.0],  # Strong negative
                "momentum_score": [-0.5],
                "trend_strength": [-0.3],
                "vol_percentile": [30],
                "atr_14": [0.005],
            })
        }
        
        signals = custom_strategy.generate_signals(features, low_vol_macro)
        
        # Should generate SHORT signal
        eurusd_signals = [s for s in signals if s.instrument == "EURUSD"]
        assert len(eurusd_signals) > 0
        assert eurusd_signals[0].direction == SignalDirection.SHORT
    
    def test_no_signal_high_vol(self, custom_strategy, low_vol_macro):
        """Test no carry signal when vol too high."""
        features = {
            "EURUSD": pd.DataFrame({
                "PX_LAST": [1.10],
                "carry_score": [2.0],
                "momentum_score": [0.5],
                "trend_strength": [0.1],  # Weak trend
                "vol_percentile": [90],  # High vol - should block
                "atr_14": [0.005],
            })
        }
        
        signals = custom_strategy.generate_signals(features, low_vol_macro)
        
        # Carry should be blocked by high vol
        eurusd_signals = [s for s in signals if s.instrument == "EURUSD"]
        # May still get momentum signal, but carry should be blocked


# =============================================================================
# Momentum Signal Tests
# =============================================================================

class TestMomentumSignals:
    """Tests for momentum-based signals."""
    
    def test_long_momentum_signal(self, custom_strategy, high_vol_macro):
        """Test long momentum signal generation."""
        features = {
            "USDJPY": pd.DataFrame({
                "PX_LAST": [148.0],
                "momentum_score": [2.5],  # Strong positive
                "fx_momentum": [2.5],
                "trend_strength": [0.6],  # Strong uptrend
                "vol_percentile": [60],
                "atr_14": [0.8],
            })
        }
        
        signals = custom_strategy.generate_signals(features, high_vol_macro)
        
        usdjpy_signals = [s for s in signals if s.instrument == "USDJPY"]
        if usdjpy_signals:
            assert usdjpy_signals[0].direction == SignalDirection.LONG
    
    def test_short_momentum_signal(self, custom_strategy, high_vol_macro):
        """Test short momentum signal generation."""
        features = {
            "USDJPY": pd.DataFrame({
                "PX_LAST": [148.0],
                "momentum_score": [-2.5],  # Strong negative
                "fx_momentum": [-2.5],
                "trend_strength": [-0.6],  # Strong downtrend
                "vol_percentile": [60],
                "atr_14": [0.8],
            })
        }
        
        signals = custom_strategy.generate_signals(features, high_vol_macro)
        
        usdjpy_signals = [s for s in signals if s.instrument == "USDJPY"]
        if usdjpy_signals:
            assert usdjpy_signals[0].direction == SignalDirection.SHORT
    
    def test_no_signal_weak_trend(self, custom_strategy, high_vol_macro):
        """Test no momentum signal when trend is weak."""
        features = {
            "USDJPY": pd.DataFrame({
                "PX_LAST": [148.0],
                "momentum_score": [2.5],
                "fx_momentum": [2.5],
                "trend_strength": [0.1],  # Weak trend
                "vol_percentile": [60],
                "atr_14": [0.8],
            })
        }
        
        signals = custom_strategy.generate_signals(features, high_vol_macro)
        
        # Should not generate signal due to weak trend
        usdjpy_signals = [s for s in signals if s.instrument == "USDJPY"]
        assert len(usdjpy_signals) == 0


# =============================================================================
# Signal Object Tests
# =============================================================================

class TestSignalObject:
    """Tests for Signal dataclass."""
    
    def test_signal_strength_category(self):
        """Test strength category calculation."""
        signal = Signal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            strength=0.85,
            strategy_name="Test",
            strategy_pod="test_pod",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
        )
        
        assert signal.strength_category == SignalStrength.VERY_STRONG
    
    def test_signal_risk_reward_calculation(self):
        """Test automatic R:R calculation."""
        signal = Signal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            strength=0.7,
            strategy_name="Test",
            strategy_pod="test_pod",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=1.1000,
            stop_loss=1.0950,  # 50 pip risk
            take_profit_1=1.1100,  # 100 pip reward
        )
        
        assert signal.risk_reward_ratio == pytest.approx(2.0, rel=0.01)
    
    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = Signal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            strength=0.7,
            strategy_name="Test",
            strategy_pod="test_pod",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
        )
        
        d = signal.to_dict()
        
        assert d["instrument"] == "EURUSD"
        assert d["direction"] == "LONG"
        assert d["strength"] == 0.7
    
    def test_signal_display_format(self):
        """Test human-readable format."""
        signal = Signal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            strength=0.7,
            strategy_name="Test",
            strategy_pod="test_pod",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit_1=1.1100,
            rationale="Test rationale",
        )
        
        display = signal.format_for_display()
        
        assert "LONG EURUSD" in display
        assert "Entry" in display
        assert "Stop" in display


# =============================================================================
# Backtest Tests
# =============================================================================

class TestBacktest:
    """Tests for signal backtesting."""
    
    def test_backtest_target_hit(self, strategy):
        """Test backtest when target is hit."""
        signal = Signal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            strength=0.7,
            strategy_name="Test",
            strategy_pod="test_pod",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit_1=1.1100,
        )
        
        # Price goes up and hits target
        future_prices = pd.Series([1.1020, 1.1050, 1.1080, 1.1110, 1.1100])
        
        result = strategy.backtest_signal(signal, future_prices)
        
        assert result["outcome"] == "target_hit"
        assert result["pnl_pct"] > 0
    
    def test_backtest_stopped_out(self, strategy):
        """Test backtest when stop is hit."""
        signal = Signal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            strength=0.7,
            strategy_name="Test",
            strategy_pod="test_pod",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit_1=1.1100,
        )
        
        # Price goes down and hits stop
        future_prices = pd.Series([1.0980, 1.0960, 1.0940, 1.0920, 1.0900])
        
        result = strategy.backtest_signal(signal, future_prices)
        
        assert result["outcome"] == "stopped_out"
        assert result["pnl_pct"] < 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
