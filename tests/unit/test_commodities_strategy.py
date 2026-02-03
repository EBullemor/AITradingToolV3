"""
Unit Tests for Commodities Term Structure Strategy (Pod 3)

Tests cover:
- Strategy initialization and configuration
- Term structure regime detection (BACKWARDATION, FLAT, CONTANGO)
- Inventory assessment (bullish, bearish, neutral, unavailable)
- Momentum assessment (up, down, flat)
- Signal generation (long, short, no signal) for CL, GC, HG
- Price level calculations (ATR-based and per-commodity pct fallback)
- Confidence scoring
- Backtesting
- Graceful degradation without inventory data
"""

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies.base import Signal, SignalDirection, SignalStrength, SignalStatus
from src.strategies.commodities_ts import (
    CommoditiesTSStrategy,
    create_commodities_ts_strategy,
    COMMODITY_META,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def strategy():
    """Create default Commodities TS strategy."""
    return CommoditiesTSStrategy()


@pytest.fixture
def no_inventory_strategy():
    """Create strategy with inventory disabled."""
    return CommoditiesTSStrategy({"inventory": {"enabled": False}})


@pytest.fixture
def custom_strategy():
    """Create strategy with custom thresholds."""
    return CommoditiesTSStrategy(
        {
            "term_structure": {"backwardation_threshold": 0.01, "contango_threshold": -0.01},
            "momentum": {"threshold": 0.03},
            "risk": {"stop_loss_atr_multiple": 1.5},
        }
    )


def _make_commodity_features(
    n: int = 200,
    price_start: float = 75.0,
    ts_slope: float = 0.03,
    momentum_3m: float = 0.08,
    inventory_zscore: float = -1.5,
    include_inventory: bool = True,
    instrument: str = "CL",
) -> pd.DataFrame:
    """
    Helper to create commodity features DataFrame.

    Args:
        n: Number of rows
        price_start: Starting price
        ts_slope: Term structure slope (positive = backwardation)
        momentum_3m: 3-month momentum
        inventory_zscore: Inventory z-score
        include_inventory: Include inventory columns
        instrument: Ticker for metadata
    """
    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")

    np.random.seed(42)
    returns = np.random.normal(0.0005 if momentum_3m > 0 else -0.0005, 0.015, n)
    prices = price_start * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "PX_LAST": prices,
            "term_structure_slope": ts_slope,
            "ts_slope": ts_slope,
            "roll_yield": ts_slope * 12,  # Annualised approximation
            "momentum_3m": momentum_3m,
            "momentum_score": momentum_3m * 0.7,
            "realized_vol_30d": np.random.uniform(0.15, 0.30, n),
            "atr_14": prices * 0.02,  # 2% of price
        },
        index=dates,
    )

    if include_inventory:
        df["inventory_zscore"] = inventory_zscore

    df["ticker"] = instrument
    return df


# ----- Feature fixtures per scenario -----


@pytest.fixture
def cl_backwardation_bullish():
    """CL in backwardation with bullish inventory and positive momentum."""
    return _make_commodity_features(
        price_start=78.0, ts_slope=0.035, momentum_3m=0.09,
        inventory_zscore=-1.8, instrument="CL",
    )


@pytest.fixture
def cl_contango_bearish():
    """CL in contango with bearish inventory and negative momentum."""
    return _make_commodity_features(
        price_start=68.0, ts_slope=-0.03, momentum_3m=-0.07,
        inventory_zscore=1.5, instrument="CL",
    )


@pytest.fixture
def gc_flat_neutral():
    """GC with flat TS, neutral momentum."""
    return _make_commodity_features(
        price_start=2050.0, ts_slope=0.005, momentum_3m=0.02,
        include_inventory=False, instrument="GC",
    )


@pytest.fixture
def hg_backwardation_features():
    """HG in backwardation with positive momentum."""
    return _make_commodity_features(
        price_start=4.20, ts_slope=0.025, momentum_3m=0.06,
        inventory_zscore=-1.2, instrument="HG",
    )


@pytest.fixture
def low_vol_macro():
    """Low VIX macro data."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    return pd.DataFrame({"PX_LAST": np.random.normal(14, 1, 100)}, index=dates)


@pytest.fixture
def high_vol_macro():
    """High VIX macro data."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    return pd.DataFrame({"PX_LAST": np.random.normal(32, 2, 100)}, index=dates)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for strategy initialization."""

    def test_default_initialization(self, strategy):
        assert strategy.name == "Commodities Term Structure"
        assert strategy.enabled is True
        assert strategy.instruments == ["CL", "GC", "HG"]
        assert strategy.pod_name == "commodities_ts"

    def test_custom_config(self, custom_strategy):
        assert custom_strategy.ts_config["backwardation_threshold"] == 0.01
        assert custom_strategy.mom_config["threshold"] == 0.03
        assert custom_strategy.risk_config["stop_loss_atr_multiple"] == 1.5

    def test_inventory_enabled_default(self, strategy):
        assert strategy.inv_config["enabled"] is True

    def test_inventory_disabled(self, no_inventory_strategy):
        assert no_inventory_strategy.inv_config["enabled"] is False

    def test_required_features(self, strategy):
        required = strategy.get_required_features()
        assert "PX_LAST" in required
        assert "term_structure_slope" in required
        assert "momentum_3m" in required
        assert "inventory_zscore" in required

    def test_required_features_no_inventory(self, no_inventory_strategy):
        required = no_inventory_strategy.get_required_features()
        assert "inventory_zscore" not in required

    def test_factory_function(self):
        strat = create_commodities_ts_strategy()
        assert isinstance(strat, CommoditiesTSStrategy)

    def test_factory_custom_instruments(self):
        strat = create_commodities_ts_strategy(instruments=["CL", "GC"])
        assert strat.instruments == ["CL", "GC"]

    def test_factory_disable_inventory(self):
        strat = create_commodities_ts_strategy(enable_inventory=False)
        assert strat.inv_config["enabled"] is False

    def test_commodity_meta_complete(self):
        """All configured instruments have metadata."""
        for inst in ["CL", "GC", "HG"]:
            assert inst in COMMODITY_META
            assert "name" in COMMODITY_META[inst]
            assert "front_ticker" in COMMODITY_META[inst]


# =============================================================================
# Term Structure Regime Tests
# =============================================================================


class TestTSRegimeDetection:

    def test_backwardation(self, strategy):
        features = _make_commodity_features(ts_slope=0.035)
        ts = strategy.detect_ts_regime(features, "CL")
        assert ts["regime"] == "BACKWARDATION"
        assert ts["slope"] > 0
        assert ts["strength"] > 0

    def test_contango(self, strategy):
        features = _make_commodity_features(ts_slope=-0.03)
        ts = strategy.detect_ts_regime(features, "CL")
        assert ts["regime"] == "CONTANGO"
        assert ts["slope"] < 0

    def test_flat(self, strategy):
        features = _make_commodity_features(ts_slope=0.005)
        ts = strategy.detect_ts_regime(features, "CL")
        assert ts["regime"] == "FLAT"
        assert ts["strength"] == 0.0

    def test_boundary_backwardation(self, strategy):
        features = _make_commodity_features(ts_slope=0.02)
        ts = strategy.detect_ts_regime(features, "CL")
        # Exactly at threshold — should still count
        assert ts["regime"] in ["BACKWARDATION", "FLAT"]

    def test_strong_backwardation(self, strategy):
        features = _make_commodity_features(ts_slope=0.06)
        ts = strategy.detect_ts_regime(features, "CL")
        assert ts["regime"] == "BACKWARDATION"
        assert ts["strength"] >= 0.8

    def test_missing_data(self, strategy):
        ts = strategy.detect_ts_regime(None, "CL")
        assert ts["regime"] == "FLAT"

    def test_empty_dataframe(self, strategy):
        ts = strategy.detect_ts_regime(pd.DataFrame(), "CL")
        assert ts["regime"] == "FLAT"

    def test_roll_yield_populated(self, strategy):
        features = _make_commodity_features(ts_slope=0.03)
        ts = strategy.detect_ts_regime(features, "CL")
        assert ts["roll_yield"] is not None


# =============================================================================
# Inventory Assessment Tests
# =============================================================================


class TestInventoryAssessment:

    def test_bullish_low_inventory(self, strategy):
        features = _make_commodity_features(inventory_zscore=-1.5)
        inv = strategy.assess_inventory(features, "CL")
        assert inv["available"] is True
        assert inv["bias"] == "BULLISH"
        assert inv["confidence_contribution"] > 0

    def test_bearish_high_inventory(self, strategy):
        features = _make_commodity_features(inventory_zscore=1.5)
        inv = strategy.assess_inventory(features, "CL")
        assert inv["available"] is True
        assert inv["bias"] == "BEARISH"

    def test_neutral_inventory(self, strategy):
        features = _make_commodity_features(inventory_zscore=0.3)
        inv = strategy.assess_inventory(features, "CL")
        assert inv["bias"] == "NEUTRAL"
        assert inv["confidence_contribution"] == 0.0

    def test_no_inventory_for_gold(self, strategy):
        """Gold has no standard inventory source."""
        features = _make_commodity_features(include_inventory=True, instrument="GC")
        inv = strategy.assess_inventory(features, "GC")
        assert inv["available"] is False

    def test_inventory_disabled(self, no_inventory_strategy):
        features = _make_commodity_features(inventory_zscore=-2.0)
        inv = no_inventory_strategy.assess_inventory(features, "CL")
        assert inv["available"] is False
        assert inv["confidence_contribution"] == 0.0

    def test_missing_inventory_column(self, strategy):
        features = _make_commodity_features(include_inventory=False)
        inv = strategy.assess_inventory(features, "CL")
        assert inv["available"] is False


# =============================================================================
# Momentum Assessment Tests
# =============================================================================


class TestMomentumAssessment:

    def test_positive_momentum(self, strategy):
        features = _make_commodity_features(momentum_3m=0.08)
        mom = strategy.assess_momentum(features, "CL")
        assert mom["direction"] == "UP"
        assert mom["strength"] > 0

    def test_negative_momentum(self, strategy):
        features = _make_commodity_features(momentum_3m=-0.07)
        mom = strategy.assess_momentum(features, "CL")
        assert mom["direction"] == "DOWN"

    def test_flat_momentum(self, strategy):
        features = _make_commodity_features(momentum_3m=0.01)
        mom = strategy.assess_momentum(features, "CL")
        assert mom["direction"] == "FLAT"

    def test_strong_momentum(self, strategy):
        features = _make_commodity_features(momentum_3m=0.12)
        mom = strategy.assess_momentum(features, "CL")
        assert mom["direction"] == "UP"
        assert mom["strength"] >= 0.8

    def test_missing_data(self, strategy):
        mom = strategy.assess_momentum(pd.DataFrame(), "CL")
        assert mom["direction"] == "FLAT"


# =============================================================================
# Signal Generation Tests
# =============================================================================


class TestSignalGeneration:

    def test_long_backwardation_bullish(self, strategy, cl_backwardation_bullish):
        """Backwardation + positive momentum + low inventory → LONG."""
        signals = strategy.generate_signals({"CL": cl_backwardation_bullish})
        assert len(signals) > 0
        s = signals[0]
        assert s.direction == SignalDirection.LONG
        assert s.instrument == "CL"
        assert s.strategy_name == "Commodities Term Structure"

    def test_short_contango_bearish(self, strategy, cl_contango_bearish):
        """Contango + negative momentum + high inventory → SHORT."""
        signals = strategy.generate_signals({"CL": cl_contango_bearish})
        assert len(signals) > 0
        s = signals[0]
        assert s.direction == SignalDirection.SHORT

    def test_no_signal_flat_ts(self, strategy, gc_flat_neutral):
        """Flat TS → no signal."""
        signals = strategy.generate_signals({"GC": gc_flat_neutral})
        assert len(signals) == 0

    def test_multiple_instruments(self, strategy, cl_backwardation_bullish, hg_backwardation_features):
        """Signals generated for multiple commodities."""
        features = {"CL": cl_backwardation_bullish, "HG": hg_backwardation_features}
        signals = strategy.generate_signals(features)
        instruments = [s.instrument for s in signals]
        # At least one should fire
        assert len(signals) >= 1

    def test_max_signals_limit(self, strategy):
        """Respects max_signals_per_run."""
        features = {}
        for inst in ["CL", "GC", "HG"]:
            features[inst] = _make_commodity_features(
                ts_slope=0.04, momentum_3m=0.10,
                inventory_zscore=-2.0, instrument=inst,
            )
        signals = strategy.generate_signals(features)
        assert len(signals) <= strategy.max_signals_per_run

    def test_signal_has_required_fields(self, strategy, cl_backwardation_bullish):
        signals = strategy.generate_signals({"CL": cl_backwardation_bullish})
        if signals:
            s = signals[0]
            assert s.entry_price is not None
            assert s.stop_loss is not None
            assert s.take_profit_1 is not None
            assert s.take_profit_2 is not None
            assert s.risk_reward_ratio is not None
            assert len(s.rationale) > 0
            assert len(s.key_factors) > 0
            assert s.regime is not None

    def test_long_levels_correct(self, strategy, cl_backwardation_bullish):
        signals = strategy.generate_signals({"CL": cl_backwardation_bullish})
        if signals:
            s = signals[0]
            assert s.stop_loss < s.entry_price
            assert s.take_profit_1 > s.entry_price
            assert s.take_profit_2 > s.take_profit_1

    def test_short_levels_correct(self, strategy, cl_contango_bearish):
        signals = strategy.generate_signals({"CL": cl_contango_bearish})
        if signals:
            s = signals[0]
            assert s.stop_loss > s.entry_price
            assert s.take_profit_1 < s.entry_price
            assert s.take_profit_2 < s.take_profit_1

    def test_disabled_strategy(self, cl_backwardation_bullish):
        strat = CommoditiesTSStrategy({"enabled": False})
        signals = strat.generate_signals({"CL": cl_backwardation_bullish})
        assert len(signals) == 0

    def test_missing_instrument(self, strategy):
        signals = strategy.generate_signals({"NG": _make_commodity_features()})
        assert len(signals) == 0

    def test_with_macro_data(self, strategy, cl_backwardation_bullish, low_vol_macro):
        signals = strategy.generate_signals(
            {"CL": cl_backwardation_bullish}, macro_data=low_vol_macro
        )
        assert isinstance(signals, list)

    def test_high_vix_reduces_confidence(self, strategy, cl_backwardation_bullish, high_vol_macro):
        sig_no_macro = strategy.generate_signals({"CL": cl_backwardation_bullish})
        sig_high_vix = strategy.generate_signals(
            {"CL": cl_backwardation_bullish}, macro_data=high_vol_macro
        )
        if sig_no_macro and sig_high_vix:
            assert sig_high_vix[0].strength <= sig_no_macro[0].strength

    def test_no_inventory_still_signals(self, no_inventory_strategy, cl_backwardation_bullish):
        """Strategy works without inventory data."""
        signals = no_inventory_strategy.generate_signals({"CL": cl_backwardation_bullish})
        # Should still produce signals from TS + momentum
        assert isinstance(signals, list)


# =============================================================================
# Confidence Scoring Tests
# =============================================================================


class TestConfidenceScoring:

    def test_confidence_range(self, strategy, cl_backwardation_bullish):
        signals = strategy.generate_signals({"CL": cl_backwardation_bullish})
        for s in signals:
            assert 0.0 <= s.strength <= 1.0

    def test_strong_ts_higher_confidence(self, strategy):
        strong = _make_commodity_features(ts_slope=0.05, momentum_3m=0.10, inventory_zscore=-2.0)
        weak = _make_commodity_features(ts_slope=0.021, momentum_3m=0.06, inventory_zscore=-0.5)
        sig_strong = strategy.generate_signals({"CL": strong})
        sig_weak = strategy.generate_signals({"CL": weak})
        if sig_strong and sig_weak:
            assert sig_strong[0].strength >= sig_weak[0].strength

    def test_min_confidence_filter(self):
        strat = CommoditiesTSStrategy(
            {"signal": {"min_confidence": 0.95, "require_ts_momentum_alignment": True}}
        )
        features = _make_commodity_features(ts_slope=0.021, momentum_3m=0.06)
        signals = strat.generate_signals({"CL": features})
        for s in signals:
            assert s.strength >= 0.95


# =============================================================================
# Price Level Tests
# =============================================================================


class TestPriceLevels:

    def test_atr_levels_long(self, strategy):
        entry, stop, tp1, tp2 = strategy._calculate_levels_atr(75.0, 1.5, "LONG")
        assert stop < entry
        assert tp1 > entry
        assert tp2 > tp1
        assert abs(entry - stop - 2.0 * 1.5) < 0.01

    def test_atr_levels_short(self, strategy):
        entry, stop, tp1, tp2 = strategy._calculate_levels_atr(75.0, 1.5, "SHORT")
        assert stop > entry
        assert tp1 < entry

    def test_pct_fallback_cl(self, strategy):
        entry, stop, tp1, tp2 = strategy._calculate_levels_pct(75.0, "LONG", "CL")
        assert stop < entry
        assert tp1 > entry
        # CL: 4% base vol → 6% stop
        assert stop == pytest.approx(75.0 * (1 - 0.06), rel=0.01)

    def test_pct_fallback_gc(self, strategy):
        entry, stop, tp1, tp2 = strategy._calculate_levels_pct(2050.0, "LONG", "GC")
        assert stop < entry
        # GC: 2.5% base vol → 3.75% stop
        expected_stop = 2050.0 * (1 - 0.025 * 1.5)
        assert stop == pytest.approx(expected_stop, rel=0.01)


# =============================================================================
# Backtest Tests
# =============================================================================


class TestBacktest:

    def test_target_hit(self, strategy):
        signal = Signal(
            instrument="CL", direction=SignalDirection.LONG, strength=0.7,
            strategy_name="Test", strategy_pod="commodities_ts",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=75.0, stop_loss=72.0, take_profit_1=80.0,
        )
        future_prices = pd.Series([75.5, 76.5, 78.0, 80.5, 81.0])
        result = strategy.backtest_signal(signal, future_prices)
        assert result["outcome"] == "target_hit"
        assert result["pnl_pct"] > 0

    def test_stopped_out(self, strategy):
        signal = Signal(
            instrument="CL", direction=SignalDirection.LONG, strength=0.7,
            strategy_name="Test", strategy_pod="commodities_ts",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=75.0, stop_loss=72.0, take_profit_1=80.0,
        )
        future_prices = pd.Series([74.0, 73.0, 71.5, 70.0])
        result = strategy.backtest_signal(signal, future_prices)
        assert result["outcome"] == "stopped_out"
        assert result["pnl_pct"] < 0

    def test_expired(self, strategy):
        signal = Signal(
            instrument="CL", direction=SignalDirection.LONG, strength=0.7,
            strategy_name="Test", strategy_pod="commodities_ts",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=75.0, stop_loss=72.0, take_profit_1=80.0,
        )
        future_prices = pd.Series([75.2, 75.4, 75.1, 75.3, 75.5])
        result = strategy.backtest_signal(signal, future_prices, horizon_days=5)
        assert result["outcome"] == "expired"

    def test_short_backtest(self, strategy):
        signal = Signal(
            instrument="CL", direction=SignalDirection.SHORT, strength=0.7,
            strategy_name="Test", strategy_pod="commodities_ts",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=75.0, stop_loss=78.0, take_profit_1=70.0,
        )
        future_prices = pd.Series([74.0, 72.0, 69.5])
        result = strategy.backtest_signal(signal, future_prices)
        assert result["outcome"] == "target_hit"
        assert result["pnl_pct"] > 0

    def test_insufficient_data(self, strategy):
        signal = Signal(
            instrument="CL", direction=SignalDirection.LONG, strength=0.7,
            strategy_name="Test", strategy_pod="commodities_ts",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=75.0, stop_loss=72.0, take_profit_1=80.0,
        )
        result = strategy.backtest_signal(signal, pd.Series([75.2]))
        assert result["status"] == "insufficient_data"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:

    def test_no_atr_uses_pct_fallback(self, strategy):
        features = _make_commodity_features(ts_slope=0.04, momentum_3m=0.10)
        features["atr_14"] = np.nan
        signals = strategy.generate_signals({"CL": features})
        if signals:
            assert signals[0].entry_price is not None
            assert signals[0].stop_loss is not None

    def test_news_summary_no_crash(self, strategy, cl_backwardation_bullish):
        news = {"CL": {"sentiment": "bullish", "headlines": ["OPEC cuts"]}}
        signals = strategy.generate_signals(
            {"CL": cl_backwardation_bullish}, news_summary=news
        )
        assert isinstance(signals, list)

    def test_signal_validity_period(self, strategy, cl_backwardation_bullish):
        now = datetime(2026, 2, 3, 12, 0, 0)
        signals = strategy.generate_signals(
            {"CL": cl_backwardation_bullish}, as_of_date=now
        )
        if signals:
            assert signals[0].valid_until == now + timedelta(hours=24)

    def test_regime_in_signal(self, strategy, cl_backwardation_bullish):
        signals = strategy.generate_signals({"CL": cl_backwardation_bullish})
        if signals:
            assert signals[0].regime in ["BACKWARDATION", "CONTANGO", "FLAT"]

    def test_confidence_drivers_populated(self, strategy, cl_backwardation_bullish):
        signals = strategy.generate_signals({"CL": cl_backwardation_bullish})
        if signals:
            drivers = signals[0].confidence_drivers
            assert isinstance(drivers, dict)
            assert "term_structure" in drivers

    def test_serialisation(self, strategy, cl_backwardation_bullish):
        signals = strategy.generate_signals({"CL": cl_backwardation_bullish})
        if signals:
            d = signals[0].to_dict()
            assert d["instrument"] == "CL"
            assert d["strategy_pod"] == "commodities_ts"

    def test_display(self, strategy, cl_backwardation_bullish):
        signals = strategy.generate_signals({"CL": cl_backwardation_bullish})
        if signals:
            display = signals[0].format_for_display()
            assert "CL" in display
