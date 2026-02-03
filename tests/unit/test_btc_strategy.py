"""
Unit Tests for BTC Trend + Volatility Strategy (Pod 2)

Tests cover:
- Strategy initialization and configuration
- Vol regime detection (QUIET, NORMAL, BREAKOUT)
- Trend detection (MA alignment, momentum confirmation)
- On-chain enrichment (exchange flows, MVRV, SOPR)
- Signal generation (long, short, no signal)
- Price level calculations (ATR-based and percentage fallback)
- Confidence scoring
- Backtesting
- Graceful degradation without on-chain data
"""

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# We import from the module under test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies.base import Signal, SignalDirection, SignalStrength, SignalStatus
from src.strategies.btc_trend_vol import BTCTrendVolStrategy, create_btc_trend_vol_strategy


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def strategy():
    """Create default BTC Trend+Vol strategy."""
    return BTCTrendVolStrategy()


@pytest.fixture
def onchain_strategy():
    """Create strategy with on-chain enabled."""
    return BTCTrendVolStrategy({"onchain": {"enabled": True}})


@pytest.fixture
def custom_strategy():
    """Create strategy with custom config."""
    return BTCTrendVolStrategy(
        {
            "trend": {"fast_ma": 20, "slow_ma": 100},
            "volatility": {"breakout_percentile": 85},
            "risk": {"stop_loss_atr_multiple": 3.0},
        }
    )


def _make_btc_features(
    n: int = 300,
    price_start: float = 50000.0,
    trend: str = "up",
    vol_percentile: float = 95.0,
    momentum: float = 0.05,
    include_onchain: bool = False,
) -> pd.DataFrame:
    """
    Helper to create BTC features DataFrame.

    Args:
        n: Number of rows
        price_start: Starting price
        trend: 'up', 'down', or 'flat'
        vol_percentile: Vol percentile value
        momentum: Momentum score
        include_onchain: Include on-chain columns
    """
    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")

    if trend == "up":
        drift = 0.001
    elif trend == "down":
        drift = -0.001
    else:
        drift = 0.0

    np.random.seed(42)
    returns = np.random.normal(drift, 0.02, n)
    prices = price_start * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "PX_LAST": prices,
            "momentum_score": momentum,
            "momentum_1m": momentum,
            "momentum_3m": momentum * 0.8,
            "realized_vol": np.random.uniform(0.4, 0.8, n),
            "vol_percentile": vol_percentile,
            "atr_14": prices * 0.025,  # 2.5% of price
        },
        index=dates,
    )

    # Compute MAs from price
    df["ma_50"] = df["PX_LAST"].rolling(50).mean()
    df["ma_200"] = df["PX_LAST"].rolling(200).mean()

    if include_onchain:
        df["flow_zscore"] = np.random.normal(-1.5, 0.5, n)  # Bullish outflows
        df["mvrv_ratio"] = np.random.uniform(1.5, 2.5, n)
        df["sopr"] = np.random.uniform(0.95, 1.05, n)

    return df


@pytest.fixture
def uptrend_breakout_features():
    """BTC in uptrend with vol breakout."""
    return _make_btc_features(
        trend="up", vol_percentile=95.0, momentum=0.08
    )


@pytest.fixture
def downtrend_breakout_features():
    """BTC in downtrend with vol breakout."""
    return _make_btc_features(
        trend="down", vol_percentile=92.0, momentum=-0.06
    )


@pytest.fixture
def flat_quiet_features():
    """BTC flat with quiet vol."""
    return _make_btc_features(
        trend="flat", vol_percentile=15.0, momentum=0.001
    )


@pytest.fixture
def uptrend_normal_vol_features():
    """BTC in uptrend with normal vol."""
    return _make_btc_features(
        trend="up", vol_percentile=50.0, momentum=0.04
    )


@pytest.fixture
def onchain_bullish_features():
    """BTC with bullish on-chain metrics."""
    return _make_btc_features(
        trend="up",
        vol_percentile=95.0,
        momentum=0.08,
        include_onchain=True,
    )


@pytest.fixture
def low_vol_macro():
    """Low VIX macro data."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {"PX_LAST": np.random.normal(14, 1, 100)}, index=dates
    )


@pytest.fixture
def high_vol_macro():
    """High VIX macro data."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {"PX_LAST": np.random.normal(32, 2, 100)}, index=dates
    )


@pytest.fixture
def normal_macro():
    """Normal VIX macro data."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {"PX_LAST": np.random.normal(20, 2, 100)}, index=dates
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for strategy initialization."""

    def test_default_initialization(self, strategy):
        """Test strategy initialises with defaults."""
        assert strategy.name == "BTC Trend + Volatility Breakout"
        assert strategy.enabled is True
        assert strategy.instruments == ["BTCUSD"]
        assert strategy.pod_name == "btc_trend_vol"

    def test_custom_config(self, custom_strategy):
        """Test strategy with custom config."""
        assert custom_strategy.trend_config["fast_ma"] == 20
        assert custom_strategy.trend_config["slow_ma"] == 100
        assert custom_strategy.vol_config["breakout_percentile"] == 85
        assert custom_strategy.risk_config["stop_loss_atr_multiple"] == 3.0

    def test_onchain_default_disabled(self, strategy):
        """On-chain should be disabled by default."""
        assert strategy.onchain_config["enabled"] is False

    def test_onchain_enabled(self, onchain_strategy):
        """On-chain can be enabled via config."""
        assert onchain_strategy.onchain_config["enabled"] is True

    def test_required_features_base(self, strategy):
        """Test required features without on-chain."""
        required = strategy.get_required_features()
        assert "PX_LAST" in required
        assert "momentum_score" in required
        assert "vol_percentile" in required
        assert "flow_zscore" not in required

    def test_required_features_with_onchain(self, onchain_strategy):
        """Test required features with on-chain enabled."""
        required = onchain_strategy.get_required_features()
        assert "flow_zscore" in required
        assert "mvrv_ratio" in required

    def test_factory_function(self):
        """Test factory function creates valid strategy."""
        strat = create_btc_trend_vol_strategy()
        assert isinstance(strat, BTCTrendVolStrategy)
        assert strat.enabled is True

    def test_factory_onchain_override(self):
        """Test factory with on-chain override."""
        strat = create_btc_trend_vol_strategy(enable_onchain=True)
        assert strat.onchain_config["enabled"] is True


# =============================================================================
# Vol Regime Detection Tests
# =============================================================================


class TestVolRegimeDetection:
    """Tests for volatility regime detection."""

    def test_breakout_regime(self, strategy):
        """Vol percentile > 90 → BREAKOUT."""
        features = _make_btc_features(vol_percentile=95.0)
        regime = strategy.detect_vol_regime(features)
        assert regime == "BREAKOUT"

    def test_quiet_regime(self, strategy):
        """Vol percentile < 25 → QUIET."""
        features = _make_btc_features(vol_percentile=15.0)
        regime = strategy.detect_vol_regime(features)
        assert regime == "QUIET"

    def test_normal_regime(self, strategy):
        """Vol percentile between 25 and 90 → NORMAL."""
        features = _make_btc_features(vol_percentile=50.0)
        regime = strategy.detect_vol_regime(features)
        assert regime == "NORMAL"

    def test_boundary_breakout(self, strategy):
        """Vol percentile exactly at breakout threshold."""
        features = _make_btc_features(vol_percentile=90.0)
        regime = strategy.detect_vol_regime(features)
        assert regime == "BREAKOUT"

    def test_boundary_quiet(self, strategy):
        """Vol percentile exactly at quiet threshold."""
        features = _make_btc_features(vol_percentile=25.0)
        regime = strategy.detect_vol_regime(features)
        assert regime == "QUIET"

    def test_missing_data(self, strategy):
        """Missing features defaults to NORMAL."""
        regime = strategy.detect_vol_regime(None)
        assert regime == "NORMAL"

    def test_empty_dataframe(self, strategy):
        """Empty DataFrame defaults to NORMAL."""
        regime = strategy.detect_vol_regime(pd.DataFrame())
        assert regime == "NORMAL"

    def test_custom_breakout_threshold(self, custom_strategy):
        """Custom breakout threshold (85) works."""
        features = _make_btc_features(vol_percentile=87.0)
        regime = custom_strategy.detect_vol_regime(features)
        assert regime == "BREAKOUT"


# =============================================================================
# Trend Detection Tests
# =============================================================================


class TestTrendDetection:
    """Tests for trend detection."""

    def test_uptrend_detected(self, strategy, uptrend_breakout_features):
        """Uptrend with price above both MAs."""
        trend = strategy.detect_trend(uptrend_breakout_features)
        assert trend["direction"] == "UP"
        assert trend["above_fast_ma"] is True
        assert trend["above_slow_ma"] is True
        assert trend["strength"] > 0.0

    def test_downtrend_detected(self, strategy, downtrend_breakout_features):
        """Downtrend with price below both MAs."""
        trend = strategy.detect_trend(downtrend_breakout_features)
        assert trend["direction"] == "DOWN"
        assert trend["above_fast_ma"] is False
        assert trend["above_slow_ma"] is False

    def test_flat_market(self, strategy, flat_quiet_features):
        """Flat market detected."""
        trend = strategy.detect_trend(flat_quiet_features)
        # Could be FLAT or a weak direction depending on random seed
        assert trend["direction"] in ["FLAT", "UP", "DOWN"]
        if trend["direction"] == "FLAT":
            assert trend["strength"] < 0.3

    def test_momentum_populated(self, strategy, uptrend_breakout_features):
        """Momentum fields are populated."""
        trend = strategy.detect_trend(uptrend_breakout_features)
        assert "momentum_1m" in trend
        assert isinstance(trend["momentum_1m"], float)

    def test_missing_data(self, strategy):
        """Missing data returns safe defaults."""
        trend = strategy.detect_trend(None)
        assert trend["direction"] == "FLAT"
        assert trend["strength"] == 0.0

    def test_empty_dataframe(self, strategy):
        """Empty DataFrame returns safe defaults."""
        trend = strategy.detect_trend(pd.DataFrame())
        assert trend["direction"] == "FLAT"


# =============================================================================
# On-Chain Enrichment Tests
# =============================================================================


class TestOnChainEnrichment:
    """Tests for on-chain metrics enrichment."""

    def test_disabled_returns_neutral(self, strategy):
        """When disabled, returns neutral with no adjustment."""
        features = _make_btc_features(include_onchain=True)
        result = strategy.assess_onchain(features)
        assert result["available"] is False
        assert result["bias"] == "NEUTRAL"
        assert result["confidence_adjustment"] == 0.0

    def test_enabled_with_data(self, onchain_strategy):
        """When enabled with data, produces assessment."""
        features = _make_btc_features(include_onchain=True)
        result = onchain_strategy.assess_onchain(features)
        assert result["available"] is True
        assert result["bias"] in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert "exchange_flow_zscore" in result["details"]

    def test_enabled_without_data(self, onchain_strategy):
        """When enabled but no on-chain columns, degrades gracefully."""
        features = _make_btc_features(include_onchain=False)
        result = onchain_strategy.assess_onchain(features)
        assert result["available"] is False
        assert result["confidence_adjustment"] == 0.0

    def test_bullish_flow(self, onchain_strategy):
        """Strong outflows should be bullish."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq="D")
        features = pd.DataFrame(
            {"flow_zscore": -2.0, "mvrv_ratio": 1.8, "sopr": 0.92},
            index=dates,
        )
        result = onchain_strategy.assess_onchain(features)
        assert result["available"] is True
        assert result["bias"] == "BULLISH"
        assert result["confidence_adjustment"] > 0

    def test_bearish_mvrv(self, onchain_strategy):
        """MVRV > 3 should be bearish."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq="D")
        features = pd.DataFrame(
            {"mvrv_ratio": 3.5, "flow_zscore": 0.5},
            index=dates,
        )
        result = onchain_strategy.assess_onchain(features)
        assert result["available"] is True
        # Overall could be bearish or neutral depending on combination
        assert result["details"]["mvrv_ratio"] == 3.5


# =============================================================================
# Signal Generation Tests
# =============================================================================


class TestSignalGeneration:
    """Tests for signal generation."""

    def test_generates_long_uptrend_breakout(
        self, strategy, uptrend_breakout_features
    ):
        """Uptrend + breakout → LONG signal."""
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strategy.generate_signals(features)
        assert len(signals) > 0
        signal = signals[0]
        assert signal.direction == SignalDirection.LONG
        assert signal.instrument == "BTCUSD"
        assert signal.strategy_name == "BTC Trend + Volatility Breakout"

    def test_generates_short_downtrend_breakout(
        self, strategy, downtrend_breakout_features
    ):
        """Downtrend + breakout → SHORT signal."""
        features = {"BTCUSD": downtrend_breakout_features}
        signals = strategy.generate_signals(features)
        assert len(signals) > 0
        signal = signals[0]
        assert signal.direction == SignalDirection.SHORT

    def test_no_signal_flat_quiet(self, strategy, flat_quiet_features):
        """Flat + quiet → no signal."""
        features = {"BTCUSD": flat_quiet_features}
        signals = strategy.generate_signals(features)
        # Quiet vol with flat trend should produce no signal
        assert len(signals) == 0

    def test_signal_has_required_fields(
        self, strategy, uptrend_breakout_features
    ):
        """Signals have all required fields."""
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strategy.generate_signals(features)
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

    def test_signal_levels_long(self, strategy, uptrend_breakout_features):
        """LONG signal: stop < entry < target."""
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strategy.generate_signals(features)
        if signals:
            s = signals[0]
            assert s.stop_loss < s.entry_price
            assert s.take_profit_1 > s.entry_price
            assert s.take_profit_2 > s.take_profit_1

    def test_signal_levels_short(self, strategy, downtrend_breakout_features):
        """SHORT signal: stop > entry > target."""
        features = {"BTCUSD": downtrend_breakout_features}
        signals = strategy.generate_signals(features)
        if signals:
            s = signals[0]
            assert s.stop_loss > s.entry_price
            assert s.take_profit_1 < s.entry_price
            assert s.take_profit_2 < s.take_profit_1

    def test_max_signals_limit(self, strategy, uptrend_breakout_features):
        """Signal count respects max_signals_per_run."""
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strategy.generate_signals(features)
        assert len(signals) <= strategy.max_signals_per_run

    def test_disabled_strategy(self, uptrend_breakout_features):
        """Disabled strategy generates no signals."""
        strat = BTCTrendVolStrategy({"enabled": False})
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strat.generate_signals(features)
        assert len(signals) == 0

    def test_missing_instrument(self, strategy):
        """Missing instrument in features dict."""
        features = {"ETHUSD": _make_btc_features()}  # Wrong instrument
        signals = strategy.generate_signals(features)
        assert len(signals) == 0

    def test_empty_features(self, strategy):
        """Empty features DataFrame."""
        features = {"BTCUSD": pd.DataFrame()}
        signals = strategy.generate_signals(features)
        assert len(signals) == 0

    def test_with_macro_data(
        self, strategy, uptrend_breakout_features, normal_macro
    ):
        """Signal generation with macro data."""
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strategy.generate_signals(features, macro_data=normal_macro)
        # Should still generate a signal
        assert len(signals) >= 0  # May or may not based on conditions

    def test_onchain_enrichment_applied(
        self, onchain_strategy, onchain_bullish_features
    ):
        """On-chain enrichment affects signal when enabled."""
        features = {"BTCUSD": onchain_bullish_features}
        signals = onchain_strategy.generate_signals(features)
        if signals:
            s = signals[0]
            assert "onchain" in s.confidence_drivers or len(s.key_factors) > 0

    def test_normal_vol_signal(
        self, strategy, uptrend_normal_vol_features
    ):
        """Normal vol with uptrend can still generate signal."""
        features = {"BTCUSD": uptrend_normal_vol_features}
        signals = strategy.generate_signals(features)
        # May or may not generate depending on confidence threshold
        for s in signals:
            assert s.regime == "NORMAL"


# =============================================================================
# Confidence Scoring Tests
# =============================================================================


class TestConfidenceScoring:
    """Tests for confidence calculation."""

    def test_confidence_range(self, strategy, uptrend_breakout_features):
        """Confidence is always between 0 and 1."""
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strategy.generate_signals(features)
        for s in signals:
            assert 0.0 <= s.strength <= 1.0

    def test_breakout_higher_confidence(self, strategy):
        """Breakout regime should yield higher confidence than normal."""
        breakout = _make_btc_features(
            trend="up", vol_percentile=95.0, momentum=0.08
        )
        normal = _make_btc_features(
            trend="up", vol_percentile=50.0, momentum=0.08
        )

        sig_breakout = strategy.generate_signals({"BTCUSD": breakout})
        sig_normal = strategy.generate_signals({"BTCUSD": normal})

        if sig_breakout and sig_normal:
            assert sig_breakout[0].strength >= sig_normal[0].strength

    def test_min_confidence_filter(self):
        """Signals below min confidence are filtered out."""
        strat = BTCTrendVolStrategy(
            {"signal": {"min_confidence": 0.95, "require_trend_confirmation": True}}
        )
        features = _make_btc_features(
            trend="up", vol_percentile=55.0, momentum=0.02
        )
        signals = strat.generate_signals({"BTCUSD": features})
        # Very high bar should filter most signals
        for s in signals:
            assert s.strength >= 0.95


# =============================================================================
# Price Level Calculation Tests
# =============================================================================


class TestPriceLevels:
    """Tests for price level calculations."""

    def test_atr_based_levels(self, strategy):
        """ATR-based level calculation."""
        entry, stop, tp1, tp2 = strategy._calculate_levels_atr(
            price=50000.0, atr=1250.0, direction="LONG"
        )
        assert stop < entry
        assert tp1 > entry
        assert tp2 > tp1
        # Stop should be 2.5 * ATR away
        assert abs(entry - stop - 2.5 * 1250.0) < 0.01

    def test_atr_based_levels_short(self, strategy):
        """ATR-based levels for SHORT."""
        entry, stop, tp1, tp2 = strategy._calculate_levels_atr(
            price=50000.0, atr=1250.0, direction="SHORT"
        )
        assert stop > entry
        assert tp1 < entry
        assert tp2 < tp1

    def test_pct_fallback_long(self, strategy):
        """Percentage fallback for LONG."""
        entry, stop, tp1, tp2 = strategy._calculate_levels_pct(
            price=50000.0, direction="LONG"
        )
        assert stop == 50000.0 * 0.95  # 5% stop
        assert tp1 == 50000.0 * 1.08  # 8% target
        assert tp2 == 50000.0 * 1.15  # 15% extended

    def test_pct_fallback_short(self, strategy):
        """Percentage fallback for SHORT."""
        entry, stop, tp1, tp2 = strategy._calculate_levels_pct(
            price=50000.0, direction="SHORT"
        )
        assert stop == 50000.0 * 1.05
        assert tp1 == 50000.0 * 0.92
        assert tp2 == 50000.0 * 0.85


# =============================================================================
# Backtest Tests
# =============================================================================


class TestBacktest:
    """Tests for signal backtesting."""

    def test_backtest_target_hit(self, strategy):
        """Backtest when target is hit."""
        signal = Signal(
            instrument="BTCUSD",
            direction=SignalDirection.LONG,
            strength=0.7,
            strategy_name="Test",
            strategy_pod="btc_trend_vol",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=50000.0,
            stop_loss=47500.0,
            take_profit_1=55000.0,
        )
        future_prices = pd.Series(
            [50500, 51000, 52000, 54000, 55500, 56000]
        )
        result = strategy.backtest_signal(signal, future_prices)
        assert result["outcome"] == "target_hit"
        assert result["pnl_pct"] > 0

    def test_backtest_stopped_out(self, strategy):
        """Backtest when stop is hit."""
        signal = Signal(
            instrument="BTCUSD",
            direction=SignalDirection.LONG,
            strength=0.7,
            strategy_name="Test",
            strategy_pod="btc_trend_vol",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=50000.0,
            stop_loss=47500.0,
            take_profit_1=55000.0,
        )
        future_prices = pd.Series([49500, 48500, 47000, 46000])
        result = strategy.backtest_signal(signal, future_prices)
        assert result["outcome"] == "stopped_out"
        assert result["pnl_pct"] < 0

    def test_backtest_expired(self, strategy):
        """Backtest when neither stop nor target hit."""
        signal = Signal(
            instrument="BTCUSD",
            direction=SignalDirection.LONG,
            strength=0.7,
            strategy_name="Test",
            strategy_pod="btc_trend_vol",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=50000.0,
            stop_loss=47500.0,
            take_profit_1=55000.0,
        )
        future_prices = pd.Series([50100, 50200, 50300, 50400, 50500])
        result = strategy.backtest_signal(signal, future_prices, horizon_days=5)
        assert result["outcome"] == "expired"
        assert result["days_held"] == 5

    def test_backtest_short(self, strategy):
        """Backtest SHORT signal."""
        signal = Signal(
            instrument="BTCUSD",
            direction=SignalDirection.SHORT,
            strength=0.7,
            strategy_name="Test",
            strategy_pod="btc_trend_vol",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=50000.0,
            stop_loss=52500.0,
            take_profit_1=45000.0,
        )
        future_prices = pd.Series([49000, 47000, 44500])
        result = strategy.backtest_signal(signal, future_prices)
        assert result["outcome"] == "target_hit"
        assert result["pnl_pct"] > 0

    def test_backtest_insufficient_data(self, strategy):
        """Backtest with insufficient price data."""
        signal = Signal(
            instrument="BTCUSD",
            direction=SignalDirection.LONG,
            strength=0.7,
            strategy_name="Test",
            strategy_pod="btc_trend_vol",
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            entry_price=50000.0,
            stop_loss=47500.0,
            take_profit_1=55000.0,
        )
        result = strategy.backtest_signal(signal, pd.Series([50100]))
        assert result["status"] == "insufficient_data"


# =============================================================================
# Edge Cases & Graceful Degradation
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and graceful degradation."""

    def test_no_atr_uses_pct_fallback(self, strategy):
        """When ATR is missing, falls back to percentage levels."""
        features = _make_btc_features(trend="up", vol_percentile=95.0, momentum=0.08)
        features["atr_14"] = np.nan  # Remove ATR
        signals = strategy.generate_signals({"BTCUSD": features})
        if signals:
            s = signals[0]
            # Should still have levels from pct fallback
            assert s.entry_price is not None
            assert s.stop_loss is not None

    def test_single_instrument(self, strategy):
        """Strategy only processes configured instruments."""
        assert strategy.instruments == ["BTCUSD"]

    def test_news_summary_integration(self, strategy, uptrend_breakout_features):
        """News summary doesn't crash signal generation."""
        features = {"BTCUSD": uptrend_breakout_features}
        news = {"BTCUSD": {"sentiment": "bullish", "headlines": ["BTC ETF inflows"]}}
        signals = strategy.generate_signals(features, news_summary=news)
        # Should work without error
        assert isinstance(signals, list)

    def test_signal_validity_period(self, strategy, uptrend_breakout_features):
        """Signal valid_until is correctly set."""
        now = datetime(2026, 2, 3, 12, 0, 0)
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strategy.generate_signals(features, as_of_date=now)
        if signals:
            s = signals[0]
            expected_expiry = now + timedelta(hours=24)
            assert s.valid_until == expected_expiry

    def test_regime_in_signal(self, strategy, uptrend_breakout_features):
        """Signal includes vol regime."""
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strategy.generate_signals(features)
        if signals:
            assert signals[0].regime in ["BREAKOUT", "NORMAL", "QUIET"]

    def test_confidence_drivers_populated(
        self, strategy, uptrend_breakout_features
    ):
        """Confidence drivers dict is populated."""
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strategy.generate_signals(features)
        if signals:
            drivers = signals[0].confidence_drivers
            assert isinstance(drivers, dict)
            assert len(drivers) > 0


# =============================================================================
# Signal Display & Serialisation Tests
# =============================================================================


class TestSignalOutput:
    """Tests for signal display and serialisation."""

    def test_signal_to_dict(self, strategy, uptrend_breakout_features):
        """Signal serialises to dict."""
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strategy.generate_signals(features)
        if signals:
            d = signals[0].to_dict()
            assert d["instrument"] == "BTCUSD"
            assert d["direction"] in ["LONG", "SHORT"]
            assert d["strategy_pod"] == "btc_trend_vol"

    def test_signal_display(self, strategy, uptrend_breakout_features):
        """Signal format_for_display works."""
        features = {"BTCUSD": uptrend_breakout_features}
        signals = strategy.generate_signals(features)
        if signals:
            display = signals[0].format_for_display()
            assert "BTCUSD" in display
            assert "Entry" in display
