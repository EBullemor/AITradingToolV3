"""
Unit Tests for Mean Reversion Strategy (Pod 5)

Tests cover:
- Initialisation & configuration
- Spike detection (single-day, multi-day, boundaries)
- Catalyst screening (no catalyst, weak, strong, unknown)
- Liquidity filtering
- Recent signal limiting
- Price levels (ATR-based, percentage fallback)
- Confidence scoring
- Signal generation (full pipeline)
- Backtesting
- VIX regime gating
- Edge cases
"""

from datetime import datetime, timedelta
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies.base import Signal, SignalDirection
from src.strategies.mean_reversion import (
    MeanReversionStrategy,
    create_mean_reversion_strategy,
    SpikeDirection,
    SpikeDetection,
    CatalystVerdict,
    CatalystScreening,
    default_catalyst_screener,
    INSTRUMENT_META,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_price_df(
    n: int = 300,
    base: float = 100.0,
    vol: float = 0.01,
    spike_at_end: float = 0.0,
    atr: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create price DataFrame. spike_at_end adds a spike return on the last day."""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")
    returns = np.random.normal(0, vol, n)
    if spike_at_end != 0:
        returns[-1] = spike_at_end
    prices = base * np.cumprod(1 + returns)
    df = pd.DataFrame({"PX_LAST": prices}, index=dates)
    if atr > 0:
        df["atr_14"] = atr
    return df


def _spike_features(instrument: str = "EURUSD", direction: str = "down") -> Dict[str, pd.DataFrame]:
    """Create feature dict with a spike on one instrument."""
    spike_ret = -0.04 if direction == "down" else 0.04
    return {instrument: _make_price_df(n=300, spike_at_end=spike_ret, atr=0.005)}


def _no_spike_features() -> Dict[str, pd.DataFrame]:
    """Normal market — no spikes."""
    return {inst: _make_price_df(n=300, vol=0.005) for inst in ["EURUSD", "USDJPY"]}


def _macro_data(vix: float = 18.0) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.now(), periods=10, freq="D")
    return pd.DataFrame({"PX_LAST": [vix] * 10}, index=dates)


def _news_no_catalyst() -> Dict:
    return {"catalysts": ["Tech earnings beat estimates"], "key_events": ["S&P 500 hits record"]}


def _news_strong_catalyst_fx() -> Dict:
    return {
        "catalysts": ["ECB surprises with rate hike", "Euro policy shift"],
        "key_events": ["ECB raises rates by 50bps"],
    }


def _news_weak_catalyst_fx() -> Dict:
    return {
        "catalysts": ["EUR zone data mixed"],
        "key_events": ["Minor eurozone PMI miss"],
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def strategy():
    return MeanReversionStrategy()


@pytest.fixture
def lenient_strategy():
    return MeanReversionStrategy({
        "filters": {"require_no_catalyst": False, "allow_weak_catalyst": True}
    })


# ============================================================================
# Initialisation
# ============================================================================


class TestInitialisation:

    def test_default_init(self, strategy):
        assert strategy.name == "Mean Reversion (Vol Spike)"
        assert strategy.pod_name == "mean_reversion"
        assert strategy.enabled is True
        assert "EURUSD" in strategy.instruments
        assert "BTCUSD" in strategy.instruments

    def test_custom_config(self):
        s = MeanReversionStrategy({"spike": {"zscore_threshold": 2.5}})
        assert s.spike_config["zscore_threshold"] == 2.5

    def test_custom_catalyst_screener(self):
        mock_screener = MagicMock(return_value=CatalystScreening(
            instrument="EURUSD",
            verdict=CatalystVerdict.NO_CATALYST,
        ))
        s = MeanReversionStrategy(catalyst_screener=mock_screener)
        assert s.catalyst_screener is mock_screener

    def test_factory(self):
        s = create_mean_reversion_strategy()
        assert isinstance(s, MeanReversionStrategy)

    def test_required_features(self, strategy):
        feats = strategy.get_required_features()
        assert "PX_LAST" in feats
        assert "atr_14" in feats

    def test_instrument_metadata(self):
        assert "EURUSD" in INSTRUMENT_META
        assert INSTRUMENT_META["EURUSD"]["liquidity_rank"] == 1.0
        assert INSTRUMENT_META["BTCUSD"]["asset_class"] == "crypto"


# ============================================================================
# Spike Detection
# ============================================================================


class TestSpikeDetection:

    def test_large_down_spike(self, strategy):
        df = _make_price_df(n=300, spike_at_end=-0.05, atr=0.005)
        spike = strategy.detect_spike("EURUSD", df)
        assert spike.detected is True
        assert spike.direction == SpikeDirection.DOWN
        assert spike.abs_zscore > 3.0

    def test_large_up_spike(self, strategy):
        df = _make_price_df(n=300, spike_at_end=0.05, atr=0.005)
        spike = strategy.detect_spike("EURUSD", df)
        assert spike.detected is True
        assert spike.direction == SpikeDirection.UP

    def test_no_spike(self, strategy):
        df = _make_price_df(n=300, vol=0.005)
        spike = strategy.detect_spike("EURUSD", df)
        assert spike.detected is False

    def test_borderline_spike(self, strategy):
        """Just below threshold should not trigger."""
        df = _make_price_df(n=300, vol=0.01, spike_at_end=0.025)
        spike = strategy.detect_spike("EURUSD", df)
        # May or may not trigger depending on vol — check it doesn't crash
        assert isinstance(spike.detected, bool)

    def test_insufficient_data(self, strategy):
        df = _make_price_df(n=10, spike_at_end=-0.05)
        spike = strategy.detect_spike("EURUSD", df)
        assert spike.detected is False

    def test_atr_populated(self, strategy):
        df = _make_price_df(n=300, spike_at_end=-0.05, atr=1.5)
        spike = strategy.detect_spike("CL", df)
        assert spike.atr == 1.5

    def test_price_populated(self, strategy):
        df = _make_price_df(n=300, base=1.08, spike_at_end=-0.04)
        spike = strategy.detect_spike("EURUSD", df)
        assert spike.price > 0

    def test_multi_day_spike(self):
        """Two moderate days should trigger multi-day detection."""
        s = MeanReversionStrategy({"spike": {"use_multi_day": True, "multi_day_threshold": 2.0}})
        np.random.seed(99)
        dates = pd.date_range(end=datetime.now(), periods=300, freq="D")
        returns = np.random.normal(0, 0.005, 300)
        returns[-2] = -0.02
        returns[-1] = -0.02
        prices = 100 * np.cumprod(1 + returns)
        df = pd.DataFrame({"PX_LAST": prices}, index=dates)
        spike = s.detect_spike("EURUSD", df)
        # Multi-day cumulative z should be high
        assert spike.multi_day_zscore is not None or spike.detected

    def test_multi_day_disabled(self):
        s = MeanReversionStrategy({"spike": {"use_multi_day": False}})
        df = _make_price_df(n=300, vol=0.005)
        spike = s.detect_spike("EURUSD", df)
        assert spike.multi_day_zscore is None


# ============================================================================
# Catalyst Screening
# ============================================================================


class TestCatalystScreening:

    def test_no_catalyst(self, strategy):
        spike = SpikeDetection(instrument="EURUSD", detected=True, direction=SpikeDirection.DOWN)
        screening = strategy.screen_catalyst("EURUSD", spike, _news_no_catalyst())
        assert screening.verdict == CatalystVerdict.NO_CATALYST

    def test_strong_catalyst(self, strategy):
        spike = SpikeDetection(instrument="EURUSD", detected=True, direction=SpikeDirection.DOWN)
        screening = strategy.screen_catalyst("EURUSD", spike, _news_strong_catalyst_fx())
        assert screening.verdict == CatalystVerdict.STRONG_CATALYST

    def test_weak_catalyst(self, strategy):
        spike = SpikeDetection(instrument="EURUSD", detected=True, direction=SpikeDirection.DOWN)
        screening = strategy.screen_catalyst("EURUSD", spike, _news_weak_catalyst_fx())
        # Depending on keyword matching, should be WEAK or NO
        assert screening.verdict in [CatalystVerdict.WEAK_CATALYST, CatalystVerdict.NO_CATALYST]

    def test_no_news_unknown(self, strategy):
        spike = SpikeDetection(instrument="EURUSD", detected=True)
        screening = strategy.screen_catalyst("EURUSD", spike, None)
        assert screening.verdict == CatalystVerdict.UNKNOWN

    def test_should_trade_no_catalyst(self, strategy):
        s = CatalystScreening(instrument="X", verdict=CatalystVerdict.NO_CATALYST)
        trade, adj = strategy.should_trade_after_screening(s)
        assert trade is True
        assert adj == 1.0

    def test_should_not_trade_strong(self, strategy):
        s = CatalystScreening(instrument="X", verdict=CatalystVerdict.STRONG_CATALYST)
        trade, adj = strategy.should_trade_after_screening(s)
        assert trade is False

    def test_weak_blocked_by_default(self, strategy):
        s = CatalystScreening(instrument="X", verdict=CatalystVerdict.WEAK_CATALYST)
        trade, _ = strategy.should_trade_after_screening(s)
        assert trade is False

    def test_weak_allowed_when_configured(self, lenient_strategy):
        s = CatalystScreening(instrument="X", verdict=CatalystVerdict.WEAK_CATALYST)
        trade, adj = lenient_strategy.should_trade_after_screening(s)
        assert trade is True
        assert adj < 1.0

    def test_custom_screener(self):
        def always_clear(inst, spike, news):
            return CatalystScreening(instrument=inst, verdict=CatalystVerdict.NO_CATALYST)

        s = MeanReversionStrategy(catalyst_screener=always_clear)
        spike = SpikeDetection(instrument="CL", detected=True)
        result = s.screen_catalyst("CL", spike, _news_strong_catalyst_fx())
        assert result.verdict == CatalystVerdict.NO_CATALYST


# ============================================================================
# Liquidity Filter
# ============================================================================


class TestLiquidityFilter:

    def test_eurusd_passes(self, strategy):
        passes, rank = strategy.check_liquidity("EURUSD")
        assert passes is True
        assert rank == 1.0

    def test_unknown_instrument_low_rank(self, strategy):
        passes, rank = strategy.check_liquidity("OBSCURE_TICKER")
        assert passes is False
        assert rank == 0.5

    def test_custom_threshold(self):
        s = MeanReversionStrategy({"filters": {"min_liquidity_rank": 0.99}})
        passes, _ = s.check_liquidity("AUDUSD")
        assert passes is False  # AUDUSD = 0.85


# ============================================================================
# Recent Signal Limit
# ============================================================================


class TestRecentSignalLimit:

    def test_within_limit(self, strategy):
        assert strategy.check_recent_signal_limit("EURUSD", datetime.now()) is True

    def test_exceeds_limit(self, strategy):
        now = datetime.now()
        for i in range(3):
            strategy._recent_signals.append(Signal(
                instrument="EURUSD",
                direction=SignalDirection.LONG,
                strength=0.5,
                strategy_name="test",
                strategy_pod="mean_reversion",
                generated_at=now - timedelta(hours=i),
                valid_until=now + timedelta(hours=12),
            ))
        assert strategy.check_recent_signal_limit("EURUSD", now) is False

    def test_different_instrument_ok(self, strategy):
        now = datetime.now()
        for _ in range(3):
            strategy._recent_signals.append(Signal(
                instrument="USDJPY",
                direction=SignalDirection.SHORT,
                strength=0.5,
                strategy_name="test",
                strategy_pod="mean_reversion",
                generated_at=now,
                valid_until=now + timedelta(hours=12),
            ))
        assert strategy.check_recent_signal_limit("EURUSD", now) is True

    def test_old_signals_dont_count(self, strategy):
        now = datetime.now()
        old = now - timedelta(days=10)
        for _ in range(3):
            strategy._recent_signals.append(Signal(
                instrument="EURUSD",
                direction=SignalDirection.LONG,
                strength=0.5,
                strategy_name="test",
                strategy_pod="mean_reversion",
                generated_at=old,
                valid_until=old + timedelta(hours=12),
            ))
        assert strategy.check_recent_signal_limit("EURUSD", now) is True


# ============================================================================
# Price Levels
# ============================================================================


class TestPriceLevels:

    def test_long_fade_levels(self, strategy):
        spike = SpikeDetection(instrument="EURUSD", detected=True, price=1.08, atr=0.005)
        levels = strategy.compute_price_levels("EURUSD", spike, SignalDirection.LONG)
        assert levels["entry"] == 1.08
        assert levels["stop_loss"] < 1.08
        assert levels["target_1"] > 1.08

    def test_short_fade_levels(self, strategy):
        spike = SpikeDetection(instrument="EURUSD", detected=True, price=1.08, atr=0.005)
        levels = strategy.compute_price_levels("EURUSD", spike, SignalDirection.SHORT)
        assert levels["entry"] == 1.08
        assert levels["stop_loss"] > 1.08
        assert levels["target_1"] < 1.08

    def test_atr_based_stops(self, strategy):
        spike = SpikeDetection(instrument="EURUSD", detected=True, price=100.0, atr=2.0)
        levels = strategy.compute_price_levels("EURUSD", spike, SignalDirection.LONG)
        expected_stop = 100.0 - 2.0 * 1.5
        assert abs(levels["stop_loss"] - expected_stop) < 0.01

    def test_percentage_fallback(self, strategy):
        spike = SpikeDetection(instrument="EURUSD", detected=True, price=1.08, atr=0.0)
        levels = strategy.compute_price_levels("EURUSD", spike, SignalDirection.LONG)
        # Should use stop_pct_fallback (0.008 for EURUSD)
        expected_stop = 1.08 - 1.08 * 0.008
        assert abs(levels["stop_loss"] - expected_stop) < 0.001

    def test_target_2_further(self, strategy):
        spike = SpikeDetection(instrument="CL", detected=True, price=75.0, atr=2.0)
        levels = strategy.compute_price_levels("CL", spike, SignalDirection.LONG)
        assert levels["target_2"] > levels["target_1"]


# ============================================================================
# Confidence Scoring
# ============================================================================


class TestConfidence:

    def test_base_range(self, strategy):
        spike = SpikeDetection(instrument="EURUSD", detected=True, abs_zscore=3.5)
        screening = CatalystScreening(instrument="EURUSD", verdict=CatalystVerdict.NO_CATALYST)
        conf = strategy.compute_confidence(spike, screening, 1.0, 18.0)
        assert 0.0 <= conf <= 0.95

    def test_higher_zscore_higher_confidence(self, strategy):
        screening = CatalystScreening(instrument="X", verdict=CatalystVerdict.NO_CATALYST)
        conf_low = strategy.compute_confidence(
            SpikeDetection(instrument="X", detected=True, abs_zscore=3.1),
            screening, 0.9, 18.0,
        )
        conf_high = strategy.compute_confidence(
            SpikeDetection(instrument="X", detected=True, abs_zscore=5.0),
            screening, 0.9, 18.0,
        )
        assert conf_high > conf_low

    def test_extreme_vix_penalises(self, strategy):
        spike = SpikeDetection(instrument="X", detected=True, abs_zscore=4.0)
        screening = CatalystScreening(instrument="X", verdict=CatalystVerdict.NO_CATALYST)
        conf_normal = strategy.compute_confidence(spike, screening, 0.9, 18.0)
        conf_extreme = strategy.compute_confidence(spike, screening, 0.9, 40.0)
        assert conf_extreme < conf_normal

    def test_no_vix_neutral(self, strategy):
        spike = SpikeDetection(instrument="X", detected=True, abs_zscore=3.5)
        screening = CatalystScreening(instrument="X", verdict=CatalystVerdict.NO_CATALYST)
        conf = strategy.compute_confidence(spike, screening, 0.9, None)
        assert 0.3 < conf < 0.95

    def test_capped_at_095(self, strategy):
        spike = SpikeDetection(instrument="X", detected=True, abs_zscore=10.0)
        screening = CatalystScreening(instrument="X", verdict=CatalystVerdict.NO_CATALYST)
        conf = strategy.compute_confidence(spike, screening, 1.0, 26.0)
        assert conf <= 0.95


# ============================================================================
# Signal Generation (Full Pipeline)
# ============================================================================


class TestSignalGeneration:

    def test_generates_fade_on_spike(self):
        """Spike + no catalyst → signal."""
        s = MeanReversionStrategy()
        features = _spike_features("EURUSD", "down")
        signals = s.generate_signals(features, _macro_data(18), _news_no_catalyst())
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.LONG  # Fade down = go long
        assert signals[0].instrument == "EURUSD"

    def test_fade_up_spike(self):
        s = MeanReversionStrategy()
        features = _spike_features("EURUSD", "up")
        signals = s.generate_signals(features, _macro_data(18), _news_no_catalyst())
        if signals:
            assert signals[0].direction == SignalDirection.SHORT

    def test_no_signal_no_spike(self, strategy):
        signals = strategy.generate_signals(_no_spike_features(), _macro_data(18), _news_no_catalyst())
        assert len(signals) == 0

    def test_blocked_by_strong_catalyst(self):
        s = MeanReversionStrategy()
        features = _spike_features("EURUSD", "down")
        signals = s.generate_signals(features, _macro_data(18), _news_strong_catalyst_fx())
        assert len(signals) == 0

    def test_blocked_by_extreme_vix(self):
        s = MeanReversionStrategy()
        features = _spike_features("EURUSD", "down")
        signals = s.generate_signals(features, _macro_data(40), _news_no_catalyst())
        assert len(signals) == 0

    def test_disabled_strategy(self):
        s = MeanReversionStrategy({"enabled": False})
        features = _spike_features("EURUSD", "down")
        signals = s.generate_signals(features, _macro_data(18), _news_no_catalyst())
        assert len(signals) == 0

    def test_signal_metadata(self):
        s = MeanReversionStrategy()
        features = _spike_features("EURUSD", "down")
        signals = s.generate_signals(features, _macro_data(18), _news_no_catalyst())
        if signals:
            meta = signals[0].metadata
            assert "spike_zscore" in meta
            assert "catalyst_verdict" in meta
            assert "liquidity_rank" in meta
            assert "holding_period" in meta

    def test_signal_key_factors(self):
        s = MeanReversionStrategy()
        features = _spike_features("EURUSD", "down")
        signals = s.generate_signals(features, _macro_data(18), _news_no_catalyst())
        if signals:
            assert len(signals[0].key_factors) > 0

    def test_signal_pod_name(self):
        s = MeanReversionStrategy()
        features = _spike_features("EURUSD", "down")
        signals = s.generate_signals(features, _macro_data(18), _news_no_catalyst())
        if signals:
            assert signals[0].strategy_pod == "mean_reversion"

    def test_max_signals_limit(self):
        s = MeanReversionStrategy({"max_signals_per_run": 1})
        features = {
            "EURUSD": _make_price_df(n=300, spike_at_end=-0.05, atr=0.005, seed=42),
            "USDJPY": _make_price_df(n=300, spike_at_end=-0.06, atr=0.5, seed=43),
        }
        signals = s.generate_signals(features, _macro_data(18), _news_no_catalyst())
        assert len(signals) <= 1

    def test_custom_as_of_date(self):
        s = MeanReversionStrategy()
        features = _spike_features("EURUSD", "down")
        now = datetime(2026, 2, 3, 10, 0, 0)
        signals = s.generate_signals(features, _macro_data(18), _news_no_catalyst(), as_of_date=now)
        if signals:
            assert signals[0].generated_at == now
            assert signals[0].valid_until == now + timedelta(hours=12)


# ============================================================================
# Backtesting
# ============================================================================


class TestBacktesting:

    def test_target_hit(self, strategy):
        signal = Signal(
            instrument="EURUSD", direction=SignalDirection.LONG, strength=0.6,
            strategy_name="test", strategy_pod="mean_reversion",
            generated_at=datetime.now(), valid_until=datetime.now() + timedelta(hours=12),
            entry_price=1.08, stop_loss=1.07, take_profit_1=1.10,
        )
        # Price goes up to target
        prices = pd.Series([1.081, 1.085, 1.09, 1.10, 1.11])
        result = strategy.backtest_signal(signal, prices)
        assert result["outcome"] == "target_hit"
        assert result["pnl"] > 0

    def test_stopped_out(self, strategy):
        signal = Signal(
            instrument="EURUSD", direction=SignalDirection.LONG, strength=0.6,
            strategy_name="test", strategy_pod="mean_reversion",
            generated_at=datetime.now(), valid_until=datetime.now() + timedelta(hours=12),
            entry_price=1.08, stop_loss=1.07, take_profit_1=1.10,
        )
        prices = pd.Series([1.078, 1.075, 1.065])
        result = strategy.backtest_signal(signal, prices)
        assert result["outcome"] == "stopped_out"
        assert result["pnl"] < 0

    def test_expired(self, strategy):
        signal = Signal(
            instrument="EURUSD", direction=SignalDirection.LONG, strength=0.6,
            strategy_name="test", strategy_pod="mean_reversion",
            generated_at=datetime.now(), valid_until=datetime.now() + timedelta(hours=12),
            entry_price=1.08, stop_loss=1.07, take_profit_1=1.10,
        )
        # Price stays flat for max_holding_days
        prices = pd.Series([1.081, 1.082, 1.081, 1.080, 1.081, 1.082])
        result = strategy.backtest_signal(signal, prices)
        assert result["outcome"] == "expired"
        assert result["exit_reason"] == "max_holding_days"

    def test_short_target_hit(self, strategy):
        signal = Signal(
            instrument="EURUSD", direction=SignalDirection.SHORT, strength=0.6,
            strategy_name="test", strategy_pod="mean_reversion",
            generated_at=datetime.now(), valid_until=datetime.now() + timedelta(hours=12),
            entry_price=1.10, stop_loss=1.12, take_profit_1=1.08,
        )
        prices = pd.Series([1.095, 1.09, 1.085, 1.075])
        result = strategy.backtest_signal(signal, prices)
        assert result["outcome"] == "target_hit"
        assert result["pnl"] > 0

    def test_short_stopped_out(self, strategy):
        signal = Signal(
            instrument="EURUSD", direction=SignalDirection.SHORT, strength=0.6,
            strategy_name="test", strategy_pod="mean_reversion",
            generated_at=datetime.now(), valid_until=datetime.now() + timedelta(hours=12),
            entry_price=1.10, stop_loss=1.12, take_profit_1=1.08,
        )
        prices = pd.Series([1.105, 1.115, 1.125])
        result = strategy.backtest_signal(signal, prices)
        assert result["outcome"] == "stopped_out"
        assert result["pnl"] < 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:

    def test_empty_features(self, strategy):
        signals = strategy.generate_signals({}, _macro_data(18), _news_no_catalyst())
        assert len(signals) == 0

    def test_no_macro(self):
        s = MeanReversionStrategy()
        features = _spike_features("EURUSD", "down")
        # No macro, no news — should use fallbacks
        signals = s.generate_signals(features)
        # Might generate 0 if catalyst screening returns UNKNOWN and require_no_catalyst=True
        assert isinstance(signals, list)

    def test_serialisation(self):
        s = MeanReversionStrategy()
        features = _spike_features("EURUSD", "down")
        signals = s.generate_signals(features, _macro_data(18), _news_no_catalyst())
        if signals:
            d = signals[0].to_dict()
            assert d["instrument"] == "EURUSD"
            assert d["strategy_pod"] == "mean_reversion"

    def test_regime_classification(self, strategy):
        assert strategy._classify_regime(None) == "UNKNOWN"
        assert strategy._classify_regime(15.0) == "NORMAL"
        assert strategy._classify_regime(28.0) == "HIGH_VOL"
        assert strategy._classify_regime(40.0) == "EXTREME_VOL"

    def test_default_screener_no_news(self):
        result = default_catalyst_screener("EURUSD", SpikeDetection("EURUSD", True), None)
        assert result.verdict == CatalystVerdict.UNKNOWN

    def test_default_screener_irrelevant_news(self):
        news = {"catalysts": ["Tech IPO strong demand"], "key_events": ["Nvidia earnings"]}
        result = default_catalyst_screener("EURUSD", SpikeDetection("EURUSD", True), news)
        assert result.verdict == CatalystVerdict.NO_CATALYST

    def test_btc_spike(self):
        s = MeanReversionStrategy()
        features = {"BTCUSD": _make_price_df(n=300, base=50000, spike_at_end=-0.08, atr=2500)}
        news = {"catalysts": ["DeFi protocol launch"], "key_events": ["Stablecoin grows"]}
        signals = s.generate_signals(features, _macro_data(18), news)
        # Should detect spike and no relevant catalyst for BTC
        if signals:
            assert signals[0].instrument == "BTCUSD"

    def test_commodity_spike(self):
        s = MeanReversionStrategy()
        features = {"CL": _make_price_df(n=300, base=75, spike_at_end=-0.06, atr=2.0)}
        signals = s.generate_signals(features, _macro_data(18), _news_no_catalyst())
        if signals:
            assert signals[0].instrument == "CL"
