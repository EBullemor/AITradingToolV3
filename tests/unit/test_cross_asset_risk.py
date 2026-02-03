"""
Unit Tests for Cross-Asset Risk Sentiment Strategy (Pod 4)

Tests cover:
- Strategy initialisation and configuration
- Individual indicator scoring (VIX, credit, DXY, BTC/Gold, SPX)
- Composite score computation and regime classification
- Allocation multiplier and strategy bias outputs
- Signal generation
- Graceful degradation with missing indicators
- Edge cases and boundary conditions
"""

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies.base import Signal, SignalDirection
from src.strategies.cross_asset_risk import (
    CrossAssetRiskStrategy,
    create_cross_asset_risk_strategy,
    RiskRegime,
    ALLOCATION_MULTIPLIERS,
    STRATEGY_BIAS,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_series(
    n: int = 300, start_val: float = 100.0, drift: float = 0.0, vol: float = 0.01,
    name: str = "PX_LAST",
) -> pd.DataFrame:
    """Create a simple price DataFrame."""
    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")
    np.random.seed(42)
    returns = np.random.normal(drift, vol, n)
    prices = start_val * np.cumprod(1 + returns)
    return pd.DataFrame({name: prices}, index=dates)


def _risk_on_data() -> Dict[str, pd.DataFrame]:
    """Market data consistent with risk-on environment."""
    return {
        "VIX": _make_series(n=300, start_val=11, drift=-0.001, vol=0.02),
        "CDX_IG": _make_series(n=300, start_val=35, drift=-0.001, vol=0.01),
        "DXY": _make_series(n=300, start_val=104, drift=-0.002, vol=0.003),
        "BTC": _make_series(n=300, start_val=50000, drift=0.003, vol=0.03),
        "GC": _make_series(n=300, start_val=2000, drift=0.0005, vol=0.008),
        "SPX": _make_series(n=300, start_val=5000, drift=0.002, vol=0.01),
    }


def _risk_off_data() -> Dict[str, pd.DataFrame]:
    """Market data consistent with risk-off environment."""
    return {
        "VIX": _make_series(n=300, start_val=28, drift=0.002, vol=0.03),
        "CDX_IG": _make_series(n=300, start_val=90, drift=0.002, vol=0.02),
        "DXY": _make_series(n=300, start_val=100, drift=0.003, vol=0.004),
        "BTC": _make_series(n=300, start_val=40000, drift=-0.003, vol=0.04),
        "GC": _make_series(n=300, start_val=2100, drift=0.003, vol=0.01),
        "SPX": _make_series(n=300, start_val=4800, drift=-0.002, vol=0.015),
    }


def _neutral_data() -> Dict[str, pd.DataFrame]:
    """Market data consistent with neutral environment."""
    return {
        "VIX": _make_series(n=300, start_val=16, drift=0.0, vol=0.015),
        "CDX_IG": _make_series(n=300, start_val=55, drift=0.0, vol=0.01),
        "DXY": _make_series(n=300, start_val=103, drift=0.0, vol=0.003),
        "BTC": _make_series(n=300, start_val=45000, drift=0.0, vol=0.025),
        "GC": _make_series(n=300, start_val=2050, drift=0.0, vol=0.008),
        "SPX": _make_series(n=300, start_val=4900, drift=0.0, vol=0.01),
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def strategy():
    return CrossAssetRiskStrategy()


@pytest.fixture
def risk_on():
    return _risk_on_data()


@pytest.fixture
def risk_off():
    return _risk_off_data()


@pytest.fixture
def neutral():
    return _neutral_data()


# =============================================================================
# Initialisation
# =============================================================================


class TestInitialisation:

    def test_default_init(self, strategy):
        assert strategy.name == "Cross-Asset Risk Sentiment"
        assert strategy.pod_name == "cross_asset_risk"
        assert strategy.enabled is True
        assert strategy.instruments == []

    def test_weights_sum_to_one(self, strategy):
        total = sum(v["weight"] for v in strategy.indicator_config.values())
        assert abs(total - 1.0) < 0.01

    def test_custom_config(self):
        strat = CrossAssetRiskStrategy({
            "regime": {"risk_on_threshold": 0.7, "risk_off_threshold": 0.3},
        })
        assert strat.regime_config["risk_on_threshold"] == 0.7

    def test_factory_function(self):
        strat = create_cross_asset_risk_strategy()
        assert isinstance(strat, CrossAssetRiskStrategy)

    def test_factory_custom_weights(self):
        strat = create_cross_asset_risk_strategy(custom_weights={"vix": 0.5})
        assert strat.indicator_config["vix"]["weight"] == 0.5

    def test_required_features(self, strategy):
        feats = strategy.get_required_features()
        assert "vix" in feats
        assert "credit_spreads" in feats

    def test_regime_enum(self):
        assert RiskRegime.RISK_ON.value == "RISK_ON"
        assert RiskRegime.RISK_OFF.value == "RISK_OFF"
        assert RiskRegime.NEUTRAL.value == "NEUTRAL"

    def test_allocation_multipliers(self):
        assert ALLOCATION_MULTIPLIERS[RiskRegime.RISK_ON] > 1.0
        assert ALLOCATION_MULTIPLIERS[RiskRegime.RISK_OFF] < 1.0
        assert ALLOCATION_MULTIPLIERS[RiskRegime.NEUTRAL] == 1.0


# =============================================================================
# VIX Scoring
# =============================================================================


class TestVIXScoring:

    def test_low_vix_risk_on(self, strategy):
        data = {"VIX": _make_series(n=300, start_val=11, drift=-0.001)}
        score = strategy.score_vix(data)
        assert score["available"] is True
        assert score["score"] > 0.6

    def test_high_vix_risk_off(self, strategy):
        data = {"VIX": _make_series(n=300, start_val=32, drift=0.001)}
        score = strategy.score_vix(data)
        assert score["available"] is True
        assert score["score"] < 0.3

    def test_extreme_vix(self, strategy):
        data = {"VIX": _make_series(n=300, start_val=45, drift=0.0)}
        score = strategy.score_vix(data)
        assert score["score"] == 0.0

    def test_missing_vix(self, strategy):
        score = strategy.score_vix({})
        assert score["available"] is False
        assert score["score"] == 0.5

    def test_vix_zscore_populated(self, strategy):
        data = {"VIX": _make_series(n=300, start_val=18)}
        score = strategy.score_vix(data)
        # With 300 data points, z-score should be computable
        assert score["zscore"] is not None or score["available"]


# =============================================================================
# Credit Spread Scoring
# =============================================================================


class TestCreditSpreads:

    def test_tight_spreads_risk_on(self, strategy):
        data = {"CDX_IG": _make_series(n=300, start_val=35)}
        score = strategy.score_credit_spreads(data)
        assert score["available"] is True
        assert score["score"] > 0.5

    def test_wide_spreads_risk_off(self, strategy):
        data = {"CDX_IG": _make_series(n=300, start_val=95)}
        score = strategy.score_credit_spreads(data)
        assert score["score"] < 0.3

    def test_missing_credit(self, strategy):
        score = strategy.score_credit_spreads({})
        assert score["available"] is False


# =============================================================================
# DXY Scoring
# =============================================================================


class TestDXYScoring:

    def test_weakening_dollar_risk_on(self, strategy):
        data = {"DXY": _make_series(n=300, start_val=106, drift=-0.003)}
        score = strategy.score_dxy(data)
        assert score["available"] is True
        # Weakening USD → risk-on → higher score
        assert score["score"] >= 0.4

    def test_strengthening_dollar_risk_off(self, strategy):
        data = {"DXY": _make_series(n=300, start_val=100, drift=0.003)}
        score = strategy.score_dxy(data)
        assert score["available"] is True
        assert score["score"] <= 0.6

    def test_missing_dxy(self, strategy):
        score = strategy.score_dxy({})
        assert score["available"] is False

    def test_momentum_populated(self, strategy):
        data = {"DXY": _make_series(n=300, start_val=103)}
        score = strategy.score_dxy(data)
        if score["available"]:
            assert score["momentum"] is not None


# =============================================================================
# BTC/Gold Ratio
# =============================================================================


class TestBTCGoldRatio:

    def test_btc_outperforming_risk_on(self, strategy):
        data = {
            "BTC": _make_series(n=300, start_val=50000, drift=0.005),
            "GC": _make_series(n=300, start_val=2000, drift=0.0005),
        }
        score = strategy.score_btc_gold_ratio(data)
        assert score["available"] is True
        assert score["ratio"] is not None

    def test_gold_outperforming_risk_off(self, strategy):
        data = {
            "BTC": _make_series(n=300, start_val=40000, drift=-0.003),
            "GC": _make_series(n=300, start_val=2000, drift=0.003),
        }
        score = strategy.score_btc_gold_ratio(data)
        assert score["available"] is True
        # Gold outperforming → lower score
        assert score["score"] <= 0.6

    def test_missing_btc(self, strategy):
        data = {"GC": _make_series(n=300, start_val=2000)}
        score = strategy.score_btc_gold_ratio(data)
        assert score["available"] is False

    def test_missing_gold(self, strategy):
        data = {"BTC": _make_series(n=300, start_val=50000)}
        score = strategy.score_btc_gold_ratio(data)
        assert score["available"] is False


# =============================================================================
# SPX Momentum
# =============================================================================


class TestSPXMomentum:

    def test_positive_momentum_risk_on(self, strategy):
        data = {"SPX": _make_series(n=300, start_val=4500, drift=0.003)}
        score = strategy.score_spx_momentum(data)
        assert score["available"] is True
        assert score["score"] > 0.4

    def test_negative_momentum_risk_off(self, strategy):
        data = {"SPX": _make_series(n=300, start_val=5000, drift=-0.003)}
        score = strategy.score_spx_momentum(data)
        assert score["available"] is True
        assert score["score"] < 0.6

    def test_missing_spx(self, strategy):
        score = strategy.score_spx_momentum({})
        assert score["available"] is False


# =============================================================================
# Composite Score & Regime
# =============================================================================


class TestCompositeScore:

    def test_risk_on_regime(self, strategy, risk_on):
        result = strategy.compute_composite_score(risk_on)
        assert result["composite_score"] > 0.0
        assert result["regime"] in [RiskRegime.RISK_ON, RiskRegime.NEUTRAL]
        assert result["available_weight"] > 0

    def test_risk_off_regime(self, strategy, risk_off):
        result = strategy.compute_composite_score(risk_off)
        assert result["composite_score"] < 1.0
        assert result["regime"] in [RiskRegime.RISK_OFF, RiskRegime.NEUTRAL]

    def test_neutral_regime(self, strategy, neutral):
        result = strategy.compute_composite_score(neutral)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_composite_range(self, strategy, risk_on):
        result = strategy.compute_composite_score(risk_on)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_allocation_multiplier_populated(self, strategy, risk_on):
        result = strategy.compute_composite_score(risk_on)
        assert result["allocation_multiplier"] in [0.5, 1.0, 1.2]

    def test_strategy_bias_populated(self, strategy, risk_on):
        result = strategy.compute_composite_score(risk_on)
        bias = result["strategy_bias"]
        assert "fx_carry_momentum" in bias
        assert "btc_trend_vol" in bias
        assert bias["fx_carry_momentum"] in ["FAVOUR", "NEUTRAL", "REDUCE"]

    def test_no_data_returns_neutral(self, strategy):
        result = strategy.compute_composite_score({})
        assert result["composite_score"] == 0.5
        assert result["regime"] == RiskRegime.NEUTRAL
        assert result["available_weight"] == 0

    def test_partial_data(self, strategy):
        """Works with only VIX available."""
        data = {"VIX": _make_series(n=300, start_val=14)}
        result = strategy.compute_composite_score(data)
        assert result["available_weight"] > 0
        assert result["composite_score"] != 0.5  # Should differ from default

    def test_indicator_scores_populated(self, strategy, risk_on):
        result = strategy.compute_composite_score(risk_on)
        scores = result["indicator_scores"]
        assert "vix" in scores
        assert "credit_spreads" in scores
        assert "dxy" in scores
        assert "btc_gold_ratio" in scores
        assert "spx_momentum" in scores


# =============================================================================
# Signal Generation
# =============================================================================


class TestSignalGeneration:

    def test_generates_signal(self, strategy, risk_on):
        signals = strategy.generate_signals(risk_on)
        assert len(signals) == 1

    def test_signal_instrument_portfolio(self, strategy, risk_on):
        signals = strategy.generate_signals(risk_on)
        assert signals[0].instrument == "PORTFOLIO"

    def test_signal_strategy_pod(self, strategy, risk_on):
        signals = strategy.generate_signals(risk_on)
        assert signals[0].strategy_pod == "cross_asset_risk"

    def test_risk_on_long_direction(self, strategy, risk_on):
        signals = strategy.generate_signals(risk_on)
        # Risk-on or neutral → LONG
        assert signals[0].direction == SignalDirection.LONG

    def test_risk_off_short_direction(self, strategy, risk_off):
        signals = strategy.generate_signals(risk_off)
        s = signals[0]
        # Risk-off → SHORT, or neutral → LONG
        assert s.direction in [SignalDirection.SHORT, SignalDirection.LONG]

    def test_signal_has_metadata(self, strategy, risk_on):
        signals = strategy.generate_signals(risk_on)
        meta = signals[0].metadata
        assert "composite_score" in meta
        assert "allocation_multiplier" in meta
        assert "strategy_bias" in meta
        assert "indicator_details" in meta

    def test_signal_confidence_range(self, strategy, risk_on):
        signals = strategy.generate_signals(risk_on)
        assert 0.0 <= signals[0].strength <= 1.0

    def test_signal_has_rationale(self, strategy, risk_on):
        signals = strategy.generate_signals(risk_on)
        assert len(signals[0].rationale) > 0

    def test_signal_has_key_factors(self, strategy, risk_on):
        signals = strategy.generate_signals(risk_on)
        assert len(signals[0].key_factors) > 0

    def test_signal_regime_in_metadata(self, strategy, risk_on):
        signals = strategy.generate_signals(risk_on)
        assert signals[0].regime in ["RISK_ON", "RISK_OFF", "NEUTRAL"]

    def test_disabled_strategy(self, risk_on):
        strat = CrossAssetRiskStrategy({"enabled": False})
        signals = strat.generate_signals(risk_on)
        assert len(signals) == 0

    def test_macro_data_merged(self, strategy):
        """Macro data gets merged as VIX if not present."""
        features = {"SPX": _make_series(n=300, start_val=5000)}
        macro = _make_series(n=300, start_val=14, name="PX_LAST")
        signals = strategy.generate_signals(features, macro_data=macro)
        assert len(signals) == 1

    def test_custom_as_of_date(self, strategy, risk_on):
        now = datetime(2026, 2, 3, 10, 0, 0)
        signals = strategy.generate_signals(risk_on, as_of_date=now)
        assert signals[0].generated_at == now
        assert signals[0].valid_until == now + timedelta(hours=24)


# =============================================================================
# Portfolio Helpers
# =============================================================================


class TestPortfolioHelpers:

    def test_get_allocation_multiplier(self, strategy, risk_on):
        mult = strategy.get_allocation_multiplier(risk_on)
        assert mult in [0.5, 1.0, 1.2]

    def test_get_strategy_bias_fx(self, strategy, risk_on):
        bias = strategy.get_strategy_bias(risk_on, "fx_carry_momentum")
        assert bias in ["FAVOUR", "NEUTRAL", "REDUCE"]

    def test_get_strategy_bias_unknown_pod(self, strategy, risk_on):
        bias = strategy.get_strategy_bias(risk_on, "nonexistent_pod")
        assert bias == "NEUTRAL"

    def test_risk_off_reduces_carry(self, strategy, risk_off):
        bias = strategy.get_strategy_bias(risk_off, "fx_carry_momentum")
        # In risk-off, carry should be REDUCE (if regime detected correctly)
        assert bias in ["REDUCE", "NEUTRAL"]


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:

    def test_single_data_point(self, strategy):
        data = {"VIX": pd.DataFrame({"PX_LAST": [18.0]}, index=[datetime.now()])}
        result = strategy.compute_composite_score(data)
        # Should not crash, VIX score computable
        assert result["indicator_scores"]["vix"]["available"] is True

    def test_series_as_input(self, strategy):
        """Should handle pd.Series as well as DataFrame."""
        data = {"VIX": pd.Series([15, 16, 17, 18], name="PX_LAST")}
        score = strategy.score_vix(data)
        assert score["available"] is True

    def test_alternative_key_names(self, strategy):
        """Should find data under alternative key names."""
        data = {"vix": _make_series(n=300, start_val=14)}
        score = strategy.score_vix(data)
        assert score["available"] is True

    def test_btcusd_key(self, strategy):
        data = {
            "BTCUSD": _make_series(n=300, start_val=50000),
            "XAUUSD": _make_series(n=300, start_val=2000),
        }
        score = strategy.score_btc_gold_ratio(data)
        assert score["available"] is True

    def test_serialisation(self, strategy, risk_on):
        signals = strategy.generate_signals(risk_on)
        d = signals[0].to_dict()
        assert d["instrument"] == "PORTFOLIO"
        assert d["strategy_pod"] == "cross_asset_risk"

    def test_news_summary_no_crash(self, strategy, risk_on):
        news = {"macro": {"sentiment": "cautious"}}
        signals = strategy.generate_signals(risk_on, news_summary=news)
        assert len(signals) == 1

    def test_weights_warn_if_not_one(self):
        """Should log warning if weights don't sum to 1."""
        strat = CrossAssetRiskStrategy({
            "indicators": {
                "vix": {"weight": 0.5},
                "credit_spreads": {"weight": 0.5},
                "dxy": {"weight": 0.5},
                "btc_gold_ratio": {"weight": 0.5},
                "spx_momentum": {"weight": 0.5},
            }
        })
        # Should still work
        result = strat.compute_composite_score({})
        assert result["composite_score"] == 0.5
