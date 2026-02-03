"""
Cross-Asset Risk Sentiment Strategy (Pod 4)

Portfolio-level risk-on/off allocation strategy.

Unlike Pods 1-3 which generate directional signals on specific instruments,
Pod 4 produces a composite risk score that acts as an allocation multiplier
for the entire portfolio.

Indicators:
    - VIX level and z-score (weight: 0.30)
    - Credit spreads / CDX IG proxy (weight: 0.20)
    - DXY momentum (weight: 0.20)
    - BTC/Gold ratio momentum (weight: 0.15)
    - SPX momentum (weight: 0.15)

Output:
    RISK_ON  (composite > 0.6): Full allocation, favour carry/momentum
    NEUTRAL  (0.4 - 0.6):      Normal allocation
    RISK_OFF (composite < 0.4): Reduce exposure, favour hedges

Holding period: 1-2 weeks
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .base import (
    BaseStrategy,
    Signal,
    SignalDirection,
    SignalStrength,
    SignalStatus,
    load_strategy_config,
)


class RiskRegime(str, Enum):
    """Risk regime classification."""
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"


# Allocation multiplier by regime
ALLOCATION_MULTIPLIERS = {
    RiskRegime.RISK_ON: 1.2,   # Scale up exposure
    RiskRegime.NEUTRAL: 1.0,   # Normal
    RiskRegime.RISK_OFF: 0.5,  # Halve exposure
}

# Strategy bias map: which pods benefit from each regime
STRATEGY_BIAS = {
    RiskRegime.RISK_ON: {
        "fx_carry_momentum": "FAVOUR",      # Carry works in risk-on
        "btc_trend_vol": "FAVOUR",           # BTC correlates with risk appetite
        "commodities_ts": "NEUTRAL",         # Commodity-specific drivers dominate
        "mean_reversion": "NEUTRAL",
    },
    RiskRegime.NEUTRAL: {
        "fx_carry_momentum": "NEUTRAL",
        "btc_trend_vol": "NEUTRAL",
        "commodities_ts": "NEUTRAL",
        "mean_reversion": "NEUTRAL",
    },
    RiskRegime.RISK_OFF: {
        "fx_carry_momentum": "REDUCE",       # Carry unwinds in risk-off
        "btc_trend_vol": "REDUCE",           # BTC sells off
        "commodities_ts": "NEUTRAL",
        "mean_reversion": "FAVOUR",           # Spikes more common
    },
}


class CrossAssetRiskStrategy(BaseStrategy):
    """
    Cross-Asset Risk Sentiment Strategy.

    Computes a composite risk score from multiple cross-asset indicators
    and classifies the market into RISK_ON, NEUTRAL, or RISK_OFF.

    This strategy does NOT generate instrument-specific directional signals.
    Instead, it produces:
      1. A risk regime classification
      2. An allocation multiplier (scalar)
      3. Per-strategy bias recommendations

    Other pods should consume these outputs to scale their exposure.
    """

    DEFAULT_CONFIG = {
        "pod_name": "cross_asset_risk",
        "enabled": True,
        "instruments": [],  # Portfolio-level, no specific instruments
        "signal_validity_hours": 24,
        "max_signals_per_run": 1,  # Only one regime signal per run

        # Indicator weights (must sum to 1.0)
        "indicators": {
            "vix": {
                "weight": 0.30,
                "risk_off_threshold": 20,
                "risk_on_threshold": 12,
                "extreme_threshold": 30,
                "lookback_days": 252,
            },
            "credit_spreads": {
                "weight": 0.20,
                "risk_off_threshold_bps": 80,
                "risk_on_threshold_bps": 40,
                "lookback_days": 252,
            },
            "dxy": {
                "weight": 0.20,
                "momentum_period": 21,
                "lookback_days": 252,
            },
            "btc_gold_ratio": {
                "weight": 0.15,
                "momentum_period": 21,
                "lookback_days": 252,
            },
            "spx_momentum": {
                "weight": 0.15,
                "period": 63,
                "lookback_days": 252,
            },
        },

        # Regime classification
        "regime": {
            "risk_on_threshold": 0.6,
            "risk_off_threshold": 0.4,
        },

        # Signal generation
        "signal": {
            "holding_period_days": [7, 14],
            "output_type": "allocation",
        },
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise Cross-Asset Risk strategy.

        Args:
            config: Strategy configuration (uses defaults if not provided)
        """
        merged_config = self.DEFAULT_CONFIG.copy()
        if config:
            self._deep_merge(merged_config, config)

        super().__init__(merged_config, name="Cross-Asset Risk Sentiment")

        self.indicator_config = merged_config["indicators"]
        self.regime_config = merged_config["regime"]
        self.signal_config = merged_config["signal"]

        # Validate weights sum to ~1.0
        total_weight = sum(v["weight"] for v in self.indicator_config.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Indicator weights sum to {total_weight:.2f}, expected 1.0"
            )

        logger.info(
            f"Cross-Asset Risk strategy initialised | "
            f"Indicators: {list(self.indicator_config.keys())} | "
            f"Risk-on: >{self.regime_config['risk_on_threshold']} | "
            f"Risk-off: <{self.regime_config['risk_off_threshold']}"
        )

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Deep merge override into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get_required_features(self) -> List[str]:
        """Features required by this strategy."""
        return [
            "vix",
            "credit_spreads",
            "dxy",
            "btc_price",
            "gold_price",
            "spx_price",
        ]

    # =========================================================================
    # Individual Indicator Scoring
    # =========================================================================

    def score_vix(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Score VIX indicator.

        Low VIX → risk-on (score near 1.0)
        High VIX → risk-off (score near 0.0)

        Args:
            data: Market data dict (expects 'VIX' key)

        Returns:
            Dict with score, raw value, z-score, and detail
        """
        cfg = self.indicator_config["vix"]
        result = {"score": 0.5, "value": None, "zscore": None, "available": False}

        vix_series = self._extract_series(data, ["VIX", "vix"], "PX_LAST")
        if vix_series is None or vix_series.empty:
            logger.debug("VIX data not available")
            return result

        vix = float(vix_series.iloc[-1])
        result["value"] = vix
        result["available"] = True

        # Z-score
        lookback = min(cfg["lookback_days"], len(vix_series))
        if lookback > 20:
            mean = vix_series.iloc[-lookback:].mean()
            std = vix_series.iloc[-lookback:].std()
            if std > 0:
                result["zscore"] = (vix - mean) / std

        # Score: map VIX to 0-1 (inverted — low VIX = high score = risk-on)
        risk_on = cfg["risk_on_threshold"]
        risk_off = cfg["risk_off_threshold"]

        if vix <= risk_on:
            result["score"] = 1.0
        elif vix >= risk_off:
            # Further penalise extreme VIX
            extreme = cfg.get("extreme_threshold", 30)
            if vix >= extreme:
                result["score"] = 0.0
            else:
                result["score"] = max(0.0, 1.0 - (vix - risk_on) / (extreme - risk_on))
        else:
            result["score"] = 1.0 - (vix - risk_on) / (risk_off - risk_on)

        logger.debug(f"VIX: {vix:.1f} → score {result['score']:.2f}")
        return result

    def score_credit_spreads(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Score credit spreads indicator.

        Tight spreads → risk-on (score near 1.0)
        Wide spreads → risk-off (score near 0.0)

        Args:
            data: Market data dict (expects 'CDX_IG' or 'credit_spreads' key)

        Returns:
            Dict with score, raw value, z-score
        """
        cfg = self.indicator_config["credit_spreads"]
        result = {"score": 0.5, "value": None, "zscore": None, "available": False}

        series = self._extract_series(
            data, ["CDX_IG", "credit_spreads", "CDX", "IG_SPREAD"], "PX_LAST"
        )
        if series is None or series.empty:
            logger.debug("Credit spread data not available")
            return result

        spread = float(series.iloc[-1])
        result["value"] = spread
        result["available"] = True

        # Z-score
        lookback = min(cfg["lookback_days"], len(series))
        if lookback > 20:
            mean = series.iloc[-lookback:].mean()
            std = series.iloc[-lookback:].std()
            if std > 0:
                result["zscore"] = (spread - mean) / std

        # Score: map spreads to 0-1 (inverted — tight = high score)
        risk_on_bps = cfg["risk_on_threshold_bps"]
        risk_off_bps = cfg["risk_off_threshold_bps"]

        if spread <= risk_on_bps:
            result["score"] = 1.0
        elif spread >= risk_off_bps:
            result["score"] = 0.0
        else:
            result["score"] = 1.0 - (spread - risk_on_bps) / (risk_off_bps - risk_on_bps)

        logger.debug(f"Credit spreads: {spread:.0f}bps → score {result['score']:.2f}")
        return result

    def score_dxy(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Score DXY (US Dollar Index) momentum.

        Falling DXY → risk-on (score near 1.0)
        Rising DXY → risk-off (score near 0.0)

        Risk-off episodes typically feature a strengthening USD.
        """
        cfg = self.indicator_config["dxy"]
        result = {"score": 0.5, "value": None, "momentum": None, "available": False}

        series = self._extract_series(data, ["DXY", "dxy", "USDX"], "PX_LAST")
        if series is None or len(series) < cfg["momentum_period"] + 1:
            logger.debug("DXY data not available")
            return result

        result["value"] = float(series.iloc[-1])
        result["available"] = True

        # Momentum: % change over period
        momentum = (series.iloc[-1] / series.iloc[-cfg["momentum_period"] - 1]) - 1
        result["momentum"] = float(momentum)

        # Z-score of momentum
        lookback = min(cfg["lookback_days"], len(series))
        if lookback > cfg["momentum_period"] + 20:
            mom_series = series.pct_change(cfg["momentum_period"]).iloc[-lookback:]
            mean = mom_series.mean()
            std = mom_series.std()
            if std > 0:
                mom_z = (momentum - mean) / std
                # Inverted: negative DXY momentum = risk-on
                # Map z-score of -2 to +2 → score of 1.0 to 0.0
                result["score"] = np.clip(0.5 - mom_z * 0.25, 0.0, 1.0)
            else:
                result["score"] = 0.5
        else:
            # Simple linear mapping: -2% to +2% → 1.0 to 0.0
            result["score"] = np.clip(0.5 - momentum * 25, 0.0, 1.0)

        logger.debug(f"DXY: {result['value']:.1f}, mom={momentum:+.3f} → score {result['score']:.2f}")
        return result

    def score_btc_gold_ratio(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Score BTC/Gold ratio momentum.

        Rising BTC/Gold → risk appetite increasing (risk-on, score near 1.0)
        Falling BTC/Gold → flight to safety (risk-off, score near 0.0)
        """
        cfg = self.indicator_config["btc_gold_ratio"]
        result = {"score": 0.5, "ratio": None, "momentum": None, "available": False}

        btc = self._extract_series(data, ["BTC", "BTCUSD", "btc"], "PX_LAST")
        gold = self._extract_series(data, ["GC", "XAUUSD", "gold", "GC1"], "PX_LAST")

        if btc is None or gold is None or btc.empty or gold.empty:
            logger.debug("BTC or Gold data not available for ratio")
            return result

        # Align indices
        common = btc.index.intersection(gold.index)
        if len(common) < cfg["momentum_period"] + 1:
            return result

        btc_aligned = btc.loc[common]
        gold_aligned = gold.loc[common]

        ratio = btc_aligned / gold_aligned
        result["ratio"] = float(ratio.iloc[-1])
        result["available"] = True

        # Momentum
        period = cfg["momentum_period"]
        if len(ratio) > period:
            momentum = (ratio.iloc[-1] / ratio.iloc[-period - 1]) - 1
            result["momentum"] = float(momentum)

            # Z-score approach
            lookback = min(cfg["lookback_days"], len(ratio))
            if lookback > period + 20:
                mom_series = ratio.pct_change(period).iloc[-lookback:]
                mean = mom_series.mean()
                std = mom_series.std()
                if std > 0:
                    mom_z = (momentum - mean) / std
                    # Positive ratio momentum = risk-on
                    result["score"] = np.clip(0.5 + mom_z * 0.25, 0.0, 1.0)
                else:
                    result["score"] = 0.5
            else:
                result["score"] = np.clip(0.5 + momentum * 10, 0.0, 1.0)

        logger.debug(f"BTC/Gold: {result['ratio']:.2f}, mom={result['momentum']} → score {result['score']:.2f}")
        return result

    def score_spx_momentum(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Score S&P 500 momentum.

        Positive SPX momentum → risk-on (score near 1.0)
        Negative SPX momentum → risk-off (score near 0.0)
        """
        cfg = self.indicator_config["spx_momentum"]
        result = {"score": 0.5, "value": None, "momentum": None, "available": False}

        series = self._extract_series(data, ["SPX", "SPY", "spx", "ES"], "PX_LAST")
        if series is None or len(series) < cfg["period"] + 1:
            logger.debug("SPX data not available")
            return result

        result["value"] = float(series.iloc[-1])
        result["available"] = True

        period = cfg["period"]
        momentum = (series.iloc[-1] / series.iloc[-period - 1]) - 1
        result["momentum"] = float(momentum)

        # Z-score approach
        lookback = min(cfg["lookback_days"], len(series))
        if lookback > period + 20:
            mom_series = series.pct_change(period).iloc[-lookback:]
            mean = mom_series.mean()
            std = mom_series.std()
            if std > 0:
                mom_z = (momentum - mean) / std
                result["score"] = np.clip(0.5 + mom_z * 0.25, 0.0, 1.0)
            else:
                result["score"] = 0.5
        else:
            # Simple: +10% = 1.0, -10% = 0.0
            result["score"] = np.clip(0.5 + momentum * 5, 0.0, 1.0)

        logger.debug(f"SPX: {result['value']:.0f}, mom={momentum:+.3f} → score {result['score']:.2f}")
        return result

    # =========================================================================
    # Composite Score & Regime
    # =========================================================================

    def compute_composite_score(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Compute weighted composite risk score from all indicators.

        Args:
            data: Dict mapping ticker/name to DataFrame

        Returns:
            Dict with:
                composite_score: float (0-1, higher = more risk-on)
                regime: RiskRegime
                allocation_multiplier: float
                indicator_scores: Dict per indicator
                available_weight: float (sum of weights with data)
        """
        scores = {
            "vix": self.score_vix(data),
            "credit_spreads": self.score_credit_spreads(data),
            "dxy": self.score_dxy(data),
            "btc_gold_ratio": self.score_btc_gold_ratio(data),
            "spx_momentum": self.score_spx_momentum(data),
        }

        # Weight-adjusted composite (only include available indicators)
        weighted_sum = 0.0
        available_weight = 0.0

        for name, score_data in scores.items():
            weight = self.indicator_config[name]["weight"]
            if score_data["available"]:
                weighted_sum += weight * score_data["score"]
                available_weight += weight

        # Normalise by available weight
        if available_weight > 0:
            composite = weighted_sum / available_weight
        else:
            composite = 0.5  # No data → neutral
            logger.warning("No indicator data available, defaulting to NEUTRAL")

        # Classify regime
        on_threshold = self.regime_config["risk_on_threshold"]
        off_threshold = self.regime_config["risk_off_threshold"]

        if composite > on_threshold:
            regime = RiskRegime.RISK_ON
        elif composite < off_threshold:
            regime = RiskRegime.RISK_OFF
        else:
            regime = RiskRegime.NEUTRAL

        multiplier = ALLOCATION_MULTIPLIERS[regime]
        bias = STRATEGY_BIAS[regime]

        logger.info(
            f"Composite risk score: {composite:.3f} → {regime.value} | "
            f"Allocation multiplier: {multiplier:.1f} | "
            f"Available weight: {available_weight:.0%}"
        )

        return {
            "composite_score": composite,
            "regime": regime,
            "allocation_multiplier": multiplier,
            "strategy_bias": bias,
            "indicator_scores": scores,
            "available_weight": available_weight,
        }

    # =========================================================================
    # Signal Generation
    # =========================================================================

    def generate_signals(
        self,
        features: Dict[str, pd.DataFrame],
        macro_data: Optional[pd.DataFrame] = None,
        news_summary: Optional[Dict] = None,
        as_of_date: Optional[datetime] = None,
    ) -> List[Signal]:
        """
        Generate cross-asset risk signal.

        Unlike other pods, this produces a single portfolio-level signal
        indicating risk regime and allocation guidance.

        Args:
            features: Dict mapping ticker to features DataFrame.
                      Expected keys: VIX, CDX_IG, DXY, BTC/BTCUSD, GC/gold, SPX
            macro_data: Additional macro data (merged into features if provided)
            news_summary: News context (optional)
            as_of_date: Reference date

        Returns:
            List with 0 or 1 Signal
        """
        if not self.enabled:
            logger.info("Cross-Asset Risk strategy disabled, skipping")
            return []

        if as_of_date is None:
            as_of_date = datetime.now()

        # Merge macro_data into features if provided separately
        merged = dict(features)
        if macro_data is not None and not macro_data.empty:
            if "VIX" not in merged:
                merged["VIX"] = macro_data

        # Compute composite
        result = self.compute_composite_score(merged)

        # Build signal
        regime = result["regime"]
        composite = result["composite_score"]
        multiplier = result["allocation_multiplier"]
        bias = result["strategy_bias"]

        # Direction: LONG = risk-on, SHORT = risk-off
        if regime == RiskRegime.RISK_ON:
            direction = SignalDirection.LONG
        elif regime == RiskRegime.RISK_OFF:
            direction = SignalDirection.SHORT
        else:
            # NEUTRAL — still emit a signal for tracking, low confidence
            direction = SignalDirection.LONG

        # Confidence based on how far from neutral
        if regime == RiskRegime.NEUTRAL:
            confidence = 0.3
        else:
            distance = abs(composite - 0.5)
            confidence = min(0.4 + distance * 1.5, 0.95)

        # Rationale
        rationale = self._build_rationale(result)
        key_factors = self._build_key_factors(result)

        # Confidence drivers
        confidence_drivers = {}
        for name, score_data in result["indicator_scores"].items():
            if score_data["available"]:
                weight = self.indicator_config[name]["weight"]
                confidence_drivers[name] = score_data["score"] * weight

        holding_min, holding_max = self.signal_config["holding_period_days"]

        signal = Signal(
            instrument="PORTFOLIO",
            direction=direction,
            strength=confidence,
            strategy_name=self.name,
            strategy_pod=self.pod_name,
            generated_at=as_of_date,
            valid_until=as_of_date + timedelta(hours=self.signal_validity_hours),
            entry_price=composite,       # Composite score as "price"
            stop_loss=self.regime_config["risk_off_threshold"],
            take_profit_1=multiplier,    # Allocation multiplier
            rationale=rationale,
            key_factors=key_factors,
            regime=regime.value,
            confidence_drivers=confidence_drivers,
            metadata={
                "composite_score": composite,
                "allocation_multiplier": multiplier,
                "strategy_bias": {k: v for k, v in bias.items()},
                "available_weight": result["available_weight"],
                "indicator_details": {
                    name: {
                        "score": s["score"],
                        "value": s.get("value"),
                        "available": s["available"],
                    }
                    for name, s in result["indicator_scores"].items()
                },
            },
        )

        logger.info(
            f"Risk signal: {regime.value} | composite={composite:.3f} | "
            f"multiplier={multiplier:.1f} | confidence={confidence:.2f}"
        )
        return [signal]

    # =========================================================================
    # Portfolio Helpers (for other pods to consume)
    # =========================================================================

    def get_allocation_multiplier(
        self, data: Dict[str, pd.DataFrame]
    ) -> float:
        """
        Quick method to get the current allocation multiplier.

        Args:
            data: Market data dict

        Returns:
            Allocation multiplier (0.5 to 1.2)
        """
        result = self.compute_composite_score(data)
        return result["allocation_multiplier"]

    def get_strategy_bias(
        self, data: Dict[str, pd.DataFrame], strategy_pod: str
    ) -> str:
        """
        Get bias recommendation for a specific strategy pod.

        Args:
            data: Market data dict
            strategy_pod: Pod name (e.g., 'fx_carry_momentum')

        Returns:
            'FAVOUR', 'NEUTRAL', or 'REDUCE'
        """
        result = self.compute_composite_score(data)
        return result["strategy_bias"].get(strategy_pod, "NEUTRAL")

    # =========================================================================
    # Rationale Builders
    # =========================================================================

    def _build_rationale(self, result: Dict) -> str:
        """Build human-readable rationale."""
        regime = result["regime"]
        composite = result["composite_score"]
        multiplier = result["allocation_multiplier"]

        parts = [
            f"{regime.value} regime (composite={composite:.2f}, "
            f"allocation={multiplier:.1f}x)"
        ]

        scores = result["indicator_scores"]
        for name, data in scores.items():
            if data["available"]:
                val_str = ""
                if data.get("value") is not None:
                    val_str = f"={data['value']:.1f}"
                parts.append(f"{name}{val_str} → {data['score']:.2f}")

        return " | ".join(parts)

    def _build_key_factors(self, result: Dict) -> List[str]:
        """Build list of key factors."""
        factors = []
        scores = result["indicator_scores"]

        # VIX
        if scores["vix"]["available"]:
            vix = scores["vix"]["value"]
            if vix > 25:
                factors.append(f"Elevated VIX ({vix:.0f}) — risk-off pressure")
            elif vix < 15:
                factors.append(f"Low VIX ({vix:.0f}) — complacency/risk-on")
            else:
                factors.append(f"VIX at {vix:.0f}")

        # Credit
        if scores["credit_spreads"]["available"]:
            spread = scores["credit_spreads"]["value"]
            factors.append(f"Credit spreads at {spread:.0f}bps")

        # DXY
        if scores["dxy"]["available"]:
            mom = scores["dxy"].get("momentum")
            if mom is not None:
                direction = "strengthening" if mom > 0 else "weakening"
                factors.append(f"USD {direction} ({mom:+.1%})")

        # BTC/Gold
        if scores["btc_gold_ratio"]["available"]:
            mom = scores["btc_gold_ratio"].get("momentum")
            if mom is not None:
                if mom > 0.05:
                    factors.append("BTC outperforming Gold — risk appetite")
                elif mom < -0.05:
                    factors.append("Gold outperforming BTC — safe haven demand")

        # SPX
        if scores["spx_momentum"]["available"]:
            mom = scores["spx_momentum"].get("momentum")
            if mom is not None:
                if mom > 0:
                    factors.append(f"SPX positive momentum ({mom:+.1%})")
                else:
                    factors.append(f"SPX negative momentum ({mom:+.1%})")

        return factors

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _extract_series(
        data: Dict[str, pd.DataFrame],
        possible_keys: List[str],
        column: str = "PX_LAST",
    ) -> Optional[pd.Series]:
        """
        Try to extract a price series from the data dict.

        Searches through possible_keys for a match, then tries to get
        the specified column. Falls back to first numeric column.
        """
        for key in possible_keys:
            if key in data:
                df = data[key]
                if isinstance(df, pd.Series):
                    return df
                if isinstance(df, pd.DataFrame):
                    if column in df.columns:
                        return df[column].dropna()
                    # Fallback: first numeric column
                    numeric = df.select_dtypes(include=[np.number])
                    if not numeric.empty:
                        return numeric.iloc[:, 0].dropna()
        return None


# =============================================================================
# Factory
# =============================================================================


def create_cross_asset_risk_strategy(
    config_path: Optional[str] = None,
    custom_weights: Optional[Dict[str, float]] = None,
) -> CrossAssetRiskStrategy:
    """
    Factory function to create Cross-Asset Risk strategy.

    Args:
        config_path: Path to config file (optional)
        custom_weights: Override indicator weights {name: weight}

    Returns:
        Configured strategy instance
    """
    if config_path:
        config = load_strategy_config("cross_asset_risk")
    else:
        config = {}

    if custom_weights:
        config.setdefault("indicators", {})
        for name, weight in custom_weights.items():
            config["indicators"].setdefault(name, {})["weight"] = weight

    return CrossAssetRiskStrategy(config)
