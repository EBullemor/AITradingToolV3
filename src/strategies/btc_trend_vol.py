"""
BTC Trend + Volatility Breakout Strategy (Pod 2)

Trend following with vol regime detection for Bitcoin.

Signal Logic:
    LONG if:
        - Price > 50-day MA AND price > 200-day MA (uptrend)
        - Vol percentile > 90th (breakout environment)
        - Momentum 1m > 0 (confirmation)
        - Optional: exchange flow z-score < -1 (accumulation via on-chain)
    SHORT if:
        - Price < 50-day MA AND price < 200-day MA (downtrend)
        - Vol percentile > 90th (breakout environment)
        - Momentum 1m < 0 (confirmation)
        - Optional: MVRV > 3.0 (overvaluation via on-chain)

Holding period: 2-8 weeks
Expected Sharpe: 0.7
"""

from datetime import datetime, timedelta
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


class BTCTrendVolStrategy(BaseStrategy):
    """
    BTC Trend + Volatility Breakout Strategy Pod.

    Strategy Logic:
    1. Determine vol regime (QUIET, NORMAL, BREAKOUT)
    2. Assess trend via moving average alignment and trend strength
    3. In BREAKOUT regime: generate directional signal aligned with trend
    4. Apply on-chain enrichment if Glassnode data available
    5. Generate signals with entry/stop/target levels
    """

    # Default configuration
    DEFAULT_CONFIG = {
        "pod_name": "btc_trend_vol",
        "enabled": True,
        "instruments": ["BTCUSD"],
        "signal_validity_hours": 24,
        "max_signals_per_run": 2,

        # Trend parameters
        "trend": {
            "fast_ma": 50,
            "slow_ma": 200,
            "trend_strength_threshold": 0.02,  # Price 2% above/below MA
            "momentum_confirmation_period": 21,  # 1-month momentum
        },

        # Volatility breakout parameters
        "volatility": {
            "lookback_days": 90,
            "breakout_percentile": 90,  # Vol > 90th pctile = breakout
            "rv_period": 30,  # 30-day realized vol
            "expanding_vol_multiplier": 1.5,  # Vol must be 1.5x 5-day ago
            "quiet_percentile": 25,  # Vol < 25th = quiet
        },

        # On-chain parameters (optional, requires Glassnode)
        "onchain": {
            "enabled": False,
            "exchange_flow_zscore_threshold": -1.0,  # Negative = bullish outflows
            "mvrv_overbought_threshold": 3.0,
            "mvrv_oversold_threshold": 1.0,
            "sopr_threshold": 1.0,  # Below 1 = selling at loss = capitulation
            "onchain_confidence_weight": 0.15,
        },

        # Risk management
        "risk": {
            "stop_loss_atr_multiple": 2.5,  # Wider stops for BTC
            "take_profit_atr_multiple": 4.0,
            "take_profit_2_atr_multiple": 6.0,
            "max_position_size_pct": 3.0,
            "trailing_stop_activation": 0.5,  # Activate after 50% of target
        },

        # Signal generation
        "signal": {
            "holding_period_days": [14, 56],  # 2-8 weeks
            "require_trend_confirmation": True,
            "min_confidence": 0.4,
        },
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize BTC Trend+Volatility strategy.

        Args:
            config: Strategy configuration (uses defaults if not provided)
        """
        merged_config = self.DEFAULT_CONFIG.copy()
        if config:
            self._deep_merge(merged_config, config)

        super().__init__(merged_config, name="BTC Trend + Volatility Breakout")

        self.trend_config = merged_config["trend"]
        self.vol_config = merged_config["volatility"]
        self.onchain_config = merged_config["onchain"]
        self.risk_config = merged_config["risk"]
        self.signal_config = merged_config["signal"]

        logger.info(
            f"BTC Trend+Vol strategy initialized | "
            f"MA: {self.trend_config['fast_ma']}/{self.trend_config['slow_ma']} | "
            f"Breakout: >{self.vol_config['breakout_percentile']}th pctile | "
            f"On-chain: {'ON' if self.onchain_config['enabled'] else 'OFF'}"
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
        base_features = [
            "PX_LAST",
            "momentum_score",
            "realized_vol",
            "vol_percentile",
            "atr_14",
        ]
        if self.onchain_config["enabled"]:
            base_features.extend([
                "flow_zscore",
                "mvrv_ratio",
            ])
        return base_features

    # =========================================================================
    # Regime Detection
    # =========================================================================

    def detect_vol_regime(
        self,
        features: pd.DataFrame,
        as_of_date: Optional[datetime] = None,
    ) -> str:
        """
        Detect volatility regime for BTC.

        Args:
            features: BTC features DataFrame
            as_of_date: Reference date

        Returns:
            Regime string: 'QUIET', 'NORMAL', or 'BREAKOUT'
        """
        if features is None or features.empty:
            logger.warning("No features data, defaulting to NORMAL vol regime")
            return "NORMAL"

        # Get latest vol percentile
        vol_pctile = self._get_latest_value(features, "vol_percentile")
        if vol_pctile is None:
            logger.warning("Vol percentile not found, defaulting to NORMAL")
            return "NORMAL"

        breakout_threshold = self.vol_config["breakout_percentile"]
        quiet_threshold = self.vol_config["quiet_percentile"]

        if vol_pctile >= breakout_threshold:
            regime = "BREAKOUT"
        elif vol_pctile <= quiet_threshold:
            regime = "QUIET"
        else:
            regime = "NORMAL"

        logger.info(f"Vol regime: {regime} (percentile={vol_pctile:.1f})")
        return regime

    def detect_trend(
        self,
        features: pd.DataFrame,
        as_of_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Assess BTC trend direction and strength.

        Args:
            features: BTC features DataFrame
            as_of_date: Reference date

        Returns:
            Dict with trend assessment:
                direction: 'UP', 'DOWN', 'FLAT'
                strength: float (0-1)
                above_fast_ma: bool
                above_slow_ma: bool
                momentum_1m: float
        """
        result = {
            "direction": "FLAT",
            "strength": 0.0,
            "above_fast_ma": False,
            "above_slow_ma": False,
            "momentum_1m": 0.0,
            "momentum_3m": 0.0,
        }

        if features is None or features.empty:
            return result

        price = self._get_latest_value(features, "PX_LAST")
        if price is None:
            return result

        # Moving average alignment
        fast_ma = self.trend_config["fast_ma"]
        slow_ma = self.trend_config["slow_ma"]

        # Try to get MAs from features or compute from price
        fast_ma_val = self._get_latest_value(features, f"ma_{fast_ma}")
        slow_ma_val = self._get_latest_value(features, f"ma_{slow_ma}")

        # Fallback: compute from PX_LAST if not in features
        if fast_ma_val is None and "PX_LAST" in features.columns:
            fast_ma_series = features["PX_LAST"].rolling(fast_ma).mean()
            fast_ma_val = fast_ma_series.iloc[-1] if len(fast_ma_series) > 0 else None

        if slow_ma_val is None and "PX_LAST" in features.columns:
            slow_ma_series = features["PX_LAST"].rolling(slow_ma).mean()
            slow_ma_val = slow_ma_series.iloc[-1] if len(slow_ma_series) > 0 else None

        if fast_ma_val is not None:
            result["above_fast_ma"] = price > fast_ma_val
        if slow_ma_val is not None:
            result["above_slow_ma"] = price > slow_ma_val

        # Trend strength: distance from slow MA as a percentage
        threshold = self.trend_config["trend_strength_threshold"]
        if slow_ma_val is not None and slow_ma_val > 0:
            pct_from_ma = (price - slow_ma_val) / slow_ma_val
            result["strength"] = min(abs(pct_from_ma) / 0.20, 1.0)  # Normalise to 0-1

            if pct_from_ma > threshold:
                result["direction"] = "UP"
            elif pct_from_ma < -threshold:
                result["direction"] = "DOWN"
            else:
                result["direction"] = "FLAT"

        # Momentum confirmation
        mom_1m = self._get_latest_value(features, "momentum_1m")
        if mom_1m is None:
            mom_1m = self._get_latest_value(features, "momentum_score")
        result["momentum_1m"] = mom_1m if mom_1m is not None else 0.0

        mom_3m = self._get_latest_value(features, "momentum_3m")
        result["momentum_3m"] = mom_3m if mom_3m is not None else 0.0

        logger.info(
            f"Trend: {result['direction']} | strength={result['strength']:.2f} | "
            f"fast_MA={'above' if result['above_fast_ma'] else 'below'} | "
            f"slow_MA={'above' if result['above_slow_ma'] else 'below'} | "
            f"mom_1m={result['momentum_1m']:.3f}"
        )
        return result

    # =========================================================================
    # On-Chain Enrichment
    # =========================================================================

    def assess_onchain(
        self,
        features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Assess on-chain metrics for signal enrichment.

        Degrades gracefully if Glassnode data is not available.

        Args:
            features: BTC features DataFrame (may include on-chain cols)

        Returns:
            Dict with on-chain assessment:
                available: bool
                bias: 'BULLISH', 'BEARISH', 'NEUTRAL'
                confidence_adjustment: float (-0.15 to +0.15)
                details: dict of metric values
        """
        result = {
            "available": False,
            "bias": "NEUTRAL",
            "confidence_adjustment": 0.0,
            "details": {},
        }

        if not self.onchain_config["enabled"]:
            return result

        signals = []
        weight = self.onchain_config["onchain_confidence_weight"]

        # Exchange flow z-score
        flow_z = self._get_latest_value(features, "flow_zscore")
        if flow_z is not None:
            result["available"] = True
            result["details"]["exchange_flow_zscore"] = flow_z
            threshold = self.onchain_config["exchange_flow_zscore_threshold"]
            if flow_z < threshold:
                signals.append(1.0)  # Bullish: outflows from exchanges
            elif flow_z > -threshold:
                signals.append(-1.0)  # Bearish: inflows to exchanges
            else:
                signals.append(0.0)

        # MVRV ratio
        mvrv = self._get_latest_value(features, "mvrv_ratio")
        if mvrv is None:
            mvrv = self._get_latest_value(features, "mvrv")
        if mvrv is not None:
            result["available"] = True
            result["details"]["mvrv_ratio"] = mvrv
            overbought = self.onchain_config["mvrv_overbought_threshold"]
            oversold = self.onchain_config["mvrv_oversold_threshold"]
            if mvrv > overbought:
                signals.append(-1.0)  # Bearish: overvalued
            elif mvrv < oversold:
                signals.append(1.0)  # Bullish: undervalued
            else:
                signals.append(0.0)

        # SOPR
        sopr = self._get_latest_value(features, "sopr")
        if sopr is not None:
            result["available"] = True
            result["details"]["sopr"] = sopr
            threshold = self.onchain_config["sopr_threshold"]
            if sopr < threshold:
                signals.append(1.0)  # Bullish: selling at loss = capitulation
            elif sopr > threshold * 1.2:
                signals.append(-0.5)  # Slightly bearish: heavy profit-taking
            else:
                signals.append(0.0)

        if signals:
            avg_signal = np.mean(signals)
            result["confidence_adjustment"] = avg_signal * weight

            if avg_signal > 0.3:
                result["bias"] = "BULLISH"
            elif avg_signal < -0.3:
                result["bias"] = "BEARISH"
            else:
                result["bias"] = "NEUTRAL"

            logger.info(
                f"On-chain: {result['bias']} | adj={result['confidence_adjustment']:+.3f} | "
                f"metrics={result['details']}"
            )
        else:
            logger.info("On-chain: no data available, skipping enrichment")

        return result

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
        Generate BTC trend + volatility signals.

        Args:
            features: Dict mapping instrument to features DataFrame
            macro_data: Macro indicators (VIX, DXY — used for cross-asset context)
            news_summary: Summarised news from LLM (optional)
            as_of_date: Reference date

        Returns:
            List of Signal objects
        """
        if not self.enabled:
            logger.info("BTC Trend+Vol strategy disabled, skipping")
            return []

        if as_of_date is None:
            as_of_date = datetime.now()

        signals: List[Signal] = []

        for instrument in self.instruments:
            if instrument not in features:
                logger.warning(f"No features for {instrument}, skipping")
                continue

            inst_features = features[instrument]
            if inst_features.empty:
                logger.warning(f"Empty features for {instrument}, skipping")
                continue

            signal = self._generate_instrument_signal(
                instrument=instrument,
                features=inst_features,
                macro_data=macro_data,
                news_summary=news_summary,
                as_of_date=as_of_date,
            )

            if signal is not None:
                signals.append(signal)

        # Respect max signals limit
        if len(signals) > self.max_signals_per_run:
            signals.sort(key=lambda s: s.strength, reverse=True)
            signals = signals[: self.max_signals_per_run]

        logger.info(f"BTC Trend+Vol generated {len(signals)} signal(s)")
        return signals

    def _generate_instrument_signal(
        self,
        instrument: str,
        features: pd.DataFrame,
        macro_data: Optional[pd.DataFrame],
        news_summary: Optional[Dict],
        as_of_date: datetime,
    ) -> Optional[Signal]:
        """
        Generate signal for a single BTC instrument.

        Args:
            instrument: Instrument ticker
            features: Feature DataFrame for this instrument
            macro_data: Macro data
            news_summary: News context
            as_of_date: Reference date

        Returns:
            Signal or None
        """
        # 1. Detect vol regime
        vol_regime = self.detect_vol_regime(features, as_of_date)

        # 2. Assess trend
        trend = self.detect_trend(features, as_of_date)

        # 3. On-chain enrichment
        onchain = self.assess_onchain(features)

        # 4. Determine if conditions are met
        direction, rationale_parts, confidence_drivers = self._evaluate_conditions(
            vol_regime=vol_regime,
            trend=trend,
            onchain=onchain,
            features=features,
            macro_data=macro_data,
        )

        if direction is None:
            logger.info(f"No signal for {instrument}: conditions not met")
            return None

        # 5. Calculate confidence
        confidence = self._calculate_confidence(
            vol_regime=vol_regime,
            trend=trend,
            onchain=onchain,
            direction=direction,
            confidence_drivers=confidence_drivers,
        )

        min_confidence = self.signal_config["min_confidence"]
        if confidence < min_confidence:
            logger.info(
                f"No signal for {instrument}: confidence {confidence:.2f} < {min_confidence}"
            )
            return None

        # 6. Calculate price levels
        price = self._get_latest_value(features, "PX_LAST")
        atr = self._get_latest_value(features, "atr_14")

        if price is None:
            logger.warning(f"No price for {instrument}, cannot generate signal")
            return None

        # Use ATR for stops, or fallback to percentage-based
        if atr is not None and atr > 0:
            entry, stop, tp1, tp2 = self._calculate_levels_atr(
                price, atr, direction
            )
        else:
            entry, stop, tp1, tp2 = self._calculate_levels_pct(price, direction)

        # 7. Build rationale
        rationale = self._build_rationale(
            direction=direction,
            vol_regime=vol_regime,
            trend=trend,
            onchain=onchain,
            rationale_parts=rationale_parts,
        )

        # 8. Apply news context if available
        if news_summary and instrument in news_summary:
            news_context = news_summary[instrument]
            if isinstance(news_context, dict):
                news_sentiment = news_context.get("sentiment", "neutral")
                rationale_parts.append(f"News sentiment: {news_sentiment}")

        # 9. Create signal
        holding_min, holding_max = self.signal_config["holding_period_days"]

        signal = Signal(
            instrument=instrument,
            direction=(
                SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT
            ),
            strength=confidence,
            strategy_name=self.name,
            strategy_pod=self.pod_name,
            generated_at=as_of_date,
            valid_until=as_of_date + timedelta(hours=self.signal_validity_hours),
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            rationale=rationale,
            key_factors=rationale_parts,
            regime=vol_regime,
            confidence_drivers=confidence_drivers,
        )

        logger.info(
            f"Signal: {direction} {instrument} @ {entry:.2f} | "
            f"SL={stop:.2f} TP1={tp1:.2f} TP2={tp2:.2f} | "
            f"Confidence={confidence:.2f} | Regime={vol_regime}"
        )
        return signal

    def _evaluate_conditions(
        self,
        vol_regime: str,
        trend: Dict,
        onchain: Dict,
        features: pd.DataFrame,
        macro_data: Optional[pd.DataFrame],
    ) -> Tuple[Optional[str], List[str], Dict[str, float]]:
        """
        Evaluate whether signal conditions are met.

        Returns:
            Tuple of (direction, rationale_parts, confidence_drivers)
            direction is None if no signal.
        """
        rationale_parts: List[str] = []
        confidence_drivers: Dict[str, float] = {}

        # Condition 1: Vol regime must be BREAKOUT or NORMAL
        # In QUIET regime, we can still signal if trend is strong enough
        if vol_regime == "BREAKOUT":
            rationale_parts.append("Volatility breakout detected (>90th percentile)")
            confidence_drivers["vol_regime"] = 0.3
        elif vol_regime == "NORMAL":
            rationale_parts.append("Normal vol environment")
            confidence_drivers["vol_regime"] = 0.15
        else:
            # QUIET: only signal if trend is very strong
            if trend["strength"] < 0.6:
                return None, [], {}
            rationale_parts.append("Quiet vol but strong trend override")
            confidence_drivers["vol_regime"] = 0.05

        # Condition 2: Trend direction
        if trend["direction"] == "FLAT":
            return None, [], {}

        direction = "LONG" if trend["direction"] == "UP" else "SHORT"

        # MA alignment
        if direction == "LONG":
            if not trend["above_fast_ma"] or not trend["above_slow_ma"]:
                if self.signal_config["require_trend_confirmation"]:
                    return None, [], {}
                rationale_parts.append("Partial MA alignment (above fast only)")
                confidence_drivers["trend"] = 0.1
            else:
                rationale_parts.append(
                    f"Full uptrend: above {self.trend_config['fast_ma']}/{self.trend_config['slow_ma']} MA"
                )
                confidence_drivers["trend"] = 0.25
        else:
            if trend["above_fast_ma"] or trend["above_slow_ma"]:
                if self.signal_config["require_trend_confirmation"]:
                    return None, [], {}
                rationale_parts.append("Partial MA alignment (below fast only)")
                confidence_drivers["trend"] = 0.1
            else:
                rationale_parts.append(
                    f"Full downtrend: below {self.trend_config['fast_ma']}/{self.trend_config['slow_ma']} MA"
                )
                confidence_drivers["trend"] = 0.25

        # Condition 3: Momentum confirmation
        momentum = trend["momentum_1m"]
        if direction == "LONG" and momentum <= 0:
            if self.signal_config["require_trend_confirmation"]:
                return None, [], {}
            confidence_drivers["momentum"] = 0.0
        elif direction == "SHORT" and momentum >= 0:
            if self.signal_config["require_trend_confirmation"]:
                return None, [], {}
            confidence_drivers["momentum"] = 0.0
        else:
            mom_strength = min(abs(momentum) / 0.15, 1.0)  # Normalise
            confidence_drivers["momentum"] = 0.15 * mom_strength
            rationale_parts.append(f"Momentum confirmed ({momentum:+.3f})")

        # Condition 4: On-chain enrichment (if available)
        if onchain["available"]:
            confidence_drivers["onchain"] = onchain["confidence_adjustment"]
            if onchain["bias"] != "NEUTRAL":
                rationale_parts.append(f"On-chain: {onchain['bias']}")

                # On-chain contradicts direction → reduce confidence
                if direction == "LONG" and onchain["bias"] == "BEARISH":
                    confidence_drivers["onchain"] = -abs(
                        onchain["confidence_adjustment"]
                    )
                    rationale_parts.append("⚠ On-chain divergence (bearish)")
                elif direction == "SHORT" and onchain["bias"] == "BULLISH":
                    confidence_drivers["onchain"] = -abs(
                        onchain["confidence_adjustment"]
                    )
                    rationale_parts.append("⚠ On-chain divergence (bullish)")

        # Condition 5: Macro context (VIX gating for shorts)
        if macro_data is not None and not macro_data.empty:
            vix = self._get_latest_value(macro_data, "PX_LAST")
            if vix is not None:
                if direction == "SHORT" and vix < 15:
                    # Risk-on environment, short BTC is risky
                    confidence_drivers["macro"] = -0.05
                    rationale_parts.append("Low VIX caution on short")
                elif direction == "LONG" and vix > 30:
                    # Risk-off, long BTC is risky
                    confidence_drivers["macro"] = -0.05
                    rationale_parts.append("High VIX caution on long")
                else:
                    confidence_drivers["macro"] = 0.05

        return direction, rationale_parts, confidence_drivers

    def _calculate_confidence(
        self,
        vol_regime: str,
        trend: Dict,
        onchain: Dict,
        direction: str,
        confidence_drivers: Dict[str, float],
    ) -> float:
        """
        Calculate overall signal confidence.

        Base confidence + contributions from each factor.

        Returns:
            Confidence score 0.0 to 1.0
        """
        base = 0.35
        total = base + sum(confidence_drivers.values())

        # Trend strength bonus
        total += trend["strength"] * 0.15

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, total))

        logger.debug(
            f"Confidence calc: base={base} + drivers={confidence_drivers} + "
            f"trend_bonus={trend['strength'] * 0.15:.3f} = {confidence:.3f}"
        )
        return confidence

    # =========================================================================
    # Price Level Calculations
    # =========================================================================

    def _calculate_levels_atr(
        self,
        price: float,
        atr: float,
        direction: str,
    ) -> Tuple[float, float, float, float]:
        """Calculate entry/stop/target using ATR."""
        stop_mult = self.risk_config["stop_loss_atr_multiple"]
        tp1_mult = self.risk_config["take_profit_atr_multiple"]
        tp2_mult = self.risk_config["take_profit_2_atr_multiple"]

        if direction == "LONG":
            entry = price
            stop = price - atr * stop_mult
            tp1 = price + atr * tp1_mult
            tp2 = price + atr * tp2_mult
        else:
            entry = price
            stop = price + atr * stop_mult
            tp1 = price - atr * tp1_mult
            tp2 = price - atr * tp2_mult

        return entry, stop, tp1, tp2

    def _calculate_levels_pct(
        self,
        price: float,
        direction: str,
    ) -> Tuple[float, float, float, float]:
        """Fallback: calculate levels using fixed percentages (BTC-appropriate)."""
        stop_pct = 0.05  # 5% stop
        tp1_pct = 0.08  # 8% target
        tp2_pct = 0.15  # 15% extended target

        if direction == "LONG":
            entry = price
            stop = price * (1 - stop_pct)
            tp1 = price * (1 + tp1_pct)
            tp2 = price * (1 + tp2_pct)
        else:
            entry = price
            stop = price * (1 + stop_pct)
            tp1 = price * (1 - tp1_pct)
            tp2 = price * (1 - tp2_pct)

        return entry, stop, tp1, tp2

    # =========================================================================
    # Rationale Construction
    # =========================================================================

    def _build_rationale(
        self,
        direction: str,
        vol_regime: str,
        trend: Dict,
        onchain: Dict,
        rationale_parts: List[str],
    ) -> str:
        """Build human-readable rationale string."""
        parts = [
            f"{direction} BTCUSD: {vol_regime.lower()} vol regime with "
            f"{trend['direction'].lower()} trend (strength {trend['strength']:.0%})"
        ]
        if onchain["available"] and onchain["bias"] != "NEUTRAL":
            parts.append(f"On-chain {onchain['bias'].lower()}")
        parts.append(". ".join(rationale_parts[:3]))
        return " | ".join(parts)

    # =========================================================================
    # Backtesting
    # =========================================================================

    def backtest_signal(
        self,
        signal: Signal,
        future_prices: pd.Series,
        horizon_days: int = 56,
    ) -> Dict:
        """
        Backtest a single signal against future prices.

        Args:
            signal: Signal to test
            future_prices: Price series after signal
            horizon_days: How many days to track

        Returns:
            Dictionary with backtest results
        """
        if len(future_prices) < 2:
            return {"status": "insufficient_data"}

        entry = signal.entry_price
        stop = signal.stop_loss
        target = signal.take_profit_1

        result = {
            "signal_id": signal.signal_id,
            "instrument": signal.instrument,
            "direction": signal.direction.value,
            "entry_price": entry,
            "stop_loss": stop,
            "target": target,
            "regime": signal.regime,
        }

        for i, price in enumerate(future_prices.values[:horizon_days]):
            if signal.direction == SignalDirection.LONG:
                hit_stop = price <= stop
                hit_target = price >= target
            else:
                hit_stop = price >= stop
                hit_target = price <= target

            if hit_stop:
                result["outcome"] = "stopped_out"
                result["exit_price"] = stop
                result["pnl_pct"] = -abs(entry - stop) / entry * 100
                result["days_held"] = i + 1
                break

            if hit_target:
                result["outcome"] = "target_hit"
                result["exit_price"] = target
                result["pnl_pct"] = abs(target - entry) / entry * 100
                result["days_held"] = i + 1
                break

        if "outcome" not in result:
            last_price = future_prices.values[
                min(horizon_days - 1, len(future_prices) - 1)
            ]
            if signal.direction == SignalDirection.LONG:
                pnl_pct = (last_price - entry) / entry * 100
            else:
                pnl_pct = (entry - last_price) / entry * 100

            result["outcome"] = "expired"
            result["exit_price"] = last_price
            result["pnl_pct"] = pnl_pct
            result["days_held"] = min(horizon_days, len(future_prices))

        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _get_latest_value(
        df: pd.DataFrame, column: str
    ) -> Optional[float]:
        """Safely get the latest non-NaN value from a column."""
        if column not in df.columns:
            return None
        series = df[column].dropna()
        if series.empty:
            return None
        return float(series.iloc[-1])


# =============================================================================
# Factory
# =============================================================================


def create_btc_trend_vol_strategy(
    config_path: Optional[str] = None,
    enable_onchain: Optional[bool] = None,
) -> BTCTrendVolStrategy:
    """
    Factory function to create BTC Trend+Volatility strategy.

    Args:
        config_path: Path to config file (optional)
        enable_onchain: Override on-chain enabled flag

    Returns:
        Configured strategy instance
    """
    if config_path:
        config = load_strategy_config("btc_trend_vol")
    else:
        config = {}

    if enable_onchain is not None:
        config.setdefault("onchain", {})["enabled"] = enable_onchain

    return BTCTrendVolStrategy(config)
