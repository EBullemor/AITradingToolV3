"""
Commodities Term Structure Strategy (Pod 3)

Trades commodities based on term structure, inventory, and momentum.

Signal Logic:
    LONG if:
        - ts_slope > 0.02 (backwardation) AND
        - momentum_3m > 0 AND
        - inventory below 5yr average - 10%
    SHORT if:
        - ts_slope < -0.02 (contango) AND
        - momentum_3m < 0 AND
        - inventory above 5yr average + 10%

Instruments: CL (Crude Oil), GC (Gold), HG (Copper)
Holding period: 2-6 weeks
Expected hit rate: 52%
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


# Commodity metadata for instrument-specific behaviour
COMMODITY_META = {
    "CL": {
        "name": "WTI Crude Oil",
        "front_ticker": "CL1",
        "back_ticker": "CL4",
        "contract_size": 1000,  # barrels
        "tick_size": 0.01,
        "currency": "USD",
        "inventory_source": "EIA",
        "seasonal_peak_months": [6, 7, 8],  # Summer driving season
    },
    "GC": {
        "name": "Gold",
        "front_ticker": "GC1",
        "back_ticker": "GC4",
        "contract_size": 100,  # troy oz
        "tick_size": 0.10,
        "currency": "USD",
        "inventory_source": None,  # No standard inventory for gold
        "seasonal_peak_months": [1, 2, 9],  # Wedding/festival season
    },
    "HG": {
        "name": "Copper",
        "front_ticker": "HG1",
        "back_ticker": "HG4",
        "contract_size": 25000,  # lbs
        "tick_size": 0.0005,
        "currency": "USD",
        "inventory_source": "LME",
        "seasonal_peak_months": [3, 4, 5],  # Construction season
    },
}


class CommoditiesTSStrategy(BaseStrategy):
    """
    Commodities Term Structure Strategy Pod.

    Strategy Logic:
    1. Determine term structure regime per commodity (BACKWARDATION / FLAT / CONTANGO)
    2. Assess inventory levels vs 5-year average
    3. Confirm with 3-month momentum
    4. Generate signals where all three factors align
    5. Compute entry/stop/target using ATR
    """

    DEFAULT_CONFIG = {
        "pod_name": "commodities_ts",
        "enabled": True,
        "instruments": ["CL", "GC", "HG"],
        "signal_validity_hours": 24,
        "max_signals_per_run": 3,

        # Term structure parameters
        "term_structure": {
            "front_contract": 1,
            "back_contract": 4,
            "backwardation_threshold": 0.02,   # 2%
            "contango_threshold": -0.02,       # -2%
            "strong_backwardation": 0.04,      # 4% = very strong
            "strong_contango": -0.04,
        },

        # Inventory parameters
        "inventory": {
            "enabled": True,
            "lookback_years": 5,
            "bullish_threshold": -0.10,   # 10% below 5yr avg
            "bearish_threshold": 0.10,    # 10% above 5yr avg
            "strong_bullish": -0.20,      # 20% below
            "strong_bearish": 0.20,       # 20% above
        },

        # Momentum overlay
        "momentum": {
            "period": 63,                 # 3-month (trading days)
            "threshold": 0.05,            # 5% move minimum
            "strong_threshold": 0.10,     # 10% move = strong
        },

        # Risk management
        "risk": {
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 3.5,
            "take_profit_2_atr_multiple": 5.0,
            "max_position_size_pct": 2.5,
        },

        # Signal generation
        "signal": {
            "holding_period_days": [14, 42],  # 2-6 weeks
            "require_ts_momentum_alignment": True,
            "min_confidence": 0.4,
        },
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Commodities Term Structure strategy.

        Args:
            config: Strategy configuration (uses defaults if not provided)
        """
        merged_config = self.DEFAULT_CONFIG.copy()
        if config:
            self._deep_merge(merged_config, config)

        super().__init__(merged_config, name="Commodities Term Structure")

        self.ts_config = merged_config["term_structure"]
        self.inv_config = merged_config["inventory"]
        self.mom_config = merged_config["momentum"]
        self.risk_config = merged_config["risk"]
        self.signal_config = merged_config["signal"]

        logger.info(
            f"Commodities TS strategy initialized | "
            f"Instruments: {self.instruments} | "
            f"Backwardation: >{self.ts_config['backwardation_threshold']:.0%} | "
            f"Contango: <{self.ts_config['contango_threshold']:.0%} | "
            f"Inventory: {'ON' if self.inv_config['enabled'] else 'OFF'}"
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
        features = [
            "PX_LAST",
            "term_structure_slope",
            "momentum_3m",
            "realized_vol_30d",
            "atr_14",
        ]
        if self.inv_config["enabled"]:
            features.append("inventory_zscore")
        return features

    # =========================================================================
    # Term Structure Regime Detection
    # =========================================================================

    def detect_ts_regime(
        self,
        features: pd.DataFrame,
        instrument: str,
    ) -> Dict[str, Any]:
        """
        Detect term structure regime for a commodity.

        Args:
            features: Commodity features DataFrame
            instrument: Commodity ticker (CL, GC, HG)

        Returns:
            Dict with:
                regime: 'BACKWARDATION', 'FLAT', 'CONTANGO'
                slope: float (term structure slope)
                roll_yield: float (annualised roll yield if available)
                strength: float (0-1, how extreme the TS is)
        """
        result = {
            "regime": "FLAT",
            "slope": 0.0,
            "roll_yield": None,
            "strength": 0.0,
        }

        if features is None or features.empty:
            logger.warning(f"No features for {instrument}, defaulting to FLAT TS")
            return result

        # Get term structure slope
        slope = self._get_latest_value(features, "term_structure_slope")
        if slope is None:
            slope = self._get_latest_value(features, "ts_slope")
        if slope is None:
            logger.warning(f"TS slope not found for {instrument}, defaulting to FLAT")
            return result

        result["slope"] = slope

        # Get roll yield if available
        ry = self._get_latest_value(features, "roll_yield")
        result["roll_yield"] = ry

        # Classify regime
        back_threshold = self.ts_config["backwardation_threshold"]
        cont_threshold = self.ts_config["contango_threshold"]
        strong_back = self.ts_config["strong_backwardation"]
        strong_cont = self.ts_config["strong_contango"]

        if slope > back_threshold:
            result["regime"] = "BACKWARDATION"
            result["strength"] = min(slope / strong_back, 1.0) if strong_back > 0 else 0.5
        elif slope < cont_threshold:
            result["regime"] = "CONTANGO"
            result["strength"] = min(abs(slope) / abs(strong_cont), 1.0) if strong_cont < 0 else 0.5
        else:
            result["regime"] = "FLAT"
            result["strength"] = 0.0

        logger.info(
            f"{instrument} TS: {result['regime']} | slope={slope:.4f} | "
            f"strength={result['strength']:.2f}"
        )
        return result

    # =========================================================================
    # Inventory Assessment
    # =========================================================================

    def assess_inventory(
        self,
        features: pd.DataFrame,
        instrument: str,
    ) -> Dict[str, Any]:
        """
        Assess inventory levels vs historical average.

        Degrades gracefully if inventory data is unavailable.

        Args:
            features: Commodity features DataFrame
            instrument: Commodity ticker

        Returns:
            Dict with:
                available: bool
                bias: 'BULLISH', 'BEARISH', 'NEUTRAL'
                zscore: float (inventory z-score)
                vs_avg_pct: float (% deviation from average)
                confidence_contribution: float
        """
        result = {
            "available": False,
            "bias": "NEUTRAL",
            "zscore": 0.0,
            "vs_avg_pct": 0.0,
            "confidence_contribution": 0.0,
        }

        if not self.inv_config["enabled"]:
            return result

        # Check if commodity has an inventory source
        meta = COMMODITY_META.get(instrument, {})
        if meta.get("inventory_source") is None:
            logger.debug(f"No inventory source for {instrument}")
            return result

        # Get inventory z-score
        inv_z = self._get_latest_value(features, "inventory_zscore")
        if inv_z is None:
            inv_z = self._get_latest_value(features, "inv_zscore")
        if inv_z is None:
            logger.debug(f"No inventory data for {instrument}")
            return result

        result["available"] = True
        result["zscore"] = inv_z

        # Convert z-score to approximate % deviation
        # z-score of ~1 ≈ ~10% deviation for most commodities
        result["vs_avg_pct"] = inv_z * 0.10

        bullish = self.inv_config["bullish_threshold"]
        bearish = self.inv_config["bearish_threshold"]
        strong_bullish = self.inv_config["strong_bullish"]
        strong_bearish = self.inv_config["strong_bearish"]

        if result["vs_avg_pct"] < bullish:
            result["bias"] = "BULLISH"
            strength = min(abs(result["vs_avg_pct"]) / abs(strong_bullish), 1.0)
            result["confidence_contribution"] = 0.15 * strength
        elif result["vs_avg_pct"] > bearish:
            result["bias"] = "BEARISH"
            strength = min(result["vs_avg_pct"] / strong_bearish, 1.0)
            result["confidence_contribution"] = 0.15 * strength
        else:
            result["bias"] = "NEUTRAL"
            result["confidence_contribution"] = 0.0

        logger.info(
            f"{instrument} inventory: {result['bias']} | "
            f"z={inv_z:.2f} | vs_avg={result['vs_avg_pct']:+.1%}"
        )
        return result

    # =========================================================================
    # Momentum Assessment
    # =========================================================================

    def assess_momentum(
        self,
        features: pd.DataFrame,
        instrument: str,
    ) -> Dict[str, Any]:
        """
        Assess 3-month momentum for the commodity.

        Args:
            features: Commodity features DataFrame
            instrument: Commodity ticker

        Returns:
            Dict with:
                direction: 'UP', 'DOWN', 'FLAT'
                value: float (raw momentum)
                strength: float (0-1)
                confirms_ts: bool (whether momentum aligns with TS)
        """
        result = {
            "direction": "FLAT",
            "value": 0.0,
            "strength": 0.0,
            "confirms_ts": False,
        }

        if features is None or features.empty:
            return result

        # Get 3-month momentum
        mom = self._get_latest_value(features, "momentum_3m")
        if mom is None:
            mom = self._get_latest_value(features, "price_momentum_3m")
        if mom is None:
            mom = self._get_latest_value(features, "momentum_score")
        if mom is None:
            logger.debug(f"No momentum data for {instrument}")
            return result

        result["value"] = mom

        threshold = self.mom_config["threshold"]
        strong = self.mom_config["strong_threshold"]

        if mom > threshold:
            result["direction"] = "UP"
            result["strength"] = min(mom / strong, 1.0)
        elif mom < -threshold:
            result["direction"] = "DOWN"
            result["strength"] = min(abs(mom) / strong, 1.0)
        else:
            result["direction"] = "FLAT"
            result["strength"] = 0.0

        logger.info(
            f"{instrument} momentum: {result['direction']} | "
            f"value={mom:+.4f} | strength={result['strength']:.2f}"
        )
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
        Generate commodities term structure signals.

        Args:
            features: Dict mapping instrument to features DataFrame
            macro_data: Macro indicators (VIX for risk gating)
            news_summary: News context (optional)
            as_of_date: Reference date

        Returns:
            List of Signal objects
        """
        if not self.enabled:
            logger.info("Commodities TS strategy disabled, skipping")
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

        # Respect limit
        if len(signals) > self.max_signals_per_run:
            signals.sort(key=lambda s: s.strength, reverse=True)
            signals = signals[: self.max_signals_per_run]

        logger.info(f"Commodities TS generated {len(signals)} signal(s)")
        return signals

    def _generate_instrument_signal(
        self,
        instrument: str,
        features: pd.DataFrame,
        macro_data: Optional[pd.DataFrame],
        news_summary: Optional[Dict],
        as_of_date: datetime,
    ) -> Optional[Signal]:
        """Generate signal for a single commodity."""

        # 1. Term structure regime
        ts = self.detect_ts_regime(features, instrument)

        # 2. Inventory assessment
        inventory = self.assess_inventory(features, instrument)

        # 3. Momentum assessment
        momentum = self.assess_momentum(features, instrument)

        # 4. Evaluate conditions
        direction, rationale_parts, confidence_drivers = self._evaluate_conditions(
            instrument=instrument,
            ts=ts,
            inventory=inventory,
            momentum=momentum,
            macro_data=macro_data,
        )

        if direction is None:
            logger.info(f"No signal for {instrument}: conditions not met")
            return None

        # 5. Calculate confidence
        confidence = self._calculate_confidence(
            ts=ts,
            inventory=inventory,
            momentum=momentum,
            confidence_drivers=confidence_drivers,
        )

        min_confidence = self.signal_config["min_confidence"]
        if confidence < min_confidence:
            logger.info(
                f"No signal for {instrument}: confidence {confidence:.2f} < {min_confidence}"
            )
            return None

        # 6. Price levels
        price = self._get_latest_value(features, "PX_LAST")
        atr = self._get_latest_value(features, "atr_14")

        if price is None:
            logger.warning(f"No price for {instrument}")
            return None

        if atr is not None and atr > 0:
            entry, stop, tp1, tp2 = self._calculate_levels_atr(price, atr, direction)
        else:
            entry, stop, tp1, tp2 = self._calculate_levels_pct(price, direction, instrument)

        # 7. Build rationale
        rationale = self._build_rationale(instrument, direction, ts, inventory, momentum)

        # 8. News context
        if news_summary and instrument in news_summary:
            ctx = news_summary[instrument]
            if isinstance(ctx, dict):
                rationale_parts.append(f"News: {ctx.get('sentiment', 'neutral')}")

        # 9. Create signal
        holding_min, holding_max = self.signal_config["holding_period_days"]
        meta = COMMODITY_META.get(instrument, {})

        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT,
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
            regime=ts["regime"],
            confidence_drivers=confidence_drivers,
        )

        logger.info(
            f"Signal: {direction} {instrument} ({meta.get('name', instrument)}) @ {entry:.2f} | "
            f"SL={stop:.2f} TP1={tp1:.2f} | Confidence={confidence:.2f} | "
            f"TS={ts['regime']}"
        )
        return signal

    def _evaluate_conditions(
        self,
        instrument: str,
        ts: Dict,
        inventory: Dict,
        momentum: Dict,
        macro_data: Optional[pd.DataFrame],
    ) -> Tuple[Optional[str], List[str], Dict[str, float]]:
        """
        Evaluate whether signal conditions are met.

        Returns:
            Tuple of (direction, rationale_parts, confidence_drivers)
        """
        rationale_parts: List[str] = []
        confidence_drivers: Dict[str, float] = {}

        # Condition 1: Term structure must not be FLAT
        if ts["regime"] == "FLAT":
            return None, [], {}

        if ts["regime"] == "BACKWARDATION":
            direction = "LONG"
            rationale_parts.append(
                f"Backwardation (slope={ts['slope']:+.3f})"
            )
            confidence_drivers["term_structure"] = 0.20 * ts["strength"]
        else:  # CONTANGO
            direction = "SHORT"
            rationale_parts.append(
                f"Contango (slope={ts['slope']:+.3f})"
            )
            confidence_drivers["term_structure"] = 0.20 * ts["strength"]

        # Condition 2: Momentum alignment
        require_alignment = self.signal_config["require_ts_momentum_alignment"]

        if direction == "LONG" and momentum["direction"] == "DOWN":
            if require_alignment:
                return None, [], {}
            rationale_parts.append("⚠ Momentum divergence (negative)")
            confidence_drivers["momentum"] = -0.05
        elif direction == "SHORT" and momentum["direction"] == "UP":
            if require_alignment:
                return None, [], {}
            rationale_parts.append("⚠ Momentum divergence (positive)")
            confidence_drivers["momentum"] = -0.05
        elif momentum["direction"] == "FLAT":
            rationale_parts.append("Flat momentum")
            confidence_drivers["momentum"] = 0.0
        else:
            rationale_parts.append(
                f"Momentum confirms ({momentum['value']:+.3f})"
            )
            confidence_drivers["momentum"] = 0.15 * momentum["strength"]

        # Condition 3: Inventory (if available)
        if inventory["available"]:
            if direction == "LONG" and inventory["bias"] == "BULLISH":
                rationale_parts.append(
                    f"Low inventory ({inventory['vs_avg_pct']:+.1%} vs avg)"
                )
                confidence_drivers["inventory"] = inventory["confidence_contribution"]
            elif direction == "SHORT" and inventory["bias"] == "BEARISH":
                rationale_parts.append(
                    f"High inventory ({inventory['vs_avg_pct']:+.1%} vs avg)"
                )
                confidence_drivers["inventory"] = inventory["confidence_contribution"]
            elif inventory["bias"] == "NEUTRAL":
                rationale_parts.append("Neutral inventory")
                confidence_drivers["inventory"] = 0.0
            else:
                # Inventory contradicts direction
                rationale_parts.append(
                    f"⚠ Inventory divergence ({inventory['bias']})"
                )
                confidence_drivers["inventory"] = -inventory["confidence_contribution"]
        else:
            rationale_parts.append("Inventory data unavailable")
            confidence_drivers["inventory"] = 0.0

        # Condition 4: Macro context (VIX)
        if macro_data is not None and not macro_data.empty:
            vix = self._get_latest_value(macro_data, "PX_LAST")
            if vix is not None:
                if vix > 30:
                    rationale_parts.append("High VIX — elevated uncertainty")
                    confidence_drivers["macro"] = -0.05
                elif vix < 15:
                    confidence_drivers["macro"] = 0.05
                else:
                    confidence_drivers["macro"] = 0.0

        return direction, rationale_parts, confidence_drivers

    def _calculate_confidence(
        self,
        ts: Dict,
        inventory: Dict,
        momentum: Dict,
        confidence_drivers: Dict[str, float],
    ) -> float:
        """Calculate overall signal confidence."""
        base = 0.35
        total = base + sum(confidence_drivers.values())

        # TS strength bonus
        total += ts["strength"] * 0.10

        return max(0.0, min(1.0, total))

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
        stop_m = self.risk_config["stop_loss_atr_multiple"]
        tp1_m = self.risk_config["take_profit_atr_multiple"]
        tp2_m = self.risk_config["take_profit_2_atr_multiple"]

        if direction == "LONG":
            return price, price - atr * stop_m, price + atr * tp1_m, price + atr * tp2_m
        else:
            return price, price + atr * stop_m, price - atr * tp1_m, price - atr * tp2_m

    def _calculate_levels_pct(
        self,
        price: float,
        direction: str,
        instrument: str,
    ) -> Tuple[float, float, float, float]:
        """Fallback: percentage-based levels, adjusted per commodity."""
        # Commodity-specific volatility assumptions
        vol_map = {"CL": 0.04, "GC": 0.025, "HG": 0.035}
        base_vol = vol_map.get(instrument, 0.03)

        stop_pct = base_vol * 1.5
        tp1_pct = base_vol * 2.5
        tp2_pct = base_vol * 4.0

        if direction == "LONG":
            return price, price * (1 - stop_pct), price * (1 + tp1_pct), price * (1 + tp2_pct)
        else:
            return price, price * (1 + stop_pct), price * (1 - tp1_pct), price * (1 - tp2_pct)

    # =========================================================================
    # Rationale
    # =========================================================================

    def _build_rationale(
        self,
        instrument: str,
        direction: str,
        ts: Dict,
        inventory: Dict,
        momentum: Dict,
    ) -> str:
        """Build human-readable rationale."""
        meta = COMMODITY_META.get(instrument, {})
        name = meta.get("name", instrument)

        parts = [
            f"{direction} {instrument} ({name}): "
            f"{ts['regime'].lower()} term structure (slope {ts['slope']:+.3f})"
        ]
        if momentum["direction"] != "FLAT":
            parts.append(f"{momentum['direction'].lower()} momentum ({momentum['value']:+.3f})")
        if inventory["available"] and inventory["bias"] != "NEUTRAL":
            parts.append(f"{inventory['bias'].lower()} inventory ({inventory['vs_avg_pct']:+.1%} vs avg)")

        return " | ".join(parts)

    # =========================================================================
    # Backtesting
    # =========================================================================

    def backtest_signal(
        self,
        signal: Signal,
        future_prices: pd.Series,
        horizon_days: int = 42,
    ) -> Dict:
        """
        Backtest a single signal against future prices.

        Args:
            signal: Signal to test
            future_prices: Price series after signal
            horizon_days: How many days to track (default 6 weeks)

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
    def _get_latest_value(df: pd.DataFrame, column: str) -> Optional[float]:
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


def create_commodities_ts_strategy(
    config_path: Optional[str] = None,
    instruments: Optional[List[str]] = None,
    enable_inventory: Optional[bool] = None,
) -> CommoditiesTSStrategy:
    """
    Factory function to create Commodities Term Structure strategy.

    Args:
        config_path: Path to config file (optional)
        instruments: Override instrument list
        enable_inventory: Override inventory enabled flag

    Returns:
        Configured strategy instance
    """
    if config_path:
        config = load_strategy_config("commodities_ts")
    else:
        config = {}

    if instruments is not None:
        config["instruments"] = instruments
    if enable_inventory is not None:
        config.setdefault("inventory", {})["enabled"] = enable_inventory

    return CommoditiesTSStrategy(config)
