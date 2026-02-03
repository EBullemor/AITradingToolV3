"""
FX Carry + Momentum Strategy (Pod 1)

Trades G10 FX based on carry and momentum with regime-based switching:
- LOW_VOL regime: Trade carry (interest rate differentials)
- HIGH_VOL regime: Trade momentum (trend following)

This is one of 5 strategy pods that feed into the signal aggregator.
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
    load_strategy_config,
)


class FXCarryMomentumStrategy(BaseStrategy):
    """
    FX Carry + Momentum Strategy Pod.
    
    Strategy Logic:
    1. Determine market regime (LOW_VOL vs HIGH_VOL)
    2. In LOW_VOL: Look for carry opportunities
    3. In HIGH_VOL: Look for momentum opportunities
    4. Apply filters (volatility, trend alignment, etc.)
    5. Generate signals with entry/stop/target levels
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        "pod_name": "fx_carry_momentum",
        "enabled": True,
        "instruments": ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"],
        "signal_validity_hours": 24,
        "max_signals_per_run": 4,
        
        # Regime thresholds
        "regime": {
            "vix_low_threshold": 18.0,
            "vix_high_threshold": 25.0,
        },
        
        # Carry strategy parameters
        "carry": {
            "min_carry_score": 1.0,
            "strong_carry_score": 2.0,
            "max_vol_percentile": 75,  # Don't trade carry in high vol
        },
        
        # Momentum strategy parameters
        "momentum": {
            "min_momentum_score": 1.5,
            "strong_momentum_score": 2.5,
            "min_trend_strength": 0.3,
        },
        
        # Risk management
        "risk": {
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 3.0,
            "max_position_size_pct": 2.0,
        },
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FX Carry+Momentum strategy.
        
        Args:
            config: Strategy configuration (uses defaults if not provided)
        """
        # Merge with defaults
        merged_config = self.DEFAULT_CONFIG.copy()
        if config:
            self._deep_merge(merged_config, config)
        
        super().__init__(merged_config, name="FX Carry + Momentum")
        
        # Extract specific configs
        self.regime_config = merged_config["regime"]
        self.carry_config = merged_config["carry"]
        self.momentum_config = merged_config["momentum"]
        self.risk_config = merged_config["risk"]
        
        logger.info(f"FX Carry+Momentum strategy initialized with {len(self.instruments)} instruments")
    
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
            "PX_LAST",
            "momentum_score",
            "volatility_20d",
            "vol_percentile",
            "trend_strength",
            "atr_14",
        ]
    
    def detect_regime(
        self,
        macro_data: pd.DataFrame,
        as_of_date: Optional[datetime] = None
    ) -> str:
        """
        Detect current market regime based on VIX.
        
        Args:
            macro_data: Macro indicators including VIX
            as_of_date: Reference date
        
        Returns:
            Regime string: 'LOW_VOL', 'NORMAL', or 'HIGH_VOL'
        """
        if macro_data is None or macro_data.empty:
            logger.warning("No macro data, defaulting to NORMAL regime")
            return "NORMAL"
        
        # Get latest VIX
        if "PX_LAST" in macro_data.columns:
            vix = macro_data["PX_LAST"].iloc[-1]
        elif "vix" in macro_data.columns:
            vix = macro_data["vix"].iloc[-1]
        else:
            logger.warning("VIX not found in macro data, defaulting to NORMAL")
            return "NORMAL"
        
        low_threshold = self.regime_config["vix_low_threshold"]
        high_threshold = self.regime_config["vix_high_threshold"]
        
        if vix < low_threshold:
            regime = "LOW_VOL"
        elif vix > high_threshold:
            regime = "HIGH_VOL"
        else:
            regime = "NORMAL"
        
        logger.info(f"Detected regime: {regime} (VIX={vix:.1f})")
        return regime
    
    def generate_carry_signal(
        self,
        instrument: str,
        features: pd.DataFrame,
        regime: str,
        as_of_date: Optional[datetime] = None
    ) -> Optional[Signal]:
        """
        Generate carry-based signal for an instrument.
        
        Args:
            instrument: Currency pair
            features: Feature DataFrame for this instrument
            regime: Current market regime
            as_of_date: Reference date
        
        Returns:
            Signal if conditions met, None otherwise
        """
        if features.empty:
            return None
        
        # Get latest values
        latest = features.iloc[-1]
        
        # Check if carry score exists
        if "carry_score" not in features.columns:
            # Estimate from momentum if carry not available
            carry_score = latest.get("fx_momentum", 0) * 0.5
        else:
            carry_score = latest["carry_score"]
        
        # Get thresholds
        min_score = self.carry_config["min_carry_score"]
        strong_score = self.carry_config["strong_carry_score"]
        max_vol_pct = self.carry_config["max_vol_percentile"]
        
        # Check volatility filter
        vol_percentile = latest.get("vol_percentile", 50)
        if vol_percentile > max_vol_pct:
            logger.debug(f"{instrument}: Vol too high for carry ({vol_percentile:.0f}%)")
            return None
        
        # Determine direction and strength
        direction = None
        strength = 0.0
        
        if carry_score > min_score:
            direction = SignalDirection.LONG
            strength = min((carry_score - min_score) / (strong_score - min_score), 1.0)
        elif carry_score < -min_score:
            direction = SignalDirection.SHORT
            strength = min((-carry_score - min_score) / (strong_score - min_score), 1.0)
        
        if direction is None:
            return None
        
        # Calculate price levels
        entry_price = latest["PX_LAST"]
        atr = latest.get("atr_14", entry_price * 0.01)  # Default 1% ATR
        
        stop_multiple = self.risk_config["stop_loss_atr_multiple"]
        target_multiple = self.risk_config["take_profit_atr_multiple"]
        
        if direction == SignalDirection.LONG:
            stop_loss = entry_price - (atr * stop_multiple)
            take_profit_1 = entry_price + (atr * target_multiple)
        else:
            stop_loss = entry_price + (atr * stop_multiple)
            take_profit_1 = entry_price - (atr * target_multiple)
        
        # Build rationale
        rationale = f"Carry trade in {regime} regime. "
        rationale += f"Carry score: {carry_score:.2f}. "
        rationale += f"Vol percentile: {vol_percentile:.0f}%."
        
        key_factors = [
            f"Carry score: {carry_score:.2f}",
            f"Regime: {regime}",
            f"Vol percentile: {vol_percentile:.0f}%",
        ]
        
        confidence_drivers = {
            "carry_score": min(abs(carry_score) / strong_score, 1.0),
            "vol_regime": 1.0 - (vol_percentile / 100),
        }
        
        return self._create_signal(
            instrument=instrument,
            direction=direction,
            strength=strength * 0.8,  # Scale down for carry
            rationale=rationale,
            key_factors=key_factors,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            regime=regime,
            confidence_drivers=confidence_drivers,
            as_of_date=as_of_date,
        )
    
    def generate_momentum_signal(
        self,
        instrument: str,
        features: pd.DataFrame,
        regime: str,
        as_of_date: Optional[datetime] = None
    ) -> Optional[Signal]:
        """
        Generate momentum-based signal for an instrument.
        
        Args:
            instrument: Currency pair
            features: Feature DataFrame for this instrument
            regime: Current market regime
            as_of_date: Reference date
        
        Returns:
            Signal if conditions met, None otherwise
        """
        if features.empty:
            return None
        
        latest = features.iloc[-1]
        
        # Get momentum score
        momentum_score = latest.get("momentum_score", 0)
        if "fx_momentum" in features.columns:
            momentum_score = latest["fx_momentum"]
        
        # Get trend strength
        trend_strength = latest.get("trend_strength", 0)
        
        # Get thresholds
        min_momentum = self.momentum_config["min_momentum_score"]
        strong_momentum = self.momentum_config["strong_momentum_score"]
        min_trend = self.momentum_config["min_trend_strength"]
        
        # Trend alignment filter
        if abs(trend_strength) < min_trend:
            logger.debug(f"{instrument}: Trend too weak ({trend_strength:.2f})")
            return None
        
        # Determine direction and strength
        direction = None
        strength = 0.0
        
        # For momentum, trend must align with momentum
        if momentum_score > min_momentum and trend_strength > 0:
            direction = SignalDirection.LONG
            strength = min((momentum_score - min_momentum) / (strong_momentum - min_momentum), 1.0)
        elif momentum_score < -min_momentum and trend_strength < 0:
            direction = SignalDirection.SHORT
            strength = min((-momentum_score - min_momentum) / (strong_momentum - min_momentum), 1.0)
        
        if direction is None:
            return None
        
        # Calculate price levels
        entry_price = latest["PX_LAST"]
        atr = latest.get("atr_14", entry_price * 0.01)
        
        stop_multiple = self.risk_config["stop_loss_atr_multiple"]
        target_multiple = self.risk_config["take_profit_atr_multiple"]
        
        if direction == SignalDirection.LONG:
            stop_loss = entry_price - (atr * stop_multiple)
            take_profit_1 = entry_price + (atr * target_multiple)
        else:
            stop_loss = entry_price + (atr * stop_multiple)
            take_profit_1 = entry_price - (atr * target_multiple)
        
        # Build rationale
        rationale = f"Momentum trade in {regime} regime. "
        rationale += f"Momentum score: {momentum_score:.2f}. "
        rationale += f"Trend strength: {trend_strength:.2f}."
        
        key_factors = [
            f"Momentum score: {momentum_score:.2f}",
            f"Trend strength: {trend_strength:.2f}",
            f"Regime: {regime}",
        ]
        
        confidence_drivers = {
            "momentum_score": min(abs(momentum_score) / strong_momentum, 1.0),
            "trend_alignment": abs(trend_strength),
        }
        
        return self._create_signal(
            instrument=instrument,
            direction=direction,
            strength=strength,
            rationale=rationale,
            key_factors=key_factors,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            regime=regime,
            confidence_drivers=confidence_drivers,
            as_of_date=as_of_date,
        )
    
    def generate_signals(
        self,
        features: Dict[str, pd.DataFrame],
        macro_data: Optional[pd.DataFrame] = None,
        news_summary: Optional[Dict] = None,
        as_of_date: Optional[datetime] = None
    ) -> List[Signal]:
        """
        Generate FX signals based on carry and momentum.
        
        Args:
            features: Dict mapping instrument to features DataFrame
            macro_data: Macro indicators (VIX, etc.)
            news_summary: News summary (optional, for future use)
            as_of_date: Reference date
        
        Returns:
            List of Signal objects
        """
        if not self.enabled:
            logger.info(f"{self.name} is disabled")
            return []
        
        if as_of_date is None:
            as_of_date = datetime.now()
        
        signals = []
        
        # Detect regime
        regime = self.detect_regime(macro_data, as_of_date)
        
        # Generate signals for each instrument
        for instrument in self.instruments:
            if instrument not in features:
                logger.warning(f"No features for {instrument}")
                continue
            
            inst_features = features[instrument]
            
            # Decide which signal type based on regime
            if regime == "LOW_VOL":
                # Primary: Carry
                signal = self.generate_carry_signal(
                    instrument, inst_features, regime, as_of_date
                )
                
                # If no carry signal, try momentum
                if signal is None:
                    signal = self.generate_momentum_signal(
                        instrument, inst_features, regime, as_of_date
                    )
                    if signal:
                        signal.strength *= 0.7  # Reduce strength for non-primary
                        signal.key_factors.append("Secondary strategy (momentum in low vol)")
            
            elif regime == "HIGH_VOL":
                # Primary: Momentum
                signal = self.generate_momentum_signal(
                    instrument, inst_features, regime, as_of_date
                )
            
            else:  # NORMAL
                # Try both, take stronger
                carry_signal = self.generate_carry_signal(
                    instrument, inst_features, regime, as_of_date
                )
                momentum_signal = self.generate_momentum_signal(
                    instrument, inst_features, regime, as_of_date
                )
                
                if carry_signal and momentum_signal:
                    # Take stronger signal, but check for alignment
                    if carry_signal.direction == momentum_signal.direction:
                        # Aligned - boost strength
                        signal = carry_signal if carry_signal.strength >= momentum_signal.strength else momentum_signal
                        signal.strength = min(signal.strength * 1.2, 1.0)
                        signal.key_factors.append("Carry and momentum aligned")
                    else:
                        # Conflicting - take stronger but reduce
                        signal = carry_signal if carry_signal.strength >= momentum_signal.strength else momentum_signal
                        signal.strength *= 0.6
                        signal.key_factors.append("Carry and momentum conflicting")
                elif carry_signal:
                    signal = carry_signal
                elif momentum_signal:
                    signal = momentum_signal
                else:
                    signal = None
            
            if signal:
                signals.append(signal)
                logger.info(
                    f"Generated signal: {signal.direction.value} {instrument} "
                    f"(strength: {signal.strength:.1%})"
                )
        
        # Sort by strength and limit
        signals.sort(key=lambda s: s.strength, reverse=True)
        signals = signals[:self.max_signals_per_run]
        
        logger.info(f"{self.name} generated {len(signals)} signals")
        
        return signals
    
    def backtest_signal(
        self,
        signal: Signal,
        future_prices: pd.Series,
        horizon_days: int = 5
    ) -> Dict[str, Any]:
        """
        Backtest a signal against future prices.
        
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
        }
        
        # Track P&L
        for i, price in enumerate(future_prices.values[:horizon_days]):
            if signal.direction == SignalDirection.LONG:
                pnl_pct = (price - entry) / entry * 100
                hit_stop = price <= stop
                hit_target = price >= target
            else:
                pnl_pct = (entry - price) / entry * 100
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
            # Neither hit - use last price
            last_price = future_prices.values[min(horizon_days - 1, len(future_prices) - 1)]
            if signal.direction == SignalDirection.LONG:
                pnl_pct = (last_price - entry) / entry * 100
            else:
                pnl_pct = (entry - last_price) / entry * 100
            
            result["outcome"] = "expired"
            result["exit_price"] = last_price
            result["pnl_pct"] = pnl_pct
            result["days_held"] = horizon_days
        
        return result


def create_fx_carry_momentum_strategy(
    config_path: Optional[str] = None
) -> FXCarryMomentumStrategy:
    """
    Factory function to create FX Carry+Momentum strategy.
    
    Args:
        config_path: Path to config file (optional)
    
    Returns:
        Configured strategy instance
    """
    if config_path:
        config = load_strategy_config("fx_carry_momentum")
    else:
        config = {}
    
    return FXCarryMomentumStrategy(config)
