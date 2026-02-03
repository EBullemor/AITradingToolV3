"""
Position Sizer Module

Calculates position sizes based on risk management rules:
- Fixed risk per trade (default 1%)
- Stop-loss based sizing
- Volatility-adjusted sizing
- Kelly criterion (optional)
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED_RISK = "fixed_risk"           # Risk X% of portfolio per trade
    FIXED_FRACTION = "fixed_fraction"   # Fixed % of portfolio
    VOLATILITY_ADJUSTED = "vol_adjusted"  # Adjust for instrument volatility
    KELLY = "kelly"                     # Kelly criterion


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    # Core sizing
    position_size: float          # Units/lots to trade
    position_value: float         # Dollar value of position
    position_pct: float           # As % of portfolio
    
    # Risk metrics
    risk_amount: float            # Dollar risk on trade
    risk_pct: float               # Risk as % of portfolio
    risk_per_unit: float          # Risk per unit traded
    
    # Stops
    stop_loss: float              # Stop loss price
    stop_distance: float          # Distance to stop (absolute)
    stop_distance_pct: float      # Distance to stop (%)
    
    # Adjustments
    sizing_method: SizingMethod
    was_capped: bool = False      # True if position was reduced
    cap_reason: Optional[str] = None
    
    # Metadata
    instrument: str = ""
    direction: str = ""
    calculated_at: datetime = None
    
    def __post_init__(self):
        if self.calculated_at is None:
            self.calculated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "position_size": self.position_size,
            "position_value": self.position_value,
            "position_pct": self.position_pct,
            "risk_amount": self.risk_amount,
            "risk_pct": self.risk_pct,
            "stop_loss": self.stop_loss,
            "stop_distance_pct": self.stop_distance_pct,
            "sizing_method": self.sizing_method.value,
            "was_capped": self.was_capped,
            "cap_reason": self.cap_reason,
            "instrument": self.instrument,
            "direction": self.direction,
        }


class PositionSizer:
    """
    Calculates position sizes based on risk parameters.
    
    Default method: Fixed Risk
    - Risk 1% of portfolio per trade
    - Position size = Risk Amount / (Entry - Stop)
    """
    
    def __init__(
        self,
        default_risk_pct: float = 0.01,      # 1% risk per trade
        max_position_pct: float = 0.10,       # Max 10% per position
        max_stop_distance_pct: float = 0.03,  # Max 3% stop distance
        min_position_value: float = 1000,     # Minimum position size
        default_sizing_method: SizingMethod = SizingMethod.FIXED_RISK,
    ):
        """
        Initialize position sizer.
        
        Args:
            default_risk_pct: Default risk per trade (0.01 = 1%)
            max_position_pct: Maximum position as % of portfolio
            max_stop_distance_pct: Maximum stop loss distance
            min_position_value: Minimum position value in dollars
            default_sizing_method: Default sizing method
        """
        self.default_risk_pct = default_risk_pct
        self.max_position_pct = max_position_pct
        self.max_stop_distance_pct = max_stop_distance_pct
        self.min_position_value = min_position_value
        self.default_sizing_method = default_sizing_method
        
        logger.info(
            f"Position sizer initialized: {default_risk_pct:.1%} risk, "
            f"{max_position_pct:.0%} max position"
        )
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: Optional[float] = None,
        atr_multiplier: float = 2.0,
        fixed_pct: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            direction: "LONG" or "SHORT"
            atr: Average True Range (optional)
            atr_multiplier: ATR multiplier for stop
            fixed_pct: Fixed percentage for stop (alternative to ATR)
        
        Returns:
            Tuple of (stop_price, stop_distance_pct)
        """
        # Use ATR-based stop if available
        if atr is not None:
            stop_distance = atr * atr_multiplier
        elif fixed_pct is not None:
            stop_distance = entry_price * fixed_pct
        else:
            # Default to 2% stop
            stop_distance = entry_price * 0.02
        
        # Cap stop distance
        max_stop = entry_price * self.max_stop_distance_pct
        if stop_distance > max_stop:
            stop_distance = max_stop
        
        # Calculate stop price
        if direction.upper() == "LONG":
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance
        
        stop_distance_pct = stop_distance / entry_price
        
        return stop_price, stop_distance_pct
    
    def calculate_fixed_risk_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float,
        risk_pct: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate position size using fixed risk method.
        
        Formula: Position Size = Risk Amount / Risk Per Unit
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_pct: Risk percentage (uses default if None)
        
        Returns:
            Tuple of (position_size, risk_amount)
        """
        if risk_pct is None:
            risk_pct = self.default_risk_pct
        
        # Calculate risk amount
        risk_amount = portfolio_value * risk_pct
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            logger.warning("Invalid stop loss - using minimum distance")
            risk_per_unit = entry_price * 0.01  # 1% minimum
        
        # Calculate position size
        position_size = risk_amount / risk_per_unit
        
        return position_size, risk_amount
    
    def calculate_volatility_adjusted_size(
        self,
        portfolio_value: float,
        entry_price: float,
        volatility: float,
        target_vol: float = 0.10,
    ) -> float:
        """
        Calculate position size adjusted for volatility.
        
        Targets constant portfolio volatility contribution.
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Entry price
            volatility: Annualized volatility of instrument
            target_vol: Target volatility contribution
        
        Returns:
            Position size in units
        """
        if volatility <= 0:
            volatility = 0.10  # Default 10% vol
        
        # Position weight to achieve target vol
        position_weight = target_vol / volatility
        
        # Cap at max position
        position_weight = min(position_weight, self.max_position_pct)
        
        # Calculate position value and size
        position_value = portfolio_value * position_weight
        position_size = position_value / entry_price
        
        return position_size
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        direction: str = "LONG",
        instrument: str = "",
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
        risk_pct: Optional[float] = None,
        method: Optional[SizingMethod] = None,
    ) -> PositionSizeResult:
        """
        Calculate position size with full result.
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Entry price
            stop_loss: Stop loss price (calculated if not provided)
            direction: "LONG" or "SHORT"
            instrument: Instrument name
            atr: Average True Range
            volatility: Annualized volatility
            risk_pct: Risk percentage override
            method: Sizing method override
        
        Returns:
            PositionSizeResult with all details
        """
        sizing_method = method or self.default_sizing_method
        
        # Calculate stop loss if not provided
        if stop_loss is None:
            stop_loss, stop_distance_pct = self.calculate_stop_loss(
                entry_price, direction, atr
            )
        else:
            stop_distance_pct = abs(entry_price - stop_loss) / entry_price
        
        # Calculate position size based on method
        if sizing_method == SizingMethod.FIXED_RISK:
            position_size, risk_amount = self.calculate_fixed_risk_size(
                portfolio_value, entry_price, stop_loss, risk_pct
            )
        
        elif sizing_method == SizingMethod.VOLATILITY_ADJUSTED:
            position_size = self.calculate_volatility_adjusted_size(
                portfolio_value, entry_price, volatility or 0.10
            )
            risk_amount = position_size * abs(entry_price - stop_loss)
        
        elif sizing_method == SizingMethod.FIXED_FRACTION:
            position_value = portfolio_value * (risk_pct or self.default_risk_pct)
            position_size = position_value / entry_price
            risk_amount = position_size * abs(entry_price - stop_loss)
        
        else:
            # Default to fixed risk
            position_size, risk_amount = self.calculate_fixed_risk_size(
                portfolio_value, entry_price, stop_loss, risk_pct
            )
        
        # Calculate position value
        position_value = position_size * entry_price
        position_pct = position_value / portfolio_value
        
        # Check caps
        was_capped = False
        cap_reason = None
        
        # Cap 1: Max position size
        if position_pct > self.max_position_pct:
            position_pct = self.max_position_pct
            position_value = portfolio_value * position_pct
            position_size = position_value / entry_price
            risk_amount = position_size * abs(entry_price - stop_loss)
            was_capped = True
            cap_reason = f"Max position {self.max_position_pct:.0%}"
        
        # Cap 2: Minimum position value
        if position_value < self.min_position_value:
            # Either size up to minimum or skip
            was_capped = True
            cap_reason = f"Below min size ${self.min_position_value}"
        
        risk_per_unit = abs(entry_price - stop_loss)
        risk_pct_actual = risk_amount / portfolio_value
        
        return PositionSizeResult(
            position_size=round(position_size, 4),
            position_value=round(position_value, 2),
            position_pct=round(position_pct, 4),
            risk_amount=round(risk_amount, 2),
            risk_pct=round(risk_pct_actual, 4),
            risk_per_unit=round(risk_per_unit, 6),
            stop_loss=round(stop_loss, 5),
            stop_distance=round(abs(entry_price - stop_loss), 5),
            stop_distance_pct=round(stop_distance_pct, 4),
            sizing_method=sizing_method,
            was_capped=was_capped,
            cap_reason=cap_reason,
            instrument=instrument,
            direction=direction,
        )


def calculate_position_size(
    portfolio_value: float,
    entry_price: float,
    stop_loss: float,
    risk_pct: float = 0.01,
) -> float:
    """
    Simple position size calculation.
    
    Args:
        portfolio_value: Total portfolio value
        entry_price: Entry price
        stop_loss: Stop loss price
        risk_pct: Risk per trade (0.01 = 1%)
    
    Returns:
        Position size in units
    """
    risk_amount = portfolio_value * risk_pct
    risk_per_unit = abs(entry_price - stop_loss)
    
    if risk_per_unit <= 0:
        return 0
    
    return risk_amount / risk_per_unit


def calculate_stop_from_atr(
    entry_price: float,
    atr: float,
    direction: str,
    multiplier: float = 2.0,
) -> float:
    """
    Calculate stop loss using ATR.
    
    Args:
        entry_price: Entry price
        atr: Average True Range
        direction: "LONG" or "SHORT"
        multiplier: ATR multiplier
    
    Returns:
        Stop loss price
    """
    stop_distance = atr * multiplier
    
    if direction.upper() == "LONG":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance
