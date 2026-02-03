"""
Portfolio Risk Module

Manages portfolio-level risk constraints:
- Maximum exposure limits
- Correlation-based position limits
- Drawdown monitoring
- Risk budget allocation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Position:
    """Represents a portfolio position."""
    instrument: str
    direction: str  # LONG or SHORT
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    
    # Optional
    entry_date: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    @property
    def market_value(self) -> float:
        return abs(self.size * self.current_price)
    
    @property
    def is_long(self) -> bool:
        return self.direction.upper() == "LONG"
    
    def calculate_pnl(self) -> Tuple[float, float]:
        """Calculate current P&L."""
        if self.is_long:
            pnl = (self.current_price - self.entry_price) * self.size
        else:
            pnl = (self.entry_price - self.current_price) * abs(self.size)
        
        pnl_pct = pnl / (self.entry_price * abs(self.size))
        
        self.pnl = pnl
        self.pnl_pct = pnl_pct
        
        return pnl, pnl_pct


@dataclass
class PortfolioRiskReport:
    """Portfolio risk assessment report."""
    # Portfolio value
    portfolio_value: float
    cash: float
    invested: float
    
    # Exposure
    gross_exposure: float
    net_exposure: float
    gross_exposure_pct: float
    net_exposure_pct: float
    
    # P&L
    total_pnl: float
    total_pnl_pct: float
    daily_pnl: float
    
    # Risk metrics
    risk_level: RiskLevel
    position_count: int
    long_count: int
    short_count: int
    
    # Limits
    exposure_limit_remaining: float
    max_correlated_warning: bool
    
    # Per-position risk
    position_risks: List[Dict] = field(default_factory=list)
    
    # Timestamp
    calculated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "invested": self.invested,
            "gross_exposure_pct": self.gross_exposure_pct,
            "net_exposure_pct": self.net_exposure_pct,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "risk_level": self.risk_level.value,
            "position_count": self.position_count,
            "exposure_limit_remaining": self.exposure_limit_remaining,
            "max_correlated_warning": self.max_correlated_warning,
        }


class PortfolioRiskManager:
    """
    Manages portfolio-level risk.
    
    Enforces:
    - Maximum gross/net exposure
    - Correlation limits
    - Drawdown limits
    - Position concentration
    """
    
    # Default correlation matrix for FX (simplified)
    FX_CORRELATIONS = {
        ("EURUSD", "GBPUSD"): 0.75,
        ("EURUSD", "AUDUSD"): 0.65,
        ("EURUSD", "USDCHF"): -0.85,
        ("GBPUSD", "AUDUSD"): 0.55,
        ("USDJPY", "AUDUSD"): 0.40,
        ("AUDUSD", "NZDUSD"): 0.90,
    }
    
    def __init__(
        self,
        max_gross_exposure: float = 1.0,      # 100% max gross
        max_net_exposure: float = 0.5,         # 50% max net
        max_position_pct: float = 0.10,        # 10% max per position
        max_correlated_positions: int = 3,     # Max positions with corr > threshold
        correlation_threshold: float = 0.7,    # Correlation threshold
        max_daily_loss_pct: float = 0.03,      # 3% daily loss limit
        max_drawdown_pct: float = 0.10,        # 10% max drawdown
    ):
        """
        Initialize portfolio risk manager.
        
        Args:
            max_gross_exposure: Maximum total exposure (long + short)
            max_net_exposure: Maximum net exposure (long - short)
            max_position_pct: Maximum single position size
            max_correlated_positions: Max highly correlated positions
            correlation_threshold: What counts as "correlated"
            max_daily_loss_pct: Stop trading if daily loss exceeds
            max_drawdown_pct: Maximum portfolio drawdown
        """
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.max_position_pct = max_position_pct
        self.max_correlated_positions = max_correlated_positions
        self.correlation_threshold = correlation_threshold
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        
        # State
        self.positions: Dict[str, Position] = {}
        self.portfolio_value: float = 0
        self.peak_value: float = 0
        self.daily_start_value: float = 0
        
        logger.info(
            f"Portfolio risk manager initialized: "
            f"{max_gross_exposure:.0%} max gross, "
            f"{max_net_exposure:.0%} max net"
        )
    
    def set_portfolio_value(self, value: float, is_daily_start: bool = False):
        """Set current portfolio value."""
        self.portfolio_value = value
        self.peak_value = max(self.peak_value, value)
        
        if is_daily_start:
            self.daily_start_value = value
    
    def add_position(self, position: Position):
        """Add or update a position."""
        self.positions[position.instrument] = position
    
    def remove_position(self, instrument: str):
        """Remove a position."""
        if instrument in self.positions:
            del self.positions[instrument]
    
    def get_correlation(self, inst1: str, inst2: str) -> float:
        """Get correlation between two instruments."""
        key = (inst1, inst2)
        rev_key = (inst2, inst1)
        
        if key in self.FX_CORRELATIONS:
            return self.FX_CORRELATIONS[key]
        elif rev_key in self.FX_CORRELATIONS:
            return self.FX_CORRELATIONS[rev_key]
        
        # Default low correlation for unknown pairs
        return 0.3
    
    def calculate_exposure(self) -> Tuple[float, float]:
        """
        Calculate gross and net exposure.
        
        Returns:
            Tuple of (gross_exposure, net_exposure)
        """
        long_exposure = 0.0
        short_exposure = 0.0
        
        for pos in self.positions.values():
            value = pos.market_value
            
            if pos.is_long:
                long_exposure += value
            else:
                short_exposure += value
        
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        return gross_exposure, net_exposure
    
    def count_correlated_positions(self, instrument: str) -> int:
        """Count positions correlated with given instrument."""
        count = 0
        
        for inst in self.positions:
            if inst == instrument:
                continue
            
            corr = abs(self.get_correlation(instrument, inst))
            if corr >= self.correlation_threshold:
                count += 1
        
        return count
    
    def can_add_position(
        self,
        instrument: str,
        direction: str,
        position_value: float,
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be added.
        
        Args:
            instrument: Instrument to add
            direction: LONG or SHORT
            position_value: Value of new position
        
        Returns:
            Tuple of (can_add, reason_if_not)
        """
        # Check position concentration
        position_pct = position_value / self.portfolio_value
        if position_pct > self.max_position_pct:
            return False, f"Position size {position_pct:.0%} exceeds max {self.max_position_pct:.0%}"
        
        # Check gross exposure
        gross, net = self.calculate_exposure()
        new_gross = gross + position_value
        new_gross_pct = new_gross / self.portfolio_value
        
        if new_gross_pct > self.max_gross_exposure:
            return False, f"Gross exposure {new_gross_pct:.0%} would exceed max {self.max_gross_exposure:.0%}"
        
        # Check net exposure
        if direction.upper() == "LONG":
            new_net = net + position_value
        else:
            new_net = net - position_value
        
        new_net_pct = abs(new_net) / self.portfolio_value
        if new_net_pct > self.max_net_exposure:
            return False, f"Net exposure {new_net_pct:.0%} would exceed max {self.max_net_exposure:.0%}"
        
        # Check correlated positions
        correlated_count = self.count_correlated_positions(instrument)
        if correlated_count >= self.max_correlated_positions:
            return False, f"Already {correlated_count} correlated positions (max {self.max_correlated_positions})"
        
        # Check daily loss limit
        if self.daily_start_value > 0:
            daily_pnl_pct = (self.portfolio_value - self.daily_start_value) / self.daily_start_value
            if daily_pnl_pct <= -self.max_daily_loss_pct:
                return False, f"Daily loss limit {self.max_daily_loss_pct:.0%} reached"
        
        # Check drawdown
        if self.peak_value > 0:
            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            if drawdown >= self.max_drawdown_pct:
                return False, f"Max drawdown {self.max_drawdown_pct:.0%} reached"
        
        return True, ""
    
    def get_available_risk_budget(self) -> float:
        """Get remaining risk budget (exposure room)."""
        gross, _ = self.calculate_exposure()
        gross_pct = gross / self.portfolio_value if self.portfolio_value > 0 else 0
        
        remaining = self.max_gross_exposure - gross_pct
        return max(0, remaining)
    
    def calculate_risk_report(self) -> PortfolioRiskReport:
        """Generate full portfolio risk report."""
        gross, net = self.calculate_exposure()
        
        gross_pct = gross / self.portfolio_value if self.portfolio_value > 0 else 0
        net_pct = net / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Calculate P&L
        total_pnl = 0.0
        for pos in self.positions.values():
            pos.calculate_pnl()
            total_pnl += pos.pnl
        
        total_pnl_pct = total_pnl / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Daily P&L
        daily_pnl = 0.0
        if self.daily_start_value > 0:
            daily_pnl = (self.portfolio_value - self.daily_start_value)
        
        # Count positions
        long_count = sum(1 for p in self.positions.values() if p.is_long)
        short_count = len(self.positions) - long_count
        
        # Determine risk level
        risk_level = self._determine_risk_level(gross_pct, net_pct)
        
        # Check correlation warning
        max_correlated_warning = False
        for inst in self.positions:
            if self.count_correlated_positions(inst) >= self.max_correlated_positions:
                max_correlated_warning = True
                break
        
        # Position-level risks
        position_risks = []
        for pos in self.positions.values():
            position_risks.append({
                "instrument": pos.instrument,
                "direction": pos.direction,
                "pnl": pos.pnl,
                "pnl_pct": pos.pnl_pct,
                "market_value": pos.market_value,
            })
        
        return PortfolioRiskReport(
            portfolio_value=self.portfolio_value,
            cash=self.portfolio_value - gross,
            invested=gross,
            gross_exposure=gross,
            net_exposure=net,
            gross_exposure_pct=gross_pct,
            net_exposure_pct=net_pct,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            daily_pnl=daily_pnl,
            risk_level=risk_level,
            position_count=len(self.positions),
            long_count=long_count,
            short_count=short_count,
            exposure_limit_remaining=self.get_available_risk_budget(),
            max_correlated_warning=max_correlated_warning,
            position_risks=position_risks,
        )
    
    def _determine_risk_level(
        self,
        gross_pct: float,
        net_pct: float
    ) -> RiskLevel:
        """Determine portfolio risk level."""
        # Check drawdown
        if self.peak_value > 0:
            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            if drawdown >= self.max_drawdown_pct:
                return RiskLevel.CRITICAL
            elif drawdown >= self.max_drawdown_pct * 0.75:
                return RiskLevel.HIGH
        
        # Check exposure
        if gross_pct >= self.max_gross_exposure * 0.9:
            return RiskLevel.HIGH
        elif gross_pct >= self.max_gross_exposure * 0.7:
            return RiskLevel.ELEVATED
        elif gross_pct >= self.max_gross_exposure * 0.5:
            return RiskLevel.NORMAL
        
        return RiskLevel.LOW


def check_trade_risk(
    portfolio_value: float,
    position_value: float,
    current_gross_exposure: float,
    max_position_pct: float = 0.10,
    max_gross_exposure: float = 1.0,
) -> Tuple[bool, str]:
    """
    Simple trade risk check.
    
    Args:
        portfolio_value: Total portfolio value
        position_value: Value of proposed position
        current_gross_exposure: Current gross exposure
        max_position_pct: Maximum position size
        max_gross_exposure: Maximum gross exposure
    
    Returns:
        Tuple of (allowed, reason)
    """
    # Check position size
    position_pct = position_value / portfolio_value
    if position_pct > max_position_pct:
        return False, f"Position {position_pct:.0%} > max {max_position_pct:.0%}"
    
    # Check gross exposure
    new_gross = current_gross_exposure + position_value
    new_gross_pct = new_gross / portfolio_value
    
    if new_gross_pct > max_gross_exposure:
        return False, f"Gross exposure {new_gross_pct:.0%} > max {max_gross_exposure:.0%}"
    
    return True, ""
