"""
Cross-Asset Risk Overlay (Pod 4)

Portfolio-level risk-on/risk-off signals that gate all other strategy pods.

Implements:
  - VIX hard kill switch (VIX >= 30 → no new trades)
  - Composite risk score from VIX, DXY, SPX, 10Y yield
  - Drawdown circuit breaker (-8% → halt until review)
  - Daily loss limit (-3% → halt for day)
  - Dynamic position scaling by regime

Usage:
    from src.strategies.cross_asset_risk import CrossAssetRiskOverlay

    overlay = CrossAssetRiskOverlay()
    result = overlay.compute_risk_score(
        vix=28.5,
        vix_history=vix_series,
        dxy_history=dxy_series,
        spx_history=spx_series,
        us10y_history=yield_series,
    )
    # result["regime"]          → "RISK_ON" / "NEUTRAL" / "CAUTION" / "RISK_OFF" / "KILL"
    # result["position_scalar"] → 0.0 to 1.0 multiplier for position sizes
    # result["risk_score"]      → -1.0 to +1.0 composite score
    # result["flags"]           → list of active warnings
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


# =============================================================================
# Constants
# =============================================================================

# Component weights for composite risk score
RISK_WEIGHTS = {
    "vix": 0.40,
    "spx": 0.25,
    "dxy": 0.20,
    "yield": 0.15,
}

# Regime thresholds (on composite score)
REGIME_THRESHOLDS = {
    "RISK_ON": 0.3,      # score > 0.3
    "NEUTRAL": 0.0,       # 0.0 < score <= 0.3
    "CAUTION": -0.3,      # -0.3 < score <= 0.0
    "RISK_OFF": -1.0,     # score <= -0.3
}

# Position scaling by regime
REGIME_SCALARS = {
    "RISK_ON": 1.0,
    "NEUTRAL": 0.8,
    "CAUTION": 0.5,
    "RISK_OFF": 0.25,
    "KILL": 0.0,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RiskAssessment:
    """Result from risk overlay assessment."""
    risk_score: float
    regime: str
    position_scalar: float
    components: Dict[str, float]
    flags: List[str]
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "regime": self.regime,
            "position_scalar": self.position_scalar,
            "components": self.components,
            "flags": self.flags,
            "timestamp": str(self.timestamp) if self.timestamp else None,
        }


@dataclass
class CircuitBreakerResult:
    """Result from circuit breaker check."""
    halt: bool
    flags: List[str]
    drawdown_pct: float = 0.0
    daily_loss_pct: float = 0.0


# =============================================================================
# Cross-Asset Risk Overlay
# =============================================================================

class CrossAssetRiskOverlay:
    """
    Pod 4: Cross-Asset Risk Sentiment

    Provides portfolio-level risk-on/risk-off signals that gate
    all other strategy pods.

    Risk Score Components:
        - VIX (40%): z-score relative to 252-day history, inverted
        - SPX Momentum (25%): blended 1m/3m returns
        - DXY (20%): 21-day USD momentum, inverted (strong USD = risk-off)
        - 10Y Yield (15%): 21-day yield change, inverted (rapid rises = risk-off)

    Regimes:
        - RISK_ON:  score > 0.3  → full position sizing (1.0x)
        - NEUTRAL:  0.0 to 0.3  → slightly reduced (0.8x)
        - CAUTION:  -0.3 to 0.0 → half positions (0.5x)
        - RISK_OFF: < -0.3      → quarter positions (0.25x)
        - KILL:     VIX >= 30   → no new trades (0.0x)
    """

    def __init__(
        self,
        vix_kill_threshold: float = 30.0,
        vix_caution_threshold: float = 25.0,
        max_drawdown_pct: float = 0.08,
        max_daily_loss_pct: float = 0.03,
        lookback: int = 252,
    ):
        """
        Args:
            vix_kill_threshold: VIX level that triggers kill switch (no new trades)
            vix_caution_threshold: VIX level that triggers caution flag
            max_drawdown_pct: Portfolio drawdown that triggers circuit breaker
            max_daily_loss_pct: Daily loss that triggers halt
            lookback: Days of history for z-score calculations
        """
        self.vix_kill_threshold = vix_kill_threshold
        self.vix_caution_threshold = vix_caution_threshold
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.lookback = lookback

        # State
        self.kill_switch_active = False
        self.kill_switch_activated_date: Optional[datetime] = None
        self.circuit_breaker_active = False
        self.circuit_breaker_date: Optional[datetime] = None

        logger.info(
            f"CrossAssetRiskOverlay initialized: "
            f"VIX kill={vix_kill_threshold}, "
            f"max DD={max_drawdown_pct:.0%}, "
            f"max daily loss={max_daily_loss_pct:.0%}"
        )

    def compute_risk_score(
        self,
        vix: float,
        vix_history: pd.Series,
        dxy_history: Optional[pd.Series] = None,
        spx_history: Optional[pd.Series] = None,
        us10y_history: Optional[pd.Series] = None,
        as_of_date: Optional[datetime] = None,
    ) -> RiskAssessment:
        """
        Compute composite risk score from multiple macro indicators.

        Args:
            vix: Current VIX value
            vix_history: VIX time series (at least 60 days)
            dxy_history: Dollar index time series (optional)
            spx_history: S&P 500 time series (optional)
            us10y_history: US 10Y yield time series (optional)
            as_of_date: Assessment date

        Returns:
            RiskAssessment with score, regime, scalar, components, and flags
        """
        components = {}
        flags = []

        # --- VIX Kill Switch ---
        if pd.notna(vix) and vix >= self.vix_kill_threshold:
            self.kill_switch_active = True
            self.kill_switch_activated_date = as_of_date
            flags.append(f"KILL_SWITCH: VIX={vix:.1f} >= {self.vix_kill_threshold}")
            logger.warning(f"Kill switch activated: VIX={vix:.1f}")
            return RiskAssessment(
                risk_score=-1.0,
                regime="KILL",
                position_scalar=0.0,
                components={"vix": -1.0},
                flags=flags,
                timestamp=as_of_date,
            )

        # --- VIX Component (weight: 0.40) ---
        if pd.notna(vix):
            clean_vix = vix_history.dropna()
            if len(clean_vix) >= 60:
                vix_mean = clean_vix.rolling(self.lookback).mean().iloc[-1]
                vix_std = clean_vix.rolling(self.lookback).std().iloc[-1]
                if pd.notna(vix_mean) and pd.notna(vix_std) and vix_std > 0:
                    vix_z = (vix - vix_mean) / vix_std
                    components["vix"] = np.clip(-vix_z / 2, -1, 1)
                else:
                    components["vix"] = -0.5 if vix > 20 else 0.0
            else:
                components["vix"] = -0.5 if vix > 20 else 0.0

            if vix >= self.vix_caution_threshold:
                flags.append(f"CAUTION: VIX={vix:.1f} elevated")

        # --- SPX Momentum Component (weight: 0.25) ---
        if spx_history is not None and len(spx_history.dropna()) >= 63:
            spx_ret_1m = spx_history.pct_change(21).iloc[-1]
            spx_ret_3m = spx_history.pct_change(63).iloc[-1]
            if pd.notna(spx_ret_1m) and pd.notna(spx_ret_3m):
                spx_score = (spx_ret_1m * 0.6 + spx_ret_3m * 0.4) * 10
                components["spx"] = np.clip(spx_score, -1, 1)

        # --- DXY Component (weight: 0.20) ---
        if dxy_history is not None and len(dxy_history.dropna()) >= 63:
            dxy_ret = dxy_history.pct_change(21).iloc[-1]
            if pd.notna(dxy_ret):
                # Strong USD = risk-off
                components["dxy"] = np.clip(-dxy_ret * 20, -1, 1)

        # --- Yield Component (weight: 0.15) ---
        if us10y_history is not None and len(us10y_history.dropna()) >= 63:
            yield_change = us10y_history.diff(21).iloc[-1]
            if pd.notna(yield_change):
                # Rapidly rising yields = risk-off
                components["yield"] = np.clip(-yield_change * 2, -1, 1)

        # --- Composite Score ---
        score = 0.0
        total_weight = 0.0
        for key, weight in RISK_WEIGHTS.items():
            if key in components:
                score += components[key] * weight
                total_weight += weight

        if total_weight > 0:
            score = score / total_weight

        # --- Regime Classification ---
        if score > REGIME_THRESHOLDS["RISK_ON"]:
            regime = "RISK_ON"
        elif score > REGIME_THRESHOLDS["NEUTRAL"]:
            regime = "NEUTRAL"
        elif score > REGIME_THRESHOLDS["CAUTION"]:
            regime = "CAUTION"
            flags.append("POSITION_SCALING: 50% due to risk-off signal")
        else:
            regime = "RISK_OFF"
            flags.append("POSITION_SCALING: 25% due to strong risk-off signal")

        position_scalar = REGIME_SCALARS[regime]

        # --- Kill Switch Reset ---
        if self.kill_switch_active and vix < self.vix_caution_threshold:
            self.kill_switch_active = False
            flags.append("KILL_SWITCH_RESET: VIX below caution threshold")
            logger.info("Kill switch deactivated")

        return RiskAssessment(
            risk_score=round(score, 3),
            regime=regime,
            position_scalar=position_scalar,
            components={k: round(v, 3) for k, v in components.items()},
            flags=flags,
            timestamp=as_of_date,
        )

    def check_circuit_breaker(
        self,
        current_equity: float,
        peak_equity: float,
        daily_start_equity: float,
    ) -> CircuitBreakerResult:
        """
        Check drawdown and daily loss circuit breakers.

        Args:
            current_equity: Current portfolio value
            peak_equity: High water mark
            daily_start_equity: Portfolio value at start of day

        Returns:
            CircuitBreakerResult with halt flag and diagnostics
        """
        flags = []
        halt = False
        drawdown_pct = 0.0
        daily_loss_pct = 0.0

        # Drawdown check
        if peak_equity > 0:
            drawdown_pct = (peak_equity - current_equity) / peak_equity
            if drawdown_pct >= self.max_drawdown_pct:
                halt = True
                self.circuit_breaker_active = True
                self.circuit_breaker_date = datetime.now()
                flags.append(
                    f"CIRCUIT_BREAKER: Drawdown {drawdown_pct:.1%} >= {self.max_drawdown_pct:.0%}"
                )
                logger.critical(f"Circuit breaker triggered: DD={drawdown_pct:.1%}")

        # Daily loss check
        if daily_start_equity > 0:
            daily_loss_pct = (daily_start_equity - current_equity) / daily_start_equity
            if daily_loss_pct >= self.max_daily_loss_pct:
                halt = True
                flags.append(
                    f"DAILY_LOSS_LIMIT: Loss {daily_loss_pct:.1%} >= {self.max_daily_loss_pct:.0%}"
                )
                logger.warning(f"Daily loss limit hit: {daily_loss_pct:.1%}")

        return CircuitBreakerResult(
            halt=halt,
            flags=flags,
            drawdown_pct=drawdown_pct,
            daily_loss_pct=daily_loss_pct,
        )

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker after review."""
        self.circuit_breaker_active = False
        self.circuit_breaker_date = None
        logger.info("Circuit breaker manually reset")

    @property
    def is_trading_allowed(self) -> bool:
        """Whether the overlay allows new trades."""
        return not self.kill_switch_active and not self.circuit_breaker_active
