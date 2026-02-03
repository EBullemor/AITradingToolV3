"""
Regime Detection Module

Identifies market regimes for adaptive strategy behavior:
- Volatility regimes (LOW_VOL, NORMAL, HIGH_VOL)
- Risk regimes (RISK_ON, NEUTRAL, RISK_OFF)
- Trend regimes (TRENDING, MEAN_REVERTING, CHOPPY)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class VolatilityRegime(Enum):
    """Volatility regime states."""
    LOW_VOL = "LOW_VOL"
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOL"
    EXTREME = "EXTREME"


class RiskRegime(Enum):
    """Risk sentiment regime states."""
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"


class TrendRegime(Enum):
    """Trend regime states."""
    STRONG_TREND = "STRONG_TREND"
    WEAK_TREND = "WEAK_TREND"
    MEAN_REVERTING = "MEAN_REVERTING"
    CHOPPY = "CHOPPY"


# =============================================================================
# VIX-Based Regime Detection
# =============================================================================

def compute_vix_regime(
    vix: pd.Series,
    low_threshold: float = 15.0,
    high_threshold: float = 25.0,
    extreme_threshold: float = 35.0
) -> pd.Series:
    """
    Compute volatility regime based on VIX level.
    
    Args:
        vix: VIX index values
        low_threshold: Below this = LOW_VOL
        high_threshold: Above this = HIGH_VOL
        extreme_threshold: Above this = EXTREME
    
    Returns:
        Regime series
    """
    regime = pd.Series(index=vix.index, dtype=str)
    
    regime[vix <= low_threshold] = VolatilityRegime.LOW_VOL.value
    regime[(vix > low_threshold) & (vix <= high_threshold)] = VolatilityRegime.NORMAL.value
    regime[(vix > high_threshold) & (vix <= extreme_threshold)] = VolatilityRegime.HIGH_VOL.value
    regime[vix > extreme_threshold] = VolatilityRegime.EXTREME.value
    
    return regime


def compute_vix_percentile_regime(
    vix: pd.Series,
    lookback: int = 252,
    low_percentile: float = 25,
    high_percentile: float = 75
) -> pd.Series:
    """
    Compute regime based on VIX percentile (adaptive to market conditions).
    
    Args:
        vix: VIX index values
        lookback: Lookback period for percentile calculation
        low_percentile: Below this percentile = LOW_VOL
        high_percentile: Above this percentile = HIGH_VOL
    
    Returns:
        Regime series
    """
    def rolling_percentile(x):
        if len(x) < 2:
            return 50
        return (x.iloc[-1] <= x).sum() / len(x) * 100
    
    percentile = vix.rolling(lookback).apply(rolling_percentile)
    
    regime = pd.Series(index=vix.index, dtype=str)
    regime[percentile <= low_percentile] = VolatilityRegime.LOW_VOL.value
    regime[(percentile > low_percentile) & (percentile < high_percentile)] = VolatilityRegime.NORMAL.value
    regime[percentile >= high_percentile] = VolatilityRegime.HIGH_VOL.value
    
    return regime


def compute_vix_term_structure(
    vix_spot: pd.Series,
    vix_futures: pd.Series
) -> pd.DataFrame:
    """
    Analyze VIX term structure for regime insight.
    
    Contango (futures > spot) = complacency, risk-on
    Backwardation (futures < spot) = fear, risk-off
    
    Args:
        vix_spot: VIX spot index
        vix_futures: VIX futures (e.g., 1-month)
    
    Returns:
        DataFrame with term structure metrics
    """
    result = pd.DataFrame(index=vix_spot.index)
    
    # Term structure spread
    result["vix_term_spread"] = vix_futures - vix_spot
    result["vix_term_spread_pct"] = (vix_futures - vix_spot) / vix_spot * 100
    
    # Regime based on term structure
    result["vix_contango"] = (result["vix_term_spread"] > 0).astype(int)
    
    return result


# =============================================================================
# Cross-Asset Risk Regime
# =============================================================================

def compute_risk_regime(
    vix: pd.Series,
    spx_returns: Optional[pd.Series] = None,
    credit_spreads: Optional[pd.Series] = None,
    dxy: Optional[pd.Series] = None,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Compute composite risk regime from multiple indicators.
    
    Args:
        vix: VIX index
        spx_returns: S&P 500 returns (optional)
        credit_spreads: Credit spreads like HY-IG (optional)
        dxy: Dollar index (optional)
        lookback: Lookback for momentum calculations
    
    Returns:
        DataFrame with risk metrics and regime
    """
    result = pd.DataFrame(index=vix.index)
    
    # VIX component
    vix_zscore = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    result["vix_zscore"] = vix_zscore
    
    risk_score = -vix_zscore  # High VIX = negative risk score
    
    # SPX momentum component
    if spx_returns is not None:
        spx_momentum = spx_returns.rolling(lookback).mean() * np.sqrt(252)
        spx_zscore = (spx_momentum - spx_momentum.rolling(252).mean()) / spx_momentum.rolling(252).std()
        result["spx_momentum_zscore"] = spx_zscore
        risk_score = risk_score + spx_zscore
    
    # Credit spreads component
    if credit_spreads is not None:
        credit_zscore = (credit_spreads - credit_spreads.rolling(252).mean()) / credit_spreads.rolling(252).std()
        result["credit_zscore"] = credit_zscore
        risk_score = risk_score - credit_zscore  # Wide spreads = negative
    
    # DXY component (risk-off often = strong USD)
    if dxy is not None:
        dxy_momentum = dxy.pct_change(lookback)
        dxy_zscore = (dxy_momentum - dxy_momentum.rolling(252).mean()) / dxy_momentum.rolling(252).std()
        result["dxy_zscore"] = dxy_zscore
        risk_score = risk_score - dxy_zscore * 0.5  # Strong USD = slight risk-off
    
    # Normalize risk score
    result["risk_score"] = risk_score / risk_score.rolling(252).std()
    
    # Classify regime
    regime = pd.Series(index=vix.index, dtype=str)
    regime[result["risk_score"] > 0.5] = RiskRegime.RISK_ON.value
    regime[(result["risk_score"] >= -0.5) & (result["risk_score"] <= 0.5)] = RiskRegime.NEUTRAL.value
    regime[result["risk_score"] < -0.5] = RiskRegime.RISK_OFF.value
    result["risk_regime"] = regime
    
    return result


# =============================================================================
# Trend Regime Detection
# =============================================================================

def compute_trend_regime(
    prices: pd.Series,
    short_window: int = 20,
    long_window: int = 60,
    adf_window: int = 60
) -> pd.DataFrame:
    """
    Detect trend regime using multiple methods.
    
    Args:
        prices: Price series
        short_window: Short MA window
        long_window: Long MA window
        adf_window: Window for mean-reversion tests
    
    Returns:
        DataFrame with trend metrics and regime
    """
    result = pd.DataFrame(index=prices.index)
    
    # Moving average trend
    ma_short = prices.rolling(short_window).mean()
    ma_long = prices.rolling(long_window).mean()
    
    result["ma_trend"] = (ma_short - ma_long) / ma_long
    result["ma_trend_zscore"] = (
        (result["ma_trend"] - result["ma_trend"].rolling(252).mean()) /
        result["ma_trend"].rolling(252).std()
    )
    
    # Price momentum consistency
    returns = prices.pct_change()
    pos_returns = (returns > 0).rolling(short_window).mean()
    result["trend_consistency"] = np.abs(pos_returns - 0.5) * 2  # 0 = choppy, 1 = trending
    
    # Directional movement
    up_moves = returns.clip(lower=0).rolling(short_window).sum()
    down_moves = (-returns.clip(upper=0)).rolling(short_window).sum()
    result["dmi"] = (up_moves - down_moves) / (up_moves + down_moves + 1e-10)
    
    # ADX-like trend strength
    result["trend_strength"] = result["trend_consistency"] * np.abs(result["dmi"])
    
    # Classify regime
    regime = pd.Series(index=prices.index, dtype=str)
    
    strong_trend = (result["trend_strength"] > 0.3) & (result["trend_consistency"] > 0.6)
    weak_trend = (result["trend_strength"] > 0.15) & (result["trend_consistency"] > 0.4)
    mean_rev = result["trend_consistency"] < 0.3
    
    regime[strong_trend] = TrendRegime.STRONG_TREND.value
    regime[weak_trend & ~strong_trend] = TrendRegime.WEAK_TREND.value
    regime[mean_rev] = TrendRegime.MEAN_REVERTING.value
    regime[regime.isna()] = TrendRegime.CHOPPY.value
    
    result["trend_regime"] = regime
    
    return result


def compute_hurst_exponent(
    prices: pd.Series,
    window: int = 100,
    max_lag: int = 20
) -> pd.Series:
    """
    Compute rolling Hurst exponent for mean-reversion vs trending.
    
    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    
    Args:
        prices: Price series
        window: Rolling window
        max_lag: Maximum lag for R/S calculation
    
    Returns:
        Hurst exponent series
    """
    def hurst(ts):
        if len(ts) < max_lag * 2:
            return np.nan
        
        lags = range(2, max_lag)
        rs_values = []
        
        for lag in lags:
            # Split into chunks
            chunks = [ts[i:i+lag] for i in range(0, len(ts) - lag, lag)]
            if not chunks:
                continue
            
            rs_chunk = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean = np.mean(chunk)
                std = np.std(chunk)
                if std == 0:
                    continue
                
                cumdev = np.cumsum(chunk - mean)
                r = np.max(cumdev) - np.min(cumdev)
                rs_chunk.append(r / std)
            
            if rs_chunk:
                rs_values.append((lag, np.mean(rs_chunk)))
        
        if len(rs_values) < 3:
            return np.nan
        
        lags, rs = zip(*rs_values)
        log_lags = np.log(lags)
        log_rs = np.log(rs)
        
        # Linear regression
        slope, _ = np.polyfit(log_lags, log_rs, 1)
        
        return slope
    
    hurst_series = prices.rolling(window).apply(hurst)
    
    return hurst_series


# =============================================================================
# Combined Regime Analysis
# =============================================================================

def compute_market_regime(
    vix: pd.Series,
    prices: pd.Series,
    spx_prices: Optional[pd.Series] = None,
    dxy: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute comprehensive market regime analysis.
    
    Args:
        vix: VIX index
        prices: Asset prices (for trend analysis)
        spx_prices: S&P 500 prices (optional)
        dxy: Dollar index (optional)
    
    Returns:
        DataFrame with all regime classifications
    """
    result = pd.DataFrame(index=vix.index)
    
    # Volatility regime
    result["vol_regime"] = compute_vix_regime(vix)
    result["vol_percentile_regime"] = compute_vix_percentile_regime(vix)
    
    # Risk regime
    spx_returns = spx_prices.pct_change() if spx_prices is not None else None
    risk_df = compute_risk_regime(vix, spx_returns, dxy=dxy)
    result["risk_regime"] = risk_df["risk_regime"]
    result["risk_score"] = risk_df["risk_score"]
    
    # Trend regime
    trend_df = compute_trend_regime(prices)
    result["trend_regime"] = trend_df["trend_regime"]
    result["trend_strength"] = trend_df["trend_strength"]
    
    # Composite regime score
    # Favor trending strategies in low vol + risk-on + trending
    result["strategy_regime"] = compute_strategy_regime(
        result["vol_regime"],
        result["risk_regime"],
        result["trend_regime"]
    )
    
    logger.info("Computed comprehensive market regime analysis")
    
    return result


def compute_strategy_regime(
    vol_regime: pd.Series,
    risk_regime: pd.Series,
    trend_regime: pd.Series
) -> pd.Series:
    """
    Determine which strategy type is favored.
    
    Args:
        vol_regime: Volatility regime
        risk_regime: Risk sentiment regime
        trend_regime: Trend regime
    
    Returns:
        Strategy regime recommendation
    """
    strategy = pd.Series(index=vol_regime.index, dtype=str)
    
    # Carry strategies favor low vol + risk-on
    carry_favorable = (
        (vol_regime.isin([VolatilityRegime.LOW_VOL.value, VolatilityRegime.NORMAL.value])) &
        (risk_regime.isin([RiskRegime.RISK_ON.value, RiskRegime.NEUTRAL.value]))
    )
    
    # Trend strategies favor strong trends
    trend_favorable = trend_regime.isin([
        TrendRegime.STRONG_TREND.value, 
        TrendRegime.WEAK_TREND.value
    ])
    
    # Mean reversion favors high vol + mean reverting
    mean_rev_favorable = (
        (vol_regime == VolatilityRegime.HIGH_VOL.value) &
        (trend_regime == TrendRegime.MEAN_REVERTING.value)
    )
    
    # Risk-off = reduce exposure
    risk_off = risk_regime == RiskRegime.RISK_OFF.value
    
    # Assign strategy regimes
    strategy[carry_favorable & ~trend_favorable] = "CARRY"
    strategy[trend_favorable & ~carry_favorable] = "TREND"
    strategy[carry_favorable & trend_favorable] = "CARRY_TREND"
    strategy[mean_rev_favorable] = "MEAN_REVERSION"
    strategy[risk_off] = "DEFENSIVE"
    strategy[strategy.isna()] = "NEUTRAL"
    
    return strategy


# =============================================================================
# Regime Change Detection
# =============================================================================

def detect_regime_changes(
    regime: pd.Series,
    min_duration: int = 5
) -> pd.DataFrame:
    """
    Detect regime changes and filter out noise.
    
    Args:
        regime: Regime series
        min_duration: Minimum bars before regime change is confirmed
    
    Returns:
        DataFrame with regime change information
    """
    result = pd.DataFrame(index=regime.index)
    
    # Raw regime changes
    result["regime"] = regime
    result["regime_changed"] = regime != regime.shift(1)
    
    # Confirmed changes (lasted at least min_duration)
    regime_groups = (result["regime_changed"]).cumsum()
    group_sizes = regime_groups.map(regime_groups.value_counts())
    result["regime_confirmed"] = group_sizes >= min_duration
    
    # Days in current regime
    result["days_in_regime"] = regime_groups.groupby(regime_groups).cumcount() + 1
    
    return result
