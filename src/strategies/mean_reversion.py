"""
Mean Reversion Strategy (Pod 5)

Fades extreme intraday/multiday moves that lack a fundamental catalyst.

Core logic:
    1. Detect spike: daily return z-score > threshold (default 3.0)
    2. Verify no catalyst: LLM screens recent news for structural drivers
    3. Check liquidity: only trade top-liquidity instruments
    4. Fade the move: enter opposite direction with tight stop

Key characteristics:
    - Higher hit rate (~58%) but smaller average win
    - Very short holding period (1-5 days)
    - Tighter stops than trend-following pods (1.5x ATR)
    - LLM catalyst screening is the differentiator

Instruments: EURUSD, USDJPY, GBPUSD, AUDUSD, BTCUSD, CL, GC
"""

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

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


# ============================================================================
# Enums & Data Classes
# ============================================================================


class SpikeDirection(str, Enum):
    """Direction of the price spike being faded."""
    UP = "UP"
    DOWN = "DOWN"


class CatalystVerdict(str, Enum):
    """LLM catalyst screening result."""
    NO_CATALYST = "NO_CATALYST"           # Safe to fade
    WEAK_CATALYST = "WEAK_CATALYST"       # Marginal catalyst, reduce confidence
    STRONG_CATALYST = "STRONG_CATALYST"   # Structural driver, do not fade
    UNKNOWN = "UNKNOWN"                   # LLM unavailable, use fallback


@dataclass
class SpikeDetection:
    """Result of spike detection on an instrument."""
    instrument: str
    detected: bool
    direction: Optional[SpikeDirection] = None
    daily_return: float = 0.0
    zscore: float = 0.0
    abs_zscore: float = 0.0
    lookback_mean: float = 0.0
    lookback_std: float = 0.0
    multi_day_zscore: Optional[float] = None  # 2-day cumulative
    price: float = 0.0
    atr: float = 0.0


@dataclass
class CatalystScreening:
    """Result of LLM catalyst screening."""
    instrument: str
    verdict: CatalystVerdict
    catalyst_description: str = ""
    confidence: float = 0.5
    headlines_checked: int = 0
    screening_method: str = "llm"  # llm, news_summary, fallback


# ============================================================================
# Instrument Metadata
# ============================================================================

INSTRUMENT_META = {
    "EURUSD": {
        "asset_class": "fx",
        "liquidity_rank": 1.0,
        "stop_pct_fallback": 0.008,  # 80 pips
        "typical_spread_pips": 1.0,
        "news_keywords": ["ECB", "eurozone", "EUR", "euro"],
    },
    "USDJPY": {
        "asset_class": "fx",
        "liquidity_rank": 0.95,
        "stop_pct_fallback": 0.010,
        "typical_spread_pips": 1.0,
        "news_keywords": ["BOJ", "yen", "JPY", "Japan"],
    },
    "GBPUSD": {
        "asset_class": "fx",
        "liquidity_rank": 0.90,
        "stop_pct_fallback": 0.010,
        "typical_spread_pips": 1.5,
        "news_keywords": ["BOE", "sterling", "GBP", "UK"],
    },
    "AUDUSD": {
        "asset_class": "fx",
        "liquidity_rank": 0.85,
        "stop_pct_fallback": 0.012,
        "typical_spread_pips": 1.5,
        "news_keywords": ["RBA", "AUD", "Australia", "iron ore"],
    },
    "BTCUSD": {
        "asset_class": "crypto",
        "liquidity_rank": 0.80,
        "stop_pct_fallback": 0.05,
        "typical_spread_pips": 50.0,
        "news_keywords": ["bitcoin", "BTC", "crypto", "SEC crypto", "ETF bitcoin"],
    },
    "CL": {
        "asset_class": "commodity",
        "liquidity_rank": 0.90,
        "stop_pct_fallback": 0.04,
        "typical_spread_pips": 3.0,
        "news_keywords": ["crude", "oil", "OPEC", "WTI", "petroleum"],
    },
    "GC": {
        "asset_class": "commodity",
        "liquidity_rank": 0.85,
        "stop_pct_fallback": 0.025,
        "typical_spread_pips": 2.0,
        "news_keywords": ["gold", "bullion", "safe haven", "precious metals"],
    },
}


# ============================================================================
# Default Catalyst Screener (LLM-powered)
# ============================================================================


def default_catalyst_screener(
    instrument: str,
    spike: SpikeDetection,
    news_summary: Optional[Dict] = None,
) -> CatalystScreening:
    """
    Default catalyst screening using news_summary dict.

    In production, this calls the LLM client to verify whether
    recent news contains a structural catalyst for the move.
    Here we implement a keyword-based heuristic that can be
    replaced with a full LLM call.

    Args:
        instrument: Ticker
        spike: Detected spike details
        news_summary: News summary from pipeline (optional)

    Returns:
        CatalystScreening result
    """
    meta = INSTRUMENT_META.get(instrument, {})
    keywords = meta.get("news_keywords", [])

    if news_summary is None:
        return CatalystScreening(
            instrument=instrument,
            verdict=CatalystVerdict.UNKNOWN,
            catalyst_description="No news data available",
            confidence=0.3,
            screening_method="fallback",
        )

    # Check catalysts list from news summary
    catalysts = news_summary.get("catalysts", [])
    key_events = news_summary.get("key_events", [])
    all_text = " ".join(catalysts + key_events).lower()

    matched = [kw for kw in keywords if kw.lower() in all_text]

    if not matched:
        return CatalystScreening(
            instrument=instrument,
            verdict=CatalystVerdict.NO_CATALYST,
            catalyst_description="No relevant catalysts found in recent news",
            confidence=0.7,
            headlines_checked=len(catalysts) + len(key_events),
            screening_method="news_summary",
        )

    # Assess strength based on number of matches and event count
    if len(matched) >= 2 or any("rate" in t.lower() or "policy" in t.lower() for t in catalysts):
        return CatalystScreening(
            instrument=instrument,
            verdict=CatalystVerdict.STRONG_CATALYST,
            catalyst_description=f"Structural driver detected: {', '.join(matched)}",
            confidence=0.8,
            headlines_checked=len(catalysts) + len(key_events),
            screening_method="news_summary",
        )

    return CatalystScreening(
        instrument=instrument,
        verdict=CatalystVerdict.WEAK_CATALYST,
        catalyst_description=f"Marginal catalyst: {', '.join(matched)}",
        confidence=0.5,
        headlines_checked=len(catalysts) + len(key_events),
        screening_method="news_summary",
    )


# ============================================================================
# Strategy
# ============================================================================


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy (Pod 5).

    Detects extreme price moves (z-score spikes), verifies no fundamental
    catalyst via LLM screening, and generates counter-trend fade signals
    with tight risk parameters.
    """

    DEFAULT_CONFIG = {
        "pod_name": "mean_reversion",
        "enabled": True,
        "instruments": ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "BTCUSD", "CL", "GC"],
        "signal_validity_hours": 12,  # Shorter — time-sensitive
        "max_signals_per_run": 3,

        # Spike detection
        "spike": {
            "zscore_threshold": 3.0,
            "lookback_days": 252,
            "min_lookback_days": 60,
            "use_multi_day": True,         # Also check 2-day cumulative
            "multi_day_threshold": 2.5,    # Lower threshold for 2-day
        },

        # Filters
        "filters": {
            "require_no_catalyst": True,
            "allow_weak_catalyst": False,   # If True, weak catalyst still trades (lower conf)
            "min_liquidity_rank": 0.8,
            "max_recent_signals": 2,        # Don't overload same instrument
            "recent_signal_window_days": 5,
        },

        # Risk
        "risk": {
            "max_holding_days": 5,
            "stop_multiple": 1.5,          # ATR multiplier for stop
            "target_multiple": 2.0,        # ATR multiplier for target (risk-reward ~1.3)
            "use_atr_stops": True,
        },

        # VIX gating
        "regime": {
            "vix_extreme_threshold": 35,   # Avoid fading in extreme vol
            "vix_boost_threshold": 25,     # Higher vol → more spikes, more opportunity
        },

        # Confidence
        "confidence": {
            "base": 0.30,
            "extremity_weight": 0.30,      # How far past threshold
            "no_catalyst_weight": 0.20,    # LLM verification bonus
            "liquidity_weight": 0.10,      # Liquidity bonus
            "regime_weight": 0.10,         # VIX context
        },
    }

    def __init__(
        self,
        config: Optional[Dict] = None,
        catalyst_screener: Optional[Callable] = None,
    ):
        """
        Initialise Mean Reversion strategy.

        Args:
            config: Strategy configuration
            catalyst_screener: Custom catalyst screening function.
                               Signature: (instrument, spike, news_summary) -> CatalystScreening
        """
        merged = self.DEFAULT_CONFIG.copy()
        if config:
            self._deep_merge(merged, config)

        super().__init__(merged, name="Mean Reversion (Vol Spike)")

        self.spike_config = merged["spike"]
        self.filter_config = merged["filters"]
        self.risk_config = merged["risk"]
        self.regime_config = merged["regime"]
        self.confidence_config = merged["confidence"]

        self.catalyst_screener = catalyst_screener or default_catalyst_screener
        self._recent_signals: List[Signal] = []

        logger.info(
            f"Mean Reversion strategy initialised | "
            f"Z-score threshold: {self.spike_config['zscore_threshold']} | "
            f"Instruments: {self.instruments} | "
            f"Catalyst filter: {self.filter_config['require_no_catalyst']}"
        )

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get_required_features(self) -> List[str]:
        return ["PX_LAST", "atr_14", "realized_vol_30d"]

    # ========================================================================
    # Spike Detection
    # ========================================================================

    def detect_spike(
        self,
        instrument: str,
        features: pd.DataFrame,
    ) -> SpikeDetection:
        """
        Detect whether an instrument has experienced a spike.

        A spike = daily return z-score exceeds threshold,
        measured against lookback-period standard deviation.

        Args:
            instrument: Ticker
            features: Feature DataFrame with PX_LAST column

        Returns:
            SpikeDetection result
        """
        result = SpikeDetection(instrument=instrument, detected=False)

        # Extract price series
        if "PX_LAST" in features.columns:
            prices = features["PX_LAST"].dropna()
        else:
            numeric = features.select_dtypes(include=[np.number])
            if numeric.empty:
                logger.debug(f"{instrument}: No price data")
                return result
            prices = numeric.iloc[:, 0].dropna()

        min_lookback = self.spike_config["min_lookback_days"]
        if len(prices) < min_lookback:
            logger.debug(f"{instrument}: Insufficient data ({len(prices)} < {min_lookback})")
            return result

        # Daily returns
        returns = prices.pct_change().dropna()
        if len(returns) < min_lookback:
            return result

        lookback = min(self.spike_config["lookback_days"], len(returns))
        lookback_returns = returns.iloc[-lookback:]

        mean_ret = float(lookback_returns.mean())
        std_ret = float(lookback_returns.std())

        if std_ret < 1e-10:
            return result

        latest_return = float(returns.iloc[-1])
        zscore = (latest_return - mean_ret) / std_ret

        result.daily_return = latest_return
        result.zscore = zscore
        result.abs_zscore = abs(zscore)
        result.lookback_mean = mean_ret
        result.lookback_std = std_ret
        result.price = float(prices.iloc[-1])

        # ATR
        if "atr_14" in features.columns:
            atr_val = features["atr_14"].dropna()
            if len(atr_val) > 0:
                result.atr = float(atr_val.iloc[-1])

        # Single-day threshold
        threshold = self.spike_config["zscore_threshold"]
        if result.abs_zscore >= threshold:
            result.detected = True
            result.direction = SpikeDirection.UP if zscore > 0 else SpikeDirection.DOWN
            logger.info(
                f"{instrument}: Spike detected | return={latest_return:+.4f} | "
                f"z={zscore:+.2f} | direction={result.direction.value}"
            )
            return result

        # Multi-day cumulative check
        if self.spike_config.get("use_multi_day", False) and len(returns) >= 2:
            cum_2d = float(returns.iloc[-2:].sum())
            cum_zscore = (cum_2d - mean_ret * 2) / (std_ret * np.sqrt(2))
            result.multi_day_zscore = cum_zscore

            multi_threshold = self.spike_config.get("multi_day_threshold", 2.5)
            if abs(cum_zscore) >= multi_threshold:
                result.detected = True
                result.direction = SpikeDirection.UP if cum_zscore > 0 else SpikeDirection.DOWN
                result.abs_zscore = abs(cum_zscore)
                result.zscore = cum_zscore
                logger.info(
                    f"{instrument}: Multi-day spike | 2d_return={cum_2d:+.4f} | "
                    f"z={cum_zscore:+.2f}"
                )

        return result

    # ========================================================================
    # Catalyst Filter
    # ========================================================================

    def screen_catalyst(
        self,
        instrument: str,
        spike: SpikeDetection,
        news_summary: Optional[Dict] = None,
    ) -> CatalystScreening:
        """
        Screen for fundamental catalyst behind the spike.

        Delegates to the configured catalyst_screener function.

        Args:
            instrument: Ticker
            spike: Detected spike
            news_summary: Pipeline news summary

        Returns:
            CatalystScreening result
        """
        screening = self.catalyst_screener(instrument, spike, news_summary)
        logger.info(
            f"{instrument}: Catalyst screening → {screening.verdict.value} | "
            f"{screening.catalyst_description}"
        )
        return screening

    def should_trade_after_screening(
        self, screening: CatalystScreening
    ) -> Tuple[bool, float]:
        """
        Decide whether to proceed based on catalyst screening.

        Args:
            screening: Catalyst screening result

        Returns:
            (should_trade, confidence_adjustment)
        """
        verdict = screening.verdict
        require = self.filter_config["require_no_catalyst"]
        allow_weak = self.filter_config.get("allow_weak_catalyst", False)

        if verdict == CatalystVerdict.STRONG_CATALYST:
            return False, 0.0

        if verdict == CatalystVerdict.NO_CATALYST:
            return True, 1.0

        if verdict == CatalystVerdict.WEAK_CATALYST:
            if not require:
                return True, 0.8
            if allow_weak:
                return True, 0.6
            return False, 0.0

        # UNKNOWN — LLM unavailable
        if require:
            return False, 0.0
        return True, 0.5

    # ========================================================================
    # Liquidity Filter
    # ========================================================================

    def check_liquidity(self, instrument: str) -> Tuple[bool, float]:
        """
        Check if instrument meets minimum liquidity rank.

        Args:
            instrument: Ticker

        Returns:
            (passes_filter, liquidity_rank)
        """
        meta = INSTRUMENT_META.get(instrument, {})
        rank = meta.get("liquidity_rank", 0.5)
        min_rank = self.filter_config["min_liquidity_rank"]

        passes = rank >= min_rank
        if not passes:
            logger.debug(f"{instrument}: Liquidity rank {rank} < {min_rank}")
        return passes, rank

    # ========================================================================
    # Recent Signal Tracking
    # ========================================================================

    def check_recent_signal_limit(
        self, instrument: str, as_of_date: datetime
    ) -> bool:
        """
        Check if we've already sent too many recent signals for this instrument.

        Args:
            instrument: Ticker
            as_of_date: Reference date

        Returns:
            True if within limit (can trade), False if exceeded
        """
        max_recent = self.filter_config["max_recent_signals"]
        window = timedelta(days=self.filter_config["recent_signal_window_days"])
        cutoff = as_of_date - window

        recent_count = sum(
            1 for s in self._recent_signals
            if s.instrument == instrument and s.generated_at >= cutoff
        )

        if recent_count >= max_recent:
            logger.debug(
                f"{instrument}: {recent_count} recent signals >= max {max_recent}"
            )
            return False
        return True

    # ========================================================================
    # Price Levels
    # ========================================================================

    def compute_price_levels(
        self,
        instrument: str,
        spike: SpikeDetection,
        fade_direction: SignalDirection,
    ) -> Dict[str, Optional[float]]:
        """
        Compute entry, stop, and target prices.

        Uses ATR-based stops (tighter than trend-following pods).
        Fade direction is opposite to the spike.

        Args:
            instrument: Ticker
            spike: Detected spike
            fade_direction: Direction of fade trade

        Returns:
            Dict with entry, stop_loss, target_1, target_2
        """
        price = spike.price
        atr = spike.atr
        meta = INSTRUMENT_META.get(instrument, {})
        stop_mult = self.risk_config["stop_multiple"]
        target_mult = self.risk_config["target_multiple"]

        # Use ATR if available, else percentage fallback
        if atr > 0 and self.risk_config.get("use_atr_stops", True):
            stop_dist = atr * stop_mult
            target_dist = atr * target_mult
        else:
            fallback_pct = meta.get("stop_pct_fallback", 0.02)
            stop_dist = price * fallback_pct
            target_dist = price * fallback_pct * (target_mult / stop_mult)

        if fade_direction == SignalDirection.LONG:
            # Fading a down spike → buy the dip
            return {
                "entry": price,
                "stop_loss": price - stop_dist,
                "target_1": price + target_dist,
                "target_2": price + target_dist * 1.5,
            }
        else:
            # Fading an up spike → sell the rip
            return {
                "entry": price,
                "stop_loss": price + stop_dist,
                "target_1": price - target_dist,
                "target_2": price - target_dist * 1.5,
            }

    # ========================================================================
    # Confidence Scoring
    # ========================================================================

    def compute_confidence(
        self,
        spike: SpikeDetection,
        screening: CatalystScreening,
        liquidity_rank: float,
        vix_level: Optional[float] = None,
    ) -> float:
        """
        Compute signal confidence.

        Formula:
            base (0.30)
            + extremity_weight * min((|z| - threshold) / 2, 1.0)
            + no_catalyst_weight * catalyst_adjustment
            + liquidity_weight * liquidity_rank
            + regime_weight * vix_context

        Args:
            spike: Detected spike
            screening: Catalyst screening result
            liquidity_rank: Instrument liquidity (0-1)
            vix_level: Current VIX (optional)

        Returns:
            Confidence score (0-1)
        """
        cfg = self.confidence_config
        threshold = self.spike_config["zscore_threshold"]

        # Base
        conf = cfg["base"]

        # Extremity bonus
        excess = spike.abs_zscore - threshold
        extremity_score = min(excess / 2.0, 1.0) if excess > 0 else 0.0
        conf += cfg["extremity_weight"] * extremity_score

        # Catalyst bonus
        _, catalyst_adj = self.should_trade_after_screening(screening)
        conf += cfg["no_catalyst_weight"] * catalyst_adj

        # Liquidity bonus
        conf += cfg["liquidity_weight"] * liquidity_rank

        # VIX context
        if vix_level is not None:
            extreme = self.regime_config["vix_extreme_threshold"]
            boost = self.regime_config["vix_boost_threshold"]
            if vix_level >= extreme:
                conf -= 0.1  # Penalise — too volatile to fade
            elif vix_level >= boost:
                conf += cfg["regime_weight"] * 0.5  # Moderate boost
            else:
                conf += cfg["regime_weight"] * 0.3
        else:
            conf += cfg["regime_weight"] * 0.3  # Neutral if unknown

        return float(np.clip(conf, 0.0, 0.95))

    # ========================================================================
    # Signal Generation
    # ========================================================================

    def generate_signals(
        self,
        features: Dict[str, pd.DataFrame],
        macro_data: Optional[pd.DataFrame] = None,
        news_summary: Optional[Dict] = None,
        as_of_date: Optional[datetime] = None,
    ) -> List[Signal]:
        """
        Generate mean reversion signals.

        Pipeline:
            1. For each instrument, detect spike
            2. Screen catalyst (LLM)
            3. Check liquidity
            4. Check recent signal limit
            5. Compute price levels and confidence
            6. Emit fade signal

        Args:
            features: Dict mapping ticker to feature DataFrame
            macro_data: Macro data (VIX)
            news_summary: News summary dict from LLM pipeline
            as_of_date: Reference date

        Returns:
            List of fade signals
        """
        if not self.enabled:
            logger.info("Mean Reversion strategy disabled")
            return []

        if as_of_date is None:
            as_of_date = datetime.now()

        # Extract VIX
        vix_level = self._get_vix(macro_data)
        if vix_level is not None and vix_level >= self.regime_config["vix_extreme_threshold"]:
            logger.warning(f"VIX at {vix_level:.0f} — above extreme threshold, skipping mean reversion")
            return []

        signals = []

        for instrument in self.instruments:
            if instrument not in features:
                continue

            feat_df = features[instrument]

            # 1. Spike detection
            spike = self.detect_spike(instrument, feat_df)
            if not spike.detected:
                continue

            # 2. Catalyst screening
            screening = self.screen_catalyst(instrument, spike, news_summary)
            should_trade, _ = self.should_trade_after_screening(screening)
            if not should_trade:
                logger.info(f"{instrument}: Catalyst filter blocked signal ({screening.verdict.value})")
                continue

            # 3. Liquidity check
            passes_liq, liq_rank = self.check_liquidity(instrument)
            if not passes_liq:
                continue

            # 4. Recent signal limit
            if not self.check_recent_signal_limit(instrument, as_of_date):
                continue

            # 5. Fade direction (opposite to spike)
            if spike.direction == SpikeDirection.UP:
                fade_dir = SignalDirection.SHORT
            else:
                fade_dir = SignalDirection.LONG

            # 6. Price levels
            levels = self.compute_price_levels(instrument, spike, fade_dir)

            # 7. Confidence
            confidence = self.compute_confidence(spike, screening, liq_rank, vix_level)

            # 8. Build signal
            holding_min, holding_max = 1, self.risk_config["max_holding_days"]

            signal = Signal(
                instrument=instrument,
                direction=fade_dir,
                strength=confidence,
                strategy_name=self.name,
                strategy_pod=self.pod_name,
                generated_at=as_of_date,
                valid_until=as_of_date + timedelta(hours=self.signal_validity_hours),
                entry_price=levels["entry"],
                stop_loss=levels["stop_loss"],
                take_profit_1=levels["target_1"],
                take_profit_2=levels.get("target_2"),
                rationale=self._build_rationale(spike, screening, fade_dir),
                key_factors=self._build_key_factors(spike, screening, vix_level),
                regime=self._classify_regime(vix_level),
                confidence_drivers={
                    "extremity": spike.abs_zscore,
                    "catalyst_clear": 1.0 if screening.verdict == CatalystVerdict.NO_CATALYST else 0.0,
                    "liquidity": liq_rank,
                },
                metadata={
                    "spike_zscore": spike.zscore,
                    "spike_direction": spike.direction.value if spike.direction else None,
                    "daily_return": spike.daily_return,
                    "multi_day_zscore": spike.multi_day_zscore,
                    "catalyst_verdict": screening.verdict.value,
                    "catalyst_description": screening.catalyst_description,
                    "screening_method": screening.screening_method,
                    "liquidity_rank": liq_rank,
                    "atr": spike.atr,
                    "holding_period": f"{holding_min}-{holding_max} days",
                    "stop_multiple": self.risk_config["stop_multiple"],
                    "target_multiple": self.risk_config["target_multiple"],
                },
            )

            signals.append(signal)
            self._recent_signals.append(signal)

            logger.info(
                f"{instrument}: Fade signal → {fade_dir.value} | "
                f"z={spike.zscore:+.2f} | catalyst={screening.verdict.value} | "
                f"conf={confidence:.2f}"
            )

            if len(signals) >= self.max_signals:
                break

        logger.info(f"Mean Reversion: {len(signals)} signal(s) generated")
        return signals

    # ========================================================================
    # Backtesting
    # ========================================================================

    def backtest_signal(
        self,
        signal: Signal,
        subsequent_prices: pd.Series,
    ) -> Dict[str, Any]:
        """
        Backtest a mean reversion signal.

        Checks if stop, target, or time exit was hit first.

        Args:
            signal: Generated signal
            subsequent_prices: Prices after signal date

        Returns:
            Dict with outcome, pnl, holding_days, exit_reason
        """
        entry = signal.entry_price
        stop = signal.stop_loss
        target = signal.take_profit_1
        max_days = self.risk_config["max_holding_days"]
        is_long = signal.direction == SignalDirection.LONG

        for i, price in enumerate(subsequent_prices):
            if i >= max_days:
                # Time exit
                pnl = (price - entry) if is_long else (entry - price)
                return {
                    "outcome": "expired",
                    "pnl": pnl,
                    "pnl_pct": pnl / entry,
                    "holding_days": max_days,
                    "exit_price": price,
                    "exit_reason": "max_holding_days",
                }

            if is_long:
                if price <= stop:
                    return {
                        "outcome": "stopped_out",
                        "pnl": stop - entry,
                        "pnl_pct": (stop - entry) / entry,
                        "holding_days": i + 1,
                        "exit_price": stop,
                        "exit_reason": "stop_loss",
                    }
                if price >= target:
                    return {
                        "outcome": "target_hit",
                        "pnl": target - entry,
                        "pnl_pct": (target - entry) / entry,
                        "holding_days": i + 1,
                        "exit_price": target,
                        "exit_reason": "take_profit",
                    }
            else:
                if price >= stop:
                    return {
                        "outcome": "stopped_out",
                        "pnl": entry - stop,
                        "pnl_pct": (entry - stop) / entry,
                        "holding_days": i + 1,
                        "exit_price": stop,
                        "exit_reason": "stop_loss",
                    }
                if price <= target:
                    return {
                        "outcome": "target_hit",
                        "pnl": entry - target,
                        "pnl_pct": (entry - target) / entry,
                        "holding_days": i + 1,
                        "exit_price": target,
                        "exit_reason": "take_profit",
                    }

        # Exhausted prices
        last_price = float(subsequent_prices.iloc[-1])
        pnl = (last_price - entry) if is_long else (entry - last_price)
        return {
            "outcome": "data_exhausted",
            "pnl": pnl,
            "pnl_pct": pnl / entry,
            "holding_days": len(subsequent_prices),
            "exit_price": last_price,
            "exit_reason": "end_of_data",
        }

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    @staticmethod
    def _get_vix(macro_data: Optional[pd.DataFrame]) -> Optional[float]:
        if macro_data is None or (isinstance(macro_data, pd.DataFrame) and macro_data.empty):
            return None
        if isinstance(macro_data, pd.DataFrame):
            for col in ["PX_LAST", "vix", "VIX"]:
                if col in macro_data.columns:
                    return float(macro_data[col].dropna().iloc[-1])
            numeric = macro_data.select_dtypes(include=[np.number])
            if not numeric.empty:
                return float(numeric.iloc[-1, 0])
        return None

    def _classify_regime(self, vix_level: Optional[float]) -> str:
        if vix_level is None:
            return "UNKNOWN"
        if vix_level >= self.regime_config["vix_extreme_threshold"]:
            return "EXTREME_VOL"
        if vix_level >= self.regime_config["vix_boost_threshold"]:
            return "HIGH_VOL"
        return "NORMAL"

    def _build_rationale(
        self, spike: SpikeDetection, screening: CatalystScreening, fade_dir: SignalDirection
    ) -> str:
        parts = [
            f"Fading {spike.direction.value} spike on {spike.instrument} "
            f"(z={spike.zscore:+.2f}, return={spike.daily_return:+.4f})",
            f"Catalyst: {screening.verdict.value}",
            f"Direction: {fade_dir.value}",
        ]
        if spike.multi_day_zscore is not None:
            parts.append(f"2-day z={spike.multi_day_zscore:+.2f}")
        return " | ".join(parts)

    def _build_key_factors(
        self, spike: SpikeDetection, screening: CatalystScreening, vix: Optional[float]
    ) -> List[str]:
        factors = [
            f"Return z-score: {spike.zscore:+.2f} (threshold: {self.spike_config['zscore_threshold']})",
        ]
        if screening.verdict == CatalystVerdict.NO_CATALYST:
            factors.append("No fundamental catalyst detected — safe to fade")
        elif screening.verdict == CatalystVerdict.WEAK_CATALYST:
            factors.append(f"Weak catalyst: {screening.catalyst_description}")
        if vix is not None:
            factors.append(f"VIX at {vix:.0f}")
        factors.append(f"Max holding: {self.risk_config['max_holding_days']} days")
        return factors


# ============================================================================
# Factory
# ============================================================================


def create_mean_reversion_strategy(
    config_path: Optional[str] = None,
    catalyst_screener: Optional[Callable] = None,
) -> MeanReversionStrategy:
    """
    Factory function to create Mean Reversion strategy.

    Args:
        config_path: Path to config file (optional)
        catalyst_screener: Custom catalyst screening function

    Returns:
        Configured strategy instance
    """
    config = {}
    if config_path:
        config = load_strategy_config("mean_reversion")

    return MeanReversionStrategy(config=config, catalyst_screener=catalyst_screener)
