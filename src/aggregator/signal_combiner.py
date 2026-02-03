"""
Signal Combiner Module

Core logic for combining signals from multiple strategy pods into
coherent, deduplicated recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
from loguru import logger

from src.strategies.base import Signal, SignalDirection, SignalStatus


@dataclass
class AggregatedSignal:
    """
    Final aggregated signal combining inputs from multiple strategy pods.
    
    This is what gets promoted to a recommendation for the user.
    """
    # Core fields
    instrument: str
    direction: SignalDirection
    confidence: float  # 0-1, ensemble weighted
    
    # Contributing strategies
    contributing_pods: List[str]
    contributing_signals: List[Signal]
    
    # Price levels (most conservative)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    
    # Combined rationale
    rationale: str = ""
    key_factors: List[str] = field(default_factory=list)
    
    # Conflict info
    conflict_flag: bool = False
    conflict_details: Optional[str] = None
    
    # Metadata
    aggregated_at: datetime = field(default_factory=datetime.now)
    signal_id: Optional[str] = None
    regime: Optional[str] = None
    
    # Risk metrics
    risk_reward_ratio: Optional[float] = None
    position_size_pct: Optional[float] = None
    
    def __post_init__(self):
        if self.signal_id is None:
            self.signal_id = f"AGG_{self.instrument}_{self.aggregated_at.strftime('%Y%m%d%H%M%S')}"
    
    @property
    def strength_label(self) -> str:
        """Get human-readable strength label."""
        if self.confidence >= 0.8:
            return "VERY_STRONG"
        elif self.confidence >= 0.6:
            return "STRONG"
        elif self.confidence >= 0.4:
            return "MODERATE"
        else:
            return "WEAK"
    
    @property
    def pod_count(self) -> int:
        """Number of contributing strategy pods."""
        return len(self.contributing_pods)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "signal_id": self.signal_id,
            "instrument": self.instrument,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "strength_label": self.strength_label,
            "contributing_pods": self.contributing_pods,
            "pod_count": self.pod_count,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "risk_reward_ratio": self.risk_reward_ratio,
            "rationale": self.rationale,
            "key_factors": self.key_factors,
            "conflict_flag": self.conflict_flag,
            "conflict_details": self.conflict_details,
            "regime": self.regime,
            "aggregated_at": self.aggregated_at.isoformat(),
        }
    
    def format_for_display(self) -> str:
        """Format for human-readable display."""
        emoji = "🟢" if self.direction == SignalDirection.LONG else "🔴"
        conflict_marker = " ⚠️ CONFLICTED" if self.conflict_flag else ""
        
        lines = [
            f"{emoji} {self.direction.value} {self.instrument}{conflict_marker}",
            f"   Confidence: {self.confidence:.0%} ({self.strength_label})",
            f"   Sources: {', '.join(self.contributing_pods)} ({self.pod_count} pods)",
        ]
        
        if self.entry_price:
            lines.append(f"   Entry: {self.entry_price:.5f}")
        if self.stop_loss:
            lines.append(f"   Stop: {self.stop_loss:.5f}")
        if self.take_profit_1:
            lines.append(f"   Target: {self.take_profit_1:.5f}")
        if self.risk_reward_ratio:
            lines.append(f"   R:R: {self.risk_reward_ratio:.1f}:1")
        
        if self.rationale:
            lines.append(f"   Rationale: {self.rationale[:100]}...")
        
        if self.conflict_flag and self.conflict_details:
            lines.append(f"   ⚠️ Conflict: {self.conflict_details}")
        
        return "\n".join(lines)


class SignalCombiner:
    """
    Combines signals from multiple strategy pods.
    
    Handles:
    - Same-direction signal combination
    - Opposing signal detection
    - Confidence weighting
    - Price level selection
    """
    
    def __init__(
        self,
        ensemble_weights: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.3,
        boost_aligned_signals: bool = True,
        alignment_boost: float = 0.15
    ):
        """
        Initialize signal combiner.
        
        Args:
            ensemble_weights: Weight for each strategy pod
            min_confidence: Minimum confidence to include in output
            boost_aligned_signals: Whether to boost confidence when pods agree
            alignment_boost: How much to boost aligned signals
        """
        self.ensemble_weights = ensemble_weights or {
            "fx_carry_momentum": 0.30,
            "btc_trend_vol": 0.20,
            "commodities_ts": 0.25,
            "cross_asset_risk": 0.15,
            "mean_reversion": 0.10,
        }
        self.min_confidence = min_confidence
        self.boost_aligned_signals = boost_aligned_signals
        self.alignment_boost = alignment_boost
    
    def get_weight(self, pod_name: str) -> float:
        """Get ensemble weight for a strategy pod."""
        # Try exact match first
        if pod_name in self.ensemble_weights:
            return self.ensemble_weights[pod_name]
        
        # Try partial match
        for key, weight in self.ensemble_weights.items():
            if key in pod_name.lower() or pod_name.lower() in key:
                return weight
        
        # Default weight
        return 0.10
    
    def group_signals_by_instrument(
        self,
        signals: List[Signal]
    ) -> Dict[str, List[Signal]]:
        """Group signals by instrument."""
        grouped = defaultdict(list)
        
        for signal in signals:
            if signal.is_active:
                grouped[signal.instrument].append(signal)
        
        return dict(grouped)
    
    def calculate_weighted_confidence(
        self,
        signals: List[Signal]
    ) -> float:
        """
        Calculate ensemble-weighted confidence.
        
        Args:
            signals: List of signals (should be same direction)
        
        Returns:
            Weighted average confidence
        """
        if not signals:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal in signals:
            weight = self.get_weight(signal.strategy_pod)
            weighted_sum += signal.strength * weight
            total_weight += weight
        
        if total_weight == 0:
            return sum(s.strength for s in signals) / len(signals)
        
        base_confidence = weighted_sum / total_weight
        
        # Boost if multiple pods agree
        if self.boost_aligned_signals and len(signals) > 1:
            boost = min(self.alignment_boost * (len(signals) - 1), 0.3)
            base_confidence = min(base_confidence + boost, 1.0)
        
        return base_confidence
    
    def select_price_levels(
        self,
        signals: List[Signal],
        direction: SignalDirection
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Select most conservative price levels from signals.
        
        For LONG: highest entry, lowest stop, lowest target
        For SHORT: lowest entry, highest stop, highest target
        
        This ensures the combined signal is conservative.
        """
        entries = [s.entry_price for s in signals if s.entry_price is not None]
        stops = [s.stop_loss for s in signals if s.stop_loss is not None]
        targets = [s.take_profit_1 for s in signals if s.take_profit_1 is not None]
        
        if not entries:
            return None, None, None
        
        if direction == SignalDirection.LONG:
            # Conservative for long: higher entry, lower stop, lower target
            entry = max(entries)
            stop = min(stops) if stops else None
            target = min(targets) if targets else None
        else:
            # Conservative for short: lower entry, higher stop, higher target
            entry = min(entries)
            stop = max(stops) if stops else None
            target = max(targets) if targets else None
        
        return entry, stop, target
    
    def combine_rationales(self, signals: List[Signal]) -> Tuple[str, List[str]]:
        """Combine rationales from multiple signals."""
        all_factors = []
        
        for signal in signals:
            pod_name = signal.strategy_pod
            
            # Add key factors with pod attribution
            for factor in signal.key_factors:
                all_factors.append(f"[{pod_name}] {factor}")
        
        # Build combined rationale
        pod_names = list(set(s.strategy_pod for s in signals))
        rationale = f"Combined signal from {len(pod_names)} strategy pods: {', '.join(pod_names)}. "
        
        # Add first signal's rationale as base
        if signals and signals[0].rationale:
            rationale += signals[0].rationale
        
        return rationale, all_factors
    
    def combine_same_direction(
        self,
        signals: List[Signal],
        instrument: str,
        direction: SignalDirection
    ) -> AggregatedSignal:
        """
        Combine signals that agree on direction.
        
        Args:
            signals: List of same-direction signals
            instrument: Trading instrument
            direction: Signal direction
        
        Returns:
            Combined AggregatedSignal
        """
        # Calculate weighted confidence
        confidence = self.calculate_weighted_confidence(signals)
        
        # Select price levels
        entry, stop, target = self.select_price_levels(signals, direction)
        
        # Combine rationales
        rationale, key_factors = self.combine_rationales(signals)
        
        # Get contributing pods
        contributing_pods = list(set(s.strategy_pod for s in signals))
        
        # Get regime (use most common)
        regimes = [s.regime for s in signals if s.regime]
        regime = max(set(regimes), key=regimes.count) if regimes else None
        
        # Calculate R:R
        risk_reward = None
        if entry and stop and target:
            risk = abs(entry - stop)
            reward = abs(target - entry)
            if risk > 0:
                risk_reward = reward / risk
        
        return AggregatedSignal(
            instrument=instrument,
            direction=direction,
            confidence=confidence,
            contributing_pods=contributing_pods,
            contributing_signals=signals,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=target,
            rationale=rationale,
            key_factors=key_factors,
            conflict_flag=False,
            regime=regime,
            risk_reward_ratio=risk_reward,
        )
    
    def combine_signals(
        self,
        signals: List[Signal]
    ) -> Tuple[List[AggregatedSignal], List[AggregatedSignal]]:
        """
        Combine all signals into aggregated signals.
        
        Args:
            signals: All signals from all strategy pods
        
        Returns:
            Tuple of (aligned_signals, conflicted_signals)
        """
        aligned = []
        conflicted = []
        
        # Group by instrument
        grouped = self.group_signals_by_instrument(signals)
        
        for instrument, inst_signals in grouped.items():
            # Separate by direction
            long_signals = [s for s in inst_signals if s.direction == SignalDirection.LONG]
            short_signals = [s for s in inst_signals if s.direction == SignalDirection.SHORT]
            
            if long_signals and short_signals:
                # Conflict! Different pods disagree
                long_agg = self.combine_same_direction(long_signals, instrument, SignalDirection.LONG)
                short_agg = self.combine_same_direction(short_signals, instrument, SignalDirection.SHORT)
                
                # Mark as conflicted
                long_agg.conflict_flag = True
                short_agg.conflict_flag = True
                
                conflict_detail = (
                    f"LONG signals ({long_agg.confidence:.0%}) from {long_agg.contributing_pods} "
                    f"vs SHORT signals ({short_agg.confidence:.0%}) from {short_agg.contributing_pods}"
                )
                long_agg.conflict_details = conflict_detail
                short_agg.conflict_details = conflict_detail
                
                conflicted.append(long_agg)
                conflicted.append(short_agg)
                
                logger.warning(f"Conflict detected for {instrument}: {conflict_detail}")
            
            elif long_signals:
                # All agree on LONG
                agg = self.combine_same_direction(long_signals, instrument, SignalDirection.LONG)
                if agg.confidence >= self.min_confidence:
                    aligned.append(agg)
            
            elif short_signals:
                # All agree on SHORT
                agg = self.combine_same_direction(short_signals, instrument, SignalDirection.SHORT)
                if agg.confidence >= self.min_confidence:
                    aligned.append(agg)
        
        # Sort by confidence
        aligned.sort(key=lambda x: x.confidence, reverse=True)
        conflicted.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(
            f"Combined {len(signals)} signals into "
            f"{len(aligned)} aligned + {len(conflicted)} conflicted"
        )
        
        return aligned, conflicted
