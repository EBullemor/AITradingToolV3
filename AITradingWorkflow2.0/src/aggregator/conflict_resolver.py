"""
Conflict Resolver Module

Handles opposing signals on the same instrument from different strategy pods.
Determines whether to pick a winner, flag as conflicted, or abstain.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

from loguru import logger

from src.strategies.base import Signal, SignalDirection
from .signal_combiner import AggregatedSignal


class ConflictResolution(Enum):
    """How a conflict was resolved."""
    WINNER_SELECTED = "winner_selected"      # One side clearly stronger
    FLAGGED_CONFLICT = "flagged_conflict"    # Similar strength, flag for user
    ABSTAIN = "abstain"                       # Don't output anything
    MERGED = "merged"                         # Signals somehow merged


@dataclass
class ConflictRecord:
    """Record of a conflict and its resolution."""
    instrument: str
    timestamp: datetime
    
    # The conflicting signals
    long_signal: AggregatedSignal
    short_signal: AggregatedSignal
    
    # Resolution
    resolution: ConflictResolution
    winning_signal: Optional[AggregatedSignal]
    confidence_diff: float
    
    # Explanation
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            "instrument": self.instrument,
            "timestamp": self.timestamp.isoformat(),
            "long_confidence": self.long_signal.confidence,
            "long_pods": self.long_signal.contributing_pods,
            "short_confidence": self.short_signal.confidence,
            "short_pods": self.short_signal.contributing_pods,
            "resolution": self.resolution.value,
            "winning_direction": self.winning_signal.direction.value if self.winning_signal else None,
            "confidence_diff": self.confidence_diff,
            "reason": self.reason,
        }


class ConflictResolver:
    """
    Resolves conflicts between opposing signals.
    
    Resolution strategies:
    1. If confidence difference > threshold: Pick the stronger signal
    2. If confidence similar but regime favors one: Use regime
    3. If still unclear: Flag as conflicted or abstain
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.30,
        allow_conflicted_output: bool = True,
        use_regime_tiebreaker: bool = True,
        max_signals_per_instrument: int = 1
    ):
        """
        Initialize conflict resolver.
        
        Args:
            confidence_threshold: Min confidence diff to declare winner
            allow_conflicted_output: Whether to output flagged conflicts
            use_regime_tiebreaker: Whether to use regime to break ties
            max_signals_per_instrument: Max outputs per instrument
        """
        self.confidence_threshold = confidence_threshold
        self.allow_conflicted_output = allow_conflicted_output
        self.use_regime_tiebreaker = use_regime_tiebreaker
        self.max_signals_per_instrument = max_signals_per_instrument
        
        # Track conflict history
        self.conflict_log: List[ConflictRecord] = []
    
    def resolve_conflict(
        self,
        long_signal: AggregatedSignal,
        short_signal: AggregatedSignal
    ) -> Tuple[ConflictResolution, Optional[AggregatedSignal], str]:
        """
        Resolve a conflict between long and short signals.
        
        Args:
            long_signal: Aggregated LONG signal
            short_signal: Aggregated SHORT signal
        
        Returns:
            Tuple of (resolution_type, winning_signal or None, reason)
        """
        instrument = long_signal.instrument
        confidence_diff = abs(long_signal.confidence - short_signal.confidence)
        
        logger.info(
            f"Resolving conflict for {instrument}: "
            f"LONG {long_signal.confidence:.0%} vs SHORT {short_signal.confidence:.0%}"
        )
        
        # Strategy 1: Clear winner by confidence
        if confidence_diff >= self.confidence_threshold:
            if long_signal.confidence > short_signal.confidence:
                winner = long_signal
                reason = (
                    f"LONG signal ({long_signal.confidence:.0%}) significantly stronger than "
                    f"SHORT ({short_signal.confidence:.0%}), diff: {confidence_diff:.0%}"
                )
            else:
                winner = short_signal
                reason = (
                    f"SHORT signal ({short_signal.confidence:.0%}) significantly stronger than "
                    f"LONG ({long_signal.confidence:.0%}), diff: {confidence_diff:.0%}"
                )
            
            # Clear the conflict flag on winner
            winner.conflict_flag = False
            winner.conflict_details = f"Won conflict: {reason}"
            
            self._log_conflict(
                long_signal, short_signal,
                ConflictResolution.WINNER_SELECTED,
                winner, confidence_diff, reason
            )
            
            return ConflictResolution.WINNER_SELECTED, winner, reason
        
        # Strategy 2: Use regime as tiebreaker
        if self.use_regime_tiebreaker:
            regime_winner = self._regime_tiebreaker(long_signal, short_signal)
            if regime_winner:
                reason = f"Regime-based tiebreaker: {regime_winner.regime} favors {regime_winner.direction.value}"
                regime_winner.conflict_flag = False
                regime_winner.conflict_details = f"Won by regime: {reason}"
                
                self._log_conflict(
                    long_signal, short_signal,
                    ConflictResolution.WINNER_SELECTED,
                    regime_winner, confidence_diff, reason
                )
                
                return ConflictResolution.WINNER_SELECTED, regime_winner, reason
        
        # Strategy 3: Flag or abstain
        if self.allow_conflicted_output:
            # Return the stronger one but flagged
            if long_signal.confidence >= short_signal.confidence:
                winner = long_signal
            else:
                winner = short_signal
            
            winner.conflict_flag = True
            reason = (
                f"Conflicting signals with similar confidence "
                f"(diff: {confidence_diff:.0%}). Outputting stronger signal with warning."
            )
            
            self._log_conflict(
                long_signal, short_signal,
                ConflictResolution.FLAGGED_CONFLICT,
                winner, confidence_diff, reason
            )
            
            return ConflictResolution.FLAGGED_CONFLICT, winner, reason
        
        else:
            # Abstain - don't output anything
            reason = (
                f"Conflicting signals with similar confidence "
                f"(diff: {confidence_diff:.0%}). Abstaining from recommendation."
            )
            
            self._log_conflict(
                long_signal, short_signal,
                ConflictResolution.ABSTAIN,
                None, confidence_diff, reason
            )
            
            return ConflictResolution.ABSTAIN, None, reason
    
    def _regime_tiebreaker(
        self,
        long_signal: AggregatedSignal,
        short_signal: AggregatedSignal
    ) -> Optional[AggregatedSignal]:
        """
        Use market regime to break ties.
        
        LOW_VOL regime: Favors carry (typically long high-yielders)
        HIGH_VOL regime: Favors momentum/trend
        RISK_OFF: Favors safe havens (JPY, CHF, USD)
        """
        regime = long_signal.regime or short_signal.regime
        
        if not regime:
            return None
        
        instrument = long_signal.instrument
        
        # Regime-based preferences
        if regime == "HIGH_VOL" or regime == "RISK_OFF":
            # In high vol/risk-off, favor safe haven currencies
            safe_havens = ["JPY", "CHF", "USD"]
            
            # If instrument is USD/XXX, favor long USD (short pair like EURUSD)
            if any(sh in instrument for sh in safe_havens):
                if instrument.startswith("USD"):
                    # USDJPY - long USD = long the pair
                    return long_signal
                else:
                    # EURUSD - long USD = short the pair
                    return short_signal
        
        elif regime == "LOW_VOL" or regime == "RISK_ON":
            # In low vol, favor risk currencies
            risk_currencies = ["AUD", "NZD"]
            
            if any(rc in instrument for rc in risk_currencies):
                # Long risk currencies
                return long_signal
        
        return None
    
    def _log_conflict(
        self,
        long_signal: AggregatedSignal,
        short_signal: AggregatedSignal,
        resolution: ConflictResolution,
        winner: Optional[AggregatedSignal],
        confidence_diff: float,
        reason: str
    ):
        """Log a conflict for analysis."""
        record = ConflictRecord(
            instrument=long_signal.instrument,
            timestamp=datetime.now(),
            long_signal=long_signal,
            short_signal=short_signal,
            resolution=resolution,
            winning_signal=winner,
            confidence_diff=confidence_diff,
            reason=reason,
        )
        
        self.conflict_log.append(record)
        
        logger.info(
            f"Conflict resolved for {long_signal.instrument}: "
            f"{resolution.value} - {reason[:50]}..."
        )
    
    def get_conflict_summary(self) -> Dict:
        """Get summary of all conflicts."""
        if not self.conflict_log:
            return {"total_conflicts": 0}
        
        resolutions = {}
        for record in self.conflict_log:
            res = record.resolution.value
            resolutions[res] = resolutions.get(res, 0) + 1
        
        return {
            "total_conflicts": len(self.conflict_log),
            "resolutions": resolutions,
            "instruments_affected": list(set(r.instrument for r in self.conflict_log)),
            "avg_confidence_diff": sum(r.confidence_diff for r in self.conflict_log) / len(self.conflict_log),
        }
    
    def clear_log(self):
        """Clear conflict log."""
        self.conflict_log = []


def resolve_all_conflicts(
    aligned_signals: List[AggregatedSignal],
    conflicted_signals: List[AggregatedSignal],
    resolver: Optional[ConflictResolver] = None
) -> List[AggregatedSignal]:
    """
    Resolve all conflicts and return final signal list.
    
    Args:
        aligned_signals: Signals with no conflicts
        conflicted_signals: Signals marked as conflicted
        resolver: Conflict resolver instance
    
    Returns:
        Final list of signals (aligned + resolved conflicts)
    """
    if resolver is None:
        resolver = ConflictResolver()
    
    final_signals = list(aligned_signals)
    
    # Group conflicted by instrument
    conflicts_by_instrument: Dict[str, List[AggregatedSignal]] = {}
    for signal in conflicted_signals:
        inst = signal.instrument
        if inst not in conflicts_by_instrument:
            conflicts_by_instrument[inst] = []
        conflicts_by_instrument[inst].append(signal)
    
    # Resolve each conflict
    for instrument, signals in conflicts_by_instrument.items():
        long_signals = [s for s in signals if s.direction == SignalDirection.LONG]
        short_signals = [s for s in signals if s.direction == SignalDirection.SHORT]
        
        if long_signals and short_signals:
            # Take the first of each (should only be one)
            long_agg = long_signals[0]
            short_agg = short_signals[0]
            
            resolution, winner, reason = resolver.resolve_conflict(long_agg, short_agg)
            
            if winner:
                final_signals.append(winner)
    
    # Sort by confidence
    final_signals.sort(key=lambda x: x.confidence, reverse=True)
    
    return final_signals
