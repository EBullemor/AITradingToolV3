"""
Signal Deduplication Module

Ensures only one recommendation per instrument within a time window.
Updates existing recommendations if newer signals are stronger.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.strategies.base import Signal, SignalDirection
from .signal_combiner import AggregatedSignal


class SignalDeduplicator:
    """
    Handles deduplication of signals.
    
    Rules:
    1. Same instrument + same direction within time window = duplicate
    2. If new signal stronger: Update existing
    3. If existing stronger: Keep existing, discard new
    4. Different directions = both valid (conflict handled elsewhere)
    """
    
    def __init__(
        self,
        time_window_hours: int = 24,
        update_if_stronger: bool = True,
        strength_threshold: float = 0.1  # Min improvement to update
    ):
        """
        Initialize deduplicator.
        
        Args:
            time_window_hours: Window for duplicate detection
            update_if_stronger: Whether to update if new signal is stronger
            strength_threshold: Min strength improvement to trigger update
        """
        self.time_window = timedelta(hours=time_window_hours)
        self.update_if_stronger = update_if_stronger
        self.strength_threshold = strength_threshold
        
        # Track existing signals
        self.active_signals: Dict[str, AggregatedSignal] = {}
    
    def _make_key(self, instrument: str, direction: SignalDirection) -> str:
        """Create unique key for signal."""
        return f"{instrument}_{direction.value}"
    
    def is_duplicate(
        self,
        signal: AggregatedSignal,
        existing_signals: Optional[List[AggregatedSignal]] = None
    ) -> Tuple[bool, Optional[AggregatedSignal]]:
        """
        Check if signal is a duplicate of an existing one.
        
        Args:
            signal: Signal to check
            existing_signals: List of existing signals (uses internal cache if None)
        
        Returns:
            Tuple of (is_duplicate, existing_signal or None)
        """
        key = self._make_key(signal.instrument, signal.direction)
        
        # Check internal cache
        if key in self.active_signals:
            existing = self.active_signals[key]
            
            # Check if within time window
            time_diff = signal.aggregated_at - existing.aggregated_at
            if abs(time_diff) <= self.time_window:
                return True, existing
        
        # Check provided existing signals
        if existing_signals:
            for existing in existing_signals:
                if (existing.instrument == signal.instrument and
                    existing.direction == signal.direction):
                    
                    time_diff = signal.aggregated_at - existing.aggregated_at
                    if abs(time_diff) <= self.time_window:
                        return True, existing
        
        return False, None
    
    def should_update(
        self,
        new_signal: AggregatedSignal,
        existing_signal: AggregatedSignal
    ) -> bool:
        """
        Determine if existing signal should be updated.
        
        Args:
            new_signal: Incoming signal
            existing_signal: Current signal
        
        Returns:
            True if should update
        """
        if not self.update_if_stronger:
            return False
        
        strength_improvement = new_signal.confidence - existing_signal.confidence
        
        return strength_improvement >= self.strength_threshold
    
    def deduplicate(
        self,
        signals: List[AggregatedSignal],
        existing_signals: Optional[List[AggregatedSignal]] = None
    ) -> Tuple[List[AggregatedSignal], List[AggregatedSignal]]:
        """
        Deduplicate a list of signals.
        
        Args:
            signals: New signals to process
            existing_signals: Existing active signals
        
        Returns:
            Tuple of (accepted_signals, rejected_signals)
        """
        accepted = []
        rejected = []
        
        # Build lookup from existing signals
        existing_lookup: Dict[str, AggregatedSignal] = {}
        if existing_signals:
            for sig in existing_signals:
                key = self._make_key(sig.instrument, sig.direction)
                existing_lookup[key] = sig
        
        # Also include our internal cache
        existing_lookup.update(self.active_signals)
        
        for signal in signals:
            key = self._make_key(signal.instrument, signal.direction)
            
            if key in existing_lookup:
                existing = existing_lookup[key]
                
                # Check time window
                time_diff = abs((signal.aggregated_at - existing.aggregated_at).total_seconds())
                
                if time_diff <= self.time_window.total_seconds():
                    # Within window - check if should update
                    if self.should_update(signal, existing):
                        logger.info(
                            f"Updating {signal.instrument} {signal.direction.value}: "
                            f"{existing.confidence:.0%} -> {signal.confidence:.0%}"
                        )
                        accepted.append(signal)
                        self.active_signals[key] = signal
                    else:
                        logger.debug(
                            f"Rejecting duplicate {signal.instrument} {signal.direction.value}: "
                            f"existing {existing.confidence:.0%} >= new {signal.confidence:.0%}"
                        )
                        rejected.append(signal)
                else:
                    # Outside window - accept as new
                    accepted.append(signal)
                    self.active_signals[key] = signal
            else:
                # No existing signal
                accepted.append(signal)
                self.active_signals[key] = signal
        
        logger.info(
            f"Deduplication: {len(accepted)} accepted, {len(rejected)} rejected "
            f"from {len(signals)} signals"
        )
        
        return accepted, rejected
    
    def cleanup_expired(self, as_of: Optional[datetime] = None):
        """Remove signals outside the time window."""
        if as_of is None:
            as_of = datetime.now()
        
        expired_keys = []
        for key, signal in self.active_signals.items():
            age = as_of - signal.aggregated_at
            if age > self.time_window:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.active_signals[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired signals")
    
    def get_active_signals(self) -> List[AggregatedSignal]:
        """Get all active signals."""
        self.cleanup_expired()
        return list(self.active_signals.values())
    
    def clear(self):
        """Clear all cached signals."""
        self.active_signals = {}


def deduplicate_signals(
    new_signals: List[AggregatedSignal],
    existing_signals: Optional[List[AggregatedSignal]] = None,
    time_window_hours: int = 24,
    update_if_stronger: bool = True
) -> List[AggregatedSignal]:
    """
    Convenience function to deduplicate signals.
    
    Args:
        new_signals: Signals to deduplicate
        existing_signals: Existing active signals
        time_window_hours: Deduplication window
        update_if_stronger: Whether to update on stronger signals
    
    Returns:
        List of accepted (non-duplicate) signals
    """
    deduplicator = SignalDeduplicator(
        time_window_hours=time_window_hours,
        update_if_stronger=update_if_stronger
    )
    
    accepted, _ = deduplicator.deduplicate(new_signals, existing_signals)
    
    return accepted
