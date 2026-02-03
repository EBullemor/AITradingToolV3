"""
Signal Aggregator Module

Combines signals from multiple strategy pods:
- Signal weighting and combination
- Conflict resolution
- Deduplication
"""

from .signal_combiner import (
    SignalCombiner,
    CombinedSignal,
    combine_signals,
)
from .conflict_resolver import (
    ConflictResolver,
    ConflictResolution,
    resolve_conflicts,
)
from .deduplication import (
    SignalDeduplicator,
    deduplicate_signals,
)
from .aggregator import (
    SignalAggregator,
    AggregatorConfig,
    aggregate_signals,
)

__all__ = [
    # Combiner
    "SignalCombiner",
    "CombinedSignal",
    "combine_signals",
    
    # Conflict Resolution
    "ConflictResolver",
    "ConflictResolution",
    "resolve_conflicts",
    
    # Deduplication
    "SignalDeduplicator",
    "deduplicate_signals",
    
    # Main Aggregator
    "SignalAggregator",
    "AggregatorConfig",
    "aggregate_signals",
]
