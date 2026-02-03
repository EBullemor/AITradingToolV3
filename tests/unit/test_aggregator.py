"""
Unit Tests for Signal Aggregator Module

Tests signal combination, conflict resolution, and deduplication.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies.base import Signal, SignalDirection
from src.aggregator import (
    AggregatedSignal,
    SignalCombiner,
    ConflictResolver,
    ConflictResolution,
    SignalDeduplicator,
    SignalAggregator,
    resolve_all_conflicts,
    deduplicate_signals,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_long_signal():
    """Create a sample LONG signal."""
    return Signal(
        instrument="EURUSD",
        direction=SignalDirection.LONG,
        strength=0.7,
        strategy_name="FX Carry + Momentum",
        strategy_pod="fx_carry_momentum",
        generated_at=datetime.now(),
        valid_until=datetime.now() + timedelta(hours=24),
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit_1=1.1100,
        rationale="Strong carry signal",
        key_factors=["Carry score: 2.0", "Low vol regime"],
        regime="LOW_VOL",
    )


@pytest.fixture
def sample_short_signal():
    """Create a sample SHORT signal."""
    return Signal(
        instrument="EURUSD",
        direction=SignalDirection.SHORT,
        strength=0.6,
        strategy_name="Cross Asset Risk",
        strategy_pod="cross_asset_risk",
        generated_at=datetime.now(),
        valid_until=datetime.now() + timedelta(hours=24),
        entry_price=1.1000,
        stop_loss=1.1050,
        take_profit_1=1.0900,
        rationale="Risk-off signal",
        key_factors=["VIX spike", "Dollar strength"],
        regime="HIGH_VOL",
    )


@pytest.fixture
def aligned_signals():
    """Create signals that agree on direction."""
    base_time = datetime.now()
    return [
        Signal(
            instrument="USDJPY",
            direction=SignalDirection.LONG,
            strength=0.65,
            strategy_name="FX Carry",
            strategy_pod="fx_carry_momentum",
            generated_at=base_time,
            valid_until=base_time + timedelta(hours=24),
            entry_price=148.00,
            stop_loss=147.50,
            take_profit_1=149.00,
            rationale="Carry trade",
            key_factors=["Positive carry"],
        ),
        Signal(
            instrument="USDJPY",
            direction=SignalDirection.LONG,
            strength=0.55,
            strategy_name="Momentum",
            strategy_pod="btc_trend_vol",  # Different pod
            generated_at=base_time,
            valid_until=base_time + timedelta(hours=24),
            entry_price=148.10,
            stop_loss=147.40,
            take_profit_1=149.20,
            rationale="Momentum signal",
            key_factors=["Strong trend"],
        ),
    ]


@pytest.fixture
def conflicting_signals():
    """Create signals that conflict on direction."""
    base_time = datetime.now()
    return [
        Signal(
            instrument="GBPUSD",
            direction=SignalDirection.LONG,
            strength=0.60,
            strategy_name="FX Carry",
            strategy_pod="fx_carry_momentum",
            generated_at=base_time,
            valid_until=base_time + timedelta(hours=24),
            entry_price=1.2700,
            stop_loss=1.2650,
            take_profit_1=1.2800,
            rationale="Carry long",
            key_factors=["Positive carry"],
        ),
        Signal(
            instrument="GBPUSD",
            direction=SignalDirection.SHORT,
            strength=0.55,
            strategy_name="Risk",
            strategy_pod="cross_asset_risk",
            generated_at=base_time,
            valid_until=base_time + timedelta(hours=24),
            entry_price=1.2700,
            stop_loss=1.2750,
            take_profit_1=1.2600,
            rationale="Risk-off short",
            key_factors=["Risk-off"],
        ),
    ]


@pytest.fixture
def combiner():
    """Create SignalCombiner instance."""
    return SignalCombiner()


@pytest.fixture
def resolver():
    """Create ConflictResolver instance."""
    return ConflictResolver()


@pytest.fixture
def deduplicator():
    """Create SignalDeduplicator instance."""
    return SignalDeduplicator()


@pytest.fixture
def aggregator():
    """Create SignalAggregator instance."""
    return SignalAggregator()


# =============================================================================
# SignalCombiner Tests
# =============================================================================

class TestSignalCombiner:
    """Tests for SignalCombiner."""
    
    def test_combine_aligned_signals(self, combiner, aligned_signals):
        """Test combining signals that agree."""
        aligned, conflicted = combiner.combine_signals(aligned_signals)
        
        assert len(aligned) == 1
        assert len(conflicted) == 0
        
        agg = aligned[0]
        assert agg.instrument == "USDJPY"
        assert agg.direction == SignalDirection.LONG
        assert len(agg.contributing_pods) == 2
        assert agg.conflict_flag is False
    
    def test_detect_conflicts(self, combiner, conflicting_signals):
        """Test detection of conflicting signals."""
        aligned, conflicted = combiner.combine_signals(conflicting_signals)
        
        assert len(aligned) == 0
        assert len(conflicted) == 2  # One LONG, one SHORT
        
        for sig in conflicted:
            assert sig.conflict_flag is True
    
    def test_weighted_confidence(self, combiner, aligned_signals):
        """Test ensemble-weighted confidence calculation."""
        aligned, _ = combiner.combine_signals(aligned_signals)
        
        agg = aligned[0]
        # Should have boosted confidence due to alignment
        assert agg.confidence > max(s.strength for s in aligned_signals)
    
    def test_conservative_price_levels(self, combiner, aligned_signals):
        """Test that price levels are conservative."""
        aligned, _ = combiner.combine_signals(aligned_signals)
        
        agg = aligned[0]
        
        # For LONG: highest entry (conservative)
        assert agg.entry_price == max(s.entry_price for s in aligned_signals)
        
        # For LONG: lowest stop (conservative)
        assert agg.stop_loss == min(s.stop_loss for s in aligned_signals)


# =============================================================================
# ConflictResolver Tests
# =============================================================================

class TestConflictResolver:
    """Tests for ConflictResolver."""
    
    def test_clear_winner_resolution(self, resolver):
        """Test resolution when one signal is clearly stronger."""
        base_time = datetime.now()
        
        long_agg = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.80,  # Much stronger
            contributing_pods=["fx_carry_momentum"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        short_agg = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.SHORT,
            confidence=0.40,  # Much weaker
            contributing_pods=["cross_asset_risk"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        resolution, winner, reason = resolver.resolve_conflict(long_agg, short_agg)
        
        assert resolution == ConflictResolution.WINNER_SELECTED
        assert winner.direction == SignalDirection.LONG
        assert winner.conflict_flag is False
    
    def test_similar_confidence_flagged(self):
        """Test that similar confidence results in flagged conflict."""
        resolver = ConflictResolver(
            confidence_threshold=0.30,
            allow_conflicted_output=True
        )
        
        base_time = datetime.now()
        
        long_agg = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.55,
            contributing_pods=["fx_carry_momentum"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        short_agg = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.SHORT,
            confidence=0.50,  # Only 5% diff - below threshold
            contributing_pods=["cross_asset_risk"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        resolution, winner, reason = resolver.resolve_conflict(long_agg, short_agg)
        
        assert resolution == ConflictResolution.FLAGGED_CONFLICT
        assert winner is not None
        assert winner.conflict_flag is True
    
    def test_abstain_resolution(self):
        """Test abstain when conflicts not allowed in output."""
        resolver = ConflictResolver(
            confidence_threshold=0.30,
            allow_conflicted_output=False  # Don't output conflicts
        )
        
        base_time = datetime.now()
        
        long_agg = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.55,
            contributing_pods=["fx_carry_momentum"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        short_agg = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.SHORT,
            confidence=0.50,
            contributing_pods=["cross_asset_risk"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        resolution, winner, reason = resolver.resolve_conflict(long_agg, short_agg)
        
        assert resolution == ConflictResolution.ABSTAIN
        assert winner is None
    
    def test_conflict_log(self, resolver):
        """Test that conflicts are logged."""
        base_time = datetime.now()
        
        long_agg = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.70,
            contributing_pods=["fx_carry_momentum"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        short_agg = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.SHORT,
            confidence=0.30,
            contributing_pods=["cross_asset_risk"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        resolver.resolve_conflict(long_agg, short_agg)
        
        assert len(resolver.conflict_log) == 1
        summary = resolver.get_conflict_summary()
        assert summary["total_conflicts"] == 1


# =============================================================================
# Deduplication Tests
# =============================================================================

class TestDeduplication:
    """Tests for SignalDeduplicator."""
    
    def test_detect_duplicate(self, deduplicator):
        """Test duplicate detection within time window."""
        base_time = datetime.now()
        
        signal1 = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.60,
            contributing_pods=["fx_carry_momentum"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        signal2 = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.55,  # Weaker - should be rejected
            contributing_pods=["fx_carry_momentum"],
            contributing_signals=[],
            aggregated_at=base_time + timedelta(hours=1),
        )
        
        # Process first signal
        accepted1, _ = deduplicator.deduplicate([signal1])
        assert len(accepted1) == 1
        
        # Second should be rejected (weaker duplicate)
        accepted2, rejected2 = deduplicator.deduplicate([signal2])
        assert len(accepted2) == 0
        assert len(rejected2) == 1
    
    def test_update_if_stronger(self, deduplicator):
        """Test that stronger signal updates existing."""
        base_time = datetime.now()
        
        signal1 = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.50,
            contributing_pods=["fx_carry_momentum"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        signal2 = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.75,  # Much stronger - should update
            contributing_pods=["fx_carry_momentum"],
            contributing_signals=[],
            aggregated_at=base_time + timedelta(hours=1),
        )
        
        deduplicator.deduplicate([signal1])
        accepted, _ = deduplicator.deduplicate([signal2])
        
        assert len(accepted) == 1
        assert accepted[0].confidence == 0.75
    
    def test_different_directions_not_duplicate(self, deduplicator):
        """Test that different directions are not duplicates."""
        base_time = datetime.now()
        
        long_signal = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.60,
            contributing_pods=["fx_carry_momentum"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        short_signal = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.SHORT,
            confidence=0.55,
            contributing_pods=["cross_asset_risk"],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        
        accepted, rejected = deduplicator.deduplicate([long_signal, short_signal])
        
        # Both should be accepted (different directions)
        assert len(accepted) == 2
        assert len(rejected) == 0


# =============================================================================
# SignalAggregator Integration Tests
# =============================================================================

class TestSignalAggregator:
    """Integration tests for SignalAggregator."""
    
    def test_full_aggregation(self, aggregator, aligned_signals, sample_long_signal):
        """Test full aggregation pipeline."""
        all_signals = aligned_signals + [sample_long_signal]
        
        recommendations = aggregator.aggregate(all_signals)
        
        assert len(recommendations) > 0
        for rec in recommendations:
            assert isinstance(rec, AggregatedSignal)
            assert rec.confidence >= 0.3  # min_confidence
    
    def test_aggregation_with_conflicts(self, aggregator, conflicting_signals):
        """Test aggregation handles conflicts."""
        recommendations = aggregator.aggregate(conflicting_signals)
        
        # Should have resolved the conflict
        assert len(recommendations) <= 1
        
        stats = aggregator.get_stats()
        assert stats["conflicts_detected"] > 0
    
    def test_format_recommendations(self, aggregator, aligned_signals):
        """Test recommendation formatting."""
        recommendations = aggregator.aggregate(aligned_signals)
        
        # Test text format
        text = aggregator.format_recommendations(recommendations, "text")
        assert "USDJPY" in text
        
        # Test markdown format
        markdown = aggregator.format_recommendations(recommendations, "markdown")
        assert "##" in markdown
        
        # Test JSON format
        json_str = aggregator.format_recommendations(recommendations, "json")
        assert "{" in json_str
    
    def test_max_recommendations_limit(self, aggregator):
        """Test that output respects max_recommendations."""
        # Create many signals
        signals = []
        for i in range(20):
            signals.append(Signal(
                instrument=f"PAIR{i}",
                direction=SignalDirection.LONG,
                strength=0.5 + (i * 0.02),
                strategy_name="Test",
                strategy_pod="test_pod",
                generated_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=24),
            ))
        
        recommendations = aggregator.aggregate(signals)
        
        assert len(recommendations) <= aggregator.config["max_recommendations"]


# =============================================================================
# AggregatedSignal Tests
# =============================================================================

class TestAggregatedSignal:
    """Tests for AggregatedSignal dataclass."""
    
    def test_strength_label(self):
        """Test strength label calculation."""
        base_time = datetime.now()
        
        weak = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.25,
            contributing_pods=[],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        assert weak.strength_label == "WEAK"
        
        strong = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.85,
            contributing_pods=[],
            contributing_signals=[],
            aggregated_at=base_time,
        )
        assert strong.strength_label == "VERY_STRONG"
    
    def test_to_dict(self):
        """Test serialization."""
        base_time = datetime.now()
        
        signal = AggregatedSignal(
            instrument="EURUSD",
            direction=SignalDirection.LONG,
            confidence=0.70,
            contributing_pods=["fx_carry_momentum"],
            contributing_signals=[],
            aggregated_at=base_time,
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit_1=1.1100,
        )
        
        d = signal.to_dict()
        
        assert d["instrument"] == "EURUSD"
        assert d["direction"] == "LONG"
        assert d["confidence"] == 0.70
        assert d["strength_label"] == "STRONG"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
