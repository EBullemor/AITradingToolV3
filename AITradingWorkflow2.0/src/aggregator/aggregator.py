"""
Signal Aggregator

Main orchestrator that combines signals from all strategy pods into
final recommendations.

Pipeline:
1. Collect signals from all strategy pods
2. Combine same-direction signals (weighted confidence)
3. Detect and resolve conflicts (opposing signals)
4. Deduplicate against existing recommendations
5. Output final recommendation set
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from loguru import logger

from src.strategies.base import Signal, SignalDirection
from .signal_combiner import SignalCombiner, AggregatedSignal
from .conflict_resolver import ConflictResolver, resolve_all_conflicts
from .deduplication import SignalDeduplicator, deduplicate_signals


class SignalAggregator:
    """
    Main signal aggregation orchestrator.
    
    Takes signals from multiple strategy pods and produces a coherent
    set of recommendations.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize signal aggregator.
        
        Args:
            config: Configuration dictionary
            config_path: Path to config file (alternative to config dict)
        """
        # Load config
        if config is None and config_path:
            config = self._load_config(config_path)
        elif config is None:
            config = self._default_config()
        
        self.config = config
        
        # Initialize components
        self.combiner = SignalCombiner(
            ensemble_weights=config.get("ensemble_weights", {}),
            min_confidence=config.get("min_confidence", 0.3),
            boost_aligned_signals=config.get("boost_aligned_signals", True),
        )
        
        conflict_config = config.get("conflict_resolution", {})
        self.resolver = ConflictResolver(
            confidence_threshold=conflict_config.get("confidence_threshold", 0.30),
            allow_conflicted_output=conflict_config.get("allow_conflicted_output", True),
            use_regime_tiebreaker=conflict_config.get("use_regime_tiebreaker", True),
        )
        
        dedup_config = config.get("deduplication", {})
        self.deduplicator = SignalDeduplicator(
            time_window_hours=dedup_config.get("time_window_hours", 24),
            update_if_stronger=dedup_config.get("update_if_stronger", True),
        )
        
        # Track statistics
        self.stats = {
            "total_signals_processed": 0,
            "aligned_signals": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "duplicates_rejected": 0,
            "final_recommendations": 0,
        }
        
        logger.info("Signal aggregator initialized")
    
    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file."""
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return self._default_config()
        
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "ensemble_weights": {
                "fx_carry_momentum": 0.30,
                "btc_trend_vol": 0.20,
                "commodities_ts": 0.25,
                "cross_asset_risk": 0.15,
                "mean_reversion": 0.10,
            },
            "min_confidence": 0.30,
            "boost_aligned_signals": True,
            "max_recommendations": 10,
            "conflict_resolution": {
                "confidence_threshold": 0.30,
                "allow_conflicted_output": True,
                "use_regime_tiebreaker": True,
            },
            "deduplication": {
                "time_window_hours": 24,
                "update_if_stronger": True,
            },
        }
    
    def aggregate(
        self,
        signals: List[Signal],
        existing_recommendations: Optional[List[AggregatedSignal]] = None,
        as_of_date: Optional[datetime] = None
    ) -> List[AggregatedSignal]:
        """
        Aggregate signals into final recommendations.
        
        Args:
            signals: All signals from all strategy pods
            existing_recommendations: Current active recommendations
            as_of_date: Reference timestamp
        
        Returns:
            List of final AggregatedSignal recommendations
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        logger.info(f"Aggregating {len(signals)} signals from strategy pods")
        
        self.stats["total_signals_processed"] += len(signals)
        
        # Step 1: Filter to active signals only
        active_signals = [s for s in signals if s.is_active]
        logger.debug(f"Active signals: {len(active_signals)}/{len(signals)}")
        
        if not active_signals:
            logger.warning("No active signals to aggregate")
            return []
        
        # Step 2: Combine signals (groups by instrument, combines same-direction)
        aligned, conflicted = self.combiner.combine_signals(active_signals)
        
        self.stats["aligned_signals"] += len(aligned)
        self.stats["conflicts_detected"] += len(conflicted) // 2  # Each conflict has 2 signals
        
        logger.info(f"Combined: {len(aligned)} aligned, {len(conflicted)} conflicted")
        
        # Step 3: Resolve conflicts
        resolved = resolve_all_conflicts(aligned, conflicted, self.resolver)
        
        self.stats["conflicts_resolved"] = len(self.resolver.conflict_log)
        
        # Step 4: Deduplicate against existing
        if existing_recommendations:
            accepted, rejected = self.deduplicator.deduplicate(
                resolved, existing_recommendations
            )
            self.stats["duplicates_rejected"] += len(rejected)
        else:
            accepted = resolved
        
        # Step 5: Limit output
        max_recs = self.config.get("max_recommendations", 10)
        final = accepted[:max_recs]
        
        self.stats["final_recommendations"] = len(final)
        
        logger.info(
            f"Aggregation complete: {len(final)} recommendations "
            f"(from {len(signals)} input signals)"
        )
        
        return final
    
    def aggregate_from_pods(
        self,
        pod_signals: Dict[str, List[Signal]],
        existing_recommendations: Optional[List[AggregatedSignal]] = None,
        as_of_date: Optional[datetime] = None
    ) -> List[AggregatedSignal]:
        """
        Aggregate signals organized by pod.
        
        Args:
            pod_signals: Dict mapping pod name to signals
            existing_recommendations: Current active recommendations
            as_of_date: Reference timestamp
        
        Returns:
            List of final recommendations
        """
        # Flatten all signals
        all_signals = []
        for pod_name, signals in pod_signals.items():
            logger.debug(f"Pod {pod_name}: {len(signals)} signals")
            all_signals.extend(signals)
        
        return self.aggregate(all_signals, existing_recommendations, as_of_date)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return {
            **self.stats,
            "conflict_summary": self.resolver.get_conflict_summary(),
            "active_signals": len(self.deduplicator.get_active_signals()),
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "total_signals_processed": 0,
            "aligned_signals": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "duplicates_rejected": 0,
            "final_recommendations": 0,
        }
        self.resolver.clear_log()
    
    def format_recommendations(
        self,
        recommendations: List[AggregatedSignal],
        format_type: str = "text"
    ) -> str:
        """
        Format recommendations for output.
        
        Args:
            recommendations: List of recommendations
            format_type: 'text', 'json', or 'markdown'
        
        Returns:
            Formatted string
        """
        if format_type == "json":
            import json
            return json.dumps(
                [r.to_dict() for r in recommendations],
                indent=2,
                default=str
            )
        
        elif format_type == "markdown":
            lines = ["# Trading Recommendations", ""]
            for i, rec in enumerate(recommendations, 1):
                emoji = "🟢" if rec.direction == SignalDirection.LONG else "🔴"
                conflict = " ⚠️" if rec.conflict_flag else ""
                
                lines.extend([
                    f"## {i}. {emoji} {rec.direction.value} {rec.instrument}{conflict}",
                    "",
                    f"**Confidence:** {rec.confidence:.0%} ({rec.strength_label})",
                    f"**Sources:** {', '.join(rec.contributing_pods)}",
                    "",
                ])
                
                if rec.entry_price:
                    lines.append(f"- Entry: {rec.entry_price:.5f}")
                if rec.stop_loss:
                    lines.append(f"- Stop Loss: {rec.stop_loss:.5f}")
                if rec.take_profit_1:
                    lines.append(f"- Target: {rec.take_profit_1:.5f}")
                if rec.risk_reward_ratio:
                    lines.append(f"- Risk:Reward: {rec.risk_reward_ratio:.1f}:1")
                
                if rec.rationale:
                    lines.extend(["", f"**Rationale:** {rec.rationale}", ""])
                
                lines.append("---")
                lines.append("")
            
            return "\n".join(lines)
        
        else:  # text
            lines = ["=" * 60, "TRADING RECOMMENDATIONS", "=" * 60, ""]
            
            for rec in recommendations:
                lines.append(rec.format_for_display())
                lines.append("")
            
            return "\n".join(lines)


def create_aggregator(config_path: Optional[str] = None) -> SignalAggregator:
    """
    Factory function to create aggregator.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configured SignalAggregator instance
    """
    path = Path(config_path) if config_path else None
    return SignalAggregator(config_path=path)
