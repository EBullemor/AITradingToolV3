"""
Recommendation Formatter Module

Formats aggregated signals into rich trade cards for various outputs:
- Notion pages with detailed formatting
- Slack messages
- Email digests
- Markdown reports
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.aggregator import AggregatedSignal
from src.strategies.base import SignalDirection


@dataclass
class TradeCard:
    """
    Rich trade card format for recommendations.
    
    Contains all information needed for a complete trade recommendation
    in a presentation-ready format.
    """
    # Core trade info
    instrument: str
    direction: str  # "LONG" or "SHORT"
    direction_emoji: str  # 🟢 or 🔴
    
    # Confidence
    confidence_pct: int  # 0-100
    confidence_label: str  # WEAK/MODERATE/STRONG/VERY_STRONG
    confidence_bar: str  # Visual bar ████░░░░░░
    
    # Price levels
    entry_price: Optional[str] = None
    stop_loss: Optional[str] = None
    take_profit: Optional[str] = None
    risk_reward: Optional[str] = None
    
    # Context
    strategies: List[str] = None
    rationale: str = ""
    key_factors: List[str] = None
    
    # Risk flags
    is_conflicted: bool = False
    conflict_warning: Optional[str] = None
    
    # Metadata
    signal_id: str = ""
    generated_at: str = ""
    regime: Optional[str] = None
    
    def __post_init__(self):
        if self.strategies is None:
            self.strategies = []
        if self.key_factors is None:
            self.key_factors = []


class RecommendationFormatter:
    """
    Formats recommendations for various output channels.
    """
    
    def __init__(self, price_decimals: int = 5):
        """
        Initialize formatter.
        
        Args:
            price_decimals: Decimal places for prices
        """
        self.price_decimals = price_decimals
    
    def _format_price(self, price: Optional[float], instrument: str = "") -> Optional[str]:
        """Format price with appropriate decimals."""
        if price is None:
            return None
        
        # JPY pairs use 3 decimals
        if "JPY" in instrument:
            return f"{price:.3f}"
        
        return f"{price:.{self.price_decimals}f}"
    
    def _get_confidence_bar(self, confidence: float, width: int = 10) -> str:
        """Generate visual confidence bar."""
        filled = int(confidence * width)
        empty = width - filled
        return "█" * filled + "░" * empty
    
    def _get_direction_emoji(self, direction: SignalDirection) -> str:
        """Get emoji for direction."""
        if direction == SignalDirection.LONG:
            return "🟢"
        elif direction == SignalDirection.SHORT:
            return "🔴"
        return "⚪"
    
    def format_trade_card(self, signal: AggregatedSignal) -> TradeCard:
        """
        Convert AggregatedSignal to TradeCard.
        
        Args:
            signal: Aggregated signal
        
        Returns:
            Formatted TradeCard
        """
        return TradeCard(
            instrument=signal.instrument,
            direction=signal.direction.value,
            direction_emoji=self._get_direction_emoji(signal.direction),
            confidence_pct=int(signal.confidence * 100),
            confidence_label=signal.strength_label,
            confidence_bar=self._get_confidence_bar(signal.confidence),
            entry_price=self._format_price(signal.entry_price, signal.instrument),
            stop_loss=self._format_price(signal.stop_loss, signal.instrument),
            take_profit=self._format_price(signal.take_profit_1, signal.instrument),
            risk_reward=f"{signal.risk_reward_ratio:.1f}:1" if signal.risk_reward_ratio else None,
            strategies=signal.contributing_pods,
            rationale=signal.rationale or "",
            key_factors=signal.key_factors,
            is_conflicted=signal.conflict_flag,
            conflict_warning=signal.conflict_details,
            signal_id=signal.signal_id,
            generated_at=signal.aggregated_at.strftime("%Y-%m-%d %H:%M"),
            regime=signal.regime,
        )
    
    def format_markdown(self, signal: AggregatedSignal) -> str:
        """
        Format signal as markdown.
        
        Args:
            signal: Aggregated signal
        
        Returns:
            Markdown string
        """
        card = self.format_trade_card(signal)
        
        conflict_marker = " ⚠️ CONFLICTED" if card.is_conflicted else ""
        
        lines = [
            f"## {card.direction_emoji} {card.direction} {card.instrument}{conflict_marker}",
            "",
            f"**Confidence:** {card.confidence_pct}% ({card.confidence_label})",
            f"```",
            f"{card.confidence_bar}",
            f"```",
            "",
        ]
        
        # Price levels table
        if card.entry_price:
            lines.extend([
                "| Level | Price |",
                "|-------|-------|",
            ])
            lines.append(f"| Entry | {card.entry_price} |")
            if card.stop_loss:
                lines.append(f"| Stop Loss | {card.stop_loss} |")
            if card.take_profit:
                lines.append(f"| Target | {card.take_profit} |")
            if card.risk_reward:
                lines.append(f"| R:R | {card.risk_reward} |")
            lines.append("")
        
        # Strategy sources
        if card.strategies:
            lines.append(f"**Sources:** {', '.join(card.strategies)}")
            lines.append("")
        
        # Rationale
        if card.rationale:
            lines.append(f"**Rationale:** {card.rationale}")
            lines.append("")
        
        # Key factors
        if card.key_factors:
            lines.append("**Key Factors:**")
            for factor in card.key_factors[:5]:
                lines.append(f"- {factor}")
            lines.append("")
        
        # Conflict warning
        if card.is_conflicted and card.conflict_warning:
            lines.extend([
                "> ⚠️ **Conflict Warning**",
                f"> {card.conflict_warning}",
                "",
            ])
        
        # Metadata
        lines.extend([
            "---",
            f"*Signal ID: {card.signal_id} | Generated: {card.generated_at}*",
        ])
        
        return "\n".join(lines)
    
    def format_slack(self, signal: AggregatedSignal) -> Dict:
        """
        Format signal for Slack Block Kit.
        
        Args:
            signal: Aggregated signal
        
        Returns:
            Slack blocks dictionary
        """
        card = self.format_trade_card(signal)
        
        conflict_marker = " ⚠️" if card.is_conflicted else ""
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{card.direction_emoji} {card.direction} {card.instrument}{conflict_marker}",
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Confidence:* {card.confidence_pct}% ({card.confidence_label})"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Sources:* {', '.join(card.strategies)}"
                    }
                ]
            },
        ]
        
        # Price levels
        if card.entry_price:
            fields = [{"type": "mrkdwn", "text": f"*Entry:* {card.entry_price}"}]
            if card.stop_loss:
                fields.append({"type": "mrkdwn", "text": f"*Stop:* {card.stop_loss}"})
            if card.take_profit:
                fields.append({"type": "mrkdwn", "text": f"*Target:* {card.take_profit}"})
            if card.risk_reward:
                fields.append({"type": "mrkdwn", "text": f"*R:R:* {card.risk_reward}"})
            
            blocks.append({
                "type": "section",
                "fields": fields[:4]  # Slack limit
            })
        
        # Rationale
        if card.rationale:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"_{card.rationale[:500]}_"
                }
            })
        
        # Conflict warning
        if card.is_conflicted:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"⚠️ *Conflict:* {card.conflict_warning or 'Opposing signals detected'}"
                    }
                ]
            })
        
        blocks.append({"type": "divider"})
        
        return {"blocks": blocks}
    
    def format_text(self, signal: AggregatedSignal) -> str:
        """
        Format signal as plain text.
        
        Args:
            signal: Aggregated signal
        
        Returns:
            Plain text string
        """
        card = self.format_trade_card(signal)
        
        lines = [
            f"{card.direction_emoji} {card.direction} {card.instrument}",
            f"   Confidence: {card.confidence_pct}% ({card.confidence_label})",
            f"   {card.confidence_bar}",
        ]
        
        if card.entry_price:
            lines.append(f"   Entry: {card.entry_price}")
        if card.stop_loss:
            lines.append(f"   Stop: {card.stop_loss}")
        if card.take_profit:
            lines.append(f"   Target: {card.take_profit}")
        if card.risk_reward:
            lines.append(f"   R:R: {card.risk_reward}")
        
        if card.strategies:
            lines.append(f"   Sources: {', '.join(card.strategies)}")
        
        if card.rationale:
            lines.append(f"   Rationale: {card.rationale[:100]}...")
        
        if card.is_conflicted:
            lines.append(f"   ⚠️ CONFLICTED")
        
        return "\n".join(lines)
    
    def format_notion_content(self, signal: AggregatedSignal) -> str:
        """
        Format signal as Notion page content (markdown).
        
        This is for the page body, not properties.
        
        Args:
            signal: Aggregated signal
        
        Returns:
            Notion-compatible markdown
        """
        card = self.format_trade_card(signal)
        
        lines = []
        
        # Confidence visualization
        lines.extend([
            "## Confidence",
            f"**{card.confidence_pct}%** ({card.confidence_label})",
            "",
            f"`{card.confidence_bar}`",
            "",
        ])
        
        # Price levels
        if card.entry_price or card.stop_loss or card.take_profit:
            lines.extend([
                "## Price Levels",
                "",
            ])
            if card.entry_price:
                lines.append(f"- **Entry:** {card.entry_price}")
            if card.stop_loss:
                lines.append(f"- **Stop Loss:** {card.stop_loss}")
            if card.take_profit:
                lines.append(f"- **Target:** {card.take_profit}")
            if card.risk_reward:
                lines.append(f"- **Risk:Reward:** {card.risk_reward}")
            lines.append("")
        
        # Analysis
        if card.rationale or card.key_factors:
            lines.extend([
                "## Analysis",
                "",
            ])
            if card.rationale:
                lines.append(card.rationale)
                lines.append("")
            
            if card.key_factors:
                lines.append("**Key Factors:**")
                for factor in card.key_factors:
                    lines.append(f"- {factor}")
                lines.append("")
        
        # Conflict warning
        if card.is_conflicted:
            lines.extend([
                "## ⚠️ Conflict Warning",
                "",
                f"> {card.conflict_warning or 'This signal has conflicting inputs from different strategies.'}",
                "",
            ])
        
        # Metadata
        lines.extend([
            "---",
            f"*Generated: {card.generated_at}*",
            f"*Signal ID: {card.signal_id}*",
        ])
        
        if card.regime:
            lines.append(f"*Regime: {card.regime}*")
        
        return "\n".join(lines)


def format_recommendations_report(
    recommendations: List[AggregatedSignal],
    title: str = "Trading Recommendations",
    include_summary: bool = True
) -> str:
    """
    Format multiple recommendations as a full report.
    
    Args:
        recommendations: List of recommendations
        title: Report title
        include_summary: Whether to include summary section
    
    Returns:
        Markdown report
    """
    formatter = RecommendationFormatter()
    
    lines = [
        f"# {title}",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]
    
    if include_summary:
        long_count = sum(1 for r in recommendations if r.direction == SignalDirection.LONG)
        short_count = sum(1 for r in recommendations if r.direction == SignalDirection.SHORT)
        conflicted_count = sum(1 for r in recommendations if r.conflict_flag)
        
        lines.extend([
            "## Summary",
            "",
            f"- **Total Recommendations:** {len(recommendations)}",
            f"- **Long:** {long_count}",
            f"- **Short:** {short_count}",
        ])
        
        if conflicted_count:
            lines.append(f"- **⚠️ Conflicted:** {conflicted_count}")
        
        lines.extend(["", "---", ""])
    
    # Individual recommendations
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"# {i}. Recommendation")
        lines.append("")
        lines.append(formatter.format_markdown(rec))
        lines.append("")
    
    return "\n".join(lines)
