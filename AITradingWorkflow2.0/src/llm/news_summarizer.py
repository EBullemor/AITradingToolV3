"""
News Summarizer Module

Summarizes financial news articles using Claude, extracting key events,
market catalysts, and providing source attribution.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .client import LLMClient, MockLLMClient, create_llm_client


@dataclass
class Article:
    """Input article for summarization."""
    id: str
    title: str
    text: str
    source: str
    published_at: Optional[datetime] = None
    url: Optional[str] = None
    
    def to_formatted_string(self, index: int) -> str:
        """Format article for prompt."""
        text_truncated = self.text[:2000] + "..." if len(self.text) > 2000 else self.text
        return f"""[{index}] {self.title}
Source: {self.source}
{text_truncated}"""


@dataclass
class NewsSummary:
    """Structured news summary output."""
    summary_text: str
    source_refs: List[str]  # Article IDs that were referenced
    key_events: List[str]   # Major events identified
    catalysts: List[str]    # Market catalysts
    risks: List[str]        # Risk factors mentioned
    confidence: str         # HIGH/MEDIUM/LOW
    generated_at: datetime = field(default_factory=datetime.now)
    
    # Metadata
    articles_processed: int = 0
    model_used: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "summary_text": self.summary_text,
            "source_refs": self.source_refs,
            "key_events": self.key_events,
            "catalysts": self.catalysts,
            "risks": self.risks,
            "confidence": self.confidence,
            "generated_at": self.generated_at.isoformat(),
            "articles_processed": self.articles_processed,
            "model_used": self.model_used,
        }
    
    def format_for_display(self) -> str:
        """Format for human-readable display."""
        lines = [
            "📰 NEWS SUMMARY",
            "=" * 50,
            "",
            self.summary_text,
            "",
        ]
        
        if self.key_events:
            lines.append("📌 KEY EVENTS:")
            for event in self.key_events:
                lines.append(f"  • {event}")
            lines.append("")
        
        if self.catalysts:
            lines.append("🎯 MARKET CATALYSTS:")
            for catalyst in self.catalysts:
                lines.append(f"  • {catalyst}")
            lines.append("")
        
        if self.risks:
            lines.append("⚠️ RISK FACTORS:")
            for risk in self.risks:
                lines.append(f"  • {risk}")
            lines.append("")
        
        lines.append(f"Confidence: {self.confidence}")
        lines.append(f"Sources: {len(self.source_refs)} articles referenced")
        
        return "\n".join(lines)


class NewsSummarizer:
    """
    Summarizes financial news using Claude.
    
    Features:
    - Extracts key events and market catalysts
    - Provides source attribution via citations
    - Handles multiple articles
    - Configurable prompts
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a senior financial news analyst at a major investment bank. 
Your role is to synthesize news articles into actionable intelligence for traders.

Key principles:
1. Focus on market-moving information
2. Distinguish between facts and speculation
3. Highlight timing and magnitude of impacts
4. Note when sources disagree
5. Be concise but comprehensive"""

    DEFAULT_USER_PROMPT_TEMPLATE = """Analyze the following news articles and provide a structured summary.

FOCUS AREAS:
1. Central bank policy signals
2. Economic data releases and surprises
3. Geopolitical developments
4. Corporate earnings/guidance
5. Market sentiment shifts

RULES:
- Cite article numbers [1], [2], etc. for each claim
- Mark speculation as "Hypothesis:" or "Market speculation:"
- Note conflicting information between sources
- Focus on implications for the next 1-4 weeks
- Be specific about timing when mentioned

ARTICLES:
{articles}

Respond in the following JSON format:
{{
    "summary": "2-3 paragraph synthesis of the key news",
    "key_events": ["Event 1 [citation]", "Event 2 [citation]"],
    "catalysts": ["Catalyst that could move markets [citation]"],
    "risks": ["Risk factor to monitor [citation]"],
    "confidence": "HIGH/MEDIUM/LOW based on source quality and agreement"
}}"""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None
    ):
        """
        Initialize news summarizer.
        
        Args:
            client: LLM client (creates default if not provided)
            system_prompt: Custom system prompt
            user_prompt_template: Custom user prompt template
        """
        self.client = client or create_llm_client()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or self.DEFAULT_USER_PROMPT_TEMPLATE
        
        logger.info("News summarizer initialized")
    
    def _format_articles(self, articles: List[Article]) -> str:
        """Format articles for inclusion in prompt."""
        formatted = []
        for i, article in enumerate(articles, 1):
            formatted.append(article.to_formatted_string(i))
        
        return "\n\n---\n\n".join(formatted)
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation references from text."""
        # Match patterns like [1], [2], [1,2], [1-3]
        pattern = r'\[(\d+(?:[-,]\d+)*)\]'
        matches = re.findall(pattern, text)
        
        citations = set()
        for match in matches:
            if '-' in match:
                # Range like [1-3]
                parts = match.split('-')
                start, end = int(parts[0]), int(parts[1])
                for i in range(start, end + 1):
                    citations.add(str(i))
            elif ',' in match:
                # Multiple like [1,2,3]
                for num in match.split(','):
                    citations.add(num.strip())
            else:
                citations.add(match)
        
        return sorted(citations, key=lambda x: int(x))
    
    def _parse_response(
        self,
        response_text: str,
        articles: List[Article]
    ) -> NewsSummary:
        """Parse LLM response into NewsSummary."""
        import json
        
        # Try to parse as JSON
        try:
            # Handle markdown code blocks
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            
            data = json.loads(text)
            
            # Extract citations from all text fields
            all_text = " ".join([
                data.get("summary", ""),
                " ".join(data.get("key_events", [])),
                " ".join(data.get("catalysts", [])),
                " ".join(data.get("risks", [])),
            ])
            
            citation_indices = self._extract_citations(all_text)
            
            # Map citation indices to article IDs
            source_refs = []
            for idx in citation_indices:
                i = int(idx) - 1  # Convert to 0-indexed
                if 0 <= i < len(articles):
                    source_refs.append(articles[i].id)
            
            return NewsSummary(
                summary_text=data.get("summary", ""),
                source_refs=source_refs,
                key_events=data.get("key_events", []),
                catalysts=data.get("catalysts", []),
                risks=data.get("risks", []),
                confidence=data.get("confidence", "MEDIUM"),
                articles_processed=len(articles),
            )
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            
            # Fall back to extracting what we can
            citations = self._extract_citations(response_text)
            source_refs = []
            for idx in citations:
                i = int(idx) - 1
                if 0 <= i < len(articles):
                    source_refs.append(articles[i].id)
            
            return NewsSummary(
                summary_text=response_text,
                source_refs=source_refs,
                key_events=[],
                catalysts=[],
                risks=[],
                confidence="LOW",
                articles_processed=len(articles),
            )
    
    def summarize(
        self,
        articles: List[Article],
        max_tokens: int = 2048,
        temperature: float = 0.3
    ) -> NewsSummary:
        """
        Summarize a list of news articles.
        
        Args:
            articles: List of Article objects
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
        
        Returns:
            NewsSummary with structured output
        """
        if not articles:
            logger.warning("No articles provided for summarization")
            return NewsSummary(
                summary_text="No articles provided.",
                source_refs=[],
                key_events=[],
                catalysts=[],
                risks=[],
                confidence="LOW",
                articles_processed=0,
            )
        
        logger.info(f"Summarizing {len(articles)} articles")
        
        # Format articles for prompt
        formatted_articles = self._format_articles(articles)
        
        # Build prompt
        user_prompt = self.user_prompt_template.format(articles=formatted_articles)
        
        # Call LLM
        response = self.client.complete(
            prompt=user_prompt,
            system=self.system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Parse response
        summary = self._parse_response(response.content, articles)
        summary.model_used = response.model
        
        logger.info(
            f"Summary generated: {len(summary.key_events)} events, "
            f"{len(summary.catalysts)} catalysts, "
            f"confidence={summary.confidence}"
        )
        
        return summary
    
    def summarize_from_dicts(
        self,
        article_dicts: List[Dict],
        max_tokens: int = 2048
    ) -> NewsSummary:
        """
        Summarize articles from dictionary format.
        
        Args:
            article_dicts: List of dicts with keys: id, title, text, source
            max_tokens: Maximum tokens for response
        
        Returns:
            NewsSummary
        """
        articles = []
        for i, d in enumerate(article_dicts):
            articles.append(Article(
                id=d.get("id", f"article_{i}"),
                title=d.get("title", "Untitled"),
                text=d.get("text", ""),
                source=d.get("source", "Unknown"),
                published_at=d.get("published_at"),
                url=d.get("url"),
            ))
        
        return self.summarize(articles, max_tokens)


def create_news_summarizer(
    api_key: Optional[str] = None,
    mock: bool = False
) -> NewsSummarizer:
    """
    Factory function to create news summarizer.
    
    Args:
        api_key: API key for LLM client
        mock: Whether to use mock client
    
    Returns:
        NewsSummarizer instance
    """
    client = create_llm_client(api_key=api_key, mock=mock)
    return NewsSummarizer(client=client)


# =============================================================================
# Asset-Specific Summarizers
# =============================================================================

class FXNewsSummarizer(NewsSummarizer):
    """FX-specific news summarizer."""
    
    DEFAULT_SYSTEM_PROMPT = """You are a senior FX strategist at a major bank.
Your role is to analyze news for currency market implications.

Focus on:
1. Central bank policy signals (Fed, ECB, BOJ, BOE, etc.)
2. Interest rate expectations
3. Economic data that affects rate differentials
4. Risk sentiment indicators
5. Trade/tariff developments"""

    DEFAULT_USER_PROMPT_TEMPLATE = """Analyze these articles for FX market implications.

CURRENCY FOCUS: {currencies}

ANALYZE FOR:
1. Central bank policy direction
2. Interest rate differential changes
3. Risk-on vs risk-off sentiment
4. Specific currency impacts

ARTICLES:
{articles}

Respond in JSON format:
{{
    "summary": "FX-focused synthesis",
    "key_events": ["Event with currency impact [citation]"],
    "catalysts": ["What could move currencies [citation]"],
    "risks": ["Risk factors [citation]"],
    "currency_impacts": {{
        "USD": "bullish/bearish/neutral and why",
        "EUR": "bullish/bearish/neutral and why"
    }},
    "confidence": "HIGH/MEDIUM/LOW"
}}"""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        currencies: List[str] = None
    ):
        super().__init__(client)
        self.currencies = currencies or ["USD", "EUR", "JPY", "GBP", "AUD"]
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
    
    def summarize(
        self,
        articles: List[Article],
        max_tokens: int = 2048,
        temperature: float = 0.3
    ) -> NewsSummary:
        """Summarize with FX focus."""
        # Update template with currencies
        self.user_prompt_template = self.DEFAULT_USER_PROMPT_TEMPLATE.replace(
            "{currencies}", ", ".join(self.currencies)
        )
        return super().summarize(articles, max_tokens, temperature)


class CommodityNewsSummarizer(NewsSummarizer):
    """Commodity-specific news summarizer."""
    
    DEFAULT_SYSTEM_PROMPT = """You are a senior commodities analyst.
Your role is to analyze news for commodity market implications.

Focus on:
1. Supply/demand dynamics
2. Inventory data
3. Weather impacts
4. Geopolitical supply risks
5. OPEC and producer actions"""


class CryptoNewsSummarizer(NewsSummarizer):
    """Crypto-specific news summarizer."""
    
    DEFAULT_SYSTEM_PROMPT = """You are a crypto market analyst.
Your role is to analyze news for cryptocurrency market implications.

Focus on:
1. Regulatory developments
2. Institutional adoption
3. Network/protocol updates
4. Exchange and custody news
5. Macro correlation factors"""
