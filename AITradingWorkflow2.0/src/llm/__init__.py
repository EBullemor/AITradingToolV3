"""
LLM Integration Module

Provides Claude API integration for:
- News summarization
- Trade thesis generation
- Claim extraction and grounding verification
"""

from .client import ClaudeClient, create_client
from .news_summarizer import (
    NewsSummarizer,
    NewsSummary,
    summarize_news,
)

__all__ = [
    "ClaudeClient",
    "create_client",
    "NewsSummarizer", 
    "NewsSummary",
    "summarize_news",
]
