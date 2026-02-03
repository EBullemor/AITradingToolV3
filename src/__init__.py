"""
AI Trading Workflow 2.0

A comprehensive trading recommendation system combining quantitative signals
with LLM-powered research synthesis.

Modules:
    data: Data ingestion and validation
    features: Feature engineering (FX, BTC, commodities)
    strategies: Trading strategy implementations
    aggregator: Signal combination and conflict resolution
    llm: LLM integration for news analysis and grounding
    risk: Position sizing and portfolio risk management
    outputs: Notion, Slack, and formatting
    monitoring: Health checks, metrics, and alerts
    backtest: Backtesting framework with walk-forward validation
"""

__version__ = "2.0.0"
__author__ = "Ethan Bullemor"
