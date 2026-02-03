# AI Trading Workflow 2.0

An AI-powered trading recommendation system generating 2-10 actionable trade ideas per day for FX, Bitcoin, and Commodities.

## 🎯 Overview

This platform combines quantitative trading signals with LLM-powered research synthesis to generate daily trading recommendations with:

- **Multi-Asset Coverage:** FX majors, Bitcoin, commodities (Oil, Gold, Copper)
- **5 Strategy Pods:** FX Carry+Momentum, BTC Trend+Vol, Commodities Term Structure, Cross-Asset Risk, Mean Reversion
- **Risk Management:** Position sizing, correlation limits, drawdown protection
- **LLM Integration:** Grounded news summarization, trade thesis generation
- **Automated Outputs:** Notion databases, Slack notifications

## 📁 Project Structure

```
AITradingWorkflow2.0/
├── src/                          # Source code
│   ├── data/                     # Data layer
│   │   ├── ingest/               # Bloomberg, on-chain loaders
│   │   └── validate/             # Schema, quality, bias checks
│   ├── features/                 # Feature engineering
│   │   ├── fx_features.py        # FX momentum, carry, volatility
│   │   ├── btc_features.py       # BTC trend, on-chain metrics
│   │   ├── commodity_features.py # Term structure, inventory
│   │   └── regime.py             # VIX-based regime detection
│   ├── strategies/               # Trading strategies
│   │   ├── base.py               # BaseStrategy abstract class
│   │   └── fx_carry_momentum.py  # FX Carry+Momentum (Pod 1)
│   ├── aggregator/               # Signal combination
│   │   ├── signal_combiner.py    # Weighted signal combination
│   │   ├── conflict_resolver.py  # Handle conflicting signals
│   │   └── deduplication.py      # Remove duplicate signals
│   ├── llm/                      # LLM integration
│   │   ├── client.py             # Claude API wrapper
│   │   ├── news_summarizer.py    # News analysis
│   │   └── grounding/            # Claim verification
│   ├── risk/                     # Risk management
│   │   ├── position_sizer.py     # Position sizing algorithms
│   │   └── portfolio_risk.py     # Portfolio-level constraints
│   ├── outputs/                  # Output integrations
│   │   ├── notion_client.py      # Notion API integration
│   │   ├── formatter.py          # Trade card formatting
│   │   └── slack_poster.py       # Slack notifications
│   ├── monitoring/               # System monitoring
│   │   ├── health_checks.py      # Pipeline health
│   │   ├── metrics_collector.py  # Performance metrics
│   │   └── alerter.py            # Alert management
│   └── backtest/                 # Backtesting
│       ├── engine.py             # Walk-forward validation
│       ├── costs.py              # Transaction cost models
│       └── metrics.py            # Performance calculations
├── config/                       # Configuration files
│   ├── instruments.yaml          # Tradeable instruments
│   ├── risk_limits.yaml          # Risk parameters
│   ├── strategy_params.yaml      # Strategy settings
│   ├── feature_registry.yaml     # Feature definitions
│   └── model_registry.yaml       # Strategy-feature mapping
├── pipelines/                    # Orchestration
│   ├── daily_run.py              # Main daily pipeline
│   ├── backtest.py               # Backtesting pipeline
│   └── health_check.py           # Health monitoring
├── prompts/                      # LLM prompt templates
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
└── docs/                         # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Bloomberg Terminal access (for data)
- Claude API key (for LLM features)
- Notion API key (for output)

### Installation

```bash
# Clone repository
git clone https://github.com/EBullemor/AITradingWorkflow2.0.git
cd AITradingWorkflow2.0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit `.env` with your credentials:

```bash
ANTHROPIC_API_KEY=your_claude_api_key
NOTION_API_KEY=your_notion_api_key
SLACK_WEBHOOK_URL=your_slack_webhook
```

### Running the Pipeline

```bash
# Run daily recommendation pipeline
python -m pipelines.daily_run

# Run backtesting
python -m pipelines.backtest --strategy fx_carry_momentum --start 2024-01-01

# Run health checks
python -m pipelines.health_check
```

## 📊 Strategy Pods

### Pod 1: FX Carry + Momentum (Implemented ✅)
- **Instruments:** EURUSD, USDJPY, GBPUSD, AUDUSD
- **Signals:** Carry score, momentum z-scores, regime filter
- **Holding Period:** 1-4 weeks

### Pod 2: BTC Trend + Volatility (Planned)
- **Signals:** MA crossover, volatility breakout, on-chain metrics
- **Holding Period:** 2-8 weeks

### Pod 3: Commodities Term Structure (Planned)
- **Instruments:** WTI, Brent, Gold, Copper
- **Signals:** Roll yield, inventory, momentum

### Pod 4: Cross-Asset Risk (Planned)
- **Signals:** VIX regime, credit spreads, safe haven flows

### Pod 5: Mean Reversion (Planned)
- **Signals:** Extreme moves without catalyst

## 🔧 Development

```bash
# Run tests
pytest tests/

# Run specific test file
pytest tests/unit/test_fx_strategy.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📈 Risk Management

- **Position Sizing:** 1% risk per trade
- **Max Position:** 10% of portfolio
- **Max Gross Exposure:** 100%
- **Correlation Limit:** Max 3 positions with corr > 0.7
- **Kill Switch:** -8% drawdown halts trading

## 🔗 Integrations

- **Bloomberg Terminal:** Market data export
- **Claude API:** News summarization, trade thesis
- **Notion:** Recommendation database
- **Slack:** Daily notifications

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always paper trade before using real capital.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
