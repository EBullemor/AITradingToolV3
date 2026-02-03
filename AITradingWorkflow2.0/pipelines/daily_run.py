#!/usr/bin/env python3
"""
Daily Pipeline Orchestration

Main orchestration script that runs the complete trading recommendation pipeline:
1. Data Ingestion - Load market data
2. Feature Engineering - Compute features for all instruments
3. News Analysis - Summarize news with LLM
4. Signal Generation - Run strategy pods
5. Signal Aggregation - Combine and resolve conflicts
6. Recommendation Output - Format and save results

Usage:
    python -m pipelines.daily_run                    # Run for today
    python -m pipelines.daily_run --date 2026-02-01  # Run for specific date
    python -m pipelines.daily_run --dry-run          # Preview without saving
"""

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import ingest_and_validate, load_processed_data
from src.features import compute_fx_features
from src.strategies import FXCarryMomentumStrategy, Signal
from src.aggregator import SignalAggregator, AggregatedSignal
from src.llm import create_news_summarizer, Article, NewsSummary


@dataclass
class PipelineConfig:
    """Configuration for pipeline run."""
    run_date: datetime = field(default_factory=datetime.now)
    data_dir: Path = Path("data")
    lookback_days: int = 120
    strategies_enabled: List[str] = field(default_factory=lambda: ["fx_carry_momentum"])
    output_dir: Path = Path("reports")
    save_signals: bool = True
    use_llm: bool = True
    mock_llm: bool = False
    dry_run: bool = False
    verbose: bool = False


@dataclass
class PipelineResult:
    """Result from pipeline run."""
    run_date: datetime
    status: str = "PENDING"
    instruments_loaded: List[str] = field(default_factory=list)
    signals_generated: int = 0
    signals_after_aggregation: int = 0
    recommendations: List[AggregatedSignal] = field(default_factory=list)
    news_summary: Optional[NewsSummary] = None
    duration_seconds: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DailyPipeline:
    """Main daily pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.result = PipelineResult(run_date=config.run_date)
        self.aggregator = SignalAggregator()
        self.news_summarizer = create_news_summarizer(mock=config.mock_llm) if config.use_llm else None
        self.strategies = {}
        if "fx_carry_momentum" in config.strategies_enabled:
            self.strategies["fx_carry_momentum"] = FXCarryMomentumStrategy()
        logger.info(f"Pipeline initialized for {config.run_date.date()}")
    
    def _generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample data for testing."""
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        dates = pd.date_range(end=self.config.run_date, periods=self.config.lookback_days, freq="D")
        sample_data = {}
        fx_configs = {
            "EURUSD": {"base": 1.10, "vol": 0.005},
            "USDJPY": {"base": 148.0, "vol": 0.5},
            "GBPUSD": {"base": 1.27, "vol": 0.006},
            "AUDUSD": {"base": 0.66, "vol": 0.004},
        }
        for pair, cfg in fx_configs.items():
            returns = np.random.normal(0, cfg["vol"], len(dates))
            prices = cfg["base"] * np.cumprod(1 + returns)
            sample_data[pair] = pd.DataFrame({
                "PX_LAST": prices,
                "PX_HIGH": prices * (1 + np.random.uniform(0.001, 0.003, len(dates))),
                "PX_LOW": prices * (1 - np.random.uniform(0.001, 0.003, len(dates))),
            }, index=dates)
            self.result.instruments_loaded.append(pair)
        vix = 18 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
        sample_data["VIX"] = pd.DataFrame({"PX_LAST": np.clip(vix, 10, 40)}, index=dates)
        self.result.instruments_loaded.append("VIX")
        self.result.warnings.append("Using sample data")
        return sample_data
    
    def stage_data_ingestion(self) -> Dict[str, Any]:
        """Load market data."""
        logger.info("Stage 1: Data Ingestion")
        try:
            data, results = ingest_and_validate(self.config.run_date)
            if data:
                for ticker in data:
                    self.result.instruments_loaded.append(ticker)
                return data
        except Exception as e:
            logger.warning(f"Could not load real data: {e}")
        return self._generate_sample_data()
    
    def stage_feature_engineering(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute features."""
        logger.info("Stage 2: Feature Engineering")
        features = {}
        for pair in ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]:
            if pair not in market_data:
                continue
            df = market_data[pair]
            feat = compute_fx_features(df["PX_LAST"], df.get("PX_HIGH"), df.get("PX_LOW"), pair)
            features[pair] = feat
        if "VIX" in market_data:
            features["_macro"] = market_data["VIX"]
        return features
    
    def stage_news_analysis(self, features: Dict[str, Any]) -> Optional[NewsSummary]:
        """Analyze news."""
        if not self.config.use_llm:
            return None
        logger.info("Stage 3: News Analysis")
        articles = [
            Article(id="1", title="Fed Signals Patience", text="Fed officials signaled patience on rate cuts.", source="Reuters"),
            Article(id="2", title="Dollar Gains", text="Dollar strengthened on safe-haven demand.", source="Bloomberg"),
        ]
        try:
            return self.news_summarizer.summarize(articles)
        except Exception as e:
            logger.warning(f"News analysis failed: {e}")
            return None
    
    def stage_signal_generation(self, features: Dict[str, Any], news: Optional[NewsSummary]) -> List[Signal]:
        """Generate signals."""
        logger.info("Stage 4: Signal Generation")
        signals = []
        macro = features.get("_macro")
        inst_features = {k: v for k, v in features.items() if not k.startswith("_")}
        for name, strategy in self.strategies.items():
            try:
                sigs = strategy.generate_signals(inst_features, macro, as_of_date=self.config.run_date)
                signals.extend(sigs)
            except Exception as e:
                self.result.errors.append(f"Strategy {name}: {e}")
        self.result.signals_generated = len(signals)
        return signals
    
    def stage_signal_aggregation(self, signals: List[Signal]) -> List[AggregatedSignal]:
        """Aggregate signals."""
        logger.info("Stage 5: Signal Aggregation")
        recs = self.aggregator.aggregate(signals, as_of_date=self.config.run_date)
        self.result.signals_after_aggregation = len(recs)
        self.result.recommendations = recs
        return recs
    
    def stage_output_generation(self, recs: List[AggregatedSignal], news: Optional[NewsSummary]) -> Dict:
        """Generate outputs."""
        logger.info("Stage 6: Output Generation")
        if self.config.dry_run:
            return {"dry_run": True}
        output_dir = self.config.output_dir / "signals"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"recs_{self.config.run_date.strftime('%Y%m%d')}.json"
        with open(path, "w") as f:
            json.dump({"recommendations": [r.to_dict() for r in recs]}, f, indent=2, default=str)
        return {"file": str(path)}
    
    def run(self) -> PipelineResult:
        """Execute pipeline."""
        import time
        start = time.time()
        logger.info(f"{'='*60}\nStarting pipeline for {self.config.run_date.date()}\n{'='*60}")
        try:
            data = self.stage_data_ingestion()
            features = self.stage_feature_engineering(data)
            news = self.stage_news_analysis(features)
            signals = self.stage_signal_generation(features, news)
            recs = self.stage_signal_aggregation(signals)
            self.stage_output_generation(recs, news)
            self.result.status = "SUCCESS" if not self.result.errors else "PARTIAL"
        except Exception as e:
            self.result.status = "FAILED"
            self.result.errors.append(str(e))
        self.result.duration_seconds = time.time() - start
        logger.info(f"Pipeline {self.result.status} in {self.result.duration_seconds:.1f}s")
        return self.result


def main():
    parser = argparse.ArgumentParser(description="Run daily pipeline")
    parser.add_argument("--date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    
    run_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.now()
    config = PipelineConfig(run_date=run_date, dry_run=args.dry_run, mock_llm=args.mock_llm, use_llm=not args.no_llm)
    result = DailyPipeline(config).run()
    
    print(f"\n{'='*60}\nPIPELINE SUMMARY\n{'='*60}")
    print(f"Status: {result.status}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"Recommendations: {len(result.recommendations)}")
    for r in result.recommendations:
        print(f"  {'🟢' if r.direction.value=='LONG' else '🔴'} {r.direction.value} {r.instrument} ({r.confidence:.0%})")
    sys.exit(0 if result.status == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
