#!/usr/bin/env python3
"""
Daily Pipeline V1 — Wired to Real CSV Data

Main orchestration script that runs the complete trading recommendation pipeline
against real Bloomberg-exported CSV data, with Cross-Asset Risk Overlay (Pod 4)
integrated for signal gating.

Changes from V0:
  - Reads real CSV data instead of falling back to sample data
  - Integrates CrossAssetRiskOverlay before signal generation
  - Supports configurable data directory

Usage:
    python -m pipelines.daily_run                    # Run for today
    python -m pipelines.daily_run --date 2026-02-03  # Run for specific date
    python -m pipelines.daily_run --dry-run          # Preview without saving
"""

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import ingest_and_validate, load_processed_data
from src.features import compute_fx_features
from src.strategies import FXCarryMomentumStrategy, Signal
from src.strategies.cross_asset_risk import CrossAssetRiskOverlay
from src.aggregator import SignalAggregator, AggregatedSignal
from src.llm import create_news_summarizer, Article, NewsSummary


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for pipeline run."""
    run_date: datetime = field(default_factory=datetime.now)
    data_dir: Path = Path("data")
    lookback_days: int = 120
    strategies_enabled: List[str] = field(
        default_factory=lambda: ["fx_carry_momentum", "commodities_ts"]
    )
    output_dir: Path = Path("reports")
    save_signals: bool = True
    use_llm: bool = True
    mock_llm: bool = False
    dry_run: bool = False
    verbose: bool = False
    # Risk overlay
    use_risk_overlay: bool = True
    vix_kill_threshold: float = 30.0


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
    risk_regime: str = "UNKNOWN"
    risk_score: float = 0.0
    duration_seconds: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# CSV Data Loading
# =============================================================================

CSV_FILES = {
    "fx": "fx_spots_clean.csv",
    "rates": "rates_clean.csv",
    "macros": "macros_clean.csv",
    "commodities": "commodities_clean.csv",
}


def find_csv_data_dir() -> Optional[Path]:
    """Search for CSV data files in common locations."""
    candidates = [
        Path("data/clean"),
        Path("data"),
        Path("."),
        Path(__file__).parent.parent / "data" / "clean",
        Path(__file__).parent.parent / "data",
        Path(__file__).parent.parent,
    ]
    for d in candidates:
        if (d / "fx_spots_clean.csv").exists():
            return d
    return None


def load_csv_market_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load market data from CSV files.

    Returns dict with keys: fx, rates, macros, commodities
    """
    data = {}
    for key, filename in CSV_FILES.items():
        path = data_dir / filename
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index().ffill()
            data[key] = df
            logger.info(f"  Loaded {key}: {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
        else:
            logger.warning(f"  Missing: {path}")
    return data


def reshape_for_pipeline(csv_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Reshape CSV data into per-instrument DataFrames with PX_LAST column
    expected by feature engineering and strategy modules.
    """
    instruments = {}

    # FX pairs
    fx = csv_data.get("fx", pd.DataFrame())
    for pair in ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]:
        if pair in fx.columns:
            df = pd.DataFrame({"PX_LAST": fx[pair]}, index=fx.index).dropna()
            if len(df) > 0:
                instruments[pair] = df

    # Macros
    macros = csv_data.get("macros", pd.DataFrame())
    for col in macros.columns:
        instruments[col] = pd.DataFrame({"PX_LAST": macros[col]}, index=macros.index)

    # Commodities
    commodities = csv_data.get("commodities", pd.DataFrame())
    for col in commodities.columns:
        instruments[col] = pd.DataFrame({"PX_LAST": commodities[col]}, index=commodities.index)

    return instruments


# =============================================================================
# Pipeline
# =============================================================================

class DailyPipeline:
    """Main daily pipeline orchestrator."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.result = PipelineResult(run_date=config.run_date)
        self.strategies = {}
        self.risk_overlay = None

        # Initialize strategies
        if "fx_carry_momentum" in config.strategies_enabled:
            self.strategies["fx_carry_momentum"] = FXCarryMomentumStrategy()

        # Initialize risk overlay
        if config.use_risk_overlay:
            self.risk_overlay = CrossAssetRiskOverlay(
                vix_kill_threshold=config.vix_kill_threshold
            )

        # Initialize aggregator and LLM
        self.aggregator = SignalAggregator()
        self.news_summarizer = create_news_summarizer() if config.use_llm else None

    def run(self) -> PipelineResult:
        """Execute the full pipeline."""
        start_time = time.time()

        logger.info(f"\n{'='*60}")
        logger.info(f"  DAILY PIPELINE V1 — {self.config.run_date.strftime('%Y-%m-%d')}")
        logger.info(f"{'='*60}")

        try:
            # Stage 1: Data Ingestion
            t0 = time.time()
            market_data = self.stage_data_ingestion()
            self.result.stage_timings["data_ingestion"] = time.time() - t0

            # Stage 2: Risk Assessment
            t0 = time.time()
            risk_allowed = self.stage_risk_assessment(market_data)
            self.result.stage_timings["risk_assessment"] = time.time() - t0

            if not risk_allowed:
                self.result.status = "HALTED_BY_RISK"
                logger.warning("Pipeline halted by risk overlay — no new signals generated")
                return self.result

            # Stage 3: Feature Engineering
            t0 = time.time()
            features = self.stage_feature_engineering(market_data)
            self.result.stage_timings["feature_engineering"] = time.time() - t0

            # Stage 4: News Analysis
            t0 = time.time()
            news = self.stage_news_analysis(features)
            self.result.stage_timings["news_analysis"] = time.time() - t0

            # Stage 5: Signal Generation
            t0 = time.time()
            signals = self.stage_signal_generation(features, news)
            self.result.stage_timings["signal_generation"] = time.time() - t0

            # Stage 6: Signal Aggregation
            t0 = time.time()
            recs = self.stage_signal_aggregation(signals)
            self.result.stage_timings["signal_aggregation"] = time.time() - t0

            # Stage 7: Output
            t0 = time.time()
            self.stage_output_generation(recs, news)
            self.result.stage_timings["output_generation"] = time.time() - t0

            self.result.status = "SUCCESS"

        except Exception as e:
            self.result.status = "ERROR"
            self.result.errors.append(f"{type(e).__name__}: {str(e)}")
            logger.error(f"Pipeline failed: {e}")
            traceback.print_exc()

        self.result.duration_seconds = time.time() - start_time
        logger.info(f"\nPipeline completed in {self.result.duration_seconds:.1f}s — Status: {self.result.status}")
        return self.result

    def stage_data_ingestion(self) -> Dict[str, Any]:
        """Stage 1: Load market data from CSVs (with fallback to Bloomberg ingest)."""
        logger.info("Stage 1: Data Ingestion")

        # Try CSV files first
        csv_dir = find_csv_data_dir()
        if csv_dir:
            logger.info(f"  Loading from CSV: {csv_dir}")
            csv_data = load_csv_market_data(csv_dir)
            if csv_data:
                instruments = reshape_for_pipeline(csv_data)
                self.result.instruments_loaded = list(instruments.keys())
                logger.info(f"  Loaded {len(instruments)} instruments from CSV")
                # Store raw CSV data for risk overlay
                instruments["_csv_data"] = csv_data
                return instruments

        # Fallback to Bloomberg ingest
        logger.info("  CSV not found, falling back to Bloomberg ingest...")
        try:
            data, results = ingest_and_validate(self.config.run_date)
            if data:
                for ticker in data:
                    self.result.instruments_loaded.append(ticker)
                return data
        except Exception as e:
            logger.warning(f"  Bloomberg ingest failed: {e}")

        self.result.warnings.append("No data source available")
        return {}

    def stage_risk_assessment(self, market_data: Dict[str, Any]) -> bool:
        """Stage 2: Check risk overlay — returns True if trading is allowed."""
        logger.info("Stage 2: Risk Assessment")

        if not self.risk_overlay:
            logger.info("  Risk overlay disabled — proceeding")
            self.result.risk_regime = "DISABLED"
            return True

        csv_data = market_data.get("_csv_data", {})
        macros = csv_data.get("macros", pd.DataFrame())

        if macros.empty or "VIX" not in macros.columns:
            logger.warning("  No VIX data — proceeding with caution")
            self.result.risk_regime = "NO_DATA"
            return True

        vix_hist = macros["VIX"].dropna()
        current_vix = vix_hist.iloc[-1]
        dxy_hist = macros["DXY"].dropna() if "DXY" in macros.columns else None
        spx_hist = macros["SPX"].dropna() if "SPX" in macros.columns else None
        y10_hist = macros["US_10Y_YIELD"].dropna() if "US_10Y_YIELD" in macros.columns else None

        assessment = self.risk_overlay.compute_risk_score(
            vix=current_vix,
            vix_history=vix_hist,
            dxy_history=dxy_hist,
            spx_history=spx_hist,
            us10y_history=y10_hist,
            as_of_date=self.config.run_date,
        )

        self.result.risk_regime = assessment.regime
        self.result.risk_score = assessment.risk_score

        logger.info(f"  VIX: {current_vix:.1f}")
        logger.info(f"  Risk Score: {assessment.risk_score:+.3f}")
        logger.info(f"  Regime: {assessment.regime}")
        logger.info(f"  Position Scalar: {assessment.position_scalar:.0%}")

        for flag in assessment.flags:
            logger.warning(f"  ⚠️  {flag}")

        if assessment.regime == "KILL":
            return False

        return True

    def stage_feature_engineering(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Compute features for all instruments."""
        logger.info("Stage 3: Feature Engineering")
        features = {}

        for pair in ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]:
            if pair not in market_data:
                continue
            df = market_data[pair]
            feat = compute_fx_features(
                df["PX_LAST"],
                df.get("PX_HIGH"),
                df.get("PX_LOW"),
                pair,
            )
            features[pair] = feat
            logger.info(f"  {pair}: {len(feat)} rows")

        if "VIX" in market_data:
            features["_macro"] = market_data["VIX"]

        return features

    def stage_news_analysis(self, features: Dict[str, Any]) -> Optional[NewsSummary]:
        """Stage 4: Analyze news with LLM."""
        if not self.config.use_llm:
            return None

        logger.info("Stage 4: News Analysis")
        articles = [
            Article(id="1", title="Market Update", text="Placeholder for real news feed.", source="Reuters"),
        ]
        try:
            return self.news_summarizer.summarize(articles)
        except Exception as e:
            logger.warning(f"  News analysis failed: {e}")
            return None

    def stage_signal_generation(self, features: Dict[str, Any], news: Optional[NewsSummary]) -> List[Signal]:
        """Stage 5: Generate trading signals from all strategies."""
        logger.info("Stage 5: Signal Generation")
        signals = []
        macro = features.get("_macro")
        inst_features = {k: v for k, v in features.items() if not k.startswith("_")}

        for name, strategy in self.strategies.items():
            try:
                sigs = strategy.generate_signals(
                    inst_features, macro, as_of_date=self.config.run_date
                )
                signals.extend(sigs)
                logger.info(f"  {name}: {len(sigs)} signals")
            except Exception as e:
                self.result.errors.append(f"Strategy {name}: {e}")
                logger.error(f"  {name} failed: {e}")

        self.result.signals_generated = len(signals)
        return signals

    def stage_signal_aggregation(self, signals: List[Signal]) -> List[AggregatedSignal]:
        """Stage 6: Aggregate and deduplicate signals."""
        logger.info("Stage 6: Signal Aggregation")
        recs = self.aggregator.aggregate(signals, as_of_date=self.config.run_date)
        self.result.signals_after_aggregation = len(recs)
        self.result.recommendations = recs
        logger.info(f"  {len(recs)} recommendations after aggregation")
        return recs

    def stage_output_generation(
        self, recs: List[AggregatedSignal], news: Optional[NewsSummary]
    ) -> Dict:
        """Stage 7: Format and save outputs."""
        logger.info("Stage 7: Output Generation")

        if self.config.dry_run:
            logger.info("  (DRY RUN — no files saved)")
            return {}

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save recommendations as JSON
        report = {
            "run_date": self.config.run_date.strftime("%Y-%m-%d"),
            "status": self.result.status,
            "risk_regime": self.result.risk_regime,
            "risk_score": self.result.risk_score,
            "signals_generated": self.result.signals_generated,
            "recommendations_count": len(recs),
            "recommendations": [
                {
                    "instrument": r.instrument,
                    "direction": str(r.direction),
                    "confidence": r.confidence,
                }
                for r in recs
            ],
        }

        json_path = output_dir / f"daily_report_{self.config.run_date.strftime('%Y%m%d')}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"  Report saved: {json_path}")

        return report


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Daily trading pipeline")
    parser.add_argument("--date", type=str, help="Run date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--no-llm", action="store_true", help="Skip news analysis")
    parser.add_argument("--no-risk-overlay", action="store_true", help="Disable risk overlay")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    run_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.now()

    config = PipelineConfig(
        run_date=run_date,
        dry_run=args.dry_run,
        use_llm=not args.no_llm,
        use_risk_overlay=not args.no_risk_overlay,
        verbose=args.verbose,
    )

    pipeline = DailyPipeline(config)
    result = pipeline.run()

    if result.status == "SUCCESS":
        logger.info(f"✅ Pipeline complete: {result.signals_after_aggregation} recommendations")
    elif result.status == "HALTED_BY_RISK":
        logger.warning("⛔ Pipeline halted by risk overlay")
    else:
        logger.error(f"❌ Pipeline failed: {result.errors}")


if __name__ == "__main__":
    main()
