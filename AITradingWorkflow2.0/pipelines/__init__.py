"""
Pipelines Module

Contains orchestration scripts for the trading recommendation system.

Available pipelines:
- daily_run: Main daily pipeline that generates recommendations

Usage:
    # From command line
    python -m pipelines.daily_run --date 2026-02-01
    
    # From Python
    from pipelines.daily_run import DailyPipeline, PipelineConfig
    
    config = PipelineConfig(run_date=datetime.now())
    pipeline = DailyPipeline(config)
    result = pipeline.run()
"""

from .daily_run import (
    PipelineConfig,
    PipelineResult,
    DailyPipeline,
)

__all__ = [
    "PipelineConfig",
    "PipelineResult", 
    "DailyPipeline",
]
