"""
Shared Fixtures for Integration Tests

Provides common test infrastructure:
- Pipeline configurations
- Sample data generators
- Mock services
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Path to project root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config_dir():
    """Path to config directory."""
    return PROJECT_ROOT / "config"


@pytest.fixture
def mock_pipeline_config():
    """Standard mock pipeline configuration for testing."""
    from pipelines.daily_run import PipelineConfig
    return PipelineConfig(
        run_date=datetime(2026, 2, 3),
        dry_run=True,
        mock_llm=True,
        use_llm=True,
    )


@pytest.fixture
def mock_pipeline(mock_pipeline_config):
    """Pre-configured pipeline instance for testing."""
    from pipelines.daily_run import DailyPipeline
    return DailyPipeline(mock_pipeline_config)


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    dates = pd.date_range("2025-06-01", periods=120, freq="D")

    data = {}
    configs = {
        "EURUSD": {"base": 1.10, "vol": 0.005},
        "USDJPY": {"base": 148.0, "vol": 0.5},
        "GBPUSD": {"base": 1.27, "vol": 0.006},
        "AUDUSD": {"base": 0.66, "vol": 0.004},
    }

    for pair, cfg in configs.items():
        returns = np.random.normal(0, cfg["vol"], len(dates))
        prices = cfg["base"] * np.cumprod(1 + returns)
        data[pair] = pd.DataFrame({
            "PX_LAST": prices,
            "PX_HIGH": prices * (1 + np.random.uniform(0.001, 0.003, len(dates))),
            "PX_LOW": prices * (1 - np.random.uniform(0.001, 0.003, len(dates))),
        }, index=dates)

    return data
