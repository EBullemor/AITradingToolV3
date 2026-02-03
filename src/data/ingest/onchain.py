"""
On-Chain Data Ingestion Module

Fetches on-chain metrics from Glassnode or similar providers for BTC analysis.
This is used by the BTC Trend + Volatility strategy pod.

Note: Requires Glassnode API key for full functionality.
For MVP, can use CSV exports or mock data.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

from src.data.validate import (
    validate_dataframe,
    ValidationResult,
    quarantine_bad_data,
)


# =============================================================================
# Configuration
# =============================================================================

# Glassnode API configuration
GLASSNODE_BASE_URL = "https://api.glassnode.com/v1/metrics"

# Metrics we care about for BTC strategy
GLASSNODE_METRICS = {
    "exchange_net_flow": {
        "endpoint": "/transactions/transfers_volume_exchanges_net",
        "description": "Net BTC flow to/from exchanges (negative = accumulation)",
        "asset": "BTC",
        "resolution": "24h",
    },
    "mvrv_ratio": {
        "endpoint": "/market/mvrv",
        "description": "Market Value to Realized Value ratio",
        "asset": "BTC",
        "resolution": "24h",
    },
    "sopr": {
        "endpoint": "/indicators/sopr",
        "description": "Spent Output Profit Ratio",
        "asset": "BTC",
        "resolution": "24h",
    },
    "active_addresses": {
        "endpoint": "/addresses/active_count",
        "description": "Number of unique active addresses",
        "asset": "BTC",
        "resolution": "24h",
    },
    "hash_rate": {
        "endpoint": "/mining/hash_rate_mean",
        "description": "Mean hash rate",
        "asset": "BTC", 
        "resolution": "24h",
    },
}


def get_glassnode_api_key() -> Optional[str]:
    """Get Glassnode API key from environment."""
    return os.environ.get("GLASSNODE_API_KEY")


def get_onchain_data_dir() -> Path:
    """Get on-chain data directory."""
    base = Path(__file__).parent.parent.parent / "data"
    onchain_dir = base / "raw" / "onchain"
    onchain_dir.mkdir(parents=True, exist_ok=True)
    return onchain_dir


# =============================================================================
# Glassnode API Functions
# =============================================================================

def fetch_glassnode_metric(
    metric_name: str,
    start_date: datetime,
    end_date: Optional[datetime] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch a single metric from Glassnode API.
    
    Args:
        metric_name: Name of metric (from GLASSNODE_METRICS)
        start_date: Start date for data
        end_date: End date (defaults to now)
        api_key: Glassnode API key
    
    Returns:
        DataFrame with metric data
    """
    if api_key is None:
        api_key = get_glassnode_api_key()
    
    if api_key is None:
        logger.warning("No Glassnode API key - using mock data")
        return create_mock_onchain_data(metric_name, start_date, end_date)
    
    if metric_name not in GLASSNODE_METRICS:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    if end_date is None:
        end_date = datetime.now()
    
    metric_config = GLASSNODE_METRICS[metric_name]
    
    url = f"{GLASSNODE_BASE_URL}{metric_config['endpoint']}"
    params = {
        "a": metric_config["asset"],
        "s": int(start_date.timestamp()),
        "u": int(end_date.timestamp()),
        "i": metric_config["resolution"],
        "api_key": api_key,
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s")
        df = df.rename(columns={"v": metric_name})
        df = df[["timestamp", metric_name]]
        
        logger.info(f"Fetched {len(df)} rows for {metric_name}")
        return df
        
    except requests.RequestException as e:
        logger.error(f"Error fetching {metric_name}: {e}")
        return pd.DataFrame()


def fetch_all_onchain_metrics(
    start_date: datetime,
    end_date: Optional[datetime] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch all on-chain metrics and combine into single DataFrame.
    
    Args:
        start_date: Start date
        end_date: End date
        api_key: Glassnode API key
    
    Returns:
        Combined DataFrame with all metrics
    """
    if end_date is None:
        end_date = datetime.now()
    
    dfs = []
    
    for metric_name in GLASSNODE_METRICS.keys():
        df = fetch_glassnode_metric(metric_name, start_date, end_date, api_key)
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    # Merge all metrics on timestamp
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on="timestamp", how="outer")
    
    result = result.sort_values("timestamp")
    
    logger.info(f"Combined {len(result)} rows of on-chain data")
    return result


# =============================================================================
# Mock Data for Development
# =============================================================================

def create_mock_onchain_data(
    metric_name: str,
    start_date: datetime,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Create mock on-chain data for development/testing.
    
    Args:
        metric_name: Name of metric
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with mock data
    """
    import numpy as np
    
    if end_date is None:
        end_date = datetime.now()
    
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    np.random.seed(42)
    
    # Generate realistic-looking mock data based on metric type
    if metric_name == "exchange_net_flow":
        # Net flow oscillates around 0, occasionally spikes
        values = np.random.normal(0, 5000, len(dates))
        values = np.cumsum(values * 0.1)  # Add some trend
    
    elif metric_name == "mvrv_ratio":
        # MVRV typically ranges from 0.5 to 4
        values = 1.5 + np.random.normal(0, 0.3, len(dates))
        values = np.clip(values, 0.5, 4.0)
    
    elif metric_name == "sopr":
        # SOPR oscillates around 1
        values = 1.0 + np.random.normal(0, 0.05, len(dates))
        values = np.clip(values, 0.8, 1.2)
    
    elif metric_name == "active_addresses":
        # Active addresses in hundreds of thousands
        values = 800000 + np.random.normal(0, 50000, len(dates))
        values = np.clip(values, 500000, 1200000)
    
    elif metric_name == "hash_rate":
        # Hash rate in EH/s, generally trending up
        base = 400 + np.arange(len(dates)) * 0.5
        values = base + np.random.normal(0, 10, len(dates))
    
    else:
        values = np.random.randn(len(dates))
    
    df = pd.DataFrame({
        "timestamp": dates,
        metric_name: values
    })
    
    logger.debug(f"Created mock data for {metric_name}: {len(df)} rows")
    return df


# =============================================================================
# CSV Loading (for manual exports)
# =============================================================================

def load_onchain_csv(
    file_path: Path,
    metric_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Load on-chain data from CSV export.
    
    Args:
        file_path: Path to CSV file
        metric_name: Name to give the value column
    
    Returns:
        Standardized DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Find timestamp column
    ts_cols = ["timestamp", "date", "time", "t"]
    for col in ts_cols:
        if col in df.columns:
            df["timestamp"] = pd.to_datetime(df[col])
            break
    
    # Find value column
    value_cols = ["value", "v", metric_name] if metric_name else ["value", "v"]
    for col in value_cols:
        if col in df.columns:
            if metric_name:
                df[metric_name] = df[col]
            break
    
    logger.info(f"Loaded on-chain CSV: {len(df)} rows from {file_path}")
    return df


def load_all_onchain_csvs(
    data_dir: Optional[Path] = None,
    data_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load all on-chain CSVs from a directory.
    
    Args:
        data_dir: Directory containing CSV files
        data_date: Date folder to look in
    
    Returns:
        Combined DataFrame
    """
    if data_dir is None:
        data_dir = get_onchain_data_dir()
    
    if data_date:
        data_dir = data_dir / data_date.strftime("%Y-%m-%d")
    
    if not data_dir.exists():
        logger.warning(f"On-chain data directory not found: {data_dir}")
        return pd.DataFrame()
    
    dfs = []
    
    for csv_file in data_dir.glob("*.csv"):
        metric_name = csv_file.stem.lower().replace("-", "_")
        df = load_onchain_csv(csv_file, metric_name)
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    # Merge all
    result = dfs[0]
    for df in dfs[1:]:
        if "timestamp" in df.columns:
            result = result.merge(df, on="timestamp", how="outer")
    
    return result.sort_values("timestamp")


# =============================================================================
# Main Ingestion Function
# =============================================================================

def ingest_onchain_data(
    start_date: datetime,
    end_date: Optional[datetime] = None,
    use_api: bool = True,
    use_csv: bool = True,
    validate: bool = True
) -> pd.DataFrame:
    """
    Ingest on-chain data from all available sources.
    
    Priority: API > CSV > Mock
    
    Args:
        start_date: Start date
        end_date: End date
        use_api: Whether to try Glassnode API
        use_csv: Whether to try loading CSVs
        validate: Whether to validate the data
    
    Returns:
        DataFrame with on-chain metrics
    """
    if end_date is None:
        end_date = datetime.now()
    
    df = pd.DataFrame()
    
    # Try API first
    if use_api and get_glassnode_api_key():
        logger.info("Fetching on-chain data from Glassnode API...")
        df = fetch_all_onchain_metrics(start_date, end_date)
    
    # Try CSV if API didn't work
    if df.empty and use_csv:
        logger.info("Loading on-chain data from CSV files...")
        df = load_all_onchain_csvs()
    
    # Fall back to mock data
    if df.empty:
        logger.warning("No on-chain data available - using mock data")
        df = pd.DataFrame({"timestamp": pd.date_range(start_date, end_date, freq="D")})
        for metric in GLASSNODE_METRICS.keys():
            mock = create_mock_onchain_data(metric, start_date, end_date)
            df = df.merge(mock, on="timestamp", how="left")
    
    # Add ticker column for validation
    df["ticker"] = "BTCUSD"
    
    # Validate if requested
    if validate and not df.empty:
        result = validate_dataframe(df, "BTCUSD", end_date)
        if not result.passed:
            logger.warning(f"On-chain data validation issues: {len(result.issues)}")
    
    return df


def save_onchain_data(
    df: pd.DataFrame,
    data_date: datetime,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save on-chain data to processed folder.
    
    Args:
        df: DataFrame with on-chain metrics
        data_date: Date of data
        output_dir: Output directory
    
    Returns:
        Path to saved file
    """
    if output_dir is None:
        base = Path(__file__).parent.parent.parent / "data" / "processed"
        output_dir = base
    
    date_str = data_date.strftime("%Y-%m-%d")
    output_dir = output_dir / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / f"BTCUSD_onchain_{date_str}.parquet"
    df.to_parquet(file_path, index=False)
    
    logger.info(f"Saved on-chain data to {file_path}")
    return file_path


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest on-chain data")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to fetch"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data instead of API"
    )
    
    args = parser.parse_args()
    
    logger.add("logs/onchain_ingestion_{time}.log", rotation="1 day")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    df = ingest_onchain_data(
        start_date,
        end_date,
        use_api=not args.mock
    )
    
    if not df.empty:
        save_onchain_data(df, end_date)
    
    logger.info(f"On-chain ingestion complete: {len(df)} rows")


if __name__ == "__main__":
    main()
