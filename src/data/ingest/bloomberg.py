"""
Bloomberg Data Ingestion Module

Loads data from Bloomberg Terminal CSV exports, validates, and stores
in the processed data folder.

Expected file structure:
    data/raw/bloomberg/YYYY-MM-DD/
        fx_spots_YYYY-MM-DD.csv
        fx_forwards_YYYY-MM-DD.csv
        commodities_YYYY-MM-DD.csv
        macro_YYYY-MM-DD.csv
"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml
from loguru import logger

from src.data.validate import (
    validate_dataframe,
    ValidationResult,
    ValidationStatus,
    quarantine_bad_data,
    create_point_in_time_snapshot,
)


# =============================================================================
# Configuration
# =============================================================================

def get_data_dirs() -> Dict[str, Path]:
    """Get data directory paths."""
    base = Path(__file__).parent.parent.parent / "data"
    return {
        "raw": base / "raw",
        "validated": base / "validated",
        "processed": base / "processed",
        "quarantine": base / "quarantine",
    }


def ensure_data_dirs():
    """Create data directories if they don't exist."""
    for name, path in get_data_dirs().items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")


# =============================================================================
# File Patterns and Mappings
# =============================================================================

# Expected file patterns for Bloomberg exports
BLOOMBERG_FILE_PATTERNS = {
    "fx_spots": "fx_spots_{date}.csv",
    "fx_forwards": "fx_forwards_{date}.csv", 
    "commodities": "commodities_{date}.csv",
    "macro": "macro_{date}.csv",
}

# Column mappings from Bloomberg to our standard format
BLOOMBERG_COLUMN_MAPPINGS = {
    # Bloomberg field -> Standard field
    "Ticker": "ticker",
    "Date": "date",
    "Time": "time",
    "PX_LAST": "PX_LAST",
    "PX_HIGH": "PX_HIGH",
    "PX_LOW": "PX_LOW",
    "PX_OPEN": "PX_OPEN",
    "PX_VOLUME": "PX_VOLUME",
    "OPEN_INT": "OPEN_INT",
    "FWD_PTS_1M": "FWD_POINTS_1M",
    "FWD_PTS_3M": "FWD_POINTS_3M",
    "IVOL_1M": "IMPLIED_VOL_1M",
    "IVOL_3M": "IMPLIED_VOL_3M",
}

# Ticker mappings (Bloomberg ticker -> Our ticker)
TICKER_MAPPINGS = {
    "EURUSD Curncy": "EURUSD",
    "USDJPY Curncy": "USDJPY",
    "GBPUSD Curncy": "GBPUSD",
    "AUDUSD Curncy": "AUDUSD",
    "CL1 Comdty": "CL1",
    "GC1 Comdty": "GC1",
    "HG1 Comdty": "HG1",
    "VIX Index": "VIX",
    "DXY Index": "DXY",
    "XBTUSD BGN Curncy": "BTCUSD",
}


# =============================================================================
# Data Loading Functions
# =============================================================================

def find_bloomberg_file(
    data_date: datetime,
    file_type: str,
    raw_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Find Bloomberg export file for a given date and type.
    
    Args:
        data_date: Date to load data for
        file_type: Type of file (fx_spots, commodities, etc.)
        raw_dir: Raw data directory (uses default if not specified)
    
    Returns:
        Path to file if found, None otherwise
    """
    if raw_dir is None:
        raw_dir = get_data_dirs()["raw"]
    
    date_str = data_date.strftime("%Y-%m-%d")
    date_folder = raw_dir / "bloomberg" / date_str
    
    if not date_folder.exists():
        logger.warning(f"No Bloomberg data folder for {date_str}")
        return None
    
    # Try expected pattern
    pattern = BLOOMBERG_FILE_PATTERNS.get(file_type)
    if pattern:
        expected_file = date_folder / pattern.format(date=date_str)
        if expected_file.exists():
            return expected_file
    
    # Try alternate patterns
    for f in date_folder.iterdir():
        if file_type.lower() in f.name.lower() and f.suffix == ".csv":
            return f
    
    logger.warning(f"No {file_type} file found for {date_str}")
    return None


def load_csv_with_retry(
    file_path: Path,
    encodings: List[str] = ["utf-8", "latin-1", "cp1252"]
) -> pd.DataFrame:
    """
    Load CSV with multiple encoding attempts.
    
    Args:
        file_path: Path to CSV file
        encodings: List of encodings to try
    
    Returns:
        Loaded DataFrame
    
    Raises:
        ValueError: If file cannot be loaded with any encoding
    """
    last_error = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.debug(f"Loaded {file_path} with encoding {encoding}")
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
    
    raise ValueError(f"Could not load {file_path} with any encoding: {last_error}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names from Bloomberg format.
    
    Args:
        df: Raw DataFrame from Bloomberg
    
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    
    # Apply column mappings
    rename_map = {}
    for old_name in df.columns:
        # Check exact match first
        if old_name in BLOOMBERG_COLUMN_MAPPINGS:
            rename_map[old_name] = BLOOMBERG_COLUMN_MAPPINGS[old_name]
        else:
            # Check case-insensitive match
            for bloomberg_col, standard_col in BLOOMBERG_COLUMN_MAPPINGS.items():
                if old_name.upper() == bloomberg_col.upper():
                    rename_map[old_name] = standard_col
                    break
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


def standardize_tickers(df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
    """
    Standardize ticker symbols from Bloomberg format.
    
    Args:
        df: DataFrame with Bloomberg tickers
        ticker_col: Name of ticker column
    
    Returns:
        DataFrame with standardized tickers
    """
    if ticker_col not in df.columns:
        return df
    
    df = df.copy()
    df[ticker_col] = df[ticker_col].map(
        lambda x: TICKER_MAPPINGS.get(x, x)
    )
    
    return df


def create_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create timestamp column from date and time columns.
    
    Args:
        df: DataFrame with date and optionally time columns
    
    Returns:
        DataFrame with timestamp column
    """
    df = df.copy()
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    
    if "date" in df.columns:
        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str)
            )
        else:
            df["timestamp"] = pd.to_datetime(df["date"])
    
    return df


def load_bloomberg_file(
    file_path: Path,
    file_type: str
) -> pd.DataFrame:
    """
    Load and preprocess a single Bloomberg file.
    
    Args:
        file_path: Path to Bloomberg CSV
        file_type: Type of file for context
    
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading Bloomberg {file_type} from {file_path}")
    
    # Load CSV
    df = load_csv_with_retry(file_path)
    
    # Standardize
    df = standardize_columns(df)
    df = standardize_tickers(df)
    df = create_timestamp(df)
    
    logger.info(f"Loaded {len(df)} rows from {file_path.name}")
    
    return df


# =============================================================================
# Main Ingestion Functions
# =============================================================================

def load_bloomberg_data(
    data_date: datetime,
    raw_dir: Optional[Path] = None,
    validate: bool = True,
    as_of_date: Optional[datetime] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all Bloomberg data for a given date.
    
    Args:
        data_date: Date to load data for
        raw_dir: Raw data directory
        validate: Whether to validate loaded data
        as_of_date: Reference date for validation (defaults to now)
    
    Returns:
        Dictionary mapping data type to DataFrame
    """
    ensure_data_dirs()
    
    if as_of_date is None:
        as_of_date = datetime.now()
    
    data = {}
    
    for file_type in BLOOMBERG_FILE_PATTERNS.keys():
        file_path = find_bloomberg_file(data_date, file_type, raw_dir)
        
        if file_path is None:
            logger.warning(f"Skipping {file_type} - file not found")
            continue
        
        try:
            df = load_bloomberg_file(file_path, file_type)
            data[file_type] = df
        except Exception as e:
            logger.error(f"Error loading {file_type}: {e}")
            continue
    
    logger.info(f"Loaded {len(data)} data files for {data_date.strftime('%Y-%m-%d')}")
    
    return data


def split_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> Dict[str, pd.DataFrame]:
    """
    Split a DataFrame by ticker symbol.
    
    Args:
        df: DataFrame with multiple tickers
        ticker_col: Name of ticker column
    
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    if ticker_col not in df.columns:
        return {"unknown": df}
    
    result = {}
    for ticker in df[ticker_col].unique():
        result[ticker] = df[df[ticker_col] == ticker].copy()
    
    return result


def ingest_and_validate(
    data_date: datetime,
    raw_dir: Optional[Path] = None,
    as_of_date: Optional[datetime] = None,
    quarantine_failures: bool = True
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, ValidationResult]]:
    """
    Load, validate, and process Bloomberg data.
    
    This is the main entry point for data ingestion.
    
    Args:
        data_date: Date to load data for
        raw_dir: Raw data directory
        as_of_date: Reference date for point-in-time validation
        quarantine_failures: Whether to quarantine failed data
    
    Returns:
        Tuple of (validated_data, validation_results)
    """
    ensure_data_dirs()
    dirs = get_data_dirs()
    
    if as_of_date is None:
        as_of_date = datetime.now()
    
    # Load raw data
    raw_data = load_bloomberg_data(data_date, raw_dir, validate=False)
    
    validated_data = {}
    validation_results = {}
    
    # Process each data type
    for data_type, df in raw_data.items():
        logger.info(f"Processing {data_type}...")
        
        # Split by ticker if applicable
        ticker_data = split_by_ticker(df)
        
        for ticker, ticker_df in ticker_data.items():
            # Create point-in-time snapshot
            ticker_df = create_point_in_time_snapshot(ticker_df, as_of_date)
            
            # Validate
            result = validate_dataframe(ticker_df, ticker, as_of_date)
            validation_results[ticker] = result
            
            if result.passed:
                validated_data[ticker] = ticker_df
                logger.info(f"✓ {ticker}: Validation passed")
            else:
                logger.warning(f"✗ {ticker}: Validation failed - {len(result.issues)} issues")
                
                if quarantine_failures:
                    quarantine_bad_data(ticker_df, ticker, result, dirs["quarantine"])
    
    # Summary
    passed = sum(1 for r in validation_results.values() if r.passed)
    failed = len(validation_results) - passed
    logger.info(f"Ingestion complete: {passed} passed, {failed} failed")
    
    return validated_data, validation_results


def save_validated_data(
    data: Dict[str, pd.DataFrame],
    data_date: datetime,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    Save validated data to the processed folder.
    
    Args:
        data: Dictionary mapping ticker to DataFrame
        data_date: Date of the data
        output_dir: Output directory (uses default if not specified)
    
    Returns:
        List of saved file paths
    """
    if output_dir is None:
        output_dir = get_data_dirs()["processed"]
    
    date_str = data_date.strftime("%Y-%m-%d")
    date_folder = output_dir / date_str
    date_folder.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for ticker, df in data.items():
        file_path = date_folder / f"{ticker}_{date_str}.parquet"
        df.to_parquet(file_path, index=False)
        saved_files.append(file_path)
        logger.debug(f"Saved {ticker} to {file_path}")
    
    logger.info(f"Saved {len(saved_files)} files to {date_folder}")
    
    return saved_files


def copy_raw_to_archive(
    data_date: datetime,
    raw_dir: Optional[Path] = None
):
    """
    Archive raw data files (immutable backup).
    
    Args:
        data_date: Date of data to archive
        raw_dir: Raw data directory
    """
    if raw_dir is None:
        raw_dir = get_data_dirs()["raw"]
    
    date_str = data_date.strftime("%Y-%m-%d")
    source_dir = raw_dir / "bloomberg" / date_str
    archive_dir = raw_dir / "archive" / date_str
    
    if source_dir.exists():
        shutil.copytree(source_dir, archive_dir, dirs_exist_ok=True)
        logger.info(f"Archived raw data for {date_str}")


# =============================================================================
# Utility Functions
# =============================================================================

def get_available_dates(raw_dir: Optional[Path] = None) -> List[datetime]:
    """
    Get list of dates with available Bloomberg data.
    
    Args:
        raw_dir: Raw data directory
    
    Returns:
        List of dates with data
    """
    if raw_dir is None:
        raw_dir = get_data_dirs()["raw"]
    
    bloomberg_dir = raw_dir / "bloomberg"
    
    if not bloomberg_dir.exists():
        return []
    
    dates = []
    for folder in bloomberg_dir.iterdir():
        if folder.is_dir():
            try:
                date = datetime.strptime(folder.name, "%Y-%m-%d")
                dates.append(date)
            except ValueError:
                continue
    
    return sorted(dates)


def load_processed_data(
    ticker: str,
    start_date: datetime,
    end_date: Optional[datetime] = None,
    processed_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load processed data for a ticker over a date range.
    
    Args:
        ticker: Instrument ticker
        start_date: Start date
        end_date: End date (defaults to today)
        processed_dir: Processed data directory
    
    Returns:
        Combined DataFrame for the date range
    """
    if processed_dir is None:
        processed_dir = get_data_dirs()["processed"]
    
    if end_date is None:
        end_date = datetime.now()
    
    dfs = []
    current = start_date
    
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        file_path = processed_dir / date_str / f"{ticker}_{date_str}.parquet"
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dfs.append(df)
        
        current += timedelta(days=1)
    
    if not dfs:
        logger.warning(f"No processed data found for {ticker} from {start_date} to {end_date}")
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates()
    
    logger.info(f"Loaded {len(combined)} rows for {ticker}")
    
    return combined


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for data ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest Bloomberg data")
    parser.add_argument(
        "--date",
        type=str,
        help="Date to ingest (YYYY-MM-DD), defaults to yesterday"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all available dates"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without saving"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.add("logs/ingestion_{time}.log", rotation="1 day")
    
    if args.all:
        dates = get_available_dates()
        logger.info(f"Found {len(dates)} dates to ingest")
    elif args.date:
        dates = [datetime.strptime(args.date, "%Y-%m-%d")]
    else:
        dates = [datetime.now() - timedelta(days=1)]
    
    for date in dates:
        logger.info(f"Ingesting data for {date.strftime('%Y-%m-%d')}")
        
        validated_data, results = ingest_and_validate(date)
        
        if not args.dry_run and validated_data:
            save_validated_data(validated_data, date)
            copy_raw_to_archive(date)
    
    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
