"""
Forward-Looking Bias Prevention

Checks to ensure no look-ahead bias in the data pipeline.
This is critical for backtesting and live trading consistency.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import pandas as pd
from loguru import logger

from .quality_checks import ValidationIssue, ValidationSeverity


@dataclass
class BiasCheckConfig:
    """Configuration for bias checks."""
    max_future_tolerance_seconds: int = 60  # Allow 1 minute clock drift
    check_settlement_times: bool = True
    check_announcement_dates: bool = True
    strict_mode: bool = False  # If True, any future data is CRITICAL


def check_no_future_data(
    df: pd.DataFrame,
    as_of_date: datetime,
    timestamp_col: str = "timestamp",
    config: Optional[BiasCheckConfig] = None
) -> List[ValidationIssue]:
    """
    Ensure no data points have timestamps after the as_of_date.
    
    This is the most critical bias check - using future data in backtests
    will give unrealistic results.
    
    Args:
        df: DataFrame to check
        as_of_date: The "current" date - no data should be from after this
        timestamp_col: Name of timestamp column
        config: Bias check configuration
    
    Returns:
        List of validation issues found
    """
    issues = []
    config = config or BiasCheckConfig()
    
    if timestamp_col not in df.columns:
        return issues
    
    timestamps = pd.to_datetime(df[timestamp_col])
    
    # Allow small tolerance for clock drift
    tolerance = timedelta(seconds=config.max_future_tolerance_seconds)
    cutoff = as_of_date + tolerance
    
    # Find future data
    future_mask = timestamps > cutoff
    future_count = future_mask.sum()
    
    if future_count > 0:
        future_indices = df[future_mask].index.tolist()
        future_timestamps = timestamps[future_mask].tolist()
        
        severity = ValidationSeverity.CRITICAL if config.strict_mode else ValidationSeverity.ERROR
        
        issues.append(ValidationIssue(
            check_name="forward_looking_bias",
            severity=severity,
            message=f"CRITICAL: Found {future_count} data points from the future (after {as_of_date})",
            details={
                "as_of_date": as_of_date.isoformat(),
                "future_count": future_count,
                "earliest_future": min(future_timestamps).isoformat() if future_timestamps else None,
                "latest_future": max(future_timestamps).isoformat() if future_timestamps else None,
                "tolerance_seconds": config.max_future_tolerance_seconds,
            },
            row_indices=future_indices
        ))
        
        logger.error(
            f"FORWARD-LOOKING BIAS DETECTED: {future_count} data points "
            f"are from after {as_of_date}"
        )
    
    return issues


def check_point_in_time_consistency(
    df: pd.DataFrame,
    as_of_date: datetime,
    value_date_col: str = "value_date",
    timestamp_col: str = "timestamp",
    config: Optional[BiasCheckConfig] = None
) -> List[ValidationIssue]:
    """
    Check that data reflects point-in-time values, not revised values.
    
    Economic data is often revised after initial release. We need to ensure
    we're using the value that was available at the time, not later revisions.
    
    Args:
        df: DataFrame to check
        as_of_date: Reference date
        value_date_col: Column with the economic value date (if different from timestamp)
        timestamp_col: Column with when the data was recorded/published
        config: Bias check configuration
    
    Returns:
        List of validation issues
    """
    issues = []
    config = config or BiasCheckConfig()
    
    if value_date_col not in df.columns:
        return issues  # Column doesn't exist, skip check
    
    if timestamp_col not in df.columns:
        return issues
    
    value_dates = pd.to_datetime(df[value_date_col])
    record_dates = pd.to_datetime(df[timestamp_col])
    
    # Check if any records show data before it should have been available
    # e.g., GDP data for Q1 shouldn't be recorded before Q1 ends
    suspicious = record_dates < value_dates
    
    if suspicious.any():
        suspicious_indices = df[suspicious].index.tolist()
        
        issues.append(ValidationIssue(
            check_name="point_in_time_consistency",
            severity=ValidationSeverity.WARNING,
            message=f"Found {suspicious.sum()} records where record date is before value date (possible revision)",
            details={
                "suspicious_count": int(suspicious.sum()),
                "note": "These may be using revised data instead of point-in-time data"
            },
            row_indices=suspicious_indices
        ))
    
    return issues


def check_settlement_vs_realtime(
    df: pd.DataFrame,
    ticker: str,
    price_col: str = "PX_LAST",
    timestamp_col: str = "timestamp",
    config: Optional[BiasCheckConfig] = None
) -> List[ValidationIssue]:
    """
    Check for potential settlement price vs real-time price issues.
    
    Settlement prices for commodities are determined at specific times.
    Using settlement prices in intraday backtests creates bias.
    
    Args:
        df: DataFrame to check
        ticker: Instrument ticker
        price_col: Price column name
        timestamp_col: Timestamp column name
        config: Bias check configuration
    
    Returns:
        List of validation issues
    """
    issues = []
    config = config or BiasCheckConfig()
    
    if not config.check_settlement_times:
        return issues
    
    # Settlement times for common instruments (ET)
    settlement_times = {
        "CL": "14:30",  # WTI Crude
        "CL1": "14:30",
        "GC": "13:30",  # Gold
        "GC1": "13:30",
        "HG": "13:00",  # Copper
        "HG1": "13:00",
    }
    
    ticker_upper = ticker.upper()
    if ticker_upper not in settlement_times:
        return issues  # Not a settlement-based instrument
    
    if timestamp_col not in df.columns:
        return issues
    
    timestamps = pd.to_datetime(df[timestamp_col])
    settlement_time = settlement_times[ticker_upper]
    
    # Check if all timestamps are at settlement time (suspicious for daily data)
    times_str = timestamps.dt.strftime("%H:%M")
    all_settlement = (times_str == settlement_time).all()
    
    if all_settlement and len(df) > 1:
        issues.append(ValidationIssue(
            check_name="settlement_prices",
            severity=ValidationSeverity.INFO,
            message=f"All prices appear to be settlement prices (timestamp: {settlement_time})",
            details={
                "ticker": ticker,
                "settlement_time": settlement_time,
                "note": "This is expected for EOD data but problematic for intraday"
            }
        ))
    
    return issues


def check_announcement_timing(
    df: pd.DataFrame,
    announcement_col: str = "announcement_date",
    effective_col: str = "effective_date",
    timestamp_col: str = "timestamp",
    config: Optional[BiasCheckConfig] = None
) -> List[ValidationIssue]:
    """
    Check that data uses announcement dates, not effective dates.
    
    For economic data, there's often a difference between:
    - Announcement date: When the data was released publicly
    - Effective date: When the data applies to (e.g., "January unemployment")
    
    Using effective dates creates look-ahead bias.
    
    Args:
        df: DataFrame to check
        announcement_col: Column with announcement date
        effective_col: Column with effective/reference date
        timestamp_col: Column with record timestamp
        config: Bias check configuration
    
    Returns:
        List of validation issues
    """
    issues = []
    config = config or BiasCheckConfig()
    
    if not config.check_announcement_dates:
        return issues
    
    if announcement_col not in df.columns or effective_col not in df.columns:
        return issues
    
    announcement_dates = pd.to_datetime(df[announcement_col])
    effective_dates = pd.to_datetime(df[effective_col])
    
    # Check if any records use effective date before announcement
    if timestamp_col in df.columns:
        record_dates = pd.to_datetime(df[timestamp_col])
        
        # Data shouldn't be recorded before it was announced
        early_records = record_dates < announcement_dates
        
        if early_records.any():
            issues.append(ValidationIssue(
                check_name="announcement_timing",
                severity=ValidationSeverity.ERROR,
                message=f"Found {early_records.sum()} records timestamped before their announcement date",
                details={
                    "early_count": int(early_records.sum()),
                    "note": "This is a strong indicator of look-ahead bias"
                },
                row_indices=df[early_records].index.tolist()
            ))
    
    return issues


def check_data_revisions(
    df: pd.DataFrame,
    value_col: str,
    timestamp_col: str = "timestamp",
    revision_col: Optional[str] = "revision_number",
    config: Optional[BiasCheckConfig] = None
) -> List[ValidationIssue]:
    """
    Check for and flag data revisions.
    
    Economic data gets revised. We should only use the version that was
    available at the time of the original decision.
    
    Args:
        df: DataFrame to check
        value_col: Column with the data value
        timestamp_col: Timestamp column
        revision_col: Column indicating revision number (if available)
        config: Bias check configuration
    
    Returns:
        List of validation issues
    """
    issues = []
    
    if revision_col and revision_col in df.columns:
        revisions = df[df[revision_col] > 0]
        
        if len(revisions) > 0:
            issues.append(ValidationIssue(
                check_name="data_revisions",
                severity=ValidationSeverity.INFO,
                message=f"Data contains {len(revisions)} revised values",
                details={
                    "revision_count": len(revisions),
                    "note": "Ensure backtests use original (unrevised) values"
                },
                row_indices=revisions.index.tolist()
            ))
    
    return issues


def run_all_bias_checks(
    df: pd.DataFrame,
    ticker: str,
    as_of_date: datetime,
    timestamp_col: str = "timestamp",
    config: Optional[BiasCheckConfig] = None
) -> List[ValidationIssue]:
    """
    Run all bias prevention checks on a dataframe.
    
    Args:
        df: DataFrame to check
        ticker: Instrument ticker
        as_of_date: Reference date (no data should be from after this)
        timestamp_col: Timestamp column name
        config: Bias check configuration
    
    Returns:
        List of all bias-related issues found
    """
    config = config or BiasCheckConfig()
    all_issues = []
    
    # Critical: No future data
    all_issues.extend(check_no_future_data(df, as_of_date, timestamp_col, config))
    
    # Settlement vs real-time
    all_issues.extend(check_settlement_vs_realtime(df, ticker, "PX_LAST", timestamp_col, config))
    
    # Point-in-time consistency (if columns exist)
    all_issues.extend(check_point_in_time_consistency(df, as_of_date, config=config))
    
    # Announcement timing (if columns exist)
    all_issues.extend(check_announcement_timing(df, config=config))
    
    return all_issues


def create_point_in_time_snapshot(
    df: pd.DataFrame,
    as_of_date: datetime,
    timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Create a point-in-time snapshot of data as it would have been seen
    on a specific date.
    
    This filters out any data that wasn't available at the as_of_date.
    
    Args:
        df: Full DataFrame
        as_of_date: The date to snapshot as of
        timestamp_col: Timestamp column name
    
    Returns:
        DataFrame containing only data available at as_of_date
    """
    if timestamp_col not in df.columns:
        logger.warning(f"Cannot create snapshot - column '{timestamp_col}' not found")
        return df
    
    timestamps = pd.to_datetime(df[timestamp_col])
    mask = timestamps <= as_of_date
    
    snapshot = df[mask].copy()
    
    logger.info(
        f"Created point-in-time snapshot: {len(snapshot)}/{len(df)} rows "
        f"as of {as_of_date}"
    )
    
    return snapshot
