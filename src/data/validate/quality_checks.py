"""
Data Quality Checks

Validation functions for checking data completeness, freshness, and outliers.
Returns structured results that can be logged and acted upon.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from .schema import (
    InstrumentSchema,
    InstrumentType,
    FieldSchema,
    get_schema,
    get_required_fields,
    INSTRUMENT_SCHEMAS,
)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(Enum):
    """Overall validation status."""
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    check_name: str
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    row_indices: Optional[List[int]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "check_name": self.check_name,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "row_count": len(self.row_indices) if self.row_indices else 0,
        }


@dataclass
class ValidationResult:
    """Result of validating a dataset."""
    ticker: str
    timestamp: datetime
    status: ValidationStatus
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        """Check if validation passed (possibly with warnings)."""
        return self.status in [ValidationStatus.PASSED, ValidationStatus.PASSED_WITH_WARNINGS]
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors or critical issues."""
        return any(
            i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
            for i in self.issues
        )
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "issues": [i.to_dict() for i in self.issues],
            "stats": self.stats,
            "summary": {
                "total_issues": len(self.issues),
                "errors": sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR),
                "warnings": sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING),
                "critical": sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL),
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


def load_validation_config(config_path: Optional[Path] = None) -> Dict:
    """Load validation configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "data_quality_rules.yaml"
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_default_config() -> Dict:
    """Return default validation configuration."""
    return {
        "global": {
            "missing_value_threshold": 0.01,
            "stale_data_hours": 24,
            "quarantine_on_failure": True,
            "halt_on_critical_failure": True,
        },
        "outliers": {
            "auto_remove": False,
            "max_daily_return_std": 5.0,
            "max_price_change_pct": 10.0,
        },
    }


# =============================================================================
# Completeness Checks
# =============================================================================

def check_required_fields(
    df: pd.DataFrame,
    ticker: str,
    config: Optional[Dict] = None
) -> List[ValidationIssue]:
    """Check that all required fields are present."""
    issues = []
    required = get_required_fields(ticker)
    
    missing_fields = [f for f in required if f not in df.columns]
    
    if missing_fields:
        issues.append(ValidationIssue(
            check_name="required_fields",
            severity=ValidationSeverity.CRITICAL,
            message=f"Missing required fields: {missing_fields}",
            details={"missing_fields": missing_fields, "ticker": ticker}
        ))
    
    return issues


def check_missing_values(
    df: pd.DataFrame,
    ticker: str,
    config: Optional[Dict] = None
) -> List[ValidationIssue]:
    """Check for missing values in required fields."""
    issues = []
    config = config or get_default_config()
    threshold = config.get("global", {}).get("missing_value_threshold", 0.01)
    
    schema = get_schema(ticker)
    if schema is None:
        return issues
    
    for field_schema in schema.fields:
        if field_schema.name not in df.columns:
            continue
            
        if not field_schema.required:
            continue
        
        null_count = df[field_schema.name].isna().sum()
        null_pct = null_count / len(df) if len(df) > 0 else 0
        
        if null_count > 0:
            severity = (
                ValidationSeverity.ERROR if null_pct > threshold
                else ValidationSeverity.WARNING
            )
            
            null_indices = df[df[field_schema.name].isna()].index.tolist()
            
            issues.append(ValidationIssue(
                check_name="missing_values",
                severity=severity,
                message=f"Field '{field_schema.name}' has {null_count} missing values ({null_pct:.2%})",
                details={
                    "field": field_schema.name,
                    "null_count": null_count,
                    "null_pct": null_pct,
                    "threshold": threshold,
                },
                row_indices=null_indices
            ))
    
    return issues


def check_row_count(
    df: pd.DataFrame,
    ticker: str,
    min_rows: int = 10,
    config: Optional[Dict] = None
) -> List[ValidationIssue]:
    """Check minimum row count."""
    issues = []
    
    if len(df) < min_rows:
        issues.append(ValidationIssue(
            check_name="row_count",
            severity=ValidationSeverity.ERROR,
            message=f"Insufficient rows: {len(df)} < {min_rows} minimum",
            details={"row_count": len(df), "min_required": min_rows}
        ))
    
    return issues


def check_duplicate_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    config: Optional[Dict] = None
) -> List[ValidationIssue]:
    """Check for duplicate timestamps."""
    issues = []
    
    if timestamp_col not in df.columns:
        return issues
    
    duplicates = df[df.duplicated(subset=[timestamp_col], keep=False)]
    
    if len(duplicates) > 0:
        dup_indices = duplicates.index.tolist()
        issues.append(ValidationIssue(
            check_name="duplicate_timestamps",
            severity=ValidationSeverity.WARNING,
            message=f"Found {len(duplicates)} rows with duplicate timestamps",
            details={"duplicate_count": len(duplicates)},
            row_indices=dup_indices
        ))
    
    return issues


# =============================================================================
# Freshness Checks
# =============================================================================

def check_data_freshness(
    df: pd.DataFrame,
    ticker: str,
    as_of_date: Optional[datetime] = None,
    timestamp_col: str = "timestamp",
    config: Optional[Dict] = None
) -> List[ValidationIssue]:
    """Check if data is stale."""
    issues = []
    config = config or get_default_config()
    max_age_hours = config.get("global", {}).get("stale_data_hours", 24)
    
    if timestamp_col not in df.columns:
        return issues
    
    if as_of_date is None:
        as_of_date = datetime.now()
    
    # Get most recent timestamp
    latest_ts = pd.to_datetime(df[timestamp_col]).max()
    
    if pd.isna(latest_ts):
        issues.append(ValidationIssue(
            check_name="data_freshness",
            severity=ValidationSeverity.ERROR,
            message="Cannot determine data freshness - no valid timestamps",
            details={"ticker": ticker}
        ))
        return issues
    
    # Calculate age
    age = as_of_date - latest_ts.to_pydatetime()
    age_hours = age.total_seconds() / 3600
    
    if age_hours > max_age_hours:
        # Check if it's a weekend (might be expected)
        schema = get_schema(ticker)
        is_weekend = as_of_date.weekday() >= 5
        
        if schema and not schema.trades_weekends and is_weekend:
            severity = ValidationSeverity.INFO
        else:
            severity = ValidationSeverity.WARNING
        
        issues.append(ValidationIssue(
            check_name="data_freshness",
            severity=severity,
            message=f"Data is {age_hours:.1f} hours old (threshold: {max_age_hours}h)",
            details={
                "ticker": ticker,
                "latest_timestamp": latest_ts.isoformat(),
                "age_hours": age_hours,
                "threshold_hours": max_age_hours,
                "is_weekend": is_weekend,
            }
        ))
    
    return issues


def check_gaps_in_data(
    df: pd.DataFrame,
    ticker: str,
    timestamp_col: str = "timestamp",
    max_gap_days: int = 3,
    config: Optional[Dict] = None
) -> List[ValidationIssue]:
    """Check for unexpected gaps in time series data."""
    issues = []
    
    if timestamp_col not in df.columns:
        return issues
    
    if len(df) < 2:
        return issues
    
    schema = get_schema(ticker)
    
    # Sort by timestamp
    df_sorted = df.sort_values(timestamp_col)
    timestamps = pd.to_datetime(df_sorted[timestamp_col])
    
    # Calculate gaps
    gaps = timestamps.diff()
    
    # Find large gaps (more than max_gap_days)
    threshold = timedelta(days=max_gap_days)
    large_gaps = gaps[gaps > threshold]
    
    if len(large_gaps) > 0:
        gap_details = []
        for idx in large_gaps.index:
            pos = df_sorted.index.get_loc(idx)
            if pos > 0:
                prev_idx = df_sorted.index[pos - 1]
                gap_details.append({
                    "from": str(timestamps.loc[prev_idx]),
                    "to": str(timestamps.loc[idx]),
                    "gap_days": gaps.loc[idx].days,
                })
        
        issues.append(ValidationIssue(
            check_name="data_gaps",
            severity=ValidationSeverity.WARNING,
            message=f"Found {len(large_gaps)} gaps larger than {max_gap_days} days",
            details={"gaps": gap_details[:10]}  # Limit to first 10
        ))
    
    return issues


# =============================================================================
# Outlier Detection
# =============================================================================

def check_price_outliers(
    df: pd.DataFrame,
    ticker: str,
    price_col: str = "PX_LAST",
    config: Optional[Dict] = None
) -> List[ValidationIssue]:
    """Check for outlier prices based on returns."""
    issues = []
    config = config or get_default_config()
    
    outlier_config = config.get("outliers", {})
    max_std = outlier_config.get("max_daily_return_std", 5.0)
    max_pct = outlier_config.get("max_price_change_pct", 10.0) / 100
    
    if price_col not in df.columns:
        return issues
    
    if len(df) < 2:
        return issues
    
    prices = df[price_col].astype(float)
    
    # Calculate returns
    returns = prices.pct_change()
    
    # Check for extreme returns (> max_std standard deviations)
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    if std_ret > 0:
        z_scores = (returns - mean_ret) / std_ret
        extreme_z = abs(z_scores) > max_std
        
        if extreme_z.any():
            extreme_indices = df[extreme_z].index.tolist()
            issues.append(ValidationIssue(
                check_name="return_outliers",
                severity=ValidationSeverity.WARNING,
                message=f"Found {extreme_z.sum()} returns exceeding {max_std} std devs",
                details={
                    "outlier_count": int(extreme_z.sum()),
                    "max_std_threshold": max_std,
                    "actual_max_zscore": float(abs(z_scores).max()),
                },
                row_indices=extreme_indices
            ))
    
    # Check for extreme price jumps (> max_pct)
    extreme_pct = abs(returns) > max_pct
    
    if extreme_pct.any():
        extreme_indices = df[extreme_pct].index.tolist()
        issues.append(ValidationIssue(
            check_name="price_jump",
            severity=ValidationSeverity.WARNING,
            message=f"Found {extreme_pct.sum()} price changes exceeding {max_pct:.1%}",
            details={
                "jump_count": int(extreme_pct.sum()),
                "max_pct_threshold": max_pct,
                "actual_max_pct": float(abs(returns).max()),
            },
            row_indices=extreme_indices
        ))
    
    return issues


def check_negative_prices(
    df: pd.DataFrame,
    ticker: str,
    config: Optional[Dict] = None
) -> List[ValidationIssue]:
    """Check for negative prices (usually invalid except for rates)."""
    issues = []
    
    schema = get_schema(ticker)
    if schema is None:
        return issues
    
    for field_schema in schema.fields:
        if field_schema.name not in df.columns:
            continue
        
        if field_schema.allow_negative:
            continue
        
        if field_schema.min_value is not None and field_schema.min_value >= 0:
            negatives = df[df[field_schema.name] < 0]
            
            if len(negatives) > 0:
                issues.append(ValidationIssue(
                    check_name="negative_prices",
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_schema.name}' has {len(negatives)} negative values",
                    details={
                        "field": field_schema.name,
                        "negative_count": len(negatives),
                        "min_value": float(df[field_schema.name].min()),
                    },
                    row_indices=negatives.index.tolist()
                ))
    
    return issues


def check_value_bounds(
    df: pd.DataFrame,
    ticker: str,
    config: Optional[Dict] = None
) -> List[ValidationIssue]:
    """Check that values are within expected bounds."""
    issues = []
    
    schema = get_schema(ticker)
    if schema is None:
        return issues
    
    for field_schema in schema.fields:
        if field_schema.name not in df.columns:
            continue
        
        values = df[field_schema.name]
        
        # Check minimum bound
        if field_schema.min_value is not None:
            below_min = values < field_schema.min_value
            if below_min.any():
                issues.append(ValidationIssue(
                    check_name="value_bounds",
                    severity=ValidationSeverity.WARNING,
                    message=f"Field '{field_schema.name}' has {below_min.sum()} values below minimum {field_schema.min_value}",
                    details={
                        "field": field_schema.name,
                        "bound_type": "min",
                        "bound_value": field_schema.min_value,
                        "violation_count": int(below_min.sum()),
                    },
                    row_indices=df[below_min].index.tolist()
                ))
        
        # Check maximum bound
        if field_schema.max_value is not None:
            above_max = values > field_schema.max_value
            if above_max.any():
                issues.append(ValidationIssue(
                    check_name="value_bounds",
                    severity=ValidationSeverity.WARNING,
                    message=f"Field '{field_schema.name}' has {above_max.sum()} values above maximum {field_schema.max_value}",
                    details={
                        "field": field_schema.name,
                        "bound_type": "max",
                        "bound_value": field_schema.max_value,
                        "violation_count": int(above_max.sum()),
                    },
                    row_indices=df[above_max].index.tolist()
                ))
    
    return issues


# =============================================================================
# Main Validation Function
# =============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    ticker: str,
    as_of_date: Optional[datetime] = None,
    config: Optional[Dict] = None,
    timestamp_col: str = "timestamp"
) -> ValidationResult:
    """
    Run all validation checks on a dataframe.
    
    Args:
        df: DataFrame to validate
        ticker: Instrument ticker (e.g., "EURUSD", "CL1")
        as_of_date: Reference date for freshness checks
        config: Validation configuration (loaded from YAML if not provided)
        timestamp_col: Name of timestamp column
    
    Returns:
        ValidationResult with status and any issues found
    """
    if config is None:
        config = load_validation_config()
    
    if as_of_date is None:
        as_of_date = datetime.now()
    
    all_issues: List[ValidationIssue] = []
    
    # Run all checks
    logger.debug(f"Validating {ticker} with {len(df)} rows")
    
    # Completeness checks
    all_issues.extend(check_required_fields(df, ticker, config))
    all_issues.extend(check_missing_values(df, ticker, config))
    all_issues.extend(check_row_count(df, ticker, min_rows=10, config=config))
    all_issues.extend(check_duplicate_timestamps(df, timestamp_col, config))
    
    # Freshness checks
    all_issues.extend(check_data_freshness(df, ticker, as_of_date, timestamp_col, config))
    all_issues.extend(check_gaps_in_data(df, ticker, timestamp_col, config=config))
    
    # Outlier checks
    all_issues.extend(check_price_outliers(df, ticker, config=config))
    all_issues.extend(check_negative_prices(df, ticker, config))
    all_issues.extend(check_value_bounds(df, ticker, config))
    
    # Determine overall status
    has_critical = any(i.severity == ValidationSeverity.CRITICAL for i in all_issues)
    has_errors = any(i.severity == ValidationSeverity.ERROR for i in all_issues)
    has_warnings = any(i.severity == ValidationSeverity.WARNING for i in all_issues)
    
    if has_critical or has_errors:
        status = ValidationStatus.FAILED
    elif has_warnings:
        status = ValidationStatus.PASSED_WITH_WARNINGS
    else:
        status = ValidationStatus.PASSED
    
    # Calculate stats
    stats = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }
    
    if "PX_LAST" in df.columns:
        stats["price_stats"] = {
            "min": float(df["PX_LAST"].min()),
            "max": float(df["PX_LAST"].max()),
            "mean": float(df["PX_LAST"].mean()),
            "std": float(df["PX_LAST"].std()),
        }
    
    result = ValidationResult(
        ticker=ticker,
        timestamp=datetime.now(),
        status=status,
        issues=all_issues,
        stats=stats
    )
    
    # Log summary
    log_level = "error" if result.has_errors else ("warning" if result.has_warnings else "info")
    getattr(logger, log_level)(
        f"Validation {result.status.value} for {ticker}: "
        f"{len(all_issues)} issues ({result.to_dict()['summary']})"
    )
    
    return result


def validate_all_instruments(
    data: Dict[str, pd.DataFrame],
    as_of_date: Optional[datetime] = None,
    config: Optional[Dict] = None
) -> Dict[str, ValidationResult]:
    """
    Validate data for multiple instruments.
    
    Args:
        data: Dictionary mapping ticker to DataFrame
        as_of_date: Reference date for freshness checks
        config: Validation configuration
    
    Returns:
        Dictionary mapping ticker to ValidationResult
    """
    results = {}
    
    for ticker, df in data.items():
        results[ticker] = validate_dataframe(df, ticker, as_of_date, config)
    
    # Summary logging
    passed = sum(1 for r in results.values() if r.passed)
    failed = len(results) - passed
    logger.info(f"Validated {len(results)} instruments: {passed} passed, {failed} failed")
    
    return results
