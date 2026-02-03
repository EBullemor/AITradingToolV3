"""
Data Quarantine Module

Handles moving invalid or suspicious data to quarantine for manual review.
Keeps audit trail of why data was quarantined.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from .quality_checks import ValidationResult, ValidationIssue, ValidationSeverity


class QuarantineManager:
    """Manages data quarantine operations."""
    
    def __init__(
        self,
        quarantine_dir: Optional[Path] = None,
        max_age_days: int = 30
    ):
        """
        Initialize quarantine manager.
        
        Args:
            quarantine_dir: Directory for quarantined data
            max_age_days: How long to keep quarantined data
        """
        if quarantine_dir is None:
            quarantine_dir = Path(__file__).parent.parent.parent.parent / "data" / "quarantine"
        
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        
        logger.debug(f"Quarantine manager initialized at {self.quarantine_dir}")
    
    def quarantine_dataframe(
        self,
        df: pd.DataFrame,
        ticker: str,
        validation_result: ValidationResult,
        source_file: Optional[Path] = None
    ) -> Path:
        """
        Move a DataFrame to quarantine with metadata.
        
        Args:
            df: DataFrame to quarantine
            ticker: Instrument ticker
            validation_result: Validation result explaining why
            source_file: Original source file (if applicable)
        
        Returns:
            Path to quarantined data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_name = f"{ticker}_{timestamp}"
        quarantine_path = self.quarantine_dir / quarantine_name
        quarantine_path.mkdir(parents=True, exist_ok=True)
        
        # Save the data
        data_file = quarantine_path / "data.csv"
        df.to_csv(data_file, index=False)
        
        # Save validation report
        report_file = quarantine_path / "validation_report.json"
        with open(report_file, "w") as f:
            json.dump(validation_result.to_dict(), f, indent=2, default=str)
        
        # Save metadata
        metadata = {
            "ticker": ticker,
            "quarantined_at": datetime.now().isoformat(),
            "row_count": len(df),
            "source_file": str(source_file) if source_file else None,
            "validation_status": validation_result.status.value,
            "issue_count": len(validation_result.issues),
            "critical_issues": [
                i.to_dict() for i in validation_result.issues 
                if i.severity == ValidationSeverity.CRITICAL
            ],
            "error_issues": [
                i.to_dict() for i in validation_result.issues 
                if i.severity == ValidationSeverity.ERROR
            ],
        }
        
        metadata_file = quarantine_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.warning(
            f"Quarantined {ticker} data ({len(df)} rows) to {quarantine_path}. "
            f"Reason: {validation_result.status.value} with {len(validation_result.issues)} issues"
        )
        
        return quarantine_path
    
    def quarantine_file(
        self,
        file_path: Path,
        ticker: str,
        reason: str,
        issues: Optional[List[ValidationIssue]] = None
    ) -> Path:
        """
        Move a file to quarantine.
        
        Args:
            file_path: File to quarantine
            ticker: Instrument ticker
            reason: Reason for quarantine
            issues: Validation issues (if any)
        
        Returns:
            Path to quarantined file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_name = f"{ticker}_{timestamp}"
        quarantine_path = self.quarantine_dir / quarantine_name
        quarantine_path.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        dest_file = quarantine_path / file_path.name
        shutil.copy2(file_path, dest_file)
        
        # Save metadata
        metadata = {
            "ticker": ticker,
            "quarantined_at": datetime.now().isoformat(),
            "original_path": str(file_path),
            "reason": reason,
            "issues": [i.to_dict() for i in (issues or [])],
        }
        
        metadata_file = quarantine_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.warning(f"Quarantined file {file_path.name} to {quarantine_path}. Reason: {reason}")
        
        return quarantine_path
    
    def quarantine_rows(
        self,
        df: pd.DataFrame,
        row_indices: List[int],
        ticker: str,
        reason: str,
        issue: Optional[ValidationIssue] = None
    ) -> pd.DataFrame:
        """
        Quarantine specific rows from a DataFrame.
        
        Args:
            df: Original DataFrame
            row_indices: Indices of rows to quarantine
            ticker: Instrument ticker
            reason: Reason for quarantine
            issue: Related validation issue
        
        Returns:
            DataFrame with quarantined rows removed
        """
        if not row_indices:
            return df
        
        # Extract rows to quarantine
        quarantined_df = df.loc[row_indices].copy()
        
        # Create quarantine entry
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_name = f"{ticker}_rows_{timestamp}"
        quarantine_path = self.quarantine_dir / quarantine_name
        quarantine_path.mkdir(parents=True, exist_ok=True)
        
        # Save quarantined rows
        data_file = quarantine_path / "quarantined_rows.csv"
        quarantined_df.to_csv(data_file, index=True)
        
        # Save metadata
        metadata = {
            "ticker": ticker,
            "quarantined_at": datetime.now().isoformat(),
            "row_count": len(quarantined_df),
            "original_row_indices": row_indices,
            "reason": reason,
            "issue": issue.to_dict() if issue else None,
        }
        
        metadata_file = quarantine_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Return clean DataFrame
        clean_df = df.drop(index=row_indices)
        
        logger.info(
            f"Quarantined {len(row_indices)} rows from {ticker}. "
            f"Remaining: {len(clean_df)} rows. Reason: {reason}"
        )
        
        return clean_df
    
    def list_quarantined(self) -> List[Dict]:
        """
        List all quarantined data.
        
        Returns:
            List of quarantine metadata dictionaries
        """
        quarantined = []
        
        for item_path in self.quarantine_dir.iterdir():
            if not item_path.is_dir():
                continue
            
            metadata_file = item_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    metadata["quarantine_path"] = str(item_path)
                    quarantined.append(metadata)
        
        # Sort by quarantine date (newest first)
        quarantined.sort(
            key=lambda x: x.get("quarantined_at", ""),
            reverse=True
        )
        
        return quarantined
    
    def restore_from_quarantine(
        self,
        quarantine_path: Union[str, Path],
        destination: Path
    ) -> bool:
        """
        Restore data from quarantine.
        
        Args:
            quarantine_path: Path to quarantined data
            destination: Where to restore to
        
        Returns:
            True if successful
        """
        quarantine_path = Path(quarantine_path)
        
        if not quarantine_path.exists():
            logger.error(f"Quarantine path not found: {quarantine_path}")
            return False
        
        # Find data file
        data_file = quarantine_path / "data.csv"
        if not data_file.exists():
            data_file = quarantine_path / "quarantined_rows.csv"
        
        if not data_file.exists():
            # Look for original file
            for f in quarantine_path.iterdir():
                if f.suffix in [".csv", ".parquet", ".json"]:
                    data_file = f
                    break
        
        if not data_file.exists():
            logger.error(f"No data file found in quarantine: {quarantine_path}")
            return False
        
        # Copy to destination
        shutil.copy2(data_file, destination)
        logger.info(f"Restored {data_file.name} from quarantine to {destination}")
        
        return True
    
    def cleanup_old_quarantine(self, max_age_days: Optional[int] = None) -> int:
        """
        Remove quarantined data older than max_age_days.
        
        Args:
            max_age_days: Maximum age to keep (uses instance default if not specified)
        
        Returns:
            Number of items removed
        """
        max_age_days = max_age_days or self.max_age_days
        cutoff = datetime.now() - pd.Timedelta(days=max_age_days)
        removed = 0
        
        for item_path in self.quarantine_dir.iterdir():
            if not item_path.is_dir():
                continue
            
            metadata_file = item_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                quarantined_at = datetime.fromisoformat(metadata.get("quarantined_at", ""))
                
                if quarantined_at < cutoff:
                    shutil.rmtree(item_path)
                    removed += 1
                    logger.info(f"Removed old quarantine: {item_path.name}")
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old quarantine items (older than {max_age_days} days)")
        
        return removed


# Convenience function
def quarantine_bad_data(
    df: pd.DataFrame,
    ticker: str,
    validation_result: ValidationResult,
    quarantine_dir: Optional[Path] = None
) -> Path:
    """
    Convenience function to quarantine invalid data.
    
    Args:
        df: DataFrame to quarantine
        ticker: Instrument ticker
        validation_result: Why data is being quarantined
        quarantine_dir: Quarantine directory (optional)
    
    Returns:
        Path to quarantined data
    """
    manager = QuarantineManager(quarantine_dir)
    return manager.quarantine_dataframe(df, ticker, validation_result)
