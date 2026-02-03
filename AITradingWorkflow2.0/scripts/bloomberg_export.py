#!/usr/bin/env python3
"""
Bloomberg Export Workflow

Scripts and utilities for extracting data from Bloomberg Terminal.
Supports multiple export methods:
1. BLPAPI (automated, if available)
2. Excel DDE export (semi-automated)
3. Manual CSV export (MVP approach)

This module provides:
- Data specifications (what to export)
- File naming conventions
- Validation of exported files
- Scheduling helpers
"""

import os
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


# =============================================================================
# Data Specifications
# =============================================================================

@dataclass
class BloombergField:
    """Specification for a Bloomberg data field."""
    ticker: str           # Bloomberg ticker (e.g., "EURUSD Curncy")
    field: str            # Bloomberg field (e.g., "PX_LAST")
    name: str             # Human-readable name
    category: str         # fx, commodities, rates, macro
    frequency: str        # daily, intraday
    required: bool = True


# FX Spot Rates
FX_SPOTS = [
    BloombergField("EURUSD Curncy", "PX_LAST", "EURUSD", "fx", "daily"),
    BloombergField("USDJPY Curncy", "PX_LAST", "USDJPY", "fx", "daily"),
    BloombergField("GBPUSD Curncy", "PX_LAST", "GBPUSD", "fx", "daily"),
    BloombergField("AUDUSD Curncy", "PX_LAST", "AUDUSD", "fx", "daily"),
    BloombergField("USDCHF Curncy", "PX_LAST", "USDCHF", "fx", "daily"),
    BloombergField("USDCAD Curncy", "PX_LAST", "USDCAD", "fx", "daily"),
    BloombergField("NZDUSD Curncy", "PX_LAST", "NZDUSD", "fx", "daily"),
]

# FX Additional Fields
FX_EXTENDED = [
    BloombergField("EURUSD Curncy", "PX_HIGH", "EURUSD_HIGH", "fx", "daily"),
    BloombergField("EURUSD Curncy", "PX_LOW", "EURUSD_LOW", "fx", "daily"),
    BloombergField("EURUSD Curncy", "HIST_VOL_90D", "EURUSD_VOL90", "fx", "daily", False),
    BloombergField("USDJPY Curncy", "PX_HIGH", "USDJPY_HIGH", "fx", "daily"),
    BloombergField("USDJPY Curncy", "PX_LOW", "USDJPY_LOW", "fx", "daily"),
]

# Interest Rates (for carry calculation)
INTEREST_RATES = [
    BloombergField("US0012M Index", "PX_LAST", "USD_12M_RATE", "rates", "daily"),
    BloombergField("EUR012M Index", "PX_LAST", "EUR_12M_RATE", "rates", "daily"),
    BloombergField("BP0012M Index", "PX_LAST", "GBP_12M_RATE", "rates", "daily"),
    BloombergField("JY0012M Index", "PX_LAST", "JPY_12M_RATE", "rates", "daily"),
    BloombergField("AD0012M Index", "PX_LAST", "AUD_12M_RATE", "rates", "daily"),
]

# Commodities - Futures
COMMODITY_FUTURES = [
    BloombergField("CL1 Comdty", "PX_LAST", "WTI_FRONT", "commodities", "daily"),
    BloombergField("CL4 Comdty", "PX_LAST", "WTI_4TH", "commodities", "daily"),
    BloombergField("CO1 Comdty", "PX_LAST", "BRENT_FRONT", "commodities", "daily"),
    BloombergField("GC1 Comdty", "PX_LAST", "GOLD_FRONT", "commodities", "daily"),
    BloombergField("SI1 Comdty", "PX_LAST", "SILVER_FRONT", "commodities", "daily"),
    BloombergField("HG1 Comdty", "PX_LAST", "COPPER_FRONT", "commodities", "daily"),
    BloombergField("NG1 Comdty", "PX_LAST", "NATGAS_FRONT", "commodities", "daily"),
]

# Macro Indicators
MACRO_INDICATORS = [
    BloombergField("VIX Index", "PX_LAST", "VIX", "macro", "daily"),
    BloombergField("DXY Index", "PX_LAST", "DXY", "macro", "daily"),
    BloombergField("SPX Index", "PX_LAST", "SPX", "macro", "daily"),
    BloombergField("USGG10YR Index", "PX_LAST", "US_10Y_YIELD", "macro", "daily"),
    BloombergField("USGG2YR Index", "PX_LAST", "US_2Y_YIELD", "macro", "daily"),
]

# All fields grouped
ALL_FIELDS = {
    "fx_spots": FX_SPOTS,
    "fx_extended": FX_EXTENDED,
    "rates": INTEREST_RATES,
    "commodities": COMMODITY_FUTURES,
    "macro": MACRO_INDICATORS,
}


# =============================================================================
# File Management
# =============================================================================

class BloombergExportManager:
    """
    Manages Bloomberg data exports.
    
    Handles file naming, validation, and organization.
    """
    
    def __init__(
        self,
        base_dir: Path = Path("data/raw/bloomberg"),
        processed_dir: Path = Path("data/processed")
    ):
        """
        Initialize export manager.
        
        Args:
            base_dir: Base directory for raw exports
            processed_dir: Directory for processed data
        """
        self.base_dir = Path(base_dir)
        self.processed_dir = Path(processed_dir)
        
        # Ensure directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def get_export_dir(self, export_date: date) -> Path:
        """Get directory for a specific export date."""
        dir_path = self.base_dir / export_date.strftime("%Y-%m-%d")
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def get_filename(self, category: str, export_date: date) -> str:
        """Generate standardized filename."""
        return f"{category}_{export_date.strftime('%Y%m%d')}.csv"
    
    def get_filepath(self, category: str, export_date: date) -> Path:
        """Get full path for export file."""
        export_dir = self.get_export_dir(export_date)
        filename = self.get_filename(category, export_date)
        return export_dir / filename
    
    def list_exports(self, export_date: date) -> Dict[str, Path]:
        """List all exports for a date."""
        export_dir = self.get_export_dir(export_date)
        
        exports = {}
        for category in ALL_FIELDS.keys():
            filepath = self.get_filepath(category, export_date)
            if filepath.exists():
                exports[category] = filepath
        
        return exports
    
    def validate_export(self, filepath: Path, category: str) -> Dict[str, Any]:
        """
        Validate an export file.
        
        Args:
            filepath: Path to CSV file
            category: Data category
        
        Returns:
            Validation result dict
        """
        import pandas as pd
        
        result = {
            "valid": False,
            "filepath": str(filepath),
            "category": category,
            "errors": [],
            "warnings": [],
            "row_count": 0,
            "columns": [],
        }
        
        if not filepath.exists():
            result["errors"].append(f"File not found: {filepath}")
            return result
        
        try:
            df = pd.read_csv(filepath)
            result["row_count"] = len(df)
            result["columns"] = list(df.columns)
            
            # Check for required columns
            expected_fields = ALL_FIELDS.get(category, [])
            for field in expected_fields:
                if field.required and field.name not in df.columns:
                    if "date" not in str(df.columns).lower():
                        result["warnings"].append(f"Missing field: {field.name}")
            
            # Check for data
            if len(df) == 0:
                result["errors"].append("File is empty")
            
            # Check for date column
            if "date" not in [c.lower() for c in df.columns]:
                result["warnings"].append("No 'date' column found")
            
            result["valid"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Failed to read file: {str(e)}")
        
        return result
    
    def validate_all_exports(self, export_date: date) -> Dict[str, Any]:
        """Validate all exports for a date."""
        results = {}
        
        for category in ALL_FIELDS.keys():
            filepath = self.get_filepath(category, export_date)
            results[category] = self.validate_export(filepath, category)
        
        all_valid = all(r["valid"] for r in results.values())
        
        return {
            "date": export_date.isoformat(),
            "all_valid": all_valid,
            "categories": results,
        }


# =============================================================================
# Export Procedure Generator
# =============================================================================

def generate_export_checklist(export_date: date) -> str:
    """
    Generate a checklist for manual Bloomberg export.
    
    Args:
        export_date: Date to export
    
    Returns:
        Markdown checklist
    """
    manager = BloombergExportManager()
    
    lines = [
        f"# Bloomberg Data Export Checklist",
        f"**Date:** {export_date.strftime('%Y-%m-%d')}",
        "",
        "## Pre-Export",
        "- [ ] Bloomberg Terminal is logged in",
        "- [ ] Market is closed or data is end-of-day",
        "",
        "## Export Directory",
        f"Save all files to: `{manager.get_export_dir(export_date)}`",
        "",
    ]
    
    # Generate instructions for each category
    for category, fields in ALL_FIELDS.items():
        filename = manager.get_filename(category, export_date)
        
        lines.extend([
            f"## {category.upper()}",
            f"**Filename:** `{filename}`",
            "",
            "### Tickers to Export:",
            "```",
        ])
        
        for field in fields:
            lines.append(f"{field.ticker} | {field.field}")
        
        lines.extend([
            "```",
            "",
            "### Bloomberg Steps:",
            f"1. In Bloomberg: Type `{fields[0].ticker}` <GO>",
            "2. Click 'Export' or use HP (Historical Prices)",
            "3. Set date range: Last 120 days",
            f"4. Save as: `{filename}`",
            "",
            f"- [ ] {category} exported",
            "",
        ])
    
    lines.extend([
        "## Post-Export Validation",
        "Run validation script:",
        "```bash",
        f"python -m scripts.validate_bloomberg_export --date {export_date.strftime('%Y-%m-%d')}",
        "```",
        "",
        "- [ ] All validations pass",
    ])
    
    return "\n".join(lines)


def generate_excel_formula_template() -> str:
    """
    Generate Excel DDE formulas for Bloomberg data.
    
    Returns:
        CSV content with DDE formulas
    """
    lines = [
        "# Bloomberg Excel DDE Formula Template",
        "# Copy these formulas into Excel with Bloomberg Add-in",
        "",
        "Ticker,Field,Formula",
    ]
    
    for category, fields in ALL_FIELDS.items():
        lines.append(f"# {category.upper()}")
        for field in fields:
            # DDE formula format
            formula = f'=BDP("{field.ticker}","{field.field}")'
            lines.append(f"{field.ticker},{field.field},{formula}")
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# CLI Commands
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bloomberg Export Workflow")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Checklist command
    checklist_parser = subparsers.add_parser("checklist", help="Generate export checklist")
    checklist_parser.add_argument("--date", type=str, help="Export date (YYYY-MM-DD)")
    checklist_parser.add_argument("--output", type=str, help="Output file")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate exports")
    validate_parser.add_argument("--date", type=str, required=True, help="Export date")
    
    # Excel template command
    excel_parser = subparsers.add_parser("excel", help="Generate Excel template")
    excel_parser.add_argument("--output", type=str, help="Output file")
    
    # List fields command
    list_parser = subparsers.add_parser("list", help="List all Bloomberg fields")
    
    args = parser.parse_args()
    
    if args.command == "checklist":
        export_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
        checklist = generate_export_checklist(export_date)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(checklist)
            print(f"Checklist saved to: {args.output}")
        else:
            print(checklist)
    
    elif args.command == "validate":
        export_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        manager = BloombergExportManager()
        results = manager.validate_all_exports(export_date)
        
        print(f"\nValidation Results for {export_date}")
        print("=" * 50)
        
        for category, result in results["categories"].items():
            status = "✅" if result["valid"] else "❌"
            print(f"{status} {category}: {result['row_count']} rows")
            
            for error in result["errors"]:
                print(f"   ❌ {error}")
            for warning in result["warnings"]:
                print(f"   ⚠️ {warning}")
        
        print()
        overall = "✅ ALL VALID" if results["all_valid"] else "❌ SOME INVALID"
        print(f"Overall: {overall}")
    
    elif args.command == "excel":
        template = generate_excel_formula_template()
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(template)
            print(f"Template saved to: {args.output}")
        else:
            print(template)
    
    elif args.command == "list":
        print("\nBloomberg Fields Required")
        print("=" * 50)
        
        for category, fields in ALL_FIELDS.items():
            print(f"\n{category.upper()}:")
            for field in fields:
                req = "*" if field.required else " "
                print(f"  {req} {field.ticker:<20} {field.field:<15} -> {field.name}")
        
        print("\n* = required field")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
