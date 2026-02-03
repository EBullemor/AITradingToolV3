# Bloomberg Data Export Workflow

## Overview

This document describes the workflow for extracting market data from Bloomberg Terminal for use in the trading recommendation system.

## Data Requirements

### FX Data (Daily)

| Bloomberg Ticker | Field | Output Name | Required |
|-----------------|-------|-------------|----------|
| EURUSD Curncy | PX_LAST | EURUSD | ✓ |
| USDJPY Curncy | PX_LAST | USDJPY | ✓ |
| GBPUSD Curncy | PX_LAST | GBPUSD | ✓ |
| AUDUSD Curncy | PX_LAST | AUDUSD | ✓ |
| USDCHF Curncy | PX_LAST | USDCHF | ✓ |
| USDCAD Curncy | PX_LAST | USDCAD | ✓ |
| NZDUSD Curncy | PX_LAST | NZDUSD | ✓ |

### Interest Rates (Daily)

| Bloomberg Ticker | Field | Output Name | Required |
|-----------------|-------|-------------|----------|
| US0012M Index | PX_LAST | USD_12M_RATE | ✓ |
| EUR012M Index | PX_LAST | EUR_12M_RATE | ✓ |
| BP0012M Index | PX_LAST | GBP_12M_RATE | ✓ |
| JY0012M Index | PX_LAST | JPY_12M_RATE | ✓ |
| AD0012M Index | PX_LAST | AUD_12M_RATE | ✓ |

### Commodities (Daily)

| Bloomberg Ticker | Field | Output Name | Required |
|-----------------|-------|-------------|----------|
| CL1 Comdty | PX_LAST | WTI_FRONT | ✓ |
| CL4 Comdty | PX_LAST | WTI_4TH | ✓ |
| CO1 Comdty | PX_LAST | BRENT_FRONT | ✓ |
| GC1 Comdty | PX_LAST | GOLD_FRONT | ✓ |
| HG1 Comdty | PX_LAST | COPPER_FRONT | ✓ |

### Macro Indicators (Daily)

| Bloomberg Ticker | Field | Output Name | Required |
|-----------------|-------|-------------|----------|
| VIX Index | PX_LAST | VIX | ✓ |
| DXY Index | PX_LAST | DXY | ✓ |
| SPX Index | PX_LAST | SPX | ✓ |
| USGG10YR Index | PX_LAST | US_10Y_YIELD | ✓ |

---

## Export Methods

### Method 1: Manual CSV Export (MVP)

**Best for:** Initial setup, ad-hoc exports

#### Steps:

1. **Open Bloomberg Terminal**

2. **For each data category:**
   
   a. Type the first ticker (e.g., `EURUSD Curncy` <GO>)
   
   b. Press `HP` for Historical Prices
   
   c. Set parameters:
      - Start Date: 120 days ago
      - End Date: Today
      - Periodicity: Daily
   
   d. Click **Actions** → **Export to Excel** → **Save as CSV**
   
   e. Save to: `data/raw/bloomberg/YYYY-MM-DD/`
   
   f. Use filename: `fx_spots_YYYYMMDD.csv`

3. **Repeat for all categories**

4. **Run validation:**
   ```bash
   python scripts/bloomberg_export.py validate --date 2026-02-01
   ```

### Method 2: Excel DDE Export (Semi-Automated)

**Best for:** Daily routine, multiple securities

#### Setup:

1. Open the Excel template: `templates/bloomberg_export.xlsx`

2. Ensure Bloomberg Add-in is enabled

3. The template contains DDE formulas that auto-populate

#### Daily Process:

1. Open Excel template

2. Wait for data to populate (green cells)

3. Copy data to new sheet (Paste Values)

4. Save as CSV to `data/raw/bloomberg/YYYY-MM-DD/`

### Method 3: BLPAPI (Automated)

**Best for:** Production, scheduled runs

**Requirements:**
- Bloomberg Server API license
- `blpapi` Python package

```python
# Example BLPAPI usage (requires license)
import blpapi

session = blpapi.Session()
session.start()

# Request historical data
request = service.createRequest("HistoricalDataRequest")
request.append("securities", "EURUSD Curncy")
request.append("fields", "PX_LAST")
request.set("startDate", "20260101")
request.set("endDate", "20260201")
```

---

## File Naming Convention

```
data/raw/bloomberg/
├── 2026-02-01/
│   ├── fx_spots_20260201.csv
│   ├── fx_extended_20260201.csv
│   ├── rates_20260201.csv
│   ├── commodities_20260201.csv
│   └── macro_20260201.csv
├── 2026-02-02/
│   └── ...
```

## CSV Format

All CSV files should follow this format:

```csv
date,EURUSD,USDJPY,GBPUSD,AUDUSD
2026-01-01,1.1025,148.50,1.2710,0.6550
2026-01-02,1.1030,148.75,1.2725,0.6545
...
```

**Requirements:**
- First column must be `date` in YYYY-MM-DD format
- Column headers must match output names from tables above
- No empty rows
- Decimal separator: `.` (period)

---

## Daily Workflow

### Morning Routine (7:00 AM ET)

1. **Check Market Status**
   - Ensure previous day's market is closed
   - Note any holidays

2. **Export Data**
   - Run export for previous business day
   - Save to appropriate directory

3. **Validate**
   ```bash
   python scripts/bloomberg_export.py validate --date $(date -d "yesterday" +%Y-%m-%d)
   ```

4. **Run Pipeline**
   ```bash
   python -m pipelines.daily_run
   ```

### Generate Checklist

```bash
python scripts/bloomberg_export.py checklist --date 2026-02-01 --output checklist.md
```

---

## Troubleshooting

### Missing Data

If a ticker shows N/A or #N/A:
1. Verify the ticker is valid (`EURUSD Curncy` <HELP>)
2. Check if market was open on that date
3. Try alternative data source

### Date Format Issues

Bloomberg may export dates in various formats:
- US: MM/DD/YYYY
- EU: DD/MM/YYYY
- ISO: YYYY-MM-DD

The ingestion script handles conversion automatically.

### Holiday Handling

On market holidays:
- Skip export for that market
- Or carry forward previous day's data

---

## Contact

For Bloomberg Terminal access or API questions, contact IT Support.
