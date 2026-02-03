"""
Data Schema Definitions

Defines expected schemas for different data types (FX, commodities, macro, BTC).
Used by validation layer to check data completeness and correctness.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple


class InstrumentType(Enum):
    """Types of instruments we trade."""
    FX_MAJOR = "fx_major"
    FX_EM = "fx_em"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    MACRO = "macro"  # Non-tradeable indicators like VIX


class DataFrequency(Enum):
    """Data update frequency."""
    TICK = "tick"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class FieldSchema:
    """Schema for a single data field."""
    name: str
    required: bool = True
    dtype: str = "float64"  # pandas dtype
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allow_negative: bool = False
    allow_null: bool = False
    description: str = ""


@dataclass
class InstrumentSchema:
    """Schema for an instrument's data."""
    ticker: str
    instrument_type: InstrumentType
    fields: List[FieldSchema]
    trading_hours: Tuple[time, time] = (time(0, 0), time(23, 59))
    trades_weekends: bool = False
    trades_holidays: bool = False
    timezone: str = "America/New_York"
    min_history_days: int = 252  # Minimum history required
    description: str = ""


# =============================================================================
# Field Definitions (Reusable)
# =============================================================================

FIELD_PX_LAST = FieldSchema(
    name="PX_LAST",
    required=True,
    dtype="float64",
    min_value=0.0001,  # Very small but positive
    allow_negative=False,
    description="Last traded price"
)

FIELD_PX_HIGH = FieldSchema(
    name="PX_HIGH",
    required=True,
    dtype="float64",
    min_value=0.0001,
    allow_negative=False,
    description="Daily high price"
)

FIELD_PX_LOW = FieldSchema(
    name="PX_LOW",
    required=True,
    dtype="float64",
    min_value=0.0001,
    allow_negative=False,
    description="Daily low price"
)

FIELD_PX_OPEN = FieldSchema(
    name="PX_OPEN",
    required=False,  # Not always available
    dtype="float64",
    min_value=0.0001,
    allow_negative=False,
    description="Daily open price"
)

FIELD_PX_VOLUME = FieldSchema(
    name="PX_VOLUME",
    required=False,
    dtype="float64",
    min_value=0,
    allow_negative=False,
    description="Trading volume"
)

FIELD_TIMESTAMP = FieldSchema(
    name="timestamp",
    required=True,
    dtype="datetime64[ns]",
    description="Data timestamp"
)

FIELD_DATE = FieldSchema(
    name="date",
    required=True,
    dtype="datetime64[ns]",
    description="Trading date"
)


# =============================================================================
# Instrument Schemas
# =============================================================================

# FX Majors
FX_SCHEMA_FIELDS = [
    FIELD_TIMESTAMP,
    FIELD_PX_LAST,
    FIELD_PX_HIGH,
    FIELD_PX_LOW,
    FieldSchema(name="FWD_POINTS_1M", required=False, dtype="float64", allow_negative=True),
    FieldSchema(name="FWD_POINTS_3M", required=False, dtype="float64", allow_negative=True),
    FieldSchema(name="IMPLIED_VOL_1M", required=False, dtype="float64", min_value=0),
    FieldSchema(name="IMPLIED_VOL_3M", required=False, dtype="float64", min_value=0),
]

EURUSD_SCHEMA = InstrumentSchema(
    ticker="EURUSD",
    instrument_type=InstrumentType.FX_MAJOR,
    fields=FX_SCHEMA_FIELDS.copy(),
    trading_hours=(time(17, 0), time(17, 0)),  # 24hr Sun 5pm - Fri 5pm
    trades_weekends=False,
    timezone="America/New_York",
    description="Euro / US Dollar"
)

USDJPY_SCHEMA = InstrumentSchema(
    ticker="USDJPY",
    instrument_type=InstrumentType.FX_MAJOR,
    fields=FX_SCHEMA_FIELDS.copy(),
    trading_hours=(time(17, 0), time(17, 0)),
    trades_weekends=False,
    timezone="America/New_York",
    description="US Dollar / Japanese Yen"
)

GBPUSD_SCHEMA = InstrumentSchema(
    ticker="GBPUSD",
    instrument_type=InstrumentType.FX_MAJOR,
    fields=FX_SCHEMA_FIELDS.copy(),
    trading_hours=(time(17, 0), time(17, 0)),
    trades_weekends=False,
    timezone="America/New_York",
    description="British Pound / US Dollar"
)

AUDUSD_SCHEMA = InstrumentSchema(
    ticker="AUDUSD",
    instrument_type=InstrumentType.FX_MAJOR,
    fields=FX_SCHEMA_FIELDS.copy(),
    trading_hours=(time(17, 0), time(17, 0)),
    trades_weekends=False,
    timezone="America/New_York",
    description="Australian Dollar / US Dollar"
)

# Commodities
COMMODITY_SCHEMA_FIELDS = [
    FIELD_TIMESTAMP,
    FIELD_PX_LAST,
    FIELD_PX_HIGH,
    FIELD_PX_LOW,
    FIELD_PX_OPEN,
    FIELD_PX_VOLUME,
    FieldSchema(name="OPEN_INT", required=False, dtype="float64", min_value=0),
]

CL_SCHEMA = InstrumentSchema(
    ticker="CL1",
    instrument_type=InstrumentType.COMMODITY,
    fields=COMMODITY_SCHEMA_FIELDS.copy(),
    trading_hours=(time(6, 0), time(17, 0)),
    trades_weekends=False,
    timezone="America/New_York",
    description="WTI Crude Oil Front Month"
)

GC_SCHEMA = InstrumentSchema(
    ticker="GC1",
    instrument_type=InstrumentType.COMMODITY,
    fields=COMMODITY_SCHEMA_FIELDS.copy(),
    trading_hours=(time(6, 0), time(17, 0)),
    trades_weekends=False,
    timezone="America/New_York",
    description="Gold Front Month"
)

HG_SCHEMA = InstrumentSchema(
    ticker="HG1",
    instrument_type=InstrumentType.COMMODITY,
    fields=COMMODITY_SCHEMA_FIELDS.copy(),
    trading_hours=(time(6, 0), time(17, 0)),
    trades_weekends=False,
    timezone="America/New_York",
    description="Copper Front Month"
)

# Crypto
CRYPTO_SCHEMA_FIELDS = [
    FIELD_TIMESTAMP,
    FIELD_PX_LAST,
    FIELD_PX_HIGH,
    FIELD_PX_LOW,
    FIELD_PX_OPEN,
    FIELD_PX_VOLUME,
]

BTCUSD_SCHEMA = InstrumentSchema(
    ticker="BTCUSD",
    instrument_type=InstrumentType.CRYPTO,
    fields=CRYPTO_SCHEMA_FIELDS.copy(),
    trading_hours=(time(0, 0), time(23, 59)),
    trades_weekends=True,  # BTC trades 24/7
    trades_holidays=True,
    timezone="UTC",
    description="Bitcoin / US Dollar"
)

# Macro Indicators
MACRO_SCHEMA_FIELDS = [
    FIELD_TIMESTAMP,
    FIELD_PX_LAST,
]

VIX_SCHEMA = InstrumentSchema(
    ticker="VIX",
    instrument_type=InstrumentType.MACRO,
    fields=MACRO_SCHEMA_FIELDS.copy(),
    trading_hours=(time(9, 30), time(16, 0)),
    trades_weekends=False,
    timezone="America/New_York",
    min_history_days=252,
    description="CBOE Volatility Index"
)

DXY_SCHEMA = InstrumentSchema(
    ticker="DXY",
    instrument_type=InstrumentType.MACRO,
    fields=MACRO_SCHEMA_FIELDS.copy(),
    trading_hours=(time(0, 0), time(23, 59)),
    trades_weekends=False,
    timezone="America/New_York",
    description="US Dollar Index"
)


# =============================================================================
# Schema Registry
# =============================================================================

INSTRUMENT_SCHEMAS: Dict[str, InstrumentSchema] = {
    # FX
    "EURUSD": EURUSD_SCHEMA,
    "USDJPY": USDJPY_SCHEMA,
    "GBPUSD": GBPUSD_SCHEMA,
    "AUDUSD": AUDUSD_SCHEMA,
    # Commodities
    "CL1": CL_SCHEMA,
    "CL": CL_SCHEMA,  # Alias
    "GC1": GC_SCHEMA,
    "GC": GC_SCHEMA,
    "HG1": HG_SCHEMA,
    "HG": HG_SCHEMA,
    # Crypto
    "BTCUSD": BTCUSD_SCHEMA,
    "BTC": BTCUSD_SCHEMA,
    # Macro
    "VIX": VIX_SCHEMA,
    "DXY": DXY_SCHEMA,
}


def get_schema(ticker: str) -> Optional[InstrumentSchema]:
    """Get schema for an instrument by ticker."""
    return INSTRUMENT_SCHEMAS.get(ticker.upper())


def get_required_fields(ticker: str) -> List[str]:
    """Get list of required field names for an instrument."""
    schema = get_schema(ticker)
    if schema is None:
        return []
    return [f.name for f in schema.fields if f.required]


def get_all_tickers() -> List[str]:
    """Get all registered ticker symbols."""
    # Return unique tickers (excluding aliases)
    seen = set()
    unique = []
    for ticker, schema in INSTRUMENT_SCHEMAS.items():
        if schema.ticker not in seen:
            seen.add(schema.ticker)
            unique.append(ticker)
    return unique


def get_tickers_by_type(instrument_type: InstrumentType) -> List[str]:
    """Get all tickers of a specific instrument type."""
    return [
        ticker for ticker, schema in INSTRUMENT_SCHEMAS.items()
        if schema.instrument_type == instrument_type
    ]
