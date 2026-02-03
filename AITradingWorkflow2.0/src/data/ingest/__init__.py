"""
Data Ingestion Module

Loaders for various data sources:
- Bloomberg Terminal exports
- On-chain data (Glassnode)
"""

from .bloomberg import (
    BloombergLoader,
    load_bloomberg_csv,
    load_fx_data,
    load_rates_data,
    load_commodities_data,
    load_macro_data,
)
from .onchain import (
    OnChainLoader,
    GlassnodeClient,
    load_btc_onchain,
)

__all__ = [
    # Bloomberg
    "BloombergLoader",
    "load_bloomberg_csv",
    "load_fx_data",
    "load_rates_data",
    "load_commodities_data",
    "load_macro_data",
    
    # On-chain
    "OnChainLoader",
    "GlassnodeClient",
    "load_btc_onchain",
]
