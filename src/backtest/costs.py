"""
Transaction Cost Models

Realistic transaction cost modeling per asset class.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from loguru import logger


@dataclass
class CostConfig:
    """Cost configuration for an asset class."""
    spread_bps: float = 0.0       # Bid-ask spread in basis points
    slippage_bps: float = 0.0     # Expected slippage in basis points
    commission_pct: float = 0.0    # Commission as percentage
    commission_fixed: float = 0.0  # Fixed commission per trade
    commission_per_contract: float = 0.0  # Per-contract commission


# Default cost configurations by asset class
DEFAULT_COSTS: Dict[str, CostConfig] = {
    # FX Majors - very liquid, tight spreads
    "fx_majors": CostConfig(
        spread_bps=1.0,
        slippage_bps=0.5,
        commission_pct=0.0,
    ),
    
    # FX Emerging Markets - wider spreads
    "fx_em": CostConfig(
        spread_bps=5.0,
        slippage_bps=2.0,
        commission_pct=0.0,
    ),
    
    # Commodities - futures
    "commodities": CostConfig(
        spread_bps=3.0,
        slippage_bps=2.0,
        commission_per_contract=2.50,
    ),
    
    # Bitcoin/Crypto
    "crypto": CostConfig(
        spread_bps=5.0,
        slippage_bps=5.0,
        commission_pct=0.10,  # 10 bps
    ),
    
    # Equities/ETFs
    "equities": CostConfig(
        spread_bps=2.0,
        slippage_bps=1.0,
        commission_pct=0.0,
        commission_fixed=1.0,  # $1 per trade
    ),
}

# Instrument to asset class mapping
INSTRUMENT_CLASS_MAP: Dict[str, str] = {
    # FX Majors
    "EURUSD": "fx_majors",
    "USDJPY": "fx_majors",
    "GBPUSD": "fx_majors",
    "USDCHF": "fx_majors",
    "AUDUSD": "fx_majors",
    "USDCAD": "fx_majors",
    "NZDUSD": "fx_majors",
    
    # FX Emerging Markets
    "USDMXN": "fx_em",
    "USDZAR": "fx_em",
    "USDTRY": "fx_em",
    "USDBRL": "fx_em",
    
    # Commodities
    "CL": "commodities",   # WTI Crude
    "GC": "commodities",   # Gold
    "SI": "commodities",   # Silver
    "HG": "commodities",   # Copper
    "NG": "commodities",   # Natural Gas
    "CO": "commodities",   # Brent
    
    # Crypto
    "BTC": "crypto",
    "BTCUSD": "crypto",
    "ETH": "crypto",
    "ETHUSD": "crypto",
}


class TransactionCostCalculator:
    """
    Calculates realistic transaction costs.
    
    Accounts for:
    - Bid-ask spread
    - Expected slippage
    - Commissions (fixed, percentage, or per-contract)
    """
    
    def __init__(
        self,
        custom_costs: Optional[Dict[str, CostConfig]] = None,
        custom_mappings: Optional[Dict[str, str]] = None
    ):
        """
        Initialize cost calculator.
        
        Args:
            custom_costs: Custom cost configs to override defaults
            custom_mappings: Custom instrument to class mappings
        """
        self.costs = {**DEFAULT_COSTS}
        if custom_costs:
            self.costs.update(custom_costs)
        
        self.mappings = {**INSTRUMENT_CLASS_MAP}
        if custom_mappings:
            self.mappings.update(custom_mappings)
    
    def get_asset_class(self, instrument: str) -> str:
        """Determine asset class for instrument."""
        # Direct lookup
        if instrument in self.mappings:
            return self.mappings[instrument]
        
        # Try to infer from name
        instrument_upper = instrument.upper()
        
        # FX pattern (6 letters, XXX/YYY format)
        if len(instrument_upper) == 6 and instrument_upper.isalpha():
            return "fx_majors"  # Default FX to majors
        
        # Crypto patterns
        if any(crypto in instrument_upper for crypto in ["BTC", "ETH", "CRYPTO"]):
            return "crypto"
        
        # Default to equities
        return "equities"
    
    def get_cost_config(self, instrument: str) -> CostConfig:
        """Get cost configuration for instrument."""
        asset_class = self.get_asset_class(instrument)
        return self.costs.get(asset_class, CostConfig())
    
    def calculate_cost(
        self,
        instrument: str,
        trade_value: float,
        side: str = "entry",  # "entry" or "exit"
        contracts: int = 1
    ) -> float:
        """
        Calculate transaction cost.
        
        Args:
            instrument: Instrument being traded
            trade_value: Notional value of trade
            side: "entry" or "exit"
            contracts: Number of contracts (for futures)
        
        Returns:
            Total cost in dollars
        """
        config = self.get_cost_config(instrument)
        
        total_cost = 0.0
        
        # Spread cost (half spread per side)
        spread_cost = trade_value * (config.spread_bps / 10000) / 2
        total_cost += spread_cost
        
        # Slippage cost
        slippage_cost = trade_value * (config.slippage_bps / 10000)
        total_cost += slippage_cost
        
        # Commission (percentage)
        if config.commission_pct > 0:
            commission = trade_value * (config.commission_pct / 100)
            total_cost += commission
        
        # Commission (fixed per trade)
        if config.commission_fixed > 0:
            total_cost += config.commission_fixed
        
        # Commission (per contract)
        if config.commission_per_contract > 0:
            total_cost += contracts * config.commission_per_contract
        
        return total_cost
    
    def calculate_round_trip_cost(
        self,
        instrument: str,
        trade_value: float,
        contracts: int = 1
    ) -> float:
        """Calculate total cost for entry + exit."""
        entry_cost = self.calculate_cost(instrument, trade_value, "entry", contracts)
        exit_cost = self.calculate_cost(instrument, trade_value, "exit", contracts)
        return entry_cost + exit_cost
    
    def estimate_breakeven_move(
        self,
        instrument: str,
        trade_value: float
    ) -> float:
        """
        Estimate minimum price move needed to breakeven after costs.
        
        Returns:
            Required move as percentage
        """
        round_trip_cost = self.calculate_round_trip_cost(instrument, trade_value)
        breakeven_pct = (round_trip_cost / trade_value) * 100
        return breakeven_pct


def calculate_cost(
    instrument: str,
    trade_value: float,
    side: str = "entry"
) -> float:
    """
    Convenience function to calculate transaction cost.
    
    Args:
        instrument: Instrument symbol
        trade_value: Notional value of trade
        side: "entry" or "exit"
    
    Returns:
        Cost in dollars
    """
    calculator = TransactionCostCalculator()
    return calculator.calculate_cost(instrument, trade_value, side)
