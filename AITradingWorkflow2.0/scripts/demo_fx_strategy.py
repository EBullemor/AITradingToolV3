#!/usr/bin/env python3
"""
Demo: FX Carry + Momentum Strategy

This script demonstrates the FX Carry+Momentum strategy pod
generating signals from sample data.

Run: python -m scripts.demo_fx_strategy
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies import FXCarryMomentumStrategy, SignalDirection
from src.features import compute_fx_features, compute_vix_regime


def create_sample_fx_data(pair: str, days: int = 100) -> pd.DataFrame:
    """Create realistic sample FX data."""
    np.random.seed(42 if pair == "EURUSD" else 123)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    
    # Base prices
    base_prices = {
        "EURUSD": 1.10,
        "USDJPY": 148.0,
        "GBPUSD": 1.27,
        "AUDUSD": 0.66,
    }
    
    base = base_prices.get(pair, 1.0)
    
    # Generate prices with some trend
    returns = np.random.normal(0.0001, 0.005, days)
    prices = base * np.cumprod(1 + returns)
    
    # Add high/low
    daily_range = prices * np.random.uniform(0.003, 0.008, days)
    high = prices + daily_range / 2
    low = prices - daily_range / 2
    
    df = pd.DataFrame({
        "timestamp": dates,
        "PX_LAST": prices,
        "PX_HIGH": high,
        "PX_LOW": low,
    })
    df.set_index("timestamp", inplace=True)
    
    return df


def create_sample_macro_data(vix_level: float = 18.0, days: int = 100) -> pd.DataFrame:
    """Create sample macro data (VIX)."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    
    # VIX with some noise
    vix = vix_level + np.random.normal(0, 2, days)
    vix = np.clip(vix, 10, 50)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "PX_LAST": vix,
    })
    df.set_index("timestamp", inplace=True)
    
    return df


def main():
    print("=" * 60)
    print("FX Carry + Momentum Strategy Demo")
    print("=" * 60)
    
    # Create strategy
    strategy = FXCarryMomentumStrategy()
    print(f"\nStrategy: {strategy.name}")
    print(f"Instruments: {strategy.instruments}")
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    
    pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]
    raw_data = {}
    features = {}
    
    for pair in pairs:
        raw_data[pair] = create_sample_fx_data(pair)
        
        # Compute features
        prices = raw_data[pair]["PX_LAST"]
        high = raw_data[pair]["PX_HIGH"]
        low = raw_data[pair]["PX_LOW"]
        
        feat = compute_fx_features(prices, high, low, pair)
        features[pair] = feat
        
        print(f"  {pair}: {len(feat)} rows, latest price: {prices.iloc[-1]:.4f}")
    
    # Test different regimes
    print("\n" + "=" * 60)
    print("Testing Different Market Regimes")
    print("=" * 60)
    
    regimes = [
        ("LOW_VOL", 14.0),
        ("NORMAL", 20.0),
        ("HIGH_VOL", 28.0),
    ]
    
    for regime_name, vix_level in regimes:
        print(f"\n🎯 Regime: {regime_name} (VIX ≈ {vix_level})")
        print("-" * 40)
        
        macro_data = create_sample_macro_data(vix_level)
        
        # Generate signals
        signals = strategy.generate_signals(
            features=features,
            macro_data=macro_data,
            as_of_date=datetime.now()
        )
        
        if not signals:
            print("  No signals generated")
            continue
        
        for signal in signals:
            direction_emoji = "🟢" if signal.direction == SignalDirection.LONG else "🔴"
            
            print(f"\n  {direction_emoji} {signal.direction.value} {signal.instrument}")
            print(f"     Strength: {signal.strength:.1%} ({signal.strength_category.value})")
            print(f"     Entry: {signal.entry_price:.5f}")
            print(f"     Stop: {signal.stop_loss:.5f}")
            print(f"     Target: {signal.take_profit_1:.5f}")
            
            if signal.risk_reward_ratio:
                print(f"     R:R: {signal.risk_reward_ratio:.1f}:1")
            
            print(f"     Rationale: {signal.rationale[:80]}...")
    
    # Example of signal serialization
    print("\n" + "=" * 60)
    print("Signal Serialization Example")
    print("=" * 60)
    
    macro_data = create_sample_macro_data(15.0)
    signals = strategy.generate_signals(features, macro_data)
    
    if signals:
        signal = signals[0]
        print("\n📦 Signal as Dictionary:")
        import json
        print(json.dumps(signal.to_dict(), indent=2, default=str))
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
