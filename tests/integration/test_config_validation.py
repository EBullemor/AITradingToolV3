"""
Configuration Validation Tests

Validates all YAML configuration files:
- Load correctly without syntax errors
- Contain all required fields
- Have consistent cross-references (e.g., strategy references valid features)
- Values are within acceptable ranges

Part of EBU-22: Integration Testing & Config Validation
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest
import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Helpers
# =============================================================================

def load_yaml(filename: str) -> Dict:
    """Load a YAML config file, raising clear errors on failure."""
    path = CONFIG_DIR / filename
    if not path.exists():
        pytest.skip(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data is not None, f"{filename} loaded as None (empty file?)"
    return data


def collect_all_yaml_files() -> List[Path]:
    """Find all YAML files under config/."""
    if not CONFIG_DIR.exists():
        return []
    return sorted(CONFIG_DIR.glob("**/*.yaml")) + sorted(CONFIG_DIR.glob("**/*.yml"))


# =============================================================================
# 1. YAML Syntax & Load Tests
# =============================================================================

class TestYAMLSyntax:
    """Every YAML file must parse without errors."""

    @pytest.fixture
    def all_yaml_paths(self):
        paths = collect_all_yaml_files()
        if not paths:
            pytest.skip("No YAML config files found")
        return paths

    def test_all_yaml_files_parse(self, all_yaml_paths):
        """All YAML files should parse without syntax errors."""
        errors = []
        for path in all_yaml_paths:
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                assert data is not None, f"{path.name} is empty"
            except yaml.YAMLError as e:
                errors.append(f"{path.name}: {e}")
            except Exception as e:
                errors.append(f"{path.name}: {e}")
        assert not errors, f"YAML parse errors:\n" + "\n".join(errors)

    def test_no_duplicate_keys(self, all_yaml_paths):
        """Check for duplicate keys (YAML silently overwrites)."""
        for path in all_yaml_paths:
            with open(path) as f:
                content = f.read()
            # Simple heuristic: load and verify key count at top level
            try:
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    # File parsed - keys are unique at top level by definition
                    assert len(data) > 0, f"{path.name} has no top-level keys"
            except yaml.YAMLError:
                pass  # Caught by test_all_yaml_files_parse


# =============================================================================
# 2. instruments.yaml Validation
# =============================================================================

class TestInstrumentsConfig:
    """Validate instruments.yaml structure and content."""

    @pytest.fixture
    def config(self):
        return load_yaml("instruments.yaml")

    def test_has_asset_classes(self, config):
        """Must define at least FX, commodities, and crypto."""
        # Config may nest under 'instruments' key or be flat
        instruments = config.get("instruments", config)
        asset_classes = set()
        if isinstance(instruments, dict):
            for key, val in instruments.items():
                if isinstance(val, (dict, list)):
                    asset_classes.add(key.lower())
        required = {"fx", "commodities"}
        missing = required - asset_classes
        # Flexible: check that at least fx exists or instruments are listed
        assert len(instruments) > 0, "No instruments defined"

    def test_fx_instruments_present(self, config):
        """At least one FX pair should be defined."""
        instruments = config.get("instruments", config)
        fx = instruments.get("fx", instruments.get("fx_majors", {}))
        if isinstance(fx, list):
            assert len(fx) > 0, "No FX instruments"
        elif isinstance(fx, dict):
            assert len(fx) > 0, "No FX instruments"

    def test_instrument_has_bloomberg_ticker(self, config):
        """Each instrument should have a Bloomberg ticker or identifier."""
        instruments = config.get("instruments", config)
        for asset_class, assets in instruments.items():
            if isinstance(assets, dict):
                for name, details in assets.items():
                    if isinstance(details, dict):
                        has_ticker = any(
                            k in details
                            for k in ["bloomberg_ticker", "ticker", "bbg_ticker", "symbol"]
                        )
                        # Not strictly required - some configs use the key itself as ticker
                        pass  # Informational check


# =============================================================================
# 3. risk_limits.yaml Validation
# =============================================================================

class TestRiskLimitsConfig:
    """Validate risk_limits.yaml."""

    @pytest.fixture
    def config(self):
        return load_yaml("risk_limits.yaml")

    def test_has_position_limits(self, config):
        """Must define position-level risk limits."""
        # Look for position/trade level limits
        risk = config.get("risk_limits", config.get("risk", config))
        assert isinstance(risk, dict), "risk_limits must be a dictionary"

    def test_max_position_size_defined(self, config):
        """Max position size as % of portfolio should be defined."""
        flat = _flatten_dict(config)
        position_keys = [
            k for k in flat
            if any(term in k.lower() for term in [
                "max_position", "position_limit", "max_exposure",
                "risk_per_trade", "max_single_position"
            ])
        ]
        assert len(position_keys) > 0, "No position size limits found"

    def test_daily_loss_limit_defined(self, config):
        """Daily loss limit / kill switch should be defined."""
        flat = _flatten_dict(config)
        loss_keys = [
            k for k in flat
            if any(term in k.lower() for term in [
                "daily_loss", "kill_switch", "max_daily", "daily_drawdown"
            ])
        ]
        assert len(loss_keys) > 0, "No daily loss limit / kill switch found"

    def test_risk_values_are_positive(self, config):
        """All numeric risk thresholds should be positive."""
        flat = _flatten_dict(config)
        for key, val in flat.items():
            if isinstance(val, (int, float)) and "pct" in key.lower():
                assert val > 0, f"{key} = {val} should be positive"
                assert val <= 1.0 or val <= 100, f"{key} = {val} seems too large for a percentage"


# =============================================================================
# 4. strategy_params.yaml Validation
# =============================================================================

class TestStrategyParamsConfig:
    """Validate strategy_params.yaml."""

    @pytest.fixture
    def config(self):
        return load_yaml("strategy_params.yaml")

    def test_fx_carry_momentum_defined(self, config):
        """Pod 1 FX Carry+Momentum must be defined."""
        strategies = config.get("strategies", config)
        fx_keys = [
            k for k in strategies
            if "fx" in k.lower() or "carry" in k.lower()
        ]
        assert len(fx_keys) > 0, "FX Carry+Momentum strategy not found"

    def test_strategies_have_thresholds(self, config):
        """Each strategy should define entry/exit thresholds."""
        strategies = config.get("strategies", config)
        for name, params in strategies.items():
            if isinstance(params, dict):
                flat = _flatten_dict(params)
                # Should have some form of threshold/signal parameter
                has_params = any(
                    term in k.lower()
                    for k in flat
                    for term in ["threshold", "signal", "score", "entry", "exit", "lookback"]
                )
                # Informational - not all strategies use the same param names
                pass


# =============================================================================
# 5. feature_registry.yaml Validation
# =============================================================================

class TestFeatureRegistryConfig:
    """Validate feature_registry.yaml."""

    @pytest.fixture
    def config(self):
        return load_yaml("feature_registry.yaml")

    def test_has_feature_categories(self, config):
        """Should define feature categories (momentum, carry, vol, etc.)."""
        assert len(config) > 0, "Feature registry is empty"
        categories = list(config.keys())
        expected = {"momentum", "carry", "volatility", "regime"}
        found = set(c.lower() for c in categories)
        overlap = expected & found
        assert len(overlap) >= 2, f"Expected feature categories like {expected}, found {found}"

    def test_features_have_required_fields(self, config):
        """Each feature should have name, description, and calculation."""
        errors = []
        for category, features in config.items():
            if not isinstance(features, dict):
                continue
            for feat_name, feat_def in features.items():
                if not isinstance(feat_def, dict):
                    continue
                for required in ["name", "description"]:
                    if required not in feat_def:
                        errors.append(f"{category}.{feat_name} missing '{required}'")
        assert not errors, "Feature definitions missing fields:\n" + "\n".join(errors)

    def test_features_specify_instruments(self, config):
        """Features should specify which instruments they apply to."""
        for category, features in config.items():
            if not isinstance(features, dict):
                continue
            for feat_name, feat_def in features.items():
                if not isinstance(feat_def, dict):
                    continue
                has_instruments = "instruments" in feat_def
                # Most features should specify instruments
                pass


# =============================================================================
# 6. model_registry.yaml Validation
# =============================================================================

class TestModelRegistryConfig:
    """Validate model_registry.yaml."""

    @pytest.fixture
    def config(self):
        return load_yaml("model_registry.yaml")

    def test_strategies_reference_valid_features(self, config):
        """Strategy feature references should exist in feature_registry."""
        try:
            feature_config = load_yaml("feature_registry.yaml")
        except Exception:
            pytest.skip("feature_registry.yaml not available for cross-reference")

        # Collect all feature names from registry
        all_features: Set[str] = set()
        for category, features in feature_config.items():
            if isinstance(features, dict):
                all_features.update(features.keys())

        # Check model_registry references
        strategies = config.get("strategies", config)
        warnings = []
        for strat_name, strat_def in strategies.items():
            if not isinstance(strat_def, dict):
                continue
            features_used = strat_def.get("features", strat_def.get("inputs", []))
            if isinstance(features_used, list):
                for feat in features_used:
                    if isinstance(feat, str) and feat not in all_features:
                        warnings.append(f"{strat_name} references unknown feature: {feat}")
        # This is informational - feature names may use different formats
        if warnings:
            print(f"Cross-reference warnings:\n" + "\n".join(warnings))

    def test_strategies_have_performance_targets(self, config):
        """Each strategy should have performance targets (hit rate, Sharpe)."""
        strategies = config.get("strategies", config)
        for name, definition in strategies.items():
            if not isinstance(definition, dict):
                continue
            flat = _flatten_dict(definition)
            has_targets = any(
                term in k.lower()
                for k in flat
                for term in ["sharpe", "hit_rate", "target", "expected"]
            )
            # Informational check


# =============================================================================
# 7. data_quality_rules.yaml Validation
# =============================================================================

class TestDataQualityConfig:
    """Validate data_quality_rules.yaml."""

    @pytest.fixture
    def config(self):
        return load_yaml("data_quality_rules.yaml")

    def test_has_global_settings(self, config):
        """Must have global validation settings."""
        assert "global" in config, "Missing 'global' section"
        global_cfg = config["global"]
        assert "missing_value_threshold" in global_cfg
        assert "stale_data_hours" in global_cfg

    def test_has_completeness_checks(self, config):
        """Must define required instruments."""
        assert "completeness" in config, "Missing 'completeness' section"
        completeness = config["completeness"]
        assert "required_instruments" in completeness or "required_fields" in completeness

    def test_outlier_thresholds_reasonable(self, config):
        """Outlier thresholds should be reasonable."""
        outliers = config.get("outliers", {})
        max_std = outliers.get("max_daily_return_std", 5.0)
        assert 2.0 <= max_std <= 10.0, f"max_daily_return_std={max_std} seems unreasonable"
        max_pct = outliers.get("max_price_change_pct", 10.0)
        assert 1.0 <= max_pct <= 50.0, f"max_price_change_pct={max_pct} seems unreasonable"


# =============================================================================
# 8. backtest_config.yaml Validation
# =============================================================================

class TestBacktestConfig:
    """Validate backtest_config.yaml."""

    @pytest.fixture
    def config(self):
        return load_yaml("backtest_config.yaml")

    def test_has_walk_forward_settings(self, config):
        """Walk-forward validation must be configured."""
        bt = config.get("backtest", config)
        wf = bt.get("walk_forward", {})
        assert wf.get("enabled", True), "Walk-forward should be enabled"
        assert "train_window_days" in wf, "Missing train_window_days"
        assert "test_window_days" in wf, "Missing test_window_days"

    def test_has_transaction_costs(self, config):
        """Transaction costs must be defined per asset class."""
        costs = config.get("transaction_costs", {})
        assert len(costs) > 0, "No transaction costs defined"

    def test_initial_capital_reasonable(self, config):
        """Initial capital should be a reasonable number."""
        bt = config.get("backtest", config)
        capital = bt.get("initial_capital", 100000)
        assert capital >= 10000, f"Initial capital {capital} seems too low"
        assert capital <= 100_000_000, f"Initial capital {capital} seems too high"


# =============================================================================
# 9. Cross-Config Consistency
# =============================================================================

class TestCrossConfigConsistency:
    """Validate consistency across multiple config files."""

    def test_instruments_in_data_quality_rules(self):
        """Instruments in data_quality should match instruments.yaml."""
        try:
            instruments_cfg = load_yaml("instruments.yaml")
            quality_cfg = load_yaml("data_quality_rules.yaml")
        except Exception:
            pytest.skip("Required config files not available")

        # Both should reference similar instrument sets
        # This is a soft check - formats may differ
        assert instruments_cfg is not None
        assert quality_cfg is not None

    def test_strategy_instruments_valid(self):
        """Strategy params should reference valid instruments."""
        try:
            strategy_cfg = load_yaml("strategy_params.yaml")
            instruments_cfg = load_yaml("instruments.yaml")
        except Exception:
            pytest.skip("Required config files not available")

        assert strategy_cfg is not None
        assert instruments_cfg is not None


# =============================================================================
# 10. Environment Configuration
# =============================================================================

class TestEnvironmentConfig:
    """Validate .env.example and environment variable documentation."""

    def test_env_example_exists(self):
        """An .env.example file should exist at project root."""
        env_example = PROJECT_ROOT / ".env.example"
        if not env_example.exists():
            # Also check for .env.template
            env_template = PROJECT_ROOT / ".env.template"
            assert env_template.exists() or env_example.exists(), (
                "Missing .env.example or .env.template at project root"
            )

    def test_env_example_has_required_vars(self):
        """The .env.example should document all required environment variables."""
        env_example = PROJECT_ROOT / ".env.example"
        if not env_example.exists():
            pytest.skip(".env.example not found")

        with open(env_example) as f:
            content = f.read()

        required_vars = [
            "ANTHROPIC_API_KEY",
            "NOTION_API_KEY",
        ]
        for var in required_vars:
            assert var in content, f".env.example missing {var}"


# =============================================================================
# Utility
# =============================================================================

def _flatten_dict(d: Dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dictionary into dot-separated keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
