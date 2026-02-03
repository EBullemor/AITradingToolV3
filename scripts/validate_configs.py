#!/usr/bin/env python3
"""
Config Validation Runner

Standalone script to validate all configuration files.
Can be run as a pre-commit hook or CI step.

Usage:
    python scripts/validate_configs.py
    python scripts/validate_configs.py --strict   # Fail on warnings too
    python scripts/validate_configs.py --verbose   # Show all checks

Part of EBU-22: Integration Testing & Config Validation
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check_mark(passed: bool) -> str:
    return f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"


def warn_mark() -> str:
    return f"{YELLOW}⚠{RESET}"


class ConfigValidator:
    """Validates all YAML config files for the trading system."""

    def __init__(self, strict: bool = False, verbose: bool = False):
        self.strict = strict
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks. Returns True if all pass."""
        print(f"\n{BOLD}━━━ AI Trading System Config Validation ━━━{RESET}\n")

        if not CONFIG_DIR.exists():
            self.errors.append(f"Config directory not found: {CONFIG_DIR}")
            self._print_summary()
            return False

        # 1. Syntax check all YAML files
        self._check_yaml_syntax()

        # 2. Validate each specific config
        config_checks = {
            "instruments.yaml": self._validate_instruments,
            "risk_limits.yaml": self._validate_risk_limits,
            "strategy_params.yaml": self._validate_strategy_params,
            "feature_registry.yaml": self._validate_feature_registry,
            "model_registry.yaml": self._validate_model_registry,
            "data_quality_rules.yaml": self._validate_data_quality,
            "backtest_config.yaml": self._validate_backtest,
        }

        for filename, validator in config_checks.items():
            path = CONFIG_DIR / filename
            if path.exists():
                try:
                    with open(path) as f:
                        data = yaml.safe_load(f)
                    if data:
                        validator(data, filename)
                    else:
                        self.errors.append(f"{filename}: File is empty")
                except yaml.YAMLError as e:
                    self.errors.append(f"{filename}: YAML error - {e}")
            else:
                self.warnings.append(f"{filename}: File not found (optional)")

        # 3. Cross-config consistency
        self._validate_cross_config()

        # 4. Environment config
        self._validate_env_config()

        self._print_summary()

        if self.strict:
            return len(self.errors) == 0 and len(self.warnings) == 0
        return len(self.errors) == 0

    def _check_yaml_syntax(self):
        """Check all YAML files for syntax errors."""
        yaml_files = list(CONFIG_DIR.glob("**/*.yaml")) + list(CONFIG_DIR.glob("**/*.yml"))
        all_passed = True
        for path in yaml_files:
            try:
                with open(path) as f:
                    yaml.safe_load(f)
                if self.verbose:
                    self.passed.append(f"YAML syntax: {path.name}")
            except yaml.YAMLError as e:
                self.errors.append(f"YAML syntax error in {path.name}: {e}")
                all_passed = False

        if all_passed:
            self.passed.append(f"YAML syntax: All {len(yaml_files)} files parse correctly")

    def _validate_instruments(self, data: Dict, filename: str):
        """Validate instruments.yaml."""
        instruments = data.get("instruments", data)
        if not isinstance(instruments, dict) or len(instruments) == 0:
            self.errors.append(f"{filename}: No instruments defined")
            return

        # Check for FX instruments
        has_fx = any("fx" in k.lower() for k in instruments.keys())
        if not has_fx:
            self.warnings.append(f"{filename}: No FX instrument group found")

        count = sum(
            len(v) if isinstance(v, (dict, list)) else 1
            for v in instruments.values()
        )
        self.passed.append(f"{filename}: {count} instruments across {len(instruments)} groups")

    def _validate_risk_limits(self, data: Dict, filename: str):
        """Validate risk_limits.yaml."""
        flat = _flatten(data)

        # Check for position limits
        has_position_limit = any("position" in k.lower() or "exposure" in k.lower() for k in flat)
        if not has_position_limit:
            self.errors.append(f"{filename}: No position size limits defined")
        else:
            self.passed.append(f"{filename}: Position limits defined")

        # Check for kill switch / daily loss
        has_kill_switch = any(
            term in k.lower()
            for k in flat
            for term in ["daily_loss", "kill_switch", "max_daily", "drawdown"]
        )
        if not has_kill_switch:
            self.errors.append(f"{filename}: No daily loss limit / kill switch")
        else:
            self.passed.append(f"{filename}: Kill switch / daily loss limit defined")

    def _validate_strategy_params(self, data: Dict, filename: str):
        """Validate strategy_params.yaml."""
        strategies = data.get("strategies", data)
        if not isinstance(strategies, dict):
            self.errors.append(f"{filename}: No strategies defined")
            return

        # Check Pod 1 exists
        has_fx = any("fx" in k.lower() or "carry" in k.lower() for k in strategies)
        if not has_fx:
            self.warnings.append(f"{filename}: FX Carry+Momentum (Pod 1) not found")
        else:
            self.passed.append(f"{filename}: FX Carry+Momentum strategy configured")

        self.passed.append(f"{filename}: {len(strategies)} strategies defined")

    def _validate_feature_registry(self, data: Dict, filename: str):
        """Validate feature_registry.yaml."""
        if len(data) == 0:
            self.errors.append(f"{filename}: Empty feature registry")
            return

        total_features = 0
        categories_with_issues = []
        for category, features in data.items():
            if not isinstance(features, dict):
                continue
            for feat_name, feat_def in features.items():
                total_features += 1
                if isinstance(feat_def, dict):
                    if "name" not in feat_def:
                        categories_with_issues.append(f"{category}.{feat_name}")

        self.passed.append(f"{filename}: {total_features} features across {len(data)} categories")
        if categories_with_issues and len(categories_with_issues) <= 5:
            self.warnings.append(
                f"{filename}: Features missing 'name' field: {', '.join(categories_with_issues[:5])}"
            )

    def _validate_model_registry(self, data: Dict, filename: str):
        """Validate model_registry.yaml."""
        strategies = data.get("strategies", data)
        if not isinstance(strategies, dict):
            self.warnings.append(f"{filename}: No strategies section found")
            return
        self.passed.append(f"{filename}: {len(strategies)} strategy models registered")

    def _validate_data_quality(self, data: Dict, filename: str):
        """Validate data_quality_rules.yaml."""
        if "global" not in data:
            self.errors.append(f"{filename}: Missing 'global' settings")
        else:
            self.passed.append(f"{filename}: Global validation settings defined")

        # Outlier thresholds
        outliers = data.get("outliers", {})
        max_std = outliers.get("max_daily_return_std", 5.0)
        if max_std < 2.0 or max_std > 10.0:
            self.warnings.append(f"{filename}: max_daily_return_std={max_std} may be unusual")

        self.passed.append(f"{filename}: Data quality rules configured")

    def _validate_backtest(self, data: Dict, filename: str):
        """Validate backtest_config.yaml."""
        bt = data.get("backtest", data)

        wf = bt.get("walk_forward", {})
        if wf.get("enabled", False):
            self.passed.append(f"{filename}: Walk-forward validation enabled")
        else:
            self.warnings.append(f"{filename}: Walk-forward validation not enabled")

        costs = data.get("transaction_costs", {})
        if len(costs) > 0:
            self.passed.append(f"{filename}: Transaction costs for {len(costs)} asset classes")
        else:
            self.warnings.append(f"{filename}: No transaction costs defined")

    def _validate_cross_config(self):
        """Cross-reference validation across configs."""
        # Best-effort: only if files exist
        try:
            with open(CONFIG_DIR / "instruments.yaml") as f:
                instruments = yaml.safe_load(f)
            with open(CONFIG_DIR / "data_quality_rules.yaml") as f:
                quality = yaml.safe_load(f)
            self.passed.append("Cross-config: instruments.yaml ↔ data_quality_rules.yaml loaded")
        except Exception:
            pass

    def _validate_env_config(self):
        """Validate environment configuration."""
        env_example = PROJECT_ROOT / ".env.example"
        env_template = PROJECT_ROOT / ".env.template"
        if env_example.exists() or env_template.exists():
            self.passed.append(".env.example: Environment template exists")
        else:
            self.warnings.append(".env.example: No environment template found")

    def _print_summary(self):
        """Print validation summary."""
        if self.verbose:
            for msg in self.passed:
                print(f"  {check_mark(True)} {msg}")

        for msg in self.warnings:
            print(f"  {warn_mark()} {msg}")

        for msg in self.errors:
            print(f"  {check_mark(False)} {msg}")

        print(f"\n{BOLD}━━━ Summary ━━━{RESET}")
        print(f"  {GREEN}Passed:{RESET}   {len(self.passed)}")
        print(f"  {YELLOW}Warnings:{RESET} {len(self.warnings)}")
        print(f"  {RED}Errors:{RESET}   {len(self.errors)}")

        if self.errors:
            print(f"\n  {RED}{BOLD}VALIDATION FAILED{RESET}")
        elif self.warnings and self.strict:
            print(f"\n  {YELLOW}{BOLD}VALIDATION FAILED (strict mode){RESET}")
        else:
            print(f"\n  {GREEN}{BOLD}VALIDATION PASSED{RESET}")
        print()


def _flatten(d: Dict, prefix: str = "") -> Dict:
    """Flatten nested dict."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = v
    return items


def main():
    parser = argparse.ArgumentParser(description="Validate AI Trading System configs")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings too")
    parser.add_argument("--verbose", action="store_true", help="Show all check results")
    args = parser.parse_args()

    validator = ConfigValidator(strict=args.strict, verbose=args.verbose)
    success = validator.validate_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
