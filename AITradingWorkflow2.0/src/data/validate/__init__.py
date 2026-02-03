"""
Data Validation Module

Validates incoming data for:
- Schema compliance
- Data quality (missing values, outliers)
- Look-ahead bias prevention
"""

from .schema import (
    SchemaValidator,
    InstrumentSchema,
    validate_schema,
)
from .quality_checks import (
    QualityChecker,
    QualityReport,
    check_missing_values,
    check_outliers,
    check_stale_data,
)
from .bias_checks import (
    BiasChecker,
    check_lookahead_bias,
    check_survivorship_bias,
)
from .quarantine import (
    QuarantineManager,
    quarantine_data,
)

# Convenience classes
class DataValidator:
    """Combined validator using all checks."""
    
    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.quality_checker = QualityChecker()
        self.bias_checker = BiasChecker()


class ValidationResult:
    """Result of validation."""
    
    def __init__(self, is_valid: bool, errors: list = None, warnings: list = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []


def validate_dataframe(df, instrument_type: str = "fx") -> ValidationResult:
    """Convenience function to validate a dataframe."""
    validator = DataValidator()
    errors = []
    warnings = []
    
    # Add validation logic here
    is_valid = len(errors) == 0
    
    return ValidationResult(is_valid, errors, warnings)


__all__ = [
    # Schema
    "SchemaValidator",
    "InstrumentSchema",
    "validate_schema",
    
    # Quality
    "QualityChecker",
    "QualityReport",
    "check_missing_values",
    "check_outliers",
    "check_stale_data",
    
    # Bias
    "BiasChecker",
    "check_lookahead_bias",
    "check_survivorship_bias",
    
    # Quarantine
    "QuarantineManager",
    "quarantine_data",
    
    # Convenience
    "DataValidator",
    "ValidationResult",
    "validate_dataframe",
]
