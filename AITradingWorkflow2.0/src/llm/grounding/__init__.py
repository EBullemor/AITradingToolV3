"""
LLM Grounding Module

Verifies that LLM-generated claims are supported by source documents.

Components:
- ClaimExtractor: Extracts factual claims from text
- SourceMatcher: Matches claims to source documents
- GroundingScorer: Computes confidence scores
- GroundingReportGenerator: Creates comprehensive reports

Usage:
    from src.llm.grounding import generate_grounding_report, Source
    
    sources = [
        Source(id="1", title="ECB Update", text="ECB raised rates by 25bps..."),
        Source(id="2", title="FX Report", text="EUR/USD fell 0.5% following..."),
    ]
    
    report = generate_grounding_report(llm_output, sources)
    
    if not report.meets_threshold:
        print("Warning: Low grounding score")
        for ungrounded in report.ungrounded_details:
            print(f"  - {ungrounded['claim_text']}")
"""

from .claim_extractor import (
    ClaimType,
    Claim,
    ClaimExtractor,
    extract_claims,
)

from .source_matcher import (
    Source,
    SourceMatch,
    GroundingScore,
    SourceMatcher,
    GroundingScorer,
    match_and_score,
)

from .grounding_report import (
    GroundingReport,
    GroundingReportGenerator,
    generate_grounding_report,
)


__all__ = [
    # Claim extraction
    "ClaimType",
    "Claim",
    "ClaimExtractor",
    "extract_claims",
    
    # Source matching
    "Source",
    "SourceMatch",
    "GroundingScore",
    "SourceMatcher",
    "GroundingScorer",
    "match_and_score",
    
    # Reports
    "GroundingReport",
    "GroundingReportGenerator",
    "generate_grounding_report",
]
