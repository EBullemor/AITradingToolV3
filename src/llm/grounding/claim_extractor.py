"""
Claim Extractor Module

Extracts factual claims from LLM-generated text for verification.
Identifies price claims, event claims, attribution claims, and causal claims.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class ClaimType(Enum):
    """Types of factual claims."""
    PRICE = "price"           # "EUR/USD rose 0.5%"
    EVENT = "event"           # "ECB raised rates by 25bps"
    ATTRIBUTION = "attribution"  # "Powell said..."
    CAUSAL = "causal"         # "Oil rose due to..."
    TEMPORAL = "temporal"     # "Yesterday", "Last week"
    NUMERICAL = "numerical"   # Any numerical fact
    UNKNOWN = "unknown"


@dataclass
class Claim:
    """A factual claim extracted from text."""
    text: str
    claim_type: ClaimType
    source_sentence: str
    
    # Extracted entities
    entities: List[str] = field(default_factory=list)
    numbers: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    
    # Position in source
    start_pos: int = 0
    end_pos: int = 0
    
    # Metadata
    confidence: float = 1.0  # Extraction confidence
    requires_verification: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "type": self.claim_type.value,
            "entities": self.entities,
            "numbers": self.numbers,
            "requires_verification": self.requires_verification,
        }


class ClaimExtractor:
    """
    Extracts factual claims from LLM output.
    
    Uses pattern matching and heuristics to identify
    statements that need source verification.
    """
    
    # Patterns for different claim types
    PRICE_PATTERNS = [
        r"(?P<asset>\w+(/\w+)?)\s+(rose|fell|gained|lost|dropped|surged|plunged|climbed)\s+(?P<amount>[\d.]+%?)",
        r"(?P<asset>\w+(/\w+)?)\s+(traded|trading)\s+(at|around|near)\s+(?P<price>[\d.,]+)",
        r"(?P<asset>\w+(/\w+)?)\s+(hit|reached|touched)\s+(?P<level>[\d.,]+)",
    ]
    
    EVENT_PATTERNS = [
        r"(?P<entity>(Fed|ECB|BOJ|BOE|RBA|SNB))\s+(raised|cut|held|maintained)\s+(rates?|interest rates?)",
        r"(?P<entity>\w+)\s+(announced|released|reported|published)\s+",
        r"(?P<entity>\w+)\s+(approved|rejected|passed|signed)\s+",
    ]
    
    ATTRIBUTION_PATTERNS = [
        r"(?P<person>\w+\s+\w+)\s+(said|stated|noted|mentioned|commented|remarked)",
        r"according to\s+(?P<source>\w+(\s+\w+)?)",
        r"(?P<source>\w+)\s+reported\s+that",
    ]
    
    CAUSAL_PATTERNS = [
        r"(?P<effect>.+)\s+(due to|because of|driven by|caused by|amid|following)\s+(?P<cause>.+)",
        r"(?P<cause>.+)\s+(led to|resulted in|caused|triggered)\s+(?P<effect>.+)",
    ]
    
    # Numbers and percentages
    NUMBER_PATTERN = r"[-+]?[\d,]+\.?\d*%?"
    
    # Currency/Asset patterns
    ASSET_PATTERN = r"(?:EUR|USD|GBP|JPY|AUD|CHF|CAD|NZD|BTC|ETH|XAU|XAG|CL|GC|SI|HG)"
    FX_PAIR_PATTERN = rf"({ASSET_PATTERN}/?{ASSET_PATTERN})"
    
    def __init__(self, min_claim_length: int = 10):
        """
        Initialize claim extractor.
        
        Args:
            min_claim_length: Minimum characters for a valid claim
        """
        self.min_claim_length = min_claim_length
        
        # Compile patterns
        self.price_patterns = [re.compile(p, re.IGNORECASE) for p in self.PRICE_PATTERNS]
        self.event_patterns = [re.compile(p, re.IGNORECASE) for p in self.EVENT_PATTERNS]
        self.attribution_patterns = [re.compile(p, re.IGNORECASE) for p in self.ATTRIBUTION_PATTERNS]
        self.causal_patterns = [re.compile(p, re.IGNORECASE) for p in self.CAUSAL_PATTERNS]
        
        self.number_pattern = re.compile(self.NUMBER_PATTERN)
        self.fx_pattern = re.compile(self.FX_PAIR_PATTERN)
    
    def extract_claims(self, text: str) -> List[Claim]:
        """
        Extract all claims from text.
        
        Args:
            text: LLM-generated text to analyze
        
        Returns:
            List of extracted claims
        """
        claims = []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            if len(sentence) < self.min_claim_length:
                continue
            
            # Try to classify and extract
            claim = self._analyze_sentence(sentence)
            
            if claim:
                claims.append(claim)
        
        logger.debug(f"Extracted {len(claims)} claims from text")
        
        return claims
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        # Could be improved with NLP library
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_sentence(self, sentence: str) -> Optional[Claim]:
        """Analyze a sentence and create claim if factual."""
        # Check each pattern type
        claim_type = ClaimType.UNKNOWN
        entities = []
        numbers = []
        
        # Extract numbers
        number_matches = self.number_pattern.findall(sentence)
        for num_str in number_matches:
            try:
                # Remove commas and percentage
                clean = num_str.replace(",", "").replace("%", "")
                numbers.append(float(clean))
            except ValueError:
                pass
        
        # Extract FX pairs / assets
        fx_matches = self.fx_pattern.findall(sentence)
        entities.extend(fx_matches)
        
        # Check for price claims
        for pattern in self.price_patterns:
            if pattern.search(sentence):
                claim_type = ClaimType.PRICE
                match = pattern.search(sentence)
                if match and match.groupdict().get("asset"):
                    entities.append(match.group("asset"))
                break
        
        # Check for event claims
        if claim_type == ClaimType.UNKNOWN:
            for pattern in self.event_patterns:
                if pattern.search(sentence):
                    claim_type = ClaimType.EVENT
                    match = pattern.search(sentence)
                    if match and match.groupdict().get("entity"):
                        entities.append(match.group("entity"))
                    break
        
        # Check for attribution claims
        if claim_type == ClaimType.UNKNOWN:
            for pattern in self.attribution_patterns:
                if pattern.search(sentence):
                    claim_type = ClaimType.ATTRIBUTION
                    match = pattern.search(sentence)
                    if match:
                        if match.groupdict().get("person"):
                            entities.append(match.group("person"))
                        if match.groupdict().get("source"):
                            entities.append(match.group("source"))
                    break
        
        # Check for causal claims
        if claim_type == ClaimType.UNKNOWN:
            for pattern in self.causal_patterns:
                if pattern.search(sentence):
                    claim_type = ClaimType.CAUSAL
                    break
        
        # If has numbers but no other type, it's numerical
        if claim_type == ClaimType.UNKNOWN and numbers:
            claim_type = ClaimType.NUMERICAL
        
        # Only create claim if it's a factual type
        if claim_type != ClaimType.UNKNOWN:
            return Claim(
                text=sentence,
                claim_type=claim_type,
                source_sentence=sentence,
                entities=list(set(entities)),  # Dedupe
                numbers=numbers,
                requires_verification=True,
            )
        
        return None
    
    def extract_with_context(
        self,
        text: str,
        context_sentences: int = 1
    ) -> List[Claim]:
        """
        Extract claims with surrounding context.
        
        Args:
            text: Text to analyze
            context_sentences: Number of context sentences to include
        
        Returns:
            Claims with context attached
        """
        sentences = self._split_sentences(text)
        claims = []
        
        for i, sentence in enumerate(sentences):
            claim = self._analyze_sentence(sentence)
            
            if claim:
                # Add context
                start = max(0, i - context_sentences)
                end = min(len(sentences), i + context_sentences + 1)
                
                context = " ".join(sentences[start:end])
                claim.source_sentence = context
                
                claims.append(claim)
        
        return claims


def extract_claims(text: str) -> List[Claim]:
    """
    Convenience function to extract claims.
    
    Args:
        text: LLM output text
    
    Returns:
        List of claims
    """
    extractor = ClaimExtractor()
    return extractor.extract_claims(text)
