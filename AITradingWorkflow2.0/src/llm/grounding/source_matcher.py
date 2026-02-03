"""
Source Matcher and Grounding Scorer

Matches extracted claims against source documents
and computes grounding confidence scores.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .claim_extractor import Claim, ClaimType


@dataclass
class Source:
    """A source document for verification."""
    id: str
    title: str
    text: str
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    source_type: str = "news"  # news, data, statement
    
    def get_excerpt(self, max_length: int = 200) -> str:
        """Get text excerpt."""
        if len(self.text) <= max_length:
            return self.text
        return self.text[:max_length] + "..."


@dataclass
class SourceMatch:
    """A match between a claim and a source."""
    source: Source
    similarity_score: float
    matched_text: str
    match_type: str  # exact, semantic, partial
    
    def to_dict(self) -> Dict:
        return {
            "source_id": self.source.id,
            "source_title": self.source.title,
            "similarity": self.similarity_score,
            "matched_text": self.matched_text,
            "match_type": self.match_type,
        }


@dataclass
class GroundingScore:
    """Grounding score for a claim."""
    claim: Claim
    overall_score: float  # 0-1, confidence claim is grounded
    
    # Best match details
    best_match: Optional[SourceMatch] = None
    all_matches: List[SourceMatch] = field(default_factory=list)
    
    # Verification details
    is_grounded: bool = False
    verification_method: str = ""
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "claim_text": self.claim.text,
            "claim_type": self.claim.claim_type.value,
            "score": self.overall_score,
            "is_grounded": self.is_grounded,
            "verification_method": self.verification_method,
            "explanation": self.explanation,
            "best_match": self.best_match.to_dict() if self.best_match else None,
        }


class SourceMatcher:
    """
    Matches claims to source documents.
    
    Uses text similarity and keyword matching.
    For production, would use embeddings (sentence-transformers).
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.5,
        top_k: int = 3
    ):
        """
        Initialize matcher.
        
        Args:
            similarity_threshold: Minimum similarity for match
            top_k: Number of top matches to return
        """
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
    
    def _compute_keyword_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute simple keyword-based similarity.
        
        Uses Jaccard similarity on words.
        """
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _check_entity_match(
        self,
        claim: Claim,
        source_text: str
    ) -> float:
        """Check if claim entities appear in source."""
        if not claim.entities:
            return 0.0
        
        source_lower = source_text.lower()
        matches = 0
        
        for entity in claim.entities:
            if entity.lower() in source_lower:
                matches += 1
        
        return matches / len(claim.entities)
    
    def _check_number_match(
        self,
        claim: Claim,
        source_text: str
    ) -> float:
        """Check if claim numbers appear in source."""
        import re
        
        if not claim.numbers:
            return 0.0
        
        # Extract numbers from source
        number_pattern = r"[-+]?[\d,]+\.?\d*"
        source_numbers = []
        
        for match in re.finditer(number_pattern, source_text):
            try:
                num_str = match.group().replace(",", "")
                source_numbers.append(float(num_str))
            except ValueError:
                pass
        
        if not source_numbers:
            return 0.0
        
        # Check if claim numbers appear in source
        matches = 0
        for claim_num in claim.numbers:
            # Allow small tolerance for floating point
            for source_num in source_numbers:
                if abs(claim_num - source_num) < 0.01 * max(abs(claim_num), 1):
                    matches += 1
                    break
        
        return matches / len(claim.numbers)
    
    def find_matches(
        self,
        claim: Claim,
        sources: List[Source]
    ) -> List[SourceMatch]:
        """
        Find sources that support a claim.
        
        Args:
            claim: Claim to verify
            sources: Available source documents
        
        Returns:
            List of matching sources
        """
        matches = []
        
        for source in sources:
            # Compute similarity
            text_sim = self._compute_keyword_similarity(claim.text, source.text)
            entity_score = self._check_entity_match(claim, source.text)
            number_score = self._check_number_match(claim, source.text)
            
            # Weighted score
            # Entity and number matches are strong signals
            overall_sim = (
                text_sim * 0.4 +
                entity_score * 0.4 +
                number_score * 0.2
            )
            
            if overall_sim >= self.similarity_threshold:
                # Determine match type
                if entity_score > 0.8 and number_score > 0.8:
                    match_type = "exact"
                elif entity_score > 0.5 or number_score > 0.5:
                    match_type = "partial"
                else:
                    match_type = "semantic"
                
                matches.append(SourceMatch(
                    source=source,
                    similarity_score=overall_sim,
                    matched_text=source.get_excerpt(),
                    match_type=match_type,
                ))
        
        # Sort by similarity
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        
        return matches[:self.top_k]


class GroundingScorer:
    """
    Computes grounding confidence scores.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.7,
        require_entity_match: bool = True
    ):
        """
        Initialize scorer.
        
        Args:
            min_confidence: Minimum score to be considered grounded
            require_entity_match: Require entity match for grounding
        """
        self.min_confidence = min_confidence
        self.require_entity_match = require_entity_match
    
    def score_claim(
        self,
        claim: Claim,
        matches: List[SourceMatch]
    ) -> GroundingScore:
        """
        Score how well a claim is grounded in sources.
        
        Args:
            claim: The claim to score
            matches: Source matches found
        
        Returns:
            GroundingScore
        """
        if not matches:
            return GroundingScore(
                claim=claim,
                overall_score=0.0,
                is_grounded=False,
                verification_method="no_matches",
                explanation="No supporting sources found",
            )
        
        best_match = matches[0]
        
        # Base score from similarity
        base_score = best_match.similarity_score
        
        # Boost for exact matches
        if best_match.match_type == "exact":
            base_score = min(1.0, base_score * 1.2)
        
        # Determine if grounded
        is_grounded = base_score >= self.min_confidence
        
        # Generate explanation
        if is_grounded:
            explanation = f"Claim supported by {best_match.source.title}"
            if best_match.match_type == "exact":
                explanation += " (exact match)"
        else:
            explanation = f"Weak support ({base_score:.0%} confidence)"
        
        return GroundingScore(
            claim=claim,
            overall_score=base_score,
            best_match=best_match,
            all_matches=matches,
            is_grounded=is_grounded,
            verification_method=best_match.match_type,
            explanation=explanation,
        )


def match_and_score(
    claims: List[Claim],
    sources: List[Source],
    min_confidence: float = 0.7
) -> List[GroundingScore]:
    """
    Match claims to sources and compute grounding scores.
    
    Args:
        claims: Claims to verify
        sources: Source documents
        min_confidence: Minimum confidence threshold
    
    Returns:
        List of grounding scores
    """
    matcher = SourceMatcher()
    scorer = GroundingScorer(min_confidence=min_confidence)
    
    scores = []
    
    for claim in claims:
        matches = matcher.find_matches(claim, sources)
        score = scorer.score_claim(claim, matches)
        scores.append(score)
    
    return scores
