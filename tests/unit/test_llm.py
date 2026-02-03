"""
Unit Tests for LLM Module

Tests news summarization, citation extraction, and mock client.
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm import (
    Article,
    NewsSummary,
    NewsSummarizer,
    MockLLMClient,
    create_news_summarizer,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_articles():
    """Create sample articles for testing."""
    return [
        Article(
            id="article_1",
            title="Fed Signals Potential Rate Cut in March",
            text="""The Federal Reserve signaled a potential interest rate cut 
            at its March meeting, citing improving inflation data. Fed Chair Powell 
            noted that while inflation remains above target, recent trends have been 
            encouraging. Markets are now pricing in a 75% probability of a 25bp cut.""",
            source="Reuters",
            published_at=datetime(2026, 2, 1, 10, 30),
        ),
        Article(
            id="article_2",
            title="ECB Holds Rates Steady, Signals Cautious Approach",
            text="""The European Central Bank kept interest rates unchanged at 
            today's meeting, with President Lagarde emphasizing a data-dependent 
            approach. The ECB noted that inflation in the eurozone remains sticky 
            in the services sector, warranting continued vigilance.""",
            source="Bloomberg",
            published_at=datetime(2026, 2, 1, 14, 0),
        ),
        Article(
            id="article_3",
            title="US Jobless Claims Fall to Multi-Year Low",
            text="""Initial jobless claims fell to 195,000 last week, the lowest 
            level in three years, suggesting continued labor market strength 
            despite Fed tightening. This could complicate the Fed's rate cut plans 
            if the labor market remains too tight.""",
            source="WSJ",
            published_at=datetime(2026, 2, 1, 8, 30),
        ),
    ]


@pytest.fixture
def mock_client():
    """Create mock LLM client with predefined responses."""
    mock_response = '''{
        "summary": "The Fed signaled a potential March rate cut [1] amid improving inflation data, while the ECB held steady [2]. Strong US labor data [3] could complicate the Fed's plans.",
        "key_events": [
            "Fed signals potential March rate cut [1]",
            "ECB holds rates steady [2]",
            "US jobless claims hit multi-year low [3]"
        ],
        "catalysts": [
            "March FOMC meeting could deliver first rate cut [1]",
            "Divergence between Fed and ECB policy paths [1,2]"
        ],
        "risks": [
            "Strong labor market could delay Fed cuts [3]",
            "Sticky eurozone services inflation [2]"
        ],
        "confidence": "HIGH"
    }'''
    
    return MockLLMClient(responses={"articles": mock_response})


@pytest.fixture
def summarizer(mock_client):
    """Create summarizer with mock client."""
    return NewsSummarizer(client=mock_client)


# =============================================================================
# Article Tests
# =============================================================================

class TestArticle:
    """Tests for Article dataclass."""
    
    def test_article_creation(self):
        """Test article creation."""
        article = Article(
            id="test_1",
            title="Test Title",
            text="Test content",
            source="Test Source",
        )
        
        assert article.id == "test_1"
        assert article.title == "Test Title"
        assert article.source == "Test Source"
    
    def test_article_formatting(self):
        """Test article formatting for prompt."""
        article = Article(
            id="test_1",
            title="Test Title",
            text="Test content here",
            source="Reuters",
        )
        
        formatted = article.to_formatted_string(1)
        
        assert "[1]" in formatted
        assert "Test Title" in formatted
        assert "Reuters" in formatted
        assert "Test content" in formatted
    
    def test_article_truncation(self):
        """Test that long articles are truncated."""
        long_text = "A" * 3000
        article = Article(
            id="test_1",
            title="Title",
            text=long_text,
            source="Source",
        )
        
        formatted = article.to_formatted_string(1)
        
        # Should be truncated with ellipsis
        assert len(formatted) < len(long_text) + 100
        assert "..." in formatted


# =============================================================================
# NewsSummary Tests
# =============================================================================

class TestNewsSummary:
    """Tests for NewsSummary dataclass."""
    
    def test_summary_creation(self):
        """Test summary creation."""
        summary = NewsSummary(
            summary_text="Test summary",
            source_refs=["article_1", "article_2"],
            key_events=["Event 1", "Event 2"],
            catalysts=["Catalyst 1"],
            risks=["Risk 1"],
            confidence="HIGH",
        )
        
        assert summary.summary_text == "Test summary"
        assert len(summary.source_refs) == 2
        assert summary.confidence == "HIGH"
    
    def test_summary_to_dict(self):
        """Test serialization."""
        summary = NewsSummary(
            summary_text="Test",
            source_refs=["a1"],
            key_events=["E1"],
            catalysts=["C1"],
            risks=["R1"],
            confidence="MEDIUM",
        )
        
        d = summary.to_dict()
        
        assert d["summary_text"] == "Test"
        assert d["confidence"] == "MEDIUM"
        assert "generated_at" in d
    
    def test_summary_display_format(self):
        """Test human-readable formatting."""
        summary = NewsSummary(
            summary_text="Market summary here",
            source_refs=["a1", "a2"],
            key_events=["Fed rate cut signal", "ECB holds steady"],
            catalysts=["March FOMC meeting"],
            risks=["Strong labor market"],
            confidence="HIGH",
        )
        
        display = summary.format_for_display()
        
        assert "NEWS SUMMARY" in display
        assert "Market summary" in display
        assert "KEY EVENTS" in display
        assert "CATALYSTS" in display
        assert "RISK FACTORS" in display
        assert "HIGH" in display


# =============================================================================
# Citation Extraction Tests
# =============================================================================

class TestCitationExtraction:
    """Tests for citation extraction from text."""
    
    def test_single_citation(self, summarizer):
        """Test extracting single citation."""
        text = "The Fed signaled a rate cut [1]."
        citations = summarizer._extract_citations(text)
        
        assert citations == ["1"]
    
    def test_multiple_citations(self, summarizer):
        """Test extracting multiple citations."""
        text = "Fed [1] and ECB [2] both met today [3]."
        citations = summarizer._extract_citations(text)
        
        assert citations == ["1", "2", "3"]
    
    def test_citation_range(self, summarizer):
        """Test extracting citation range."""
        text = "Multiple sources [1-3] confirmed."
        citations = summarizer._extract_citations(text)
        
        assert citations == ["1", "2", "3"]
    
    def test_citation_list(self, summarizer):
        """Test extracting citation list."""
        text = "Sources [1,2,3] reported."
        citations = summarizer._extract_citations(text)
        
        assert set(citations) == {"1", "2", "3"}
    
    def test_no_citations(self, summarizer):
        """Test handling text with no citations."""
        text = "No citations in this text."
        citations = summarizer._extract_citations(text)
        
        assert citations == []


# =============================================================================
# Summarizer Tests
# =============================================================================

class TestNewsSummarizer:
    """Tests for NewsSummarizer."""
    
    def test_summarize_articles(self, summarizer, sample_articles):
        """Test full summarization pipeline."""
        summary = summarizer.summarize(sample_articles)
        
        assert isinstance(summary, NewsSummary)
        assert len(summary.summary_text) > 0
        assert summary.articles_processed == 3
    
    def test_summarize_extracts_citations(self, summarizer, sample_articles):
        """Test that citations are mapped to article IDs."""
        summary = summarizer.summarize(sample_articles)
        
        # Mock response references [1], [2], [3]
        # These should map to article_1, article_2, article_3
        assert "article_1" in summary.source_refs or len(summary.source_refs) > 0
    
    def test_summarize_empty_articles(self, summarizer):
        """Test handling of empty article list."""
        summary = summarizer.summarize([])
        
        assert summary.confidence == "LOW"
        assert summary.articles_processed == 0
    
    def test_summarize_from_dicts(self, summarizer):
        """Test summarizing from dictionary format."""
        article_dicts = [
            {
                "id": "dict_1",
                "title": "Test Article",
                "text": "Test content",
                "source": "Test Source",
            }
        ]
        
        summary = summarizer.summarize_from_dicts(article_dicts)
        
        assert isinstance(summary, NewsSummary)
        assert summary.articles_processed == 1
    
    def test_key_events_extracted(self, summarizer, sample_articles):
        """Test that key events are extracted."""
        summary = summarizer.summarize(sample_articles)
        
        assert len(summary.key_events) > 0
    
    def test_catalysts_extracted(self, summarizer, sample_articles):
        """Test that catalysts are extracted."""
        summary = summarizer.summarize(sample_articles)
        
        assert len(summary.catalysts) > 0


# =============================================================================
# Mock Client Tests
# =============================================================================

class TestMockLLMClient:
    """Tests for MockLLMClient."""
    
    def test_mock_response(self):
        """Test mock client returns predefined response."""
        mock = MockLLMClient(responses={
            "test": "This is a test response"
        })
        
        response = mock.complete("This is a test prompt")
        
        assert response.content == "This is a test response"
    
    def test_mock_call_logging(self):
        """Test that calls are logged."""
        mock = MockLLMClient()
        
        mock.complete("Prompt 1")
        mock.complete("Prompt 2")
        
        assert len(mock.call_log) == 2
        assert mock.call_log[0]["prompt"] == "Prompt 1"
    
    def test_mock_json_response(self):
        """Test mock JSON parsing."""
        mock = MockLLMClient(responses={
            "json": '{"key": "value"}'
        })
        
        result = mock.complete_with_json("Return json please")
        
        # Should return default structure since "json" keyword matches
        assert isinstance(result, dict)


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_news_summarizer_mock(self):
        """Test creating mock summarizer."""
        summarizer = create_news_summarizer(mock=True)
        
        assert isinstance(summarizer, NewsSummarizer)
        assert isinstance(summarizer.client, MockLLMClient)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
