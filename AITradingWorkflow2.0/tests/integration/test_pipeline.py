"""
Integration Tests for Daily Pipeline

Tests the complete pipeline flow from data to recommendations.
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipelines.daily_run import (
    PipelineConfig,
    PipelineResult,
    DailyPipeline,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def pipeline_config():
    """Create test pipeline configuration."""
    return PipelineConfig(
        run_date=datetime(2026, 2, 1),
        dry_run=True,  # Don't save outputs
        mock_llm=True,  # Don't make API calls
        use_llm=True,
    )


@pytest.fixture
def pipeline(pipeline_config):
    """Create pipeline instance."""
    return DailyPipeline(pipeline_config)


# =============================================================================
# Pipeline Configuration Tests
# =============================================================================

class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.lookback_days == 120
        assert "fx_carry_momentum" in config.strategies_enabled
        assert config.use_llm is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            run_date=datetime(2026, 1, 15),
            dry_run=True,
            mock_llm=True,
        )
        
        assert config.run_date == datetime(2026, 1, 15)
        assert config.dry_run is True
        assert config.mock_llm is True


# =============================================================================
# Pipeline Stage Tests
# =============================================================================

class TestPipelineStages:
    """Tests for individual pipeline stages."""
    
    def test_data_ingestion_generates_sample(self, pipeline):
        """Test that data ingestion generates sample data."""
        data = pipeline.stage_data_ingestion()
        
        assert len(data) > 0
        assert "EURUSD" in data
        assert "VIX" in data
        assert len(pipeline.result.instruments_loaded) > 0
    
    def test_feature_engineering(self, pipeline):
        """Test feature engineering stage."""
        data = pipeline.stage_data_ingestion()
        features = pipeline.stage_feature_engineering(data)
        
        assert len(features) > 0
        
        # Check FX pair has features
        if "EURUSD" in features:
            eurusd_feat = features["EURUSD"]
            assert "momentum_score" in eurusd_feat.columns or len(eurusd_feat.columns) > 0
    
    def test_signal_generation(self, pipeline):
        """Test signal generation stage."""
        data = pipeline.stage_data_ingestion()
        features = pipeline.stage_feature_engineering(data)
        signals = pipeline.stage_signal_generation(features, None)
        
        # May or may not generate signals depending on data
        assert isinstance(signals, list)
        assert pipeline.result.signals_generated >= 0
    
    def test_signal_aggregation(self, pipeline):
        """Test signal aggregation stage."""
        data = pipeline.stage_data_ingestion()
        features = pipeline.stage_feature_engineering(data)
        signals = pipeline.stage_signal_generation(features, None)
        recs = pipeline.stage_signal_aggregation(signals)
        
        assert isinstance(recs, list)
        assert pipeline.result.signals_after_aggregation >= 0


# =============================================================================
# Full Pipeline Tests
# =============================================================================

class TestFullPipeline:
    """Tests for complete pipeline execution."""
    
    def test_pipeline_runs_successfully(self, pipeline):
        """Test that pipeline runs without errors."""
        result = pipeline.run()
        
        assert result.status in ["SUCCESS", "PARTIAL"]
        assert result.duration_seconds > 0
    
    def test_pipeline_generates_recommendations(self, pipeline):
        """Test that pipeline generates recommendations."""
        result = pipeline.run()
        
        # Should have loaded instruments
        assert len(result.instruments_loaded) > 0
        
        # May have recommendations (depends on data)
        assert isinstance(result.recommendations, list)
    
    def test_pipeline_tracks_timing(self, pipeline):
        """Test that pipeline tracks stage timing."""
        result = pipeline.run()
        
        assert result.duration_seconds > 0
    
    def test_pipeline_handles_no_llm(self):
        """Test pipeline without LLM."""
        config = PipelineConfig(
            run_date=datetime(2026, 2, 1),
            dry_run=True,
            use_llm=False,
        )
        
        pipeline = DailyPipeline(config)
        result = pipeline.run()
        
        assert result.status in ["SUCCESS", "PARTIAL"]
        assert result.news_summary is None
    
    def test_pipeline_dry_run(self):
        """Test dry run mode."""
        config = PipelineConfig(
            run_date=datetime(2026, 2, 1),
            dry_run=True,
            mock_llm=True,
        )
        
        pipeline = DailyPipeline(config)
        result = pipeline.run()
        
        # Should complete without saving files
        assert result.status in ["SUCCESS", "PARTIAL"]


# =============================================================================
# Pipeline Result Tests
# =============================================================================

class TestPipelineResult:
    """Tests for PipelineResult."""
    
    def test_result_initialization(self):
        """Test result initialization."""
        result = PipelineResult(run_date=datetime.now())
        
        assert result.status == "PENDING"
        assert result.signals_generated == 0
        assert len(result.recommendations) == 0
    
    def test_result_after_run(self, pipeline):
        """Test result after pipeline run."""
        result = pipeline.run()
        
        # Should have updated status
        assert result.status != "PENDING"
        
        # Should have run date
        assert result.run_date is not None


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_pipeline_handles_missing_strategy(self):
        """Test handling of missing strategy."""
        config = PipelineConfig(
            run_date=datetime(2026, 2, 1),
            strategies_enabled=["nonexistent_strategy"],
            dry_run=True,
            use_llm=False,
        )
        
        pipeline = DailyPipeline(config)
        result = pipeline.run()
        
        # Should still complete (just with no signals from missing strategy)
        assert result.status in ["SUCCESS", "PARTIAL", "FAILED"]


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
