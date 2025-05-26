"""
ReAgent Test Suite

This package contains all tests for the ReAgent multi-agent reasoning system,
including unit tests, integration tests, and test fixtures.
"""

import os
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_TIMEOUT = 30  # seconds
TEST_LLM_MODEL = "gpt-3.5-turbo"  # Use cheaper model for tests
TEST_REDIS_DB = 15  # Use separate Redis DB for tests
TEST_POSTGRES_DB = "reagent_test"

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["LLM_MODEL"] = TEST_LLM_MODEL

# Test markers
SLOW_TEST = "slow"
INTEGRATION_TEST = "integration"
UNIT_TEST = "unit"
REQUIRES_LLM = "requires_llm"
REQUIRES_DB = "requires_db"
REQUIRES_REDIS = "requires_redis"

# Common test utilities
class TestConfig:
    """Test configuration settings."""
    
    # Use mock LLM for most tests
    use_mock_llm = True
    
    # Timeouts
    question_timeout = 10
    agent_timeout = 5
    
    # Test data paths
    fixtures_dir = Path(__file__).parent / "fixtures"
    sample_questions_file = fixtures_dir / "sample_questions.json"
    
    # Database settings
    test_db_url = f"postgresql://reagent_user:testpass@localhost/{TEST_POSTGRES_DB}"
    
    # Redis settings
    redis_host = "localhost"
    redis_port = 6379
    redis_db = TEST_REDIS_DB
    
    # Agent settings
    max_backtrack_depth = 3
    checkpoint_retention_hours = 1

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", f"{SLOW_TEST}: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", f"{INTEGRATION_TEST}: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", f"{UNIT_TEST}: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", f"{REQUIRES_LLM}: marks tests that require LLM API access"
    )
    config.addinivalue_line(
        "markers", f"{REQUIRES_DB}: marks tests that require database access"
    )
    config.addinivalue_line(
        "markers", f"{REQUIRES_REDIS}: marks tests that require Redis access"
    )
