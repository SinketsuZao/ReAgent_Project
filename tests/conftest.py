"""
Pytest configuration and fixtures for ReAgent tests.

This module provides common fixtures and configuration for all tests.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import redis
import aioredis

from reagent.main import ReAgentSystem, MessageBus, LocalKnowledge
from reagent.models import Message, MessageType, KnowledgeAssertion
from reagent.config import settings
from db.models import Base, QuestionRecord, Agent, AgentType
from db.session import get_async_db
from worker.celery_app import app as celery_app

# Configure event loop
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Database fixtures
@pytest_asyncio.fixture(scope="function")
async def test_db_engine():
    """Create a test database engine."""
    # Use in-memory SQLite for tests
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()

@pytest_asyncio.fixture(scope="function")
async def test_db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        # Add default agents
        for agent_id, agent_type in [
            ('A_Q', AgentType.DECOMPOSER),
            ('A_R', AgentType.RETRIEVER),
            ('A_V', AgentType.VERIFIER),
            ('A_A', AgentType.ASSEMBLER),
            ('A_S', AgentType.SUPERVISOR),
            ('A_C', AgentType.CONTROLLER)
        ]:
            agent = Agent(id=agent_id, agent_type=agent_type)
            session.add(agent)
        
        await session.commit()
        yield session

# Redis fixtures
@pytest.fixture(scope="function")
def redis_client():
    """Create a test Redis client."""
    client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=15,  # Use separate DB for tests
        decode_responses=True
    )
    
    # Clear test database
    client.flushdb()
    
    yield client
    
    # Cleanup
    client.flushdb()
    client.close()

@pytest_asyncio.fixture(scope="function")
async def async_redis_client():
    """Create an async Redis client for tests."""
    client = await aioredis.from_url(
        f"redis://localhost:6379/15",
        encoding="utf-8",
        decode_responses=True
    )
    
    await client.flushdb()
    
    yield client
    
    await client.flushdb()
    await client.close()

# Message bus fixtures
@pytest_asyncio.fixture(scope="function")
async def message_bus(async_redis_client):
    """Create a test message bus."""
    bus = MessageBus(redis_client=async_redis_client)
    yield bus

# Mock LLM fixtures
@pytest.fixture(scope="function")
def mock_llm_client():
    """Create a mock OpenAI client."""
    mock_client = AsyncMock()
    
    # Default response
    mock_response = Mock()
    mock_response.choices = [
        Mock(message=Mock(content='{"result": "test response"}'))
    ]
    mock_response.usage = Mock(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )
    
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client

@pytest.fixture(scope="function")
def mock_llm_responses():
    """Predefined LLM responses for different agent types."""
    return {
        "question_decomposer": {
            "sub_questions": [
                "What is the capital of France?",
                "What is the population of that capital?"
            ],
            "decomposition_reasoning": "Breaking down into location and demographic queries",
            "dependencies": {"q2": ["q1"]},
            "question_type": "factual",
            "key_entities": ["France", "capital", "population"],
            "complexity_factors": ["geographic", "demographic"]
        },
        "retriever": {
            "retrieved_evidence": [
                {
                    "source": "Wikipedia",
                    "content": "Paris is the capital of France",
                    "confidence": 0.95,
                    "relevance": "Directly answers the capital question",
                    "metadata": {"year": 2024, "type": "article"}
                },
                {
                    "source": "Census Data",
                    "content": "Paris has a population of 2.16 million",
                    "confidence": 0.90,
                    "relevance": "Provides population information",
                    "metadata": {"year": 2023, "type": "official"}
                }
            ],
            "retrieval_reasoning": "Found authoritative sources for both facts",
            "search_terms_used": ["France capital", "Paris population"],
            "gaps_identified": []
        },
        "verifier": {
            "verified_facts": [
                "Paris is the capital of France",
                "Paris has a population of 2.16 million"
            ],
            "conflicts_detected": [],
            "local_backtracking_action": "none",
            "verification_notes": "All facts consistent and verified",
            "confidence_adjustments": {}
        },
        "answer_assembler": {
            "final_answer": "The capital of France is Paris, which has a population of 2.16 million people.",
            "partial_answer_synthesis": ["Combined geographic and demographic facts"],
            "confidence_score": 0.92,
            "supporting_facts": [
                "Paris is the capital of France",
                "Paris has a population of 2.16 million"
            ],
            "escalation_signal": "none",
            "answer_metadata": {
                "answer_type": "factual",
                "certainty": "high",
                "completeness": "complete"
            }
        }
    }

# ReAgent system fixtures
@pytest_asyncio.fixture(scope="function")
async def reagent_system(mock_llm_client, message_bus):
    """Create a test ReAgent system with mocked components."""
    with patch('reagent.main.openai.AsyncOpenAI', return_value=mock_llm_client):
        system = ReAgentSystem()
        system.llm_client = mock_llm_client
        system.message_bus = message_bus
        
        # Start the system
        await system.start()
        
        yield system
        
        # Cleanup
        await system.stop()

# Test data fixtures
@pytest.fixture(scope="session")
def sample_questions():
    """Load sample questions from fixtures."""
    fixtures_path = Path(__file__).parent / "fixtures" / "sample_questions.json"
    with open(fixtures_path, 'r') as f:
        return json.load(f)

@pytest.fixture(scope="function")
def sample_message():
    """Create a sample message."""
    return Message(
        type=MessageType.INFORM,
        sender="A_Q",
        content={
            "sub_questions": ["Question 1?", "Question 2?"],
            "reasoning": "Test reasoning"
        },
        id=str(uuid.uuid4())
    )

@pytest.fixture(scope="function")
def sample_knowledge_assertion():
    """Create a sample knowledge assertion."""
    return KnowledgeAssertion(
        content="Paris is the capital of France",
        source="Wikipedia",
        confidence=0.95,
        dependencies=[]
    )

# Local knowledge fixtures
@pytest.fixture(scope="function")
def local_knowledge():
    """Create a test LocalKnowledge instance."""
    return LocalKnowledge("test_agent")

# Celery fixtures
@pytest.fixture(scope="function")
def celery_worker():
    """Configure Celery for testing."""
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        broker_url='memory://',
        result_backend='cache+memory://'
    )
    return celery_app

# Helper fixtures
@pytest.fixture(scope="function")
def mock_question_data():
    """Generate mock question processing data."""
    return {
        "question": "What is the capital of the country that hosted the 2024 Olympics?",
        "question_id": f"q_{uuid.uuid4().hex[:8]}",
        "expected_answer": "Paris is the capital of France, which hosted the 2024 Olympics.",
        "sub_questions": [
            "Which country hosted the 2024 Olympics?",
            "What is the capital of that country?"
        ],
        "evidence": [
            {
                "content": "The 2024 Summer Olympics were held in France",
                "source": "Olympics.org",
                "confidence": 0.95
            },
            {
                "content": "Paris is the capital of France",
                "source": "Wikipedia",
                "confidence": 0.98
            }
        ]
    }

@pytest.fixture(scope="function")
def mock_metrics_collector():
    """Mock metrics collector to avoid side effects."""
    with patch('reagent.monitoring.metrics_collector') as mock:
        mock.record_question_processed = Mock()
        mock.record_agent_processing = Mock()
        mock.record_backtracking = Mock()
        mock.record_token_usage = Mock()
        mock.update_active_agents = Mock()
        yield mock

# Environment fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ.update({
        "ENVIRONMENT": "testing",
        "LOG_LEVEL": "DEBUG",
        "OPENAI_API_KEY": "test-key-123",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "reagent_test",
        "POSTGRES_USER": "reagent_user",
        "POSTGRES_PASSWORD": "testpass"
    })
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

# Async test utilities
@pytest.fixture
def async_timeout():
    """Provide a reasonable timeout for async tests."""
    return 10.0

async def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
    """Wait for a condition to become true."""
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        if await condition_func():
            return True
        await asyncio.sleep(interval)
    return False

# Export utility for tests
pytest_plugins = ["pytest_asyncio"]
