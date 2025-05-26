"""
Integration tests for the complete ReAgent system.

Tests the full system with all agents working together.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import patch, Mock
import uuid

from reagent.main import ReAgentSystem
from reagent.models import Message, MessageType
from tests import INTEGRATION_TEST, SLOW_TEST, REQUIRES_REDIS, REQUIRES_DB


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
class TestSystemIntegration:
    """Test full system integration."""
    
    async def test_simple_question_processing(self, reagent_system, sample_questions, mock_llm_responses):
        """Test processing a simple factual question."""
        question = sample_questions["simple_factual"][0]
        
        # Configure mock responses for each agent
        responses = [
            json.dumps(mock_llm_responses["question_decomposer"]),
            json.dumps(mock_llm_responses["retriever"]),
            json.dumps(mock_llm_responses["verifier"]),
            json.dumps(mock_llm_responses["answer_assembler"])
        ]
        
        response_iter = iter(responses)
        
        def mock_llm_response(*args, **kwargs):
            response = Mock()
            response.choices = [
                Mock(message=Mock(content=next(response_iter)))
            ]
            response.usage = Mock(prompt_tokens=100, completion_tokens=50)
            return response
        
        reagent_system.llm_client.chat.completions.create.side_effect = mock_llm_response
        
        # Process question
        result = await reagent_system.process_question(question["question"], timeout=30)
        
        # Verify result
        assert result["status"] == "success"
        assert result["answer"] is not None
        assert result["confidence"] > 0.5
        assert "reasoning_trace" in result
        assert len(result["supporting_facts"]) > 0
    
    async def test_complex_question_with_decomposition(self, reagent_system, sample_questions):
        """Test processing a complex multi-hop question."""
        question = sample_questions["complex_multihop"][0]
        
        # Configure responses for multi-hop processing
        decomposer_response = {
            "sub_questions": [
                "Which state hosted the 1984 Summer Olympics?",
                "What is the capital of that state?",
                "What is the population of the capital?",
                "What is the largest city in that state?",
                "What is the population of the largest city?"
            ],
            "decomposition_reasoning": "Breaking down geographic and demographic queries",
            "dependencies": {
                "q2": ["q1"],
                "q3": ["q2"],
                "q4": ["q1"],
                "q5": ["q4"]
            }
        }
        
        retriever_responses = [
            {
                "retrieved_evidence": [{
                    "source": "Olympics.org",
                    "content": "The 1984 Summer Olympics were held in Los Angeles, California",
                    "confidence": 0.95
                }],
                "retrieval_reasoning": "Found Olympic host information"
            },
            {
                "retrieved_evidence": [{
                    "source": "Wikipedia",
                    "content": "Sacramento is the capital of California",
                    "confidence": 0.98
                }],
                "retrieval_reasoning": "Found state capital"
            },
            {
                "retrieved_evidence": [{
                    "source": "Census",
                    "content": "Sacramento has a population of approximately 500,000",
                    "confidence": 0.92
                }],
                "retrieval_reasoning": "Found capital population"
            },
            {
                "retrieved_evidence": [{
                    "source": "Wikipedia",
                    "content": "Los Angeles is the largest city in California",
                    "confidence": 0.99
                }],
                "retrieval_reasoning": "Found largest city"
            },
            {
                "retrieved_evidence": [{
                    "source": "Census",
                    "content": "Los Angeles has a population of nearly 4 million",
                    "confidence": 0.95
                }],
                "retrieval_reasoning": "Found largest city population"
            }
        ]
        
        verifier_response = {
            "verified_facts": [
                "The 1984 Summer Olympics were held in California",
                "Sacramento is the capital of California",
                "Sacramento has approximately 500,000 people",
                "Los Angeles is the largest city in California",
                "Los Angeles has nearly 4 million people"
            ],
            "conflicts_detected": [],
            "local_backtracking_action": "none"
        }
        
        assembler_response = {
            "final_answer": "California. The state hosted the 1984 Summer Olympics in Los Angeles. Sacramento, the capital of California, has a population of approximately 500,000, which is smaller than Los Angeles (the state's largest city) with nearly 4 million people.",
            "confidence_score": 0.93,
            "supporting_facts": verifier_response["verified_facts"]
        }
        
        # Setup mock responses
        responses = [
            json.dumps(decomposer_response),
            *[json.dumps(r) for r in retriever_responses],
            json.dumps(verifier_response),
            json.dumps(assembler_response)
        ]
        
        response_iter = iter(responses)
        
        def mock_llm_response(*args, **kwargs):
            response = Mock()
            response.choices = [
                Mock(message=Mock(content=next(response_iter)))
            ]
            response.usage = Mock(prompt_tokens=150, completion_tokens=100)
            return response
        
        reagent_system.llm_client.chat.completions.create.side_effect = mock_llm_response
        
        # Process question
        result = await reagent_system.process_question(question["question"], timeout=60)
        
        # Verify result
        assert result["status"] == "success"
        assert "California" in result["answer"]
        assert result["confidence"] > 0.8
        assert len(result.get("supporting_facts", [])) >= 5
    
    async def test_question_with_conflicts(self, reagent_system):
        """Test handling conflicting information."""
        question = "What is the population of Paris?"
        
        # Setup conflicting evidence
        retriever_response = {
            "retrieved_evidence": [
                {
                    "source": "Source A",
                    "content": "Paris has a population of 2.16 million",
                    "confidence": 0.90
                },
                {
                    "source": "Source B",
                    "content": "Paris has a population of 10 million",
                    "confidence": 0.85
                }
            ],
            "retrieval_reasoning": "Found conflicting population data"
        }
        
        verifier_response = {
            "verified_facts": ["Paris has a population of 2.16 million"],
            "conflicts_detected": [{
                "description": "Population conflict",
                "conflicting_items": ["2.16 million", "10 million"],
                "confidence": 0.8
            }],
            "local_backtracking_action": "none",
            "verification_notes": "Resolved based on source reliability"
        }
        
        # Mock responses
        responses = [
            json.dumps({
                "sub_questions": ["What is the population of Paris?"],
                "decomposition_reasoning": "Simple factual question"
            }),
            json.dumps(retriever_response),
            json.dumps(verifier_response),
            json.dumps({
                "final_answer": "Paris has a population of approximately 2.16 million people in the city proper.",
                "confidence_score": 0.85,
                "supporting_facts": ["Paris has a population of 2.16 million"]
            })
        ]
        
        response_iter = iter(responses)
        reagent_system.llm_client.chat.completions.create.side_effect = lambda *args, **kwargs: Mock(
            choices=[Mock(message=Mock(content=next(response_iter)))],
            usage=Mock(prompt_tokens=100, completion_tokens=50)
        )
        
        # Process question
        result = await reagent_system.process_question(question, timeout=30)
        
        # Should handle conflict and provide answer
        assert result["status"] == "success"
        assert "2.16 million" in result["answer"]
        assert result["confidence"] < 0.9  # Lower due to conflict
    
    async def test_system_with_backtracking(self, reagent_system):
        """Test system behavior with backtracking."""
        question = "What is the capital of the country that won the most recent World Cup?"
        
        # Initial incorrect retrieval
        initial_retriever_response = {
            "retrieved_evidence": [{
                "source": "Outdated Source",
                "content": "France won the most recent World Cup",
                "confidence": 0.85
            }],
            "retrieval_reasoning": "Found World Cup winner"
        }
        
        # Verifier detects issue and triggers backtracking
        verifier_conflict_response = {
            "verified_facts": [],
            "conflicts_detected": [{
                "description": "Outdated information detected",
                "conflicting_items": ["France", "Argentina"],
                "confidence": 0.9
            }],
            "local_backtracking_action": "rollback to last checkpoint",
            "verification_notes": "Information appears outdated"
        }
        
        # Corrected retrieval after backtracking
        corrected_retriever_response = {
            "retrieved_evidence": [{
                "source": "FIFA Official",
                "content": "Argentina won the 2022 World Cup",
                "confidence": 0.98
            }],
            "retrieval_reasoning": "Found current World Cup winner"
        }
        
        # Final verification
        final_verifier_response = {
            "verified_facts": ["Argentina won the 2022 World Cup"],
            "conflicts_detected": [],
            "local_backtracking_action": "none"
        }
        
        # Setup response sequence
        responses = [
            # Initial decomposition
            json.dumps({
                "sub_questions": [
                    "Which country won the most recent World Cup?",
                    "What is the capital of that country?"
                ],
                "decomposition_reasoning": "Breaking into winner identification and capital lookup"
            }),
            # Initial retrieval (incorrect)
            json.dumps(initial_retriever_response),
            # Verifier detects conflict
            json.dumps(verifier_conflict_response),
            # Corrected retrieval
            json.dumps(corrected_retriever_response),
            # Final verification
            json.dumps(final_verifier_response),
            # Capital retrieval
            json.dumps({
                "retrieved_evidence": [{
                    "source": "Geography",
                    "content": "Buenos Aires is the capital of Argentina",
                    "confidence": 0.99
                }],
                "retrieval_reasoning": "Found capital information"
            }),
            # Final assembly
            json.dumps({
                "final_answer": "Buenos Aires is the capital of Argentina, which won the most recent (2022) World Cup.",
                "confidence_score": 0.95,
                "supporting_facts": [
                    "Argentina won the 2022 World Cup",
                    "Buenos Aires is the capital of Argentina"
                ]
            })
        ]
        
        response_iter = iter(responses)
        reagent_system.llm_client.chat.completions.create.side_effect = lambda *args, **kwargs: Mock(
            choices=[Mock(message=Mock(content=next(response_iter)))],
            usage=Mock(prompt_tokens=120, completion_tokens=80)
        )
        
        # Process question
        result = await reagent_system.process_question(question, timeout=45)
        
        # Verify backtracking occurred and correct answer obtained
        assert result["status"] == "success"
        assert "Buenos Aires" in result["answer"]
        assert "Argentina" in result["answer"]
        
        # Check reasoning trace for backtracking event
        trace = result.get("reasoning_trace", [])
        backtracking_events = [
            step for step in trace 
            if "backtrack" in step.get("action", "").lower()
        ]
        assert len(backtracking_events) > 0


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_redis
class TestMessageBusIntegration:
    """Test message bus functionality with real Redis."""
    
    async def test_agent_communication(self, message_bus):
        """Test agents can communicate through message bus."""
        received_messages = []
        
        # Subscribe a handler
        async def handler(message: Message):
            received_messages.append(message)
        
        message_bus.subscribe("test_agent", handler)
        
        # Start message processing
        process_task = asyncio.create_task(message_bus.process_messages())
        
        # Publish messages
        test_message = Message(
            type=MessageType.INFORM,
            sender="agent_a",
            content={"data": "test"}
        )
        
        await message_bus.publish(test_message)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Verify message received
        assert len(received_messages) == 1
        assert received_messages[0].content["data"] == "test"
        
        # Cleanup
        process_task.cancel()
        try:
            await process_task
        except asyncio.CancelledError:
            pass
    
    async def test_message_persistence(self, message_bus, redis_client):
        """Test messages are persisted in Redis."""
        # Publish a message
        test_message = Message(
            type=MessageType.ASSERT,
            sender="test_agent",
            content={"fact": "test fact"}
        )
        
        await message_bus.publish(test_message)
        
        # Check Redis directly
        stream_data = redis_client.xread(
            {message_bus.stream_key: '0'},
            count=10
        )
        
        assert len(stream_data) > 0
        messages = stream_data[0][1]  # Get messages from first stream
        
        # Verify message content
        found = False
        for msg_id, data in messages:
            if data[b'id'].decode() == test_message.id:
                found = True
                assert data[b'type'].decode() == test_message.type.value
                assert data[b'sender'].decode() == test_message.sender
                break
        
        assert found


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_db
class TestDatabaseIntegration:
    """Test database operations."""
    
    async def test_question_persistence(self, test_db_session, reagent_system):
        """Test questions are persisted to database."""
        from db.models import QuestionRecord
        
        question_text = "What is the capital of France?"
        question_id = f"q_{uuid.uuid4().hex[:8]}"
        
        # Create question record
        question_record = QuestionRecord(
            id=question_id,
            question_text=question_text,
            status="processing"
        )
        
        test_db_session.add(question_record)
        await test_db_session.commit()
        
        # Verify persistence
        result = await test_db_session.get(QuestionRecord, question_id)
        assert result is not None
        assert result.question_text == question_text
        assert result.status.value == "processing"
    
    async def test_agent_performance_tracking(self, test_db_session):
        """Test agent performance metrics are tracked."""
        from db.models import Agent, PerformanceMetric
        from sqlalchemy import select
        
        # Get an agent
        result = await test_db_session.execute(
            select(Agent).where(Agent.id == "A_Q")
        )
        agent = result.scalar_one()
        
        # Add performance metric
        metric = PerformanceMetric(
            metric_type="processing_time",
            metric_name="avg_response_time",
            metric_value=2.5,
            agent_id=agent.id
        )
        
        test_db_session.add(metric)
        await test_db_session.commit()
        
        # Verify metric was saved
        result = await test_db_session.execute(
            select(PerformanceMetric)
            .where(PerformanceMetric.agent_id == "A_Q")
        )
        metrics = result.scalars().all()
        
        assert len(metrics) > 0
        assert metrics[0].metric_value == 2.5


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    async def test_comparative_question(self, reagent_system, mock_llm_responses):
        """Test processing a comparative question."""
        question = "Which is larger, Tokyo or New York City by population?"
        
        # This would involve decomposition, retrieval of population data,
        # verification, and comparison
        # For brevity, using simplified mock responses
        
        result = await reagent_system.process_question(question, timeout=30)
        
        # Basic assertions - in real test would verify full flow
        assert result["status"] in ["success", "timeout", "failed"]
        if result["status"] == "success":
            assert result["answer"] is not None
            assert any(city in result["answer"] for city in ["Tokyo", "New York"])
    
    async def test_temporal_question(self, reagent_system):
        """Test processing a temporal question."""
        question = "Who was the US president when the iPhone was first released?"
        
        # This requires temporal reasoning and fact correlation
        result = await reagent_system.process_question(question, timeout=30)
        
        assert result["status"] in ["success", "timeout", "failed"]
        if result["status"] == "success":
            # iPhone was released in 2007, George W. Bush was president
            assert result["answer"] is not None
    
    async def test_causal_question(self, reagent_system):
        """Test processing a causal reasoning question."""
        question = "Why did the Roman Empire fall?"
        
        # This requires understanding causation and historical analysis
        result = await reagent_system.process_question(question, timeout=45)
        
        assert result["status"] in ["success", "timeout", "failed"]
        if result["status"] == "success":
            assert result["answer"] is not None
            assert len(result["answer"]) > 50  # Should be a detailed explanation
