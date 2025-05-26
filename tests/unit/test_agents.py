"""
Unit tests for ReAgent agents.

Tests individual agent functionality with mocked dependencies.
"""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import uuid

from reagent.main import (
    QuestionDecomposerAgent, RetrieverAgent, VerifierAgent,
    AnswerAssemblerAgent, SupervisorAgent, ControllerAgent,
    BaseAgent, MessageBus
)
from reagent.models import Message, MessageType, KnowledgeAssertion
from tests import UNIT_TEST


@pytest.mark.asyncio
@pytest.mark.unit
class TestBaseAgent:
    """Test the base agent functionality."""
    
    async def test_agent_initialization(self, mock_llm_client, message_bus):
        """Test base agent initialization."""
        agent = BaseAgent("test_agent", mock_llm_client, message_bus)
        
        assert agent.agent_id == "test_agent"
        assert agent.llm_client == mock_llm_client
        assert agent.message_bus == message_bus
        assert agent.temperature == 0.6
        assert agent.local_knowledge is not None
        assert agent.processing_times == []
        assert agent.token_usage == {'input': 0, 'output': 0}
    
    async def test_call_llm_success(self, mock_llm_client, message_bus):
        """Test successful LLM call."""
        # Create a test agent
        class TestAgent(BaseAgent):
            def get_prompt_template(self):
                return "Test prompt template"
            
            async def handle_message(self, message):
                pass
        
        agent = TestAgent("test_agent", mock_llm_client, message_bus)
        
        # Configure mock response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"result": "test response"}'))
        ]
        mock_response.usage = Mock(prompt_tokens=50, completion_tokens=25)
        mock_llm_client.chat.completions.create.return_value = mock_response
        
        # Call LLM
        result = await agent.call_llm("Test prompt")
        
        assert result == '{"result": "test response"}'
        assert agent.token_usage['input'] == 50
        assert agent.token_usage['output'] == 25
        assert len(agent.processing_times) == 1
    
    async def test_call_llm_retry_on_rate_limit(self, mock_llm_client, message_bus):
        """Test LLM call retry on rate limit error."""
        class TestAgent(BaseAgent):
            def get_prompt_template(self):
                return "Test prompt"
            
            async def handle_message(self, message):
                pass
        
        agent = TestAgent("test_agent", mock_llm_client, message_bus)
        
        # Configure mock to fail once then succeed
        import openai
        mock_llm_client.chat.completions.create.side_effect = [
            openai.RateLimitError("Rate limit exceeded"),
            Mock(choices=[Mock(message=Mock(content='{"result": "success"}'))], usage=Mock(prompt_tokens=10, completion_tokens=5))
        ]
        
        # Should retry and succeed
        with patch('asyncio.sleep', new=AsyncMock()):
            result = await agent.call_llm("Test prompt")
        
        assert result == '{"result": "success"}'
        assert mock_llm_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
@pytest.mark.unit
class TestQuestionDecomposerAgent:
    """Test the Question Decomposer agent."""
    
    async def test_decompose_simple_question(self, mock_llm_client, message_bus, mock_llm_responses):
        """Test decomposing a simple question."""
        agent = QuestionDecomposerAgent("A_Q", mock_llm_client, message_bus)
        
        # Configure mock response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps(mock_llm_responses["question_decomposer"])))
        ]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_llm_client.chat.completions.create.return_value = mock_response
        
        # Create message
        message = Message(
            type=MessageType.ASSERT,
            sender="user",
            content={"original_question": "What is the capital of France and its population?"}
        )
        
        # Process message
        published_messages = []
        
        async def capture_publish(msg):
            published_messages.append(msg)
        
        message_bus.publish = capture_publish
        
        await agent.handle_message(message)
        
        # Verify decomposition
        assert len(published_messages) == 1
        result = published_messages[0].content
        assert "sub_questions" in result
        assert len(result["sub_questions"]) == 2
        assert result["reasoning"] is not None
    
    async def test_complexity_analysis(self, mock_llm_client, message_bus):
        """Test question complexity analysis."""
        agent = QuestionDecomposerAgent("A_Q", mock_llm_client, message_bus)
        
        # Test various complexity levels
        simple_q = "What is the capital of France?"
        complex_q = "Compare the economic growth of Japan and Germany after World War II, considering both industrial development and technological innovation."
        
        simple_score = agent._analyze_complexity(simple_q)
        complex_score = agent._analyze_complexity(complex_q)
        
        assert simple_score < complex_score
        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1
    
    async def test_decomposition_validation(self, mock_llm_client, message_bus):
        """Test validation of decomposition results."""
        agent = QuestionDecomposerAgent("A_Q", mock_llm_client, message_bus)
        
        # Valid decomposition
        valid_result = {
            "sub_questions": ["Q1?", "Q2?"],
            "decomposition_reasoning": "Valid reasoning"
        }
        assert agent._validate_decomposition(valid_result, "Original question?")
        
        # Invalid decompositions
        assert not agent._validate_decomposition({}, "Original?")
        assert not agent._validate_decomposition({"sub_questions": []}, "Original?")
        assert not agent._validate_decomposition({"sub_questions": ["Q1"], "decomposition_reasoning": ""}, "Original?")


@pytest.mark.asyncio
@pytest.mark.unit
class TestRetrieverAgent:
    """Test the Retriever agent."""
    
    async def test_retrieve_evidence(self, mock_llm_client, message_bus, mock_llm_responses):
        """Test evidence retrieval."""
        agent = RetrieverAgent("A_R", mock_llm_client, message_bus)
        
        # Configure mock response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps(mock_llm_responses["retriever"])))
        ]
        mock_response.usage = Mock(prompt_tokens=150, completion_tokens=100)
        mock_llm_client.chat.completions.create.return_value = mock_response
        
        # Create message
        message = Message(
            type=MessageType.INFORM,
            sender="A_Q",
            content={
                "sub_questions": ["What is the capital of France?"],
                "dependencies": {}
            }
        )
        
        # Process message
        published_messages = []
        
        async def capture_publish(msg):
            published_messages.append(msg)
        
        message_bus.publish = capture_publish
        
        await agent.handle_message(message)
        
        # Verify retrieval
        assert len(published_messages) == 1
        result = published_messages[0].content
        assert "evidence" in result
        assert len(result["evidence"]) > 0
        assert all(e.get("confidence", 0) > 0 for e in result["evidence"])
    
    async def test_evidence_filtering(self, mock_llm_client, message_bus):
        """Test evidence filtering and ranking."""
        agent = RetrieverAgent("A_R", mock_llm_client, message_bus)
        
        evidence_list = [
            {"content": "Paris is the capital", "confidence": 0.9, "source": "A"},
            {"content": "London is nice", "confidence": 0.8, "source": "B"},
            {"content": "The capital is Paris", "confidence": 0.95, "source": "C"},
            {"content": "France has Paris", "confidence": 0.7, "source": "D"}
        ]
        
        filtered = agent._filter_evidence(evidence_list, "What is the capital of France?")
        
        # Should rank by relevance and confidence
        assert len(filtered) == len(evidence_list)
        assert filtered[0]["source"] in ["A", "C"]  # Most relevant
    
    async def test_query_enhancement(self, mock_llm_client, message_bus):
        """Test query enhancement functionality."""
        agent = RetrieverAgent("A_R", mock_llm_client, message_bus)
        
        query = "capital of France"
        context = {
            "temporal_context": "2024",
            "entities": ["France", "Europe"],
            "complexity": 0.8
        }
        
        enhanced = agent._enhance_query(query, context)
        
        assert "2024" in enhanced
        assert "France" in enhanced
        assert "Europe" in enhanced
        assert "multiple sources" in enhanced.lower()


@pytest.mark.asyncio
@pytest.mark.unit
class TestVerifierAgent:
    """Test the Verifier agent."""
    
    async def test_verify_consistent_evidence(self, mock_llm_client, message_bus, mock_llm_responses):
        """Test verification of consistent evidence."""
        agent = VerifierAgent("A_V", mock_llm_client, message_bus)
        
        # Configure mock response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps(mock_llm_responses["verifier"])))
        ]
        mock_response.usage = Mock(prompt_tokens=120, completion_tokens=80)
        mock_llm_client.chat.completions.create.return_value = mock_response
        
        # Create message with evidence
        message = Message(
            type=MessageType.INFORM,
            sender="A_R",
            content={
                "sub_question": "What is the capital of France?",
                "evidence": [
                    {
                        "content": "Paris is the capital of France",
                        "source": "Wikipedia",
                        "confidence": 0.95
                    }
                ]
            }
        )
        
        # Process message
        published_messages = []
        
        async def capture_publish(msg):
            published_messages.append(msg)
        
        message_bus.publish = capture_publish
        
        await agent.handle_message(message)
        
        # Should verify facts without conflicts
        assert len(published_messages) == 1
        result = published_messages[0].content
        assert result["verified_facts"] is not None
        assert len(result["verified_facts"]) > 0
    
    async def test_conflict_detection(self, mock_llm_client, message_bus):
        """Test conflict detection in local knowledge."""
        agent = VerifierAgent("A_V", mock_llm_client, message_bus)
        
        # Add conflicting assertions
        agent.local_knowledge.add_assertion(
            KnowledgeAssertion("Paris is the capital of France", "Source1", 0.9)
        )
        agent.local_knowledge.add_assertion(
            KnowledgeAssertion("Lyon is the capital of France", "Source2", 0.8)
        )
        
        conflicts = agent.local_knowledge.find_conflicts()
        assert len(conflicts) > 0
    
    async def test_backtracking_decision(self, mock_llm_client, message_bus):
        """Test backtracking decision logic."""
        agent = VerifierAgent("A_V", mock_llm_client, message_bus)
        
        # High confidence conflict should trigger backtracking
        high_conflict = {
            "conflicts_detected": [
                {"confidence": 0.9, "description": "Major conflict"}
            ],
            "local_backtracking_action": "rollback"
        }
        assert agent._should_backtrack(high_conflict)
        
        # Low confidence conflicts should not
        low_conflict = {
            "conflicts_detected": [
                {"confidence": 0.3, "description": "Minor discrepancy"}
            ],
            "local_backtracking_action": "none"
        }
        assert not agent._should_backtrack(low_conflict)


@pytest.mark.asyncio
@pytest.mark.unit
class TestAnswerAssemblerAgent:
    """Test the Answer Assembler agent."""
    
    async def test_answer_assembly(self, mock_llm_client, message_bus, mock_llm_responses):
        """Test assembling final answer from verified facts."""
        agent = AnswerAssemblerAgent("A_A", mock_llm_client, message_bus)
        
        # Configure mock response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps(mock_llm_responses["answer_assembler"])))
        ]
        mock_response.usage = Mock(prompt_tokens=200, completion_tokens=150)
        mock_llm_client.chat.completions.create.return_value = mock_response
        
        # Add verified facts
        agent.verified_facts["q1"] = ["Paris is the capital of France"]
        agent.verified_facts["q2"] = ["Paris has a population of 2.16 million"]
        
        # Trigger assembly
        await agent.assemble_answer()
        
        # Verify the assembled answer
        assert agent.local_knowledge.assertions  # Should have stored final answer
    
    async def test_assembly_readiness_check(self, mock_llm_client, message_bus):
        """Test checking if enough facts are ready for assembly."""
        agent = AnswerAssemblerAgent("A_A", mock_llm_client, message_bus)
        
        # Not ready initially
        agent.assembly_start_time = None
        assert await agent.check_assembly_ready() is None  # Won't assemble
        
        # Add enough facts
        agent.verified_facts["q1"] = ["Fact 1", "Fact 2"]
        agent.verified_facts["q2"] = ["Fact 3"]
        agent.assembly_start_time = datetime.now().timestamp()
        
        # Should be ready now
        with patch.object(agent, 'assemble_answer', new=AsyncMock()):
            await agent.check_assembly_ready()
            agent.assemble_answer.assert_called_once()
    
    async def test_fact_conflict_detection(self, mock_llm_client, message_bus):
        """Test detection of conflicts in facts before assembly."""
        agent = AnswerAssemblerAgent("A_A", mock_llm_client, message_bus)
        
        facts = [
            "Paris is the capital of France",
            "Lyon is the capital of France",
            "France is in Europe"
        ]
        
        conflicts = agent._detect_fact_conflicts(facts)
        assert len(conflicts) > 0
        assert any("capital" in str(c).lower() for c in conflicts)


@pytest.mark.asyncio
@pytest.mark.unit
class TestSupervisorAgent:
    """Test the Supervisor agent."""
    
    async def test_global_conflict_analysis(self, mock_llm_client, message_bus):
        """Test global conflict analysis."""
        agent = SupervisorAgent("A_S", mock_llm_client, message_bus)
        
        # Configure mock response
        analysis_response = {
            "conflict_summary": ["Agent A and B have conflicting facts"],
            "affected_agents": ["A_V", "A_R"],
            "rollback_strategy": "selective",
            "target_checkpoint": "latest_stable",
            "resolution_reasoning": "Conflicts are isolated to two agents",
            "expected_outcome": "Consistent state after rollback"
        }
        
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps(analysis_response)))
        ]
        mock_response.usage = Mock(prompt_tokens=300, completion_tokens=200)
        mock_llm_client.chat.completions.create.return_value = mock_response
        
        # Create conflict message
        message = Message(
            type=MessageType.CHALLENGE,
            sender="A_A",
            content={
                "reason": "Unable to resolve conflicts",
                "conflicts": [{"fact1": "A", "fact2": "B"}],
                "partial_answers": {}
            }
        )
        
        # Process message
        published_messages = []
        
        async def capture_publish(msg):
            published_messages.append(msg)
        
        message_bus.publish = capture_publish
        
        await agent.handle_message(message)
        
        # Should analyze and decide on strategy
        assert len(published_messages) > 0
    
    async def test_consensus_extraction(self, mock_llm_client, message_bus):
        """Test extracting consensus facts from global knowledge."""
        agent = SupervisorAgent("A_S", mock_llm_client, message_bus)
        
        # Simulate global knowledge
        agent.global_knowledge = {
            "A_V": {"facts": ["Paris is capital", "Population 2M"]},
            "A_R": {"facts": ["Paris is capital", "Area 105 km²"]},
            "A_A": {"facts": ["Paris is capital"]}
        }
        
        consensus = agent._extract_consensus_facts()
        assert "Paris is capital" in consensus
        assert "Population 2M" not in consensus  # Not in all agents
        assert "Area 105 km²" not in consensus  # Not in all agents


@pytest.mark.asyncio
@pytest.mark.unit
class TestControllerAgent:
    """Test the Controller agent."""
    
    async def test_performance_monitoring(self, mock_llm_client, message_bus):
        """Test agent performance monitoring."""
        agent = ControllerAgent("A_C", mock_llm_client, message_bus)
        
        # Simulate performance metrics
        agent.performance_metrics["A_Q"]["failures"] = 5
        agent.performance_metrics["A_Q"]["successes"] = 95
        agent.agent_reliability["A_Q"] = 0.95
        
        # Update performance based on message
        message = Message(
            type=MessageType.REJECT,
            sender="A_Q",
            content={"error": "Processing failed"}
        )
        
        await agent.handle_message(message)
        
        assert agent.performance_metrics["A_Q"]["failures"] == 6
        assert agent.agent_reliability["A_Q"] < 0.95
    
    async def test_intervention_decision(self, mock_llm_client, message_bus):
        """Test strategic intervention decision making."""
        agent = ControllerAgent("A_C", mock_llm_client, message_bus)
        
        # Add multiple issues to trigger intervention
        for _ in range(4):
            agent.intervention_history.append({
                "timestamp": datetime.now(),
                "sender": "A_V",
                "content": {"error": "Conflict"},
                "type": "conflict"
            })
        
        assert agent.should_intervene()
    
    async def test_system_health_assessment(self, mock_llm_client, message_bus):
        """Test system health assessment."""
        agent = ControllerAgent("A_C", mock_llm_client, message_bus)
        
        # Set up mixed health scenario
        agent.agent_reliability = {
            "A_Q": 0.9,
            "A_R": 0.8,
            "A_V": 0.4,  # Degraded
            "A_A": 0.3   # Critical
        }
        
        agent.performance_metrics = {
            "A_Q": {"successes": 90, "failures": 10},
            "A_R": {"successes": 80, "failures": 20},
            "A_V": {"successes": 40, "failures": 60},
            "A_A": {"successes": 30, "failures": 70}
        }
        
        health = agent._assess_system_health()
        assert health in ["degraded", "critical"]
    
    async def test_critical_pattern_detection(self, mock_llm_client, message_bus):
        """Test detection of critical system patterns."""
        agent = ControllerAgent("A_C", mock_llm_client, message_bus)
        
        # Pattern 1: Same agent failing repeatedly
        issues = [
            {"type": "failure", "sender": "A_V", "timestamp": datetime.now()},
            {"type": "failure", "sender": "A_V", "timestamp": datetime.now()},
            {"type": "failure", "sender": "A_V", "timestamp": datetime.now()},
        ]
        
        assert agent._detect_critical_patterns(issues)
        
        # Pattern 2: Cascading conflicts
        from datetime import timedelta
        base_time = datetime.now()
        conflict_issues = [
            {"type": "conflict", "timestamp": base_time},
            {"type": "conflict", "timestamp": base_time + timedelta(seconds=20)},
            {"type": "conflict", "timestamp": base_time + timedelta(seconds=40)},
        ]
        
        assert agent._detect_critical_patterns(conflict_issues)
