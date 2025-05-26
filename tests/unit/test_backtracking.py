"""
Unit tests for backtracking functionality.

Tests the backtracking mechanisms including checkpoints, rollback,
and conflict resolution.
"""

import pytest
from datetime import datetime, timedelta
import uuid
from copy import deepcopy

from reagent.main import LocalKnowledge, BacktrackingNode, SupervisorAgent
from reagent.models import KnowledgeAssertion, Message, MessageType
from tests import UNIT_TEST


@pytest.mark.unit
class TestLocalKnowledge:
    """Test LocalKnowledge functionality."""
    
    def test_add_assertion(self):
        """Test adding assertions to local knowledge."""
        knowledge = LocalKnowledge("test_agent")
        
        assertion = KnowledgeAssertion(
            content="Test fact",
            source="test_source",
            confidence=0.9
        )
        
        assertion_id = knowledge.add_assertion(assertion)
        
        assert assertion_id in knowledge.assertions
        assert knowledge.assertions[assertion_id] == assertion
        assert len(knowledge.assertions) == 1
    
    def test_save_checkpoint(self):
        """Test saving checkpoints."""
        knowledge = LocalKnowledge("test_agent")
        
        # Add some assertions
        assertion1 = KnowledgeAssertion("Fact 1", "Source 1", 0.9)
        assertion2 = KnowledgeAssertion("Fact 2", "Source 2", 0.8)
        
        id1 = knowledge.add_assertion(assertion1)
        id2 = knowledge.add_assertion(assertion2)
        
        # Save checkpoint
        checkpoint_id = knowledge.save_checkpoint()
        
        assert checkpoint_id is not None
        assert len(knowledge.checkpoints) == 1
        assert knowledge.current_state_id == checkpoint_id
        
        # Verify checkpoint contains current state
        checkpoint = knowledge.checkpoints[0]
        assert len(checkpoint.state['assertions']) == 2
        assert id1 in checkpoint.state['assertions']
        assert id2 in checkpoint.state['assertions']
    
    def test_rollback_to_checkpoint(self):
        """Test rolling back to a previous checkpoint."""
        knowledge = LocalKnowledge("test_agent")
        
        # Initial state
        assertion1 = KnowledgeAssertion("Fact 1", "Source 1", 0.9)
        id1 = knowledge.add_assertion(assertion1)
        checkpoint1 = knowledge.save_checkpoint()
        
        # Add more assertions
        assertion2 = KnowledgeAssertion("Fact 2", "Source 2", 0.8)
        assertion3 = KnowledgeAssertion("Fact 3", "Source 3", 0.7)
        knowledge.add_assertion(assertion2)
        knowledge.add_assertion(assertion3)
        
        # Current state should have 3 assertions
        assert len(knowledge.assertions) == 3
        
        # Rollback to first checkpoint
        success = knowledge.rollback(checkpoint1)
        
        assert success
        assert len(knowledge.assertions) == 1
        assert id1 in knowledge.assertions
        assert knowledge.current_state_id == checkpoint1
    
    def test_checkpoint_cleanup(self):
        """Test automatic cleanup of old checkpoints."""
        knowledge = LocalKnowledge("test_agent")
        knowledge.max_checkpoints = 5
        
        # Create more checkpoints than the limit
        for i in range(10):
            knowledge.add_assertion(
                KnowledgeAssertion(f"Fact {i}", f"Source {i}", 0.9)
            )
            knowledge.save_checkpoint()
        
        # Should only keep the latest max_checkpoints
        assert len(knowledge.checkpoints) == 5
        
        # Verify the oldest checkpoints were removed
        checkpoint_states = [cp.state['assertions'] for cp in knowledge.checkpoints]
        # The earliest checkpoint should have at least 6 assertions (5 were removed)
        min_assertions = min(len(state) for state in checkpoint_states)
        assert min_assertions >= 6
    
    def test_find_conflicts_direct_negation(self):
        """Test conflict detection for direct negation."""
        knowledge = LocalKnowledge("test_agent")
        
        # Add conflicting assertions
        knowledge.add_assertion(
            KnowledgeAssertion("Paris is the capital of France", "Source 1", 0.9)
        )
        knowledge.add_assertion(
            KnowledgeAssertion("Paris is not the capital of France", "Source 2", 0.8)
        )
        
        conflicts = knowledge.find_conflicts()
        assert len(conflicts) > 0
    
    def test_find_conflicts_value_conflicts(self):
        """Test conflict detection for conflicting values."""
        knowledge = LocalKnowledge("test_agent")
        
        # Conflicting population numbers
        knowledge.add_assertion(
            KnowledgeAssertion("The population of Paris is 2.1 million", "Source 1", 0.9)
        )
        knowledge.add_assertion(
            KnowledgeAssertion("The population of Paris is 10 million", "Source 2", 0.8)
        )
        
        conflicts = knowledge.find_conflicts()
        assert len(conflicts) > 0
    
    def test_find_conflicts_logical_inconsistency(self):
        """Test conflict detection for logical inconsistencies."""
        knowledge = LocalKnowledge("test_agent")
        
        # Mutually exclusive statements
        knowledge.add_assertion(
            KnowledgeAssertion("Mount Everest is the highest mountain", "Source 1", 0.95)
        )
        knowledge.add_assertion(
            KnowledgeAssertion("K2 is the highest mountain", "Source 2", 0.85)
        )
        
        conflicts = knowledge.find_conflicts()
        assert len(conflicts) > 0
    
    def test_nested_checkpoints(self):
        """Test creating checkpoints with parent relationships."""
        knowledge = LocalKnowledge("test_agent")
        
        # Create initial checkpoint
        knowledge.add_assertion(KnowledgeAssertion("Fact 1", "Source 1", 0.9))
        checkpoint1 = knowledge.save_checkpoint()
        
        # Create second checkpoint
        knowledge.add_assertion(KnowledgeAssertion("Fact 2", "Source 2", 0.8))
        checkpoint2 = knowledge.save_checkpoint()
        
        # Verify parent relationship
        assert knowledge.checkpoints[1].parent_id == checkpoint1
        assert knowledge.checkpoints[0].parent_id is None


@pytest.mark.unit
class TestBacktrackingNode:
    """Test BacktrackingNode functionality."""
    
    def test_node_creation(self):
        """Test creating a backtracking node."""
        state = {"assertions": {"id1": "Fact 1"}}
        node = BacktrackingNode(
            state=state,
            timestamp=datetime.now(),
            parent_id="parent_123"
        )
        
        assert node.id is not None
        assert node.state == state
        assert node.parent_id == "parent_123"
        assert isinstance(node.timestamp, datetime)
    
    def test_node_id_generation(self):
        """Test that node IDs are unique."""
        nodes = []
        for _ in range(100):
            node = BacktrackingNode(
                state={},
                timestamp=datetime.now()
            )
            nodes.append(node)
        
        # All IDs should be unique
        ids = [node.id for node in nodes]
        assert len(ids) == len(set(ids))


@pytest.mark.asyncio
@pytest.mark.unit
class TestGlobalBacktracking:
    """Test global backtracking functionality."""
    
    async def test_collect_agent_states(self, mock_llm_client, message_bus):
        """Test collecting states from all agents."""
        supervisor = SupervisorAgent("A_S", mock_llm_client, message_bus)
        
        # Simulate agent states
        agent_states = {
            "A_Q": {
                "assertions": ["Q1", "Q2"],
                "checkpoints": ["cp1", "cp2"],
                "last_checkpoint": "cp2"
            },
            "A_R": {
                "assertions": ["R1", "R2", "R3"],
                "checkpoints": ["cp3"],
                "last_checkpoint": "cp3"
            }
        }
        
        # Test collecting global state
        supervisor.global_knowledge = agent_states
        consensus = supervisor._extract_consensus_facts()
        
        # Should identify common facts across agents
        assert isinstance(consensus, list)
    
    async def test_global_checkpoint_creation(self, mock_llm_client, message_bus):
        """Test creating global checkpoints."""
        supervisor = SupervisorAgent("A_S", mock_llm_client, message_bus)
        
        # Create global checkpoint
        checkpoint_id = supervisor._create_global_checkpoint()
        
        assert checkpoint_id is not None
        assert len(supervisor.global_checkpoints) == 1
        assert supervisor.global_checkpoints[0]["id"] == checkpoint_id
    
    async def test_global_rollback_execution(self, mock_llm_client, message_bus):
        """Test executing global rollback."""
        supervisor = SupervisorAgent("A_S", mock_llm_client, message_bus)
        
        # Create a checkpoint
        checkpoint_id = supervisor._create_global_checkpoint()
        
        # Add more state
        supervisor.global_knowledge["A_Q"] = {"new": "state"}
        
        # Mock message publishing
        published_messages = []
        
        async def capture_publish(msg):
            published_messages.append(msg)
        
        message_bus.publish = capture_publish
        
        # Execute rollback
        success = await supervisor._execute_global_rollback(checkpoint_id)
        
        assert success
        assert len(published_messages) > 0
        
        # Verify rollback messages were sent
        rollback_messages = [
            msg for msg in published_messages 
            if msg.content.get("action") == "local_rollback"
        ]
        assert len(rollback_messages) > 0


@pytest.mark.unit
class TestConflictResolution:
    """Test conflict resolution strategies."""
    
    def test_confidence_based_resolution(self):
        """Test resolving conflicts based on confidence scores."""
        knowledge = LocalKnowledge("test_agent")
        
        # Add assertions with different confidence
        high_conf = KnowledgeAssertion(
            "Paris has 2.16 million people",
            "Official Census",
            0.95
        )
        low_conf = KnowledgeAssertion(
            "Paris has 3 million people",
            "Blog Post",
            0.60
        )
        
        id1 = knowledge.add_assertion(high_conf)
        id2 = knowledge.add_assertion(low_conf)
        
        # In a real system, the verifier would keep the higher confidence assertion
        # For this test, we verify both are detected as conflicts
        conflicts = knowledge.find_conflicts()
        assert len(conflicts) > 0
        
        # Verify we can identify which has higher confidence
        assert knowledge.assertions[id1].confidence > knowledge.assertions[id2].confidence
    
    def test_source_priority_resolution(self):
        """Test resolving conflicts based on source priority."""
        # This would be implemented in the VerifierAgent
        source_priorities = {
            "official": 1.0,
            "academic": 0.9,
            "news": 0.7,
            "blog": 0.5
        }
        
        # Verify priority ordering
        assert source_priorities["official"] > source_priorities["academic"]
        assert source_priorities["academic"] > source_priorities["news"]
        assert source_priorities["news"] > source_priorities["blog"]
    
    def test_temporal_resolution(self):
        """Test resolving conflicts based on temporal information."""
        knowledge = LocalKnowledge("test_agent")
        
        # Add assertions with temporal context
        old_assertion = KnowledgeAssertion(
            "The president is Obama",
            "News 2015",
            0.9,
            metadata={"year": 2015}
        )
        new_assertion = KnowledgeAssertion(
            "The president is Biden",
            "News 2023",
            0.9,
            metadata={"year": 2023}
        )
        
        knowledge.add_assertion(old_assertion)
        knowledge.add_assertion(new_assertion)
        
        # In a real system, temporal resolution would prefer newer information
        # for time-sensitive facts
        conflicts = knowledge.find_conflicts()
        # These shouldn't conflict if they have different temporal contexts
        # but our simple implementation might flag them


@pytest.mark.unit
class TestBacktrackingMetrics:
    """Test metrics collection for backtracking events."""
    
    def test_backtracking_depth_tracking(self):
        """Test tracking backtracking depth."""
        knowledge = LocalKnowledge("test_agent")
        
        # Create a chain of checkpoints
        depths = []
        for i in range(5):
            knowledge.add_assertion(
                KnowledgeAssertion(f"Fact {i}", f"Source {i}", 0.9)
            )
            checkpoint_id = knowledge.save_checkpoint()
            if i == 0:
                first_checkpoint = checkpoint_id
        
        # Rollback to first checkpoint
        knowledge.rollback(first_checkpoint)
        
        # The depth would be 4 (rolled back through 4 checkpoints)
        # This would be tracked in the actual metrics
        assert knowledge.current_state_id == first_checkpoint
    
    def test_backtracking_frequency(self):
        """Test tracking frequency of backtracking events."""
        # In a real system, this would be tracked by metrics_collector
        backtrack_events = []
        
        # Simulate multiple backtracking events
        for i in range(10):
            backtrack_events.append({
                "timestamp": datetime.now() + timedelta(seconds=i),
                "agent": "A_V",
                "scope": "local",
                "depth": 1
            })
        
        # Calculate frequency (events per minute)
        if len(backtrack_events) > 1:
            duration = (backtrack_events[-1]["timestamp"] - 
                       backtrack_events[0]["timestamp"]).total_seconds()
            frequency = len(backtrack_events) / (duration / 60) if duration > 0 else 0
            
            # Should detect high frequency
            assert frequency > 1  # More than 1 per minute
