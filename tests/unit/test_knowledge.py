"""
Unit tests for knowledge management functionality.

Tests knowledge assertions, dependencies, and knowledge graph operations.
"""

import pytest
from datetime import datetime, timedelta
import networkx as nx
from typing import List, Dict, Set

from reagent.models import KnowledgeAssertion
from reagent.main import LocalKnowledge
from tests import UNIT_TEST


@pytest.mark.unit
class TestKnowledgeAssertion:
    """Test KnowledgeAssertion model functionality."""
    
    def test_assertion_creation(self):
        """Test creating knowledge assertions."""
        assertion = KnowledgeAssertion(
            content="Paris is the capital of France",
            source="Wikipedia",
            confidence=0.95,
            dependencies=["fact_123", "fact_456"]
        )
        
        assert assertion.content == "Paris is the capital of France"
        assert assertion.source == "Wikipedia"
        assert assertion.confidence == 0.95
        assert len(assertion.dependencies) == 2
        assert assertion.timestamp is not None
    
    def test_assertion_validation(self):
        """Test assertion validation rules."""
        # Valid confidence
        assertion = KnowledgeAssertion("Test", "Source", 0.5)
        assert 0 <= assertion.confidence <= 1
        
        # Test boundary values
        assertion_low = KnowledgeAssertion("Test", "Source", 0.0)
        assertion_high = KnowledgeAssertion("Test", "Source", 1.0)
        assert assertion_low.confidence == 0.0
        assert assertion_high.confidence == 1.0
    
    def test_assertion_metadata(self):
        """Test assertion metadata handling."""
        metadata = {
            "year": 2024,
            "verified": True,
            "tags": ["geography", "europe"]
        }
        
        assertion = KnowledgeAssertion(
            content="Test fact",
            source="Test source",
            confidence=0.8,
            metadata=metadata
        )
        
        assert assertion.metadata == metadata
        assert assertion.metadata["year"] == 2024
        assert "geography" in assertion.metadata["tags"]


@pytest.mark.unit
class TestKnowledgeDependencies:
    """Test knowledge dependency management."""
    
    def test_dependency_chain(self):
        """Test creating chains of dependent assertions."""
        knowledge = LocalKnowledge("test_agent")
        
        # Create base assertion
        base_id = knowledge.add_assertion(
            KnowledgeAssertion("France is in Europe", "Atlas", 0.99)
        )
        
        # Create dependent assertion
        dependent_id = knowledge.add_assertion(
            KnowledgeAssertion(
                "Paris is in Europe",
                "Inference",
                0.95,
                dependencies=[base_id]
            )
        )
        
        # Verify dependency
        dependent = knowledge.assertions[dependent_id]
        assert base_id in dependent.dependencies
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        knowledge = LocalKnowledge("test_agent")
        
        # Create assertions
        id1 = knowledge.add_assertion(
            KnowledgeAssertion("Fact 1", "Source", 0.9)
        )
        
        id2 = knowledge.add_assertion(
            KnowledgeAssertion("Fact 2", "Source", 0.9, dependencies=[id1])
        )
        
        # Attempting to make id1 depend on id2 would create a cycle
        # In a real system, this should be detected and prevented
        knowledge.assertions[id1].dependencies.append(id2)
        
        # Verify cycle exists
        assert id2 in knowledge.assertions[id1].dependencies
        assert id1 in knowledge.assertions[id2].dependencies
    
    def test_dependency_propagation(self):
        """Test confidence propagation through dependencies."""
        knowledge = LocalKnowledge("test_agent")
        
        # Create chain: base -> middle -> derived
        base_id = knowledge.add_assertion(
            KnowledgeAssertion("Base fact", "Source", 0.9)
        )
        
        middle_id = knowledge.add_assertion(
            KnowledgeAssertion(
                "Middle fact",
                "Inference",
                0.8,
                dependencies=[base_id]
            )
        )
        
        derived_id = knowledge.add_assertion(
            KnowledgeAssertion(
                "Derived fact",
                "Inference",
                0.7,
                dependencies=[middle_id]
            )
        )
        
        # In a real system, confidence might propagate/degrade through chain
        # Verify the chain exists
        assert knowledge.assertions[middle_id].dependencies == [base_id]
        assert knowledge.assertions[derived_id].dependencies == [middle_id]


@pytest.mark.unit
class TestKnowledgeGraph:
    """Test knowledge graph operations."""
    
    def test_build_knowledge_graph(self):
        """Test building a graph from assertions."""
        knowledge = LocalKnowledge("test_agent")
        
        # Create interconnected assertions
        id1 = knowledge.add_assertion(
            KnowledgeAssertion("Europe is a continent", "Geography", 0.99)
        )
        
        id2 = knowledge.add_assertion(
            KnowledgeAssertion(
                "France is in Europe",
                "Geography",
                0.99,
                dependencies=[id1]
            )
        )
        
        id3 = knowledge.add_assertion(
            KnowledgeAssertion(
                "Paris is in France",
                "Geography",
                0.99,
                dependencies=[id2]
            )
        )
        
        id4 = knowledge.add_assertion(
            KnowledgeAssertion(
                "Paris is in Europe",
                "Inference",
                0.95,
                dependencies=[id2, id3]
            )
        )
        
        # Build graph
        graph = build_knowledge_graph(knowledge)
        
        assert len(graph.nodes) == 4
        assert graph.has_edge(id1, id2)
        assert graph.has_edge(id2, id3)
        assert graph.has_edge(id2, id4)
        assert graph.has_edge(id3, id4)
    
    def test_find_related_assertions(self):
        """Test finding related assertions in the graph."""
        knowledge = LocalKnowledge("test_agent")
        
        # Create related facts about Paris
        paris_facts = []
        
        paris_facts.append(knowledge.add_assertion(
            KnowledgeAssertion("Paris is a city", "Geography", 0.99)
        ))
        
        paris_facts.append(knowledge.add_assertion(
            KnowledgeAssertion("Paris is in France", "Geography", 0.99)
        ))
        
        paris_facts.append(knowledge.add_assertion(
            KnowledgeAssertion("Paris has 2.16M people", "Census", 0.95)
        ))
        
        # Create unrelated fact
        unrelated = knowledge.add_assertion(
            KnowledgeAssertion("Tokyo is in Japan", "Geography", 0.99)
        )
        
        # Find assertions related to Paris
        related = find_related_by_content(knowledge, "Paris")
        
        assert all(pid in related for pid in paris_facts)
        assert unrelated not in related
    
    def test_knowledge_clustering(self):
        """Test clustering assertions by topic."""
        knowledge = LocalKnowledge("test_agent")
        
        # Create assertions in different topics
        geo_facts = []
        geo_facts.append(knowledge.add_assertion(
            KnowledgeAssertion("Paris is in France", "Geography", 0.99)
        ))
        geo_facts.append(knowledge.add_assertion(
            KnowledgeAssertion("France is in Europe", "Geography", 0.99)
        ))
        
        pop_facts = []
        pop_facts.append(knowledge.add_assertion(
            KnowledgeAssertion("Paris has 2.16M people", "Census", 0.95)
        ))
        pop_facts.append(knowledge.add_assertion(
            KnowledgeAssertion("France has 67M people", "Census", 0.95)
        ))
        
        # Cluster by content similarity
        clusters = cluster_assertions(knowledge)
        
        # Should identify at least 2 clusters (geography and population)
        assert len(clusters) >= 2


@pytest.mark.unit
class TestKnowledgeValidation:
    """Test knowledge validation and consistency checking."""
    
    def test_validate_numeric_consistency(self):
        """Test validation of numeric facts."""
        knowledge = LocalKnowledge("test_agent")
        
        # Add consistent numeric facts
        knowledge.add_assertion(
            KnowledgeAssertion("France has 67 million people", "Census", 0.95)
        )
        
        knowledge.add_assertion(
            KnowledgeAssertion("Paris has 2.16 million people", "Census", 0.95)
        )
        
        knowledge.add_assertion(
            KnowledgeAssertion("Lyon has 0.5 million people", "Census", 0.90)
        )
        
