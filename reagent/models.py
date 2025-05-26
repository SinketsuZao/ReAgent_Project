"""
ReAgent Data Models

This module defines all data models and types used throughout the ReAgent system.
Uses dataclasses for clean, type-safe data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import hashlib
import json
from abc import ABC, abstractmethod

# ============== Enums ==============

class MessageType(Enum):
    """Types of messages that can be sent between agents."""
    ASSERT = "assert"      # Assert a fact or conclusion
    INFORM = "inform"      # Share information
    REJECT = "reject"      # Reject a proposal or fact
    CHALLENGE = "challenge"  # Challenge an assertion or trigger intervention

class AgentRole(Enum):
    """Roles that agents can play in the system."""
    DECOMPOSER = "decomposer"
    RETRIEVER = "retriever"
    VERIFIER = "verifier"
    ASSEMBLER = "assembler"
    SUPERVISOR = "supervisor"
    CONTROLLER = "controller"

class ConflictType(Enum):
    """Types of conflicts that can be detected."""
    DIRECT_NEGATION = "direct_negation"
    VALUE_CONFLICT = "value_conflict"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    TEMPORAL_CONFLICT = "temporal_conflict"
    CAUSAL_CONFLICT = "causal_conflict"

class BacktrackingScope(Enum):
    """Scope of backtracking operations."""
    LOCAL = "local"
    GLOBAL = "global"
    PARTIAL = "partial"

class SystemStatus(Enum):
    """Overall system status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"

# ============== Core Data Models ==============

@dataclass
class Message:
    """
    Message passed between agents.
    
    This is the primary communication mechanism in the ReAgent system.
    """
    type: MessageType
    sender: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: hashlib.md5(
        f"{datetime.now().isoformat()}_{hash(frozenset({}))}"
        .encode()
    ).hexdigest())
    priority: int = 0  # Higher priority messages processed first
    correlation_id: Optional[str] = None  # For tracking related messages
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps({
            'type': self.type.value,
            'sender': self.sender,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'id': self.id,
            'priority': self.priority,
            'correlation_id': self.correlation_id
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data['type']),
            sender=data['sender'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            id=data['id'],
            priority=data.get('priority', 0),
            correlation_id=data.get('correlation_id')
        )

@dataclass
class KnowledgeAssertion:
    """
    A piece of knowledge or fact asserted by an agent.
    
    These are the building blocks of the system's knowledge base.
    """
    content: str
    source: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    assertion_type: str = "fact"  # fact, inference, assumption
    valid_until: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate assertion after initialization."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        if self.valid_until and self.valid_until < self.timestamp:
            raise ValueError("valid_until must be after timestamp")
    
    def is_valid(self) -> bool:
        """Check if assertion is still valid."""
        if self.valid_until is None:
            return True
        return datetime.now() < self.valid_until
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'content': self.content,
            'source': self.source,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'dependencies': self.dependencies,
            'metadata': self.metadata,
            'assertion_type': self.assertion_type,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None
        }

@dataclass
class BacktrackingNode:
    """
    A node in the backtracking graph representing a system state.
    
    Used for both local (per-agent) and global (system-wide) backtracking.
    """
    state: Dict[str, Any]
    timestamp: datetime
    parent_id: Optional[str] = None
    id: str = field(default_factory=lambda: hashlib.md5(
        f"node_{datetime.now().isoformat()}".encode()
    ).hexdigest())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_state_hash(self) -> str:
        """Get hash of the state for comparison."""
        # Create a deterministic hash of the state
        state_str = json.dumps(self.state, sort_keys=True, default=str)
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def get_size(self) -> int:
        """Get approximate size of this node in bytes."""
        return len(json.dumps(self.state, default=str))

# ============== Question Processing Models ==============

@dataclass
class Question:
    """Represents a question being processed by the system."""
    id: str
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_questions: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    expected_answer_type: str = "factual"  # factual, comparative, explanatory, etc.

@dataclass
class SubQuestion:
    """A sub-question derived from the main question."""
    id: str
    parent_question_id: str
    text: str
    order: int
    dependencies: List[str] = field(default_factory=list)
    is_answered: bool = False
    answer: Optional[str] = None
    confidence: float = 0.0

@dataclass
class Evidence:
    """A piece of evidence retrieved for answering a question."""
    id: str
    source: str
    content: str
    relevance_score: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieved_at: datetime = field(default_factory=datetime.now)
    used_for_sub_questions: List[str] = field(default_factory=list)

@dataclass
class Conflict:
    """Represents a conflict between assertions or evidence."""
    id: str
    type: ConflictType
    conflicting_items: List[str]  # IDs of conflicting assertions/evidence
    description: str
    severity: float  # 0.0 to 1.0
    detected_by: str  # Agent ID
    detected_at: datetime = field(default_factory=datetime.now)
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

# ============== Agent State Models ==============

@dataclass
class AgentState:
    """Represents the current state of an agent."""
    agent_id: str
    role: AgentRole
    status: str = "idle"  # idle, processing, waiting, error
    current_task: Optional[str] = None
    reliability_score: float = 1.0
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    
    def update_reliability(self):
        """Update reliability score based on success/failure ratio."""
        if self.total_operations > 0:
            self.reliability_score = self.successful_operations / self.total_operations

@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""
    agent_id: str
    avg_processing_time: float
    total_messages_sent: int
    total_messages_received: int
    conflicts_detected: int
    backtracking_events: int
    token_usage: Dict[str, int] = field(default_factory=dict)  # input/output tokens
    
# ============== System State Models ==============

@dataclass
class SystemState:
    """Represents the overall state of the ReAgent system."""
    status: SystemStatus
    active_questions: List[str]
    active_agents: Dict[str, AgentState]
    message_queue_size: int
    total_questions_processed: int
    success_rate: float
    avg_processing_time: float
    last_health_check: datetime = field(default_factory=datetime.now)
    
@dataclass
class HealthCheckResult:
    """Result of a system health check."""
    timestamp: datetime
    status: SystemStatus
    checks: Dict[str, bool]
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
# ============== Result Models ==============

@dataclass
class Answer:
    """Final answer produced by the system."""
    question_id: str
    text: str
    confidence: float
    supporting_facts: List[str]
    reasoning_trace: List[Dict[str, Any]]
    answer_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ProcessingResult:
    """Result of processing a question."""
    status: str  # success, failure, timeout, partial
    question_id: str
    answer: Optional[Answer] = None
    error: Optional[str] = None
    partial_results: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    backtracking_count: int = 0
    token_usage: Dict[str, int] = field(default_factory=dict)

# ============== Configuration Models ==============

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    role: AgentRole
    temperature: float = 0.6
    max_retries: int = 3
    timeout: int = 60
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemConfig:
    """Configuration for the entire system."""
    max_concurrent_questions: int = 10
    max_backtrack_depth: int = 5
    global_timeout: int = 300
    enable_caching: bool = True
    enable_monitoring: bool = True
    log_level: str = "INFO"
    
# ============== Event Models ==============

@dataclass
class Event:
    """Base class for system events."""
    id: str
    type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktrackingEvent(Event):
    """Event fired when backtracking occurs."""
    agent_id: str
    scope: BacktrackingScope
    checkpoint_id: str
    reason: str
    
    def __post_init__(self):
        self.type = "backtracking"

@dataclass
class ConflictEvent(Event):
    """Event fired when a conflict is detected."""
    conflict: Conflict
    
    def __post_init__(self):
        self.type = "conflict_detected"

@dataclass
class QuestionCompletedEvent(Event):
    """Event fired when question processing completes."""
    result: ProcessingResult
    
    def __post_init__(self):
        self.type = "question_completed"

# ============== Abstract Base Classes ==============

class Serializable(ABC):
    """Abstract base class for serializable objects."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary."""
        pass
    
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Create object from dictionary."""
        pass
    
    def to_json(self) -> str:
        """Convert object to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Serializable':
        """Create object from JSON string."""
        return cls.from_dict(json.loads(json_str))

# ============== Utility Functions ==============

def create_correlation_id() -> str:
    """Create a unique correlation ID for tracking related messages."""
    return hashlib.md5(f"corr_{datetime.now().isoformat()}".encode()).hexdigest()

def validate_confidence(confidence: float) -> float:
    """Validate and normalize confidence scores."""
    if not isinstance(confidence, (int, float)):
        raise TypeError(f"Confidence must be numeric, got {type(confidence)}")
    
    if confidence < 0:
        return 0.0
    elif confidence > 1:
        return 1.0
    else:
        return float(confidence)

def merge_metadata(*metadata_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple metadata dictionaries."""
    result = {}
    for md in metadata_dicts:
        if md:
            result.update(md)
    return result

# ============== Type Aliases ==============

AssertionID = str
QuestionID = str
AgentID = str
CheckpointID = str
MessageID = str