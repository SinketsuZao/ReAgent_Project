"""
SQLAlchemy models for the ReAgent database.

This module defines all database models using SQLAlchemy ORM,
providing a Python interface to the database schema.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import enum
from uuid import uuid4

from sqlalchemy import (
    Column, String, Text, Float, Integer, Boolean, DateTime, 
    ForeignKey, JSON, Enum, Index, UniqueConstraint, CheckConstraint,
    Table, event, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET, ARRAY
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import expression

Base = declarative_base()

# Enums
class TaskStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class QuestionStatus(enum.Enum):
    RECEIVED = "received"
    DECOMPOSED = "decomposed"
    RETRIEVING = "retrieving"
    VERIFYING = "verifying"
    ASSEMBLING = "assembling"
    BACKTRACKING = "backtracking"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class AgentType(enum.Enum):
    DECOMPOSER = "decomposer"
    RETRIEVER = "retriever"
    VERIFIER = "verifier"
    ASSEMBLER = "assembler"
    SUPERVISOR = "supervisor"
    CONTROLLER = "controller"

class BacktrackingScope(enum.Enum):
    LOCAL = "local"
    GLOBAL = "global"

class MessageType(enum.Enum):
    ASSERT = "assert"
    INFORM = "inform"
    CHALLENGE = "challenge"
    REJECT = "reject"
    ACCEPT = "accept"

# Models
class QuestionRecord(Base):
    """Model for questions processed by the system."""
    __tablename__ = 'questions'
    __table_args__ = {'schema': 'reagent'}
    
    id = Column(String(255), primary_key=True)
    question_text = Column(Text, nullable=False)
    task_id = Column(String(255), index=True)
    status = Column(Enum(QuestionStatus), nullable=False, default=QuestionStatus.RECEIVED, index=True)
    answer = Column(Text)
    confidence = Column(Float, CheckConstraint('confidence >= 0 AND confidence <= 1'))
    processing_time = Column(Float)
    token_usage = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))
    metadata = Column(JSONB, default={})
    error = Column(Text)
    parent_question_id = Column(String(255), ForeignKey('reagent.questions.id', ondelete='CASCADE'))
    
    # Relationships
    sub_questions = relationship('SubQuestion', back_populates='parent_question', cascade='all, delete-orphan')
    parent_question = relationship('QuestionRecord', remote_side=[id], backref='child_questions')
    messages = relationship('AgentMessage', back_populates='question', cascade='all, delete-orphan')
    assertions = relationship('KnowledgeAssertion', back_populates='question', cascade='all, delete-orphan')
    checkpoints = relationship('Checkpoint', back_populates='question', cascade='all, delete-orphan')
    backtracking_events = relationship('BacktrackingEvent', back_populates='question', cascade='all, delete-orphan')
    metrics = relationship('PerformanceMetric', back_populates='question', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Question(id={self.id}, status={self.status.value})>"
    
    @validates('confidence')
    def validate_confidence(self, key, value):
        if value is not None and (value < 0 or value > 1):
            raise ValueError("Confidence must be between 0 and 1")
        return value
    
    @hybrid_property
    def is_completed(self):
        return self.status in [QuestionStatus.COMPLETED, QuestionStatus.FAILED, QuestionStatus.TIMEOUT]
    
    @hybrid_property
    def duration(self):
        if self.completed_at and self.created_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None

class SubQuestion(Base):
    """Model for decomposed sub-questions."""
    __tablename__ = 'sub_questions'
    __table_args__ = (
        UniqueConstraint('parent_question_id', 'order_index'),
        {'schema': 'reagent'}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    parent_question_id = Column(String(255), ForeignKey('reagent.questions.id', ondelete='CASCADE'), nullable=False)
    sub_question_text = Column(Text, nullable=False)
    order_index = Column(Integer, nullable=False)
    answer = Column(Text)
    confidence = Column(Float, CheckConstraint('confidence >= 0 AND confidence <= 1'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    dependencies = Column(JSONB, default=[])
    
    # Relationships
    parent_question = relationship('QuestionRecord', back_populates='sub_questions')
    
    def __repr__(self):
        return f"<SubQuestion(id={self.id}, order={self.order_index})>"

class TaskRecord(Base):
    """Model for async tasks."""
    __tablename__ = 'tasks'
    __table_args__ = {'schema': 'reagent'}
    
    id = Column(String(255), primary_key=True)
    task_type = Column(String(50), nullable=False, index=True)
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))
    result = Column(JSONB)
    error = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    metadata = Column(JSONB, default={})
    
    def __repr__(self):
        return f"<Task(id={self.id}, type={self.task_type}, status={self.status.value})>"
    
    @hybrid_property
    def can_retry(self):
        return self.retry_count < self.max_retries

class Agent(Base):
    """Model for agent registry and performance tracking."""
    __tablename__ = 'agents'
    __table_args__ = {'schema': 'reagent'}
    
    id = Column(String(50), primary_key=True)
    agent_type = Column(Enum(AgentType), nullable=False)
    status = Column(String(20), default='idle')
    total_tasks_processed = Column(Integer, default=0)
    total_tasks_failed = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    avg_processing_time = Column(Float, default=0)
    reliability_score = Column(Float, default=1.0, CheckConstraint('reliability_score >= 0 AND reliability_score <= 1'))
    last_active = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    configuration = Column(JSONB, default={})
    performance_metrics = Column(JSONB, default={})
    
    # Relationships
    messages = relationship('AgentMessage', back_populates='sender', cascade='all, delete-orphan')
    assertions = relationship('KnowledgeAssertion', back_populates='agent', cascade='all, delete-orphan')
    checkpoints = relationship('Checkpoint', back_populates='agent', cascade='all, delete-orphan')
    initiated_backtracking = relationship('BacktrackingEvent', back_populates='initiating_agent')
    metrics = relationship('PerformanceMetric', back_populates='agent')
    
    def __repr__(self):
        return f"<Agent(id={self.id}, type={self.agent_type.value})>"
    
    def update_reliability(self, success: bool):
        """Update reliability score based on task outcome."""
        if success:
            self.reliability_score = min(1.0, self.reliability_score * 1.05)
        else:
            self.reliability_score = max(0.0, self.reliability_score * 0.95)

class AgentMessage(Base):
    """Model for messages between agents."""
    __tablename__ = 'agent_messages'
    __table_args__ = {'schema': 'reagent'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    message_id = Column(String(255), unique=True, nullable=False)
    sender_agent_id = Column(String(50), ForeignKey('reagent.agents.id'), nullable=False)
    message_type = Column(Enum(MessageType), nullable=False, index=True)
    content = Column(JSONB, nullable=False)
    question_id = Column(String(255), ForeignKey('reagent.questions.id', ondelete='CASCADE'), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    sender = relationship('Agent', back_populates='messages')
    question = relationship('QuestionRecord', back_populates='messages')
    
    def __repr__(self):
        return f"<AgentMessage(id={self.message_id}, type={self.message_type.value})>"

class KnowledgeAssertion(Base):
    """Model for knowledge assertions made by agents."""
    __tablename__ = 'knowledge_assertions'
    __table_args__ = {'schema': 'reagent'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(String(50), ForeignKey('reagent.agents.id'), nullable=False, index=True)
    question_id = Column(String(255), ForeignKey('reagent.questions.id', ondelete='CASCADE'), index=True)
    content = Column(Text, nullable=False)
    source = Column(String(255))
    confidence = Column(Float, nullable=False, CheckConstraint('confidence >= 0 AND confidence <= 1'))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    invalidated_at = Column(DateTime(timezone=True))
    dependencies = Column(JSONB, default=[])
    metadata = Column(JSONB, default={})
    
    # Relationships
    agent = relationship('Agent', back_populates='assertions')
    question = relationship('QuestionRecord', back_populates='assertions')
    
    def __repr__(self):
        return f"<KnowledgeAssertion(id={self.id}, confidence={self.confidence})>"
    
    @hybrid_property
    def is_valid(self):
        return self.invalidated_at is None

class Checkpoint(Base):
    """Model for system checkpoints."""
    __tablename__ = 'checkpoints'
    __table_args__ = {'schema': 'reagent'}
    
    id = Column(String(255), primary_key=True)
    agent_id = Column(String(50), ForeignKey('reagent.agents.id'), nullable=False, index=True)
    question_id = Column(String(255), ForeignKey('reagent.questions.id', ondelete='CASCADE'), index=True)
    checkpoint_type = Column(Enum(BacktrackingScope), nullable=False)
    state_data = Column(JSONB, nullable=False)
    parent_checkpoint_id = Column(String(255), ForeignKey('reagent.checkpoints.id', ondelete='CASCADE'))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    metadata = Column(JSONB, default={})
    
    # Relationships
    agent = relationship('Agent', back_populates='checkpoints')
    question = relationship('QuestionRecord', back_populates='checkpoints')
    parent_checkpoint = relationship('Checkpoint', remote_side=[id], backref='child_checkpoints')
    backtracking_events = relationship('BacktrackingEvent', back_populates='checkpoint')
    
    def __repr__(self):
        return f"<Checkpoint(id={self.id}, type={self.checkpoint_type.value})>"

class BacktrackingEvent(Base):
    """Model for backtracking events."""
    __tablename__ = 'backtracking_events'
    __table_args__ = {'schema': 'reagent'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    question_id = Column(String(255), ForeignKey('reagent.questions.id', ondelete='CASCADE'), nullable=False, index=True)
    scope = Column(Enum(BacktrackingScope), nullable=False)
    trigger_reason = Column(Text, nullable=False)
    initiating_agent_id = Column(String(50), ForeignKey('reagent.agents.id'))
    affected_agents = Column(ARRAY(Text))
    checkpoint_id = Column(String(255), ForeignKey('reagent.checkpoints.id'))
    rollback_depth = Column(Integer, default=1)
    success = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    completed_at = Column(DateTime(timezone=True))
    metadata = Column(JSONB, default={})
    
    # Relationships
    question = relationship('QuestionRecord', back_populates='backtracking_events')
    initiating_agent = relationship('Agent', back_populates='initiated_backtracking')
    checkpoint = relationship('Checkpoint', back_populates='backtracking_events')
    
    def __repr__(self):
        return f"<BacktrackingEvent(id={self.id}, scope={self.scope.value})>"

class EvidenceCache(Base):
    """Model for caching retrieved evidence."""
    __tablename__ = 'evidence_cache'
    __table_args__ = (
        UniqueConstraint('query_hash', 'source'),
        {'schema': 'reagent'}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_hash = Column(String(64), nullable=False, index=True)
    source = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    confidence = Column(Float, CheckConstraint('confidence >= 0 AND confidence <= 1'))
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), index=True)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<EvidenceCache(hash={self.query_hash[:8]}, source={self.source})>"
    
    @hybrid_property
    def is_expired(self):
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

class PerformanceMetric(Base):
    """Model for performance metrics."""
    __tablename__ = 'performance_metrics'
    __table_args__ = {'schema': 'reagent'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    metric_type = Column(String(100), nullable=False, index=True)
    metric_name = Column(String(255), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    agent_id = Column(String(50), ForeignKey('reagent.agents.id'), index=True)
    question_id = Column(String(255), ForeignKey('reagent.questions.id', ondelete='CASCADE'))
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    tags = Column(JSONB, default={})
    
    # Relationships
    agent = relationship('Agent', back_populates='metrics')
    question = relationship('QuestionRecord', back_populates='metrics')
    
    def __repr__(self):
        return f"<PerformanceMetric(type={self.metric_type}, name={self.metric_name})>"

class SystemEvent(Base):
    """Model for system events and audit log."""
    __tablename__ = 'system_events'
    __table_args__ = {'schema': 'reagent'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    event_type = Column(String(100), nullable=False, index=True)
    event_source = Column(String(255), nullable=False)
    event_data = Column(JSONB, nullable=False)
    severity = Column(String(20), default='info', index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    user_id = Column(String(255))
    request_id = Column(String(255))
    ip_address = Column(INET)
    
    def __repr__(self):
        return f"<SystemEvent(type={self.event_type}, severity={self.severity})>"

class SystemConfiguration(Base):
    """Model for system configuration."""
    __tablename__ = 'system_configuration'
    __table_args__ = {'schema': 'reagent'}
    
    key = Column(String(255), primary_key=True)
    value = Column(JSONB, nullable=False)
    description = Column(Text)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    updated_by = Column(String(255))
    
    def __repr__(self):
        return f"<SystemConfiguration(key={self.key})>"

# Create indexes
Index('idx_questions_text_search', QuestionRecord.question_text, postgresql_using='gin')
Index('idx_questions_parent', QuestionRecord.parent_question_id, postgresql_where=QuestionRecord.parent_question_id.isnot(None))
Index('idx_evidence_cache_cleanup', EvidenceCache.expires_at, postgresql_where=EvidenceCache.expires_at.isnot(None))
Index('idx_metrics_cleanup', PerformanceMetric.timestamp)
Index('idx_events_cleanup', SystemEvent.created_at, SystemEvent.severity)

# Event listeners
@event.listens_for(AgentMessage, 'before_insert')
def update_agent_last_active(mapper, connection, target):
    """Update agent's last active timestamp when sending a message."""
    connection.execute(
        Agent.__table__.update()
        .where(Agent.id == target.sender_agent_id)
        .values(last_active=func.now())
    )

@event.listens_for(QuestionRecord, 'before_update')
def update_question_completed_at(mapper, connection, target):
    """Set completed_at when question status changes to completed state."""
    if target.status in [QuestionStatus.COMPLETED, QuestionStatus.FAILED, QuestionStatus.TIMEOUT]:
        if target.completed_at is None:
            target.completed_at = datetime.now()

# Helper functions
def get_or_create_agent(session, agent_id: str, agent_type: AgentType) -> Agent:
    """Get or create an agent record."""
    agent = session.query(Agent).filter_by(id=agent_id).first()
    if not agent:
        agent = Agent(id=agent_id, agent_type=agent_type)
        session.add(agent)
    return agent

def log_system_event(session, event_type: str, event_source: str, 
                    event_data: Dict[str, Any], severity: str = 'info',
                    user_id: Optional[str] = None, request_id: Optional[str] = None,
                    ip_address: Optional[str] = None):
    """Log a system event."""
    event = SystemEvent(
        event_type=event_type,
        event_source=event_source,
        event_data=event_data,
        severity=severity,
        user_id=user_id,
        request_id=request_id,
        ip_address=ip_address
    )
    session.add(event)
    return event

def record_metric(session, metric_type: str, metric_name: str, metric_value: float,
                 agent_id: Optional[str] = None, question_id: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None):
    """Record a performance metric."""
    metric = PerformanceMetric(
        metric_type=metric_type,
        metric_name=metric_name,
        metric_value=metric_value,
        agent_id=agent_id,
        question_id=question_id,
        tags=tags or {}
    )
    session.add(metric)
    return metric
