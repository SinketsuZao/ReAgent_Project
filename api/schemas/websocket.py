"""
WebSocket message schemas for real-time communication.

This module defines all Pydantic models used for WebSocket
message validation and serialization.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any, Literal, Union
from datetime import datetime
from enum import Enum


class WSMessageType(str, Enum):
    """WebSocket message types."""
    SYSTEM = "system"
    QUESTION_UPDATE = "question_update"
    AGENT_UPDATE = "agent_update"
    BACKTRACKING = "backtracking"
    METRICS = "metrics"
    ERROR = "error"
    SUBSCRIPTION = "subscription"


class WSQuestionStatus(BaseModel):
    """WebSocket message for question status updates."""
    
    question_id: str
    status: Literal[
        "received", "decomposed", "retrieving", 
        "verifying", "assembling", "backtracking", 
        "completed", "failed"
    ]
    progress: float = Field(ge=0.0, le=100.0)
    
    current_phase: Optional[str] = None
    current_agent: Optional[str] = None
    
    sub_questions: Optional[List[str]] = None
    processed_sub_questions: Optional[int] = None
    
    partial_answer: Optional[str] = None
    final_answer: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    error: Optional[str] = None
    timestamp: datetime


class WSAgentActivity(BaseModel):
    """WebSocket message for agent activity updates."""
    
    agent_id: str
    agent_type: Optional[str] = None
    
    action: str = Field(
        ...,
        description="Current action being performed"
    )
    
    status: Literal["active", "waiting", "backtracking", "error"]
    
    question_id: Optional[str] = None
    
    processing_time: Optional[float] = Field(
        None,
        description="Time spent on current action (seconds)"
    )
    
    token_usage: Optional[Dict[str, int]] = Field(
        None,
        example={"input": 500, "output": 200}
    )
    
    message_content: Optional[Dict[str, Any]] = Field(
        None,
        description="Content of the message being processed"
    )
    
    timestamp: datetime


class WSBacktrackingEvent(BaseModel):
    """WebSocket message for backtracking events."""
    
    event_id: str
    type: Literal["local", "global"]
    
    agent_id: Optional[str] = Field(
        None,
        description="Agent initiating backtracking (for local)"
    )
    
    question_id: Optional[str] = None
    
    checkpoint_id: Optional[str] = Field(
        None,
        description="Checkpoint being rolled back to"
    )
    
    reason: str = Field(
        ...,
        description="Reason for backtracking"
    )
    
    affected_agents: List[str] = Field(
        ...,
        description="List of affected agent IDs"
    )
    
    rollback_depth: int = Field(
        ...,
        ge=1,
        description="Number of steps being rolled back"
    )
    
    conflicts_detected: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Conflicts that triggered backtracking"
    )
    
    timestamp: datetime


class WSMetricsUpdate(BaseModel):
    """WebSocket message for system metrics updates."""
    
    active_questions: int = Field(ge=0)
    total_processed: int = Field(ge=0)
    
    success_rate: float = Field(ge=0.0, le=1.0)
    avg_processing_time: float = Field(
        ge=0.0,
        description="Average processing time in seconds"
    )
    
    active_agents: Dict[str, str] = Field(
        ...,
        description="Agent ID to status mapping",
        example={
            "A_Q": "active",
            "A_R": "idle",
            "A_V": "active"
        }
    )
    
    agent_load: Optional[Dict[str, float]] = Field(
        None,
        description="Agent load percentages",
        example={
            "A_Q": 0.75,
            "A_R": 0.90,
            "A_V": 0.45
        }
    )
    
    backtracking_count: int = Field(
        ge=0,
        description="Total backtracking events in time window"
    )
    
    token_usage_total: int = Field(
        ge=0,
        description="Total tokens used in time window"
    )
    
    token_usage_by_agent: Optional[Dict[str, int]] = None
    
    queue_lengths: Optional[Dict[str, int]] = Field(
        None,
        description="Message queue lengths",
        example={
            "questions": 5,
            "tasks": 12,
            "messages": 45
        }
    )
    
    error_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    timestamp: datetime


class WSMessage(BaseModel):
    """Generic WebSocket message wrapper."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "msg_123456",
                "type": "question_update",
                "data": {
                    "question_id": "q_abc123",
                    "status": "retrieving",
                    "progress": 45.0,
                    "current_phase": "evidence_retrieval",
                    "timestamp": "2024-01-15T10:30:00Z"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    id: Optional[str] = Field(
        default_factory=lambda: f"msg_{datetime.now().timestamp()}",
        description="Unique message ID"
    )
    
    type: WSMessageType = Field(
        ...,
        description="Type of WebSocket message"
    )
    
    data: Union[
        WSQuestionStatus,
        WSAgentActivity,
        WSBacktrackingEvent,
        WSMetricsUpdate,
        Dict[str, Any]
    ] = Field(
        ...,
        description="Message payload"
    )
    
    correlation_id: Optional[str] = Field(
        None,
        description="ID for correlating related messages"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the message was created"
    )


class WSClientCommand(BaseModel):
    """Commands that clients can send via WebSocket."""
    
    command: Literal[
        "subscribe", "unsubscribe", "ping", 
        "get_status", "set_filter"
    ]
    
    question_id: Optional[str] = Field(
        None,
        description="Question ID for subscribe/unsubscribe"
    )
    
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Filters for message types",
        example={
            "message_types": ["question_update", "backtracking"],
            "agents": ["A_Q", "A_R"]
        }
    )
    
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional command data"
    )


class WSSubscriptionConfirmation(BaseModel):
    """Confirmation message for subscription requests."""
    
    subscription_id: str
    question_id: Optional[str] = None
    subscription_type: Literal["question", "system", "metrics"]
    
    active_filters: Optional[Dict[str, Any]] = None
    
    message: str = Field(
        default="Subscription confirmed",
        description="Confirmation message"
    )
    
    timestamp: datetime = Field(default_factory=datetime.now)


class WSError(BaseModel):
    """WebSocket error message."""
    
    error_code: str
    message: str
    
    details: Optional[Dict[str, Any]] = None
    
    recoverable: bool = Field(
        default=True,
        description="Whether the client should retry"
    )
    
    suggested_action: Optional[str] = None
    
    timestamp: datetime = Field(default_factory=datetime.now)
