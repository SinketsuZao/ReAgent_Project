"""
Response schemas for the ReAgent API.

This module defines all Pydantic models used for structuring
API responses.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any, Literal, Union
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QuestionStatus(str, Enum):
    """Question processing status."""
    RECEIVED = "received"
    DECOMPOSED = "decomposed"
    RETRIEVING = "retrieving"
    VERIFYING = "verifying"
    ASSEMBLING = "assembling"
    BACKTRACKING = "backtracking"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStatus(str, Enum):
    """Agent operational status."""
    IDLE = "idle"
    ACTIVE = "active"
    BACKTRACKING = "backtracking"
    ERROR = "error"
    DISABLED = "disabled"


class ReasoningStep(BaseModel):
    """Single step in the reasoning trace."""
    
    timestamp: datetime
    agent_id: str
    action: str
    description: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class BacktrackingInfo(BaseModel):
    """Information about a backtracking event."""
    
    timestamp: datetime
    type: Literal["local", "global"]
    trigger: str
    affected_agents: List[str]
    rollback_depth: int
    checkpoint_id: Optional[str] = None
    success: bool
    details: Optional[str] = None


class QuestionResponse(BaseModel):
    """Response for single question processing."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "task_abc123",
                "question_id": "q_def456",
                "status": "completed",
                "answer": "California. The 1984 Summer Olympics were held in Los Angeles, California. Sacramento is the capital of California with a population of about 500,000, which is smaller than Los Angeles (the state's largest city) with a population of nearly 4 million.",
                "confidence": 0.92,
                "processing_time": 12.5,
                "sub_questions": [
                    "Which state hosted the 1984 Summer Olympics?",
                    "What is the capital of that state?",
                    "What is the population of the capital?",
                    "What is the largest city in that state?"
                ],
                "supporting_facts": [
                    "The 1984 Summer Olympics were held in Los Angeles, California",
                    "Sacramento is the capital city of California",
                    "Sacramento has a population of approximately 500,000",
                    "Los Angeles is California's largest city with nearly 4 million people"
                ],
                "backtracking_events": 0,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    task_id: str = Field(..., description="Unique task identifier")
    question_id: str = Field(..., description="Unique question identifier")
    status: QuestionStatus = Field(..., description="Current processing status")
    
    answer: Optional[str] = Field(None, description="Final answer (when completed)")
    confidence: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Confidence score for the answer"
    )
    
    processing_time: Optional[float] = Field(
        None,
        description="Total processing time in seconds"
    )
    
    sub_questions: Optional[List[str]] = Field(
        None,
        description="Decomposed sub-questions"
    )
    
    supporting_facts: Optional[List[str]] = Field(
        None,
        description="Key facts used to derive the answer"
    )
    
    backtracking_events: int = Field(
        default=0,
        description="Number of backtracking events during processing"
    )
    
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="When the task was created")
    completed_at: Optional[datetime] = Field(None, description="When processing completed")


class BatchQuestionResponse(BaseModel):
    """Response for batch question processing."""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    total_questions: int = Field(..., description="Total questions in batch")
    
    results: List[QuestionResponse] = Field(
        ...,
        description="Individual question results"
    )
    
    summary: Dict[str, int] = Field(
        ...,
        description="Summary statistics",
        example={
            "completed": 8,
            "failed": 1,
            "processing": 1,
            "pending": 0
        }
    )
    
    batch_processing_time: Optional[float] = Field(
        None,
        description="Total batch processing time"
    )
    
    created_at: datetime
    completed_at: Optional[datetime] = None


class TaskResponse(BaseModel):
    """Response for task status query."""
    
    task_id: str
    status: TaskStatus
    task_type: Literal["question", "batch", "backtracking"]
    
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    
    progress: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Progress percentage"
    )
    
    result: Optional[Union[QuestionResponse, BatchQuestionResponse, "BacktrackingResponse"]] = None
    error: Optional[str] = None
    
    metadata: Optional[Dict[str, Any]] = None


class TaskListResponse(BaseModel):
    """Response for listing multiple tasks."""
    
    tasks: List[TaskResponse]
    total: int = Field(..., description="Total number of matching tasks")
    offset: int = Field(..., description="Current offset for pagination")
    limit: int = Field(..., description="Number of results per page")
    
    has_more: bool = Field(..., description="Whether more results are available")


class BacktrackingResponse(BaseModel):
    """Response for backtracking operations."""
    
    backtracking_id: str
    question_id: str
    scope: Literal["local", "global"]
    status: Literal["initiated", "in_progress", "completed", "failed"]
    
    affected_agents: List[str]
    checkpoints_rolled_back: int
    
    original_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    
    initiated_at: datetime
    completed_at: Optional[datetime] = None
    
    outcome: Optional[str] = None
    details: Optional[str] = None


class SystemStatusResponse(BaseModel):
    """Response for system status endpoint."""
    
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: float
    
    agents: Dict[str, AgentStatus]
    
    active_questions: int
    questions_in_queue: int
    
    resource_usage: Dict[str, Any] = Field(
        ...,
        example={
            "cpu_percent": 45.2,
            "memory_mb": 2048,
            "redis_connections": 15,
            "postgres_connections": 10
        }
    )
    
    last_health_check: datetime
    issues: List[str] = Field(default_factory=list)


class AgentStatusResponse(BaseModel):
    """Response for individual agent status."""
    
    agent_id: str
    agent_type: str
    status: AgentStatus
    
    current_task: Optional[str] = None
    tasks_completed: int
    tasks_failed: int
    
    local_knowledge_size: int
    checkpoint_count: int
    
    performance_metrics: Dict[str, float] = Field(
        ...,
        example={
            "avg_processing_time": 2.5,
            "success_rate": 0.95,
            "token_usage_avg": 1500
        }
    )
    
    last_activity: datetime
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Response for metrics queries."""
    
    time_range: str
    metrics: Dict[str, Any]
    
    aggregations: Dict[str, float] = Field(
        ...,
        example={
            "total_questions": 1523,
            "avg_processing_time": 15.7,
            "success_rate": 0.89,
            "total_backtracking_events": 127
        }
    )
    
    time_series: Optional[Dict[str, List[Dict[str, Any]]]] = None
    
    agent_metrics: Optional[Dict[str, Dict[str, float]]] = None
    
    generated_at: datetime


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    
    status: Literal["ok", "error"]
    timestamp: datetime
    
    checks: Dict[str, Dict[str, Any]] = Field(
        ...,
        example={
            "database": {"status": "ok", "latency_ms": 5},
            "redis": {"status": "ok", "latency_ms": 2},
            "llm_api": {"status": "ok", "latency_ms": 250}
        }
    )
    
    version: str
    environment: str


class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    suggested_action: Optional[str] = Field(
        None,
        description="Suggested action to resolve the error"
    )


class ReasoningTraceResponse(BaseModel):
    """Detailed reasoning trace for a question."""
    
    question_id: str
    question: str
    
    trace: List[ReasoningStep] = Field(
        ...,
        description="Chronological list of reasoning steps"
    )
    
    backtracking_events: List[BacktrackingInfo] = Field(
        default_factory=list,
        description="All backtracking events during processing"
    )
    
    agent_interactions: Dict[str, int] = Field(
        ...,
        description="Message counts between agents",
        example={
            "A_Q->A_R": 3,
            "A_R->A_V": 5,
            "A_V->A_A": 4
        }
    )
    
    token_usage: Dict[str, int] = Field(
        ...,
        description="Token usage by agent",
        example={
            "A_Q": 1250,
            "A_R": 3500,
            "A_V": 2100,
            "A_A": 1800
        }
    )
    
    total_duration: float
    
    final_answer: Optional[str] = None
    confidence: Optional[float] = None
    
    generated_at: datetime
