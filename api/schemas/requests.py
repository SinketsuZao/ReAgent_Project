"""
Request schemas for the ReAgent API.

This module defines all Pydantic models used for validating
incoming API requests.
"""

from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from enum import Enum


class QuestionRequest(BaseModel):
    """Schema for single question processing request."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Which U.S. state has a capital city whose population is smaller than the state's largest city, given that this state hosted the 1984 Summer Olympics?",
                "timeout": 60,
                "max_backtrack_depth": 5,
                "enable_caching": True,
                "metadata": {
                    "user_id": "user123",
                    "session_id": "session456"
                }
            }
        }
    )
    
    question: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The multi-hop question to process"
    )
    
    timeout: Optional[int] = Field(
        default=60,
        ge=10,
        le=300,
        description="Maximum processing time in seconds"
    )
    
    max_backtrack_depth: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum depth for backtracking operations"
    )
    
    enable_caching: Optional[bool] = Field(
        default=True,
        description="Whether to use cached results if available"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for tracking"
    )
    
    priority: Optional[Literal["low", "normal", "high"]] = Field(
        default="normal",
        description="Processing priority"
    )
    
    @validator("question")
    def validate_question(cls, v):
        """Validate question format and content."""
        # Remove excessive whitespace
        v = " ".join(v.split())
        
        # Check for question mark (warning, not error)
        if not v.endswith("?"):
            # Optionally add question mark
            v = v + "?"
            
        return v
    
    @validator("metadata")
    def validate_metadata(cls, v):
        """Ensure metadata doesn't contain sensitive keys."""
        if v:
            sensitive_keys = {"password", "token", "secret", "api_key"}
            for key in v.keys():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    raise ValueError(f"Metadata cannot contain sensitive keys: {key}")
        return v


class BatchQuestionRequest(BaseModel):
    """Schema for batch question processing request."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "questions": [
                    "What is the capital of the country that hosted the 2024 Olympics?",
                    "Who was president when the iPhone was first released?"
                ],
                "batch_settings": {
                    "parallel_processing": True,
                    "max_concurrent": 5,
                    "fail_fast": False
                }
            }
        }
    )
    
    questions: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of questions to process"
    )
    
    batch_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "parallel_processing": True,
            "max_concurrent": 5,
            "fail_fast": False
        },
        description="Batch processing configuration"
    )
    
    common_timeout: Optional[int] = Field(
        default=60,
        ge=10,
        le=300,
        description="Timeout applied to each question"
    )
    
    priority: Optional[Literal["low", "normal", "high"]] = Field(
        default="normal",
        description="Processing priority for the batch"
    )
    
    @validator("questions")
    def validate_questions(cls, v):
        """Validate all questions in the batch."""
        if len(v) > 50:
            raise ValueError("Maximum 50 questions per batch")
            
        # Ensure questions are unique
        unique_questions = list(dict.fromkeys(v))
        if len(unique_questions) < len(v):
            # Return deduplicated list
            return unique_questions
            
        return v


class TaskQueryRequest(BaseModel):
    """Schema for querying task status."""
    
    task_ids: Optional[List[str]] = Field(
        default=None,
        max_length=100,
        description="Specific task IDs to query"
    )
    
    status: Optional[Literal["pending", "processing", "completed", "failed"]] = Field(
        default=None,
        description="Filter by task status"
    )
    
    created_after: Optional[datetime] = Field(
        default=None,
        description="Filter tasks created after this time"
    )
    
    created_before: Optional[datetime] = Field(
        default=None,
        description="Filter tasks created before this time"
    )
    
    limit: Optional[int] = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of results"
    )
    
    offset: Optional[int] = Field(
        default=0,
        ge=0,
        description="Pagination offset"
    )
    
    include_details: Optional[bool] = Field(
        default=False,
        description="Include full task details in response"
    )


class BacktrackingTrigger(str, Enum):
    """Backtracking trigger types."""
    CONFLICT = "conflict"
    TIMEOUT = "timeout"
    ERROR = "error"
    MANUAL = "manual"


class BacktrackingRequest(BaseModel):
    """Schema for manual backtracking request."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question_id": "q_123456",
                "trigger": "manual",
                "scope": "local",
                "target_agent": "A_V",
                "checkpoint_id": "ckpt_789",
                "reason": "Detected inconsistency in retrieved facts"
            }
        }
    )
    
    question_id: str = Field(
        ...,
        description="ID of the question to backtrack"
    )
    
    trigger: BacktrackingTrigger = Field(
        ...,
        description="What triggered the backtracking"
    )
    
    scope: Literal["local", "global"] = Field(
        ...,
        description="Scope of backtracking operation"
    )
    
    target_agent: Optional[str] = Field(
        default=None,
        description="Specific agent to backtrack (for local scope)"
    )
    
    checkpoint_id: Optional[str] = Field(
        default=None,
        description="Specific checkpoint to rollback to"
    )
    
    reason: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Reason for manual backtracking"
    )
    
    force: Optional[bool] = Field(
        default=False,
        description="Force backtracking even if risky"
    )
    
    @validator("target_agent")
    def validate_target_agent(cls, v, values):
        """Ensure target_agent is provided for local backtracking."""
        if values.get("scope") == "local" and not v:
            raise ValueError("target_agent is required for local backtracking")
        return v


class ConfigUpdateRequest(BaseModel):
    """Schema for updating system configuration."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config_type": "agent",
                "target": "A_Q",
                "updates": {
                    "temperature": 0.7,
                    "max_retries": 5
                },
                "temporary": True,
                "duration_minutes": 30
            }
        }
    )
    
    config_type: Literal["system", "agent", "feature"] = Field(
        ...,
        description="Type of configuration to update"
    )
    
    target: Optional[str] = Field(
        default=None,
        description="Specific target (e.g., agent ID) for the update"
    )
    
    updates: Dict[str, Any] = Field(
        ...,
        description="Configuration updates to apply"
    )
    
    temporary: Optional[bool] = Field(
        default=False,
        description="Whether this is a temporary change"
    )
    
    duration_minutes: Optional[int] = Field(
        default=None,
        ge=1,
        le=1440,  # Max 24 hours
        description="Duration for temporary changes"
    )
    
    reason: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reason for configuration change"
    )
    
    @validator("updates")
    def validate_updates(cls, v):
        """Ensure updates are not empty."""
        if not v:
            raise ValueError("Updates cannot be empty")
        return v
    
    @validator("duration_minutes")
    def validate_duration(cls, v, values):
        """Ensure duration is provided for temporary changes."""
        if values.get("temporary") and not v:
            raise ValueError("duration_minutes required for temporary changes")
        return v


class MetricsQueryRequest(BaseModel):
    """Schema for querying system metrics."""
    
    metric_types: Optional[List[str]] = Field(
        default=None,
        description="Specific metrics to retrieve"
    )
    
    agent_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter metrics by agent IDs"
    )
    
    time_range: Optional[Literal["1h", "6h", "24h", "7d", "30d"]] = Field(
        default="1h",
        description="Time range for metrics"
    )
    
    aggregation: Optional[Literal["avg", "sum", "max", "min"]] = Field(
        default="avg",
        description="Aggregation method for metrics"
    )
    
    include_percentiles: Optional[bool] = Field(
        default=False,
        description="Include percentile calculations"
    )


class FeedbackRequest(BaseModel):
    """Schema for submitting feedback on answers."""
    
    question_id: str = Field(
        ...,
        description="ID of the question being rated"
    )
    
    rating: Literal[1, 2, 3, 4, 5] = Field(
        ...,
        description="Rating from 1 (poor) to 5 (excellent)"
    )
    
    feedback_type: Literal["accuracy", "completeness", "clarity", "general"] = Field(
        default="general",
        description="Type of feedback being provided"
    )
    
    comments: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Additional feedback comments"
    )
    
    suggested_answer: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="User's suggested answer if rating is low"
    )
