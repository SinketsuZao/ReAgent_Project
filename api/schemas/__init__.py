"""
API Schemas Package

This package contains all Pydantic models used for request/response
validation and serialization in the ReAgent API.
"""

from .requests import (
    QuestionRequest,
    BatchQuestionRequest,
    TaskQueryRequest,
    BacktrackingRequest,
    ConfigUpdateRequest,
)

from .responses import (
    QuestionResponse,
    BatchQuestionResponse,
    TaskResponse,
    TaskListResponse,
    BacktrackingResponse,
    SystemStatusResponse,
    AgentStatusResponse,
    MetricsResponse,
    HealthResponse,
    ErrorResponse,
    ReasoningTraceResponse,
)

from .websocket import (
    WSMessage,
    WSMessageType,
    WSQuestionStatus,
    WSAgentActivity,
    WSBacktrackingEvent,
    WSMetricsUpdate,
)

__all__ = [
    # Requests
    "QuestionRequest",
    "BatchQuestionRequest",
    "TaskQueryRequest",
    "BacktrackingRequest",
    "ConfigUpdateRequest",
    
    # Responses
    "QuestionResponse",
    "BatchQuestionResponse",
    "TaskResponse",
    "TaskListResponse",
    "BacktrackingResponse",
    "SystemStatusResponse",
    "AgentStatusResponse",
    "MetricsResponse",
    "HealthResponse",
    "ErrorResponse",
    "ReasoningTraceResponse",
    
    # WebSocket
    "WSMessage",
    "WSMessageType",
    "WSQuestionStatus",
    "WSAgentActivity",
    "WSBacktrackingEvent",
    "WSMetricsUpdate",
]
