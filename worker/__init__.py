"""
ReAgent Worker Package

This package contains Celery workers for asynchronous task processing,
including question processing, batch operations, and scheduled maintenance tasks.
"""

from .celery_app import app as celery_app
from .tasks import (
    process_question,
    process_batch_questions,
    execute_backtracking,
    cleanup_old_checkpoints,
    collect_system_metrics,
    generate_performance_report,
    health_check_task,
)

__all__ = [
    "celery_app",
    "process_question",
    "process_batch_questions",
    "execute_backtracking",
    "cleanup_old_checkpoints",
    "collect_system_metrics",
    "generate_performance_report",
    "health_check_task",
]

# Version info
__version__ = "1.0.0"
