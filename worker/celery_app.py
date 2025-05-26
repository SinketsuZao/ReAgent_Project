"""
Celery application configuration for ReAgent workers.

This module sets up the Celery application with proper configuration,
task routing, and error handling for distributed task processing.
"""

import os
from celery import Celery, signals
from celery.schedules import crontab
from kombu import Exchange, Queue
import logging
from datetime import timedelta
from typing import Any, Dict

from reagent.config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Create Celery app
app = Celery('reagent')

# Configure Celery
app.conf.update(
    # Broker settings
    broker_url=os.getenv('CELERY_BROKER_URL', f'redis://{settings.redis_host}:{settings.redis_port}/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', f'redis://{settings.redis_host}:{settings.redis_port}/1'),
    
    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task behavior
    task_track_started=True,
    task_time_limit=300,  # 5 minutes hard limit
    task_soft_time_limit=240,  # 4 minutes soft limit
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    result_compression='gzip',
    
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=50,
    worker_disable_rate_limits=False,
    worker_send_task_events=True,
    
    # Performance optimizations
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    broker_pool_limit=None,  # Unlimited
    
    # Task routing
    task_routes={
        'worker.tasks.process_question': {'queue': 'questions', 'priority': 5},
        'worker.tasks.process_batch_questions': {'queue': 'batch', 'priority': 3},
        'worker.tasks.execute_backtracking': {'queue': 'backtracking', 'priority': 9},
        'worker.tasks.cleanup_old_checkpoints': {'queue': 'maintenance', 'priority': 1},
        'worker.tasks.collect_system_metrics': {'queue': 'monitoring', 'priority': 2},
        'worker.tasks.generate_performance_report': {'queue': 'reporting', 'priority': 1},
        'worker.tasks.health_check_task': {'queue': 'health', 'priority': 10},
    },
    
    # Queue configuration
    task_queues=(
        Queue('questions', Exchange('questions'), routing_key='questions', 
              queue_arguments={'x-max-priority': 10}),
        Queue('batch', Exchange('batch'), routing_key='batch',
              queue_arguments={'x-max-priority': 10}),
        Queue('backtracking', Exchange('backtracking'), routing_key='backtracking',
              queue_arguments={'x-max-priority': 10}),
        Queue('maintenance', Exchange('maintenance'), routing_key='maintenance'),
        Queue('monitoring', Exchange('monitoring'), routing_key='monitoring'),
        Queue('reporting', Exchange('reporting'), routing_key='reporting'),
        Queue('health', Exchange('health'), routing_key='health'),
    ),
    
    # Task annotations for rate limiting
    task_annotations={
        'worker.tasks.process_question': {'rate_limit': '100/m'},
        'worker.tasks.process_batch_questions': {'rate_limit': '10/m'},
        'worker.tasks.collect_system_metrics': {'rate_limit': '60/h'},
    },
    
    # Error handling
    task_publish_retry=True,
    task_publish_retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.2,
    },
)

# Beat schedule for periodic tasks
app.conf.beat_schedule = {
    'cleanup-old-checkpoints': {
        'task': 'worker.tasks.cleanup_old_checkpoints',
        'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
        'options': {'queue': 'maintenance', 'priority': 1}
    },
    'collect-system-metrics': {
        'task': 'worker.tasks.collect_system_metrics',
        'schedule': timedelta(minutes=1),  # Every minute
        'options': {'queue': 'monitoring', 'priority': 2}
    },
    'generate-hourly-report': {
        'task': 'worker.tasks.generate_performance_report',
        'schedule': crontab(minute=0),  # Every hour
        'args': ('hourly',),
        'options': {'queue': 'reporting', 'priority': 1}
    },
    'generate-daily-report': {
        'task': 'worker.tasks.generate_performance_report',
        'schedule': crontab(hour=0, minute=0),  # Daily at midnight
        'args': ('daily',),
        'options': {'queue': 'reporting', 'priority': 1}
    },
    'system-health-check': {
        'task': 'worker.tasks.health_check_task',
        'schedule': timedelta(seconds=30),  # Every 30 seconds
        'options': {'queue': 'health', 'priority': 10}
    },
}

# Celery signals for monitoring and logging
@signals.task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kw):
    """Log task start and update metrics."""
    logger.info(f"Task {task.name}[{task_id}] starting")
    # Could add metrics collection here

@signals.task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, 
                         retval=None, state=None, **kw):
    """Log task completion and update metrics."""
    logger.info(f"Task {task.name}[{task_id}] completed with state: {state}")
    # Could add metrics collection here

@signals.task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, args=None, 
                        kwargs=None, traceback=None, einfo=None, **kw):
    """Handle task failures."""
    logger.error(f"Task {sender.name}[{task_id}] failed: {exception}")
    # Could send alerts or update failure metrics here

@signals.task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, **kw):
    """Log task retries."""
    logger.warning(f"Task {sender.name}[{task_id}] retrying: {reason}")

@signals.worker_ready.connect
def worker_ready_handler(sender=None, **kw):
    """Handle worker startup."""
    logger.info("Worker ready to accept tasks")
    # Could perform initialization tasks here

@signals.worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kw):
    """Handle worker shutdown."""
    logger.info("Worker shutting down")
    # Could perform cleanup tasks here

# Import tasks to register them
app.autodiscover_tasks(['worker'])

# Custom task base class with additional features
from celery import Task

class ReAgentTask(Task):
    """Base task class with additional error handling and monitoring."""
    
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3, 'countdown': 5}
    retry_backoff = True
    retry_backoff_max = 300  # 5 minutes
    retry_jitter = True
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(f"Task {self.name}[{task_id}] failed permanently: {exc}")
        # Could send notifications or update dashboards
        super().on_failure(exc, task_id, args, kwargs, einfo)
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(f"Task {self.name}[{task_id}] retrying due to: {exc}")
        super().on_retry(exc, task_id, args, kwargs, einfo)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.debug(f"Task {self.name}[{task_id}] completed successfully")
        super().on_success(retval, task_id, args, kwargs)

# Set the default task base
app.Task = ReAgentTask

# Health check function
def check_celery_health() -> Dict[str, Any]:
    """Check Celery worker and broker health."""
    try:
        # Check broker connection
        conn = app.connection()
        conn.ensure_connection(max_retries=3)
        broker_ok = True
    except Exception as e:
        broker_ok = False
        logger.error(f"Broker health check failed: {e}")
    
    try:
        # Check worker status
        inspect = app.control.inspect()
        stats = inspect.stats()
        active = inspect.active()
        worker_ok = bool(stats)
    except Exception as e:
        worker_ok = False
        logger.error(f"Worker health check failed: {e}")
    
    return {
        'broker_healthy': broker_ok,
        'workers_healthy': worker_ok,
        'workers_count': len(stats) if worker_ok and stats else 0,
        'active_tasks': sum(len(tasks) for tasks in active.values()) if active else 0
    }

# CLI command for running worker
def main():
    """Entry point for running Celery worker."""
    import sys
    
    # Default to 'worker' command if none specified
    if len(sys.argv) == 1:
        sys.argv.append('worker')
    
    # Add common worker arguments
    if 'worker' in sys.argv and '--loglevel' not in sys.argv:
        sys.argv.extend(['--loglevel', settings.log_level])
    
    if 'worker' in sys.argv and '--concurrency' not in sys.argv:
        concurrency = os.getenv('CELERY_WORKER_CONCURRENCY', '4')
        sys.argv.extend(['--concurrency', concurrency])
    
    # Start the worker
    app.start(argv=sys.argv)

if __name__ == '__main__':
    main()
