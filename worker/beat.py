"""
Celery Beat scheduler configuration for periodic tasks.

This module configures and manages scheduled tasks for the ReAgent system,
including maintenance, monitoring, and reporting tasks.
"""

import os
import logging
from datetime import timedelta
from celery import Celery
from celery.schedules import crontab, solar
from typing import Dict, Any, Optional

from reagent.config import settings

logger = logging.getLogger(__name__)

# Import the Celery app
from worker.celery_app import app

# Beat configuration
beat_schedule = {
    # System maintenance tasks
    'cleanup-old-checkpoints': {
        'task': 'worker.tasks.cleanup_old_checkpoints',
        'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
        'options': {
            'queue': 'maintenance',
            'priority': 1,
            'expires': 3600  # Expire after 1 hour if not executed
        },
        'args': (),
        'kwargs': {},
        'description': 'Clean up old checkpoints and expired data'
    },
    
    'cleanup-cache': {
        'task': 'worker.tasks.cleanup_cache',
        'schedule': crontab(minute=30, hour='*/4'),  # Every 4 hours at 30 minutes
        'options': {
            'queue': 'maintenance',
            'priority': 1
        },
        'args': (),
        'kwargs': {},
        'description': 'Clean up expired cache entries'
    },
    
    'optimize-database': {
        'task': 'worker.tasks.optimize_database',
        'schedule': crontab(minute=0, hour=3, day_of_week=0),  # Sunday at 3 AM
        'options': {
            'queue': 'maintenance',
            'priority': 0,
            'time_limit': 3600  # 1 hour time limit
        },
        'args': (),
        'kwargs': {},
        'description': 'Optimize database tables and indexes'
    },
    
    # Monitoring tasks
    'collect-system-metrics': {
        'task': 'worker.tasks.collect_system_metrics',
        'schedule': timedelta(minutes=1),  # Every minute
        'options': {
            'queue': 'monitoring',
            'priority': 2,
            'expires': 30  # Expire after 30 seconds
        },
        'args': (),
        'kwargs': {},
        'description': 'Collect and publish system metrics'
    },
    
    'collect-detailed-metrics': {
        'task': 'worker.tasks.collect_detailed_metrics',
        'schedule': timedelta(minutes=5),  # Every 5 minutes
        'options': {
            'queue': 'monitoring',
            'priority': 2
        },
        'args': (),
        'kwargs': {},
        'description': 'Collect detailed performance metrics'
    },
    
    'monitor-agent-health': {
        'task': 'worker.tasks.monitor_agent_health',
        'schedule': timedelta(minutes=2),  # Every 2 minutes
        'options': {
            'queue': 'monitoring',
            'priority': 3
        },
        'args': (),
        'kwargs': {},
        'description': 'Monitor individual agent health and performance'
    },
    
    # Reporting tasks
    'generate-hourly-report': {
        'task': 'worker.tasks.generate_performance_report',
        'schedule': crontab(minute=5),  # 5 minutes past every hour
        'options': {
            'queue': 'reporting',
            'priority': 1
        },
        'args': ('hourly',),
        'kwargs': {},
        'description': 'Generate hourly performance report'
    },
    
    'generate-daily-report': {
        'task': 'worker.tasks.generate_performance_report',
        'schedule': crontab(hour=0, minute=15),  # Daily at 00:15
        'options': {
            'queue': 'reporting',
            'priority': 1
        },
        'args': ('daily',),
        'kwargs': {},
        'description': 'Generate daily performance report'
    },
    
    'generate-weekly-report': {
        'task': 'worker.tasks.generate_performance_report',
        'schedule': crontab(hour=1, minute=0, day_of_week=1),  # Monday at 1 AM
        'options': {
            'queue': 'reporting',
            'priority': 0
        },
        'args': ('weekly',),
        'kwargs': {},
        'description': 'Generate weekly performance report'
    },
    
    'export-metrics': {
        'task': 'worker.tasks.export_metrics_to_storage',
        'schedule': crontab(minute=0, hour='*/12'),  # Every 12 hours
        'options': {
            'queue': 'reporting',
            'priority': 0
        },
        'args': (),
        'kwargs': {},
        'description': 'Export metrics to long-term storage'
    },
    
    # Health check tasks
    'system-health-check': {
        'task': 'worker.tasks.health_check_task',
        'schedule': timedelta(seconds=30),  # Every 30 seconds
        'options': {
            'queue': 'health',
            'priority': 10,
            'expires': 20  # Expire after 20 seconds
        },
        'args': (),
        'kwargs': {},
        'description': 'Perform system health check'
    },
    
    'deep-health-check': {
        'task': 'worker.tasks.deep_health_check',
        'schedule': timedelta(minutes=10),  # Every 10 minutes
        'options': {
            'queue': 'health',
            'priority': 5
        },
        'args': (),
        'kwargs': {},
        'description': 'Perform comprehensive system health check'
    },
    
    # Data synchronization tasks
    'sync-knowledge-base': {
        'task': 'worker.tasks.sync_knowledge_base',
        'schedule': crontab(minute='*/30'),  # Every 30 minutes
        'options': {
            'queue': 'maintenance',
            'priority': 2
        },
        'args': (),
        'kwargs': {},
        'description': 'Synchronize knowledge base with external sources'
    },
    
    'update-embeddings': {
        'task': 'worker.tasks.update_embeddings_index',
        'schedule': crontab(hour='*/6', minute=0),  # Every 6 hours
        'options': {
            'queue': 'maintenance',
            'priority': 1,
            'time_limit': 1800  # 30 minutes time limit
        },
        'args': (),
        'kwargs': {},
        'description': 'Update vector embeddings index'
    },
    
    # Alert monitoring
    'check-alerts': {
        'task': 'worker.tasks.check_alert_conditions',
        'schedule': timedelta(minutes=5),  # Every 5 minutes
        'options': {
            'queue': 'monitoring',
            'priority': 8
        },
        'args': (),
        'kwargs': {},
        'description': 'Check alert conditions and send notifications'
    },
    
    # Token usage monitoring
    'monitor-token-usage': {
        'task': 'worker.tasks.monitor_token_usage',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
        'options': {
            'queue': 'monitoring',
            'priority': 3
        },
        'args': (),
        'kwargs': {},
        'description': 'Monitor LLM token usage and costs'
    },
    
    # Backup tasks
    'backup-checkpoints': {
        'task': 'worker.tasks.backup_checkpoints',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
        'options': {
            'queue': 'maintenance',
            'priority': 1,
            'time_limit': 3600  # 1 hour time limit
        },
        'args': (),
        'kwargs': {},
        'description': 'Backup system checkpoints'
    },
}

# Environment-specific schedule adjustments
if settings.environment == "development":
    # More frequent health checks in development
    beat_schedule['system-health-check']['schedule'] = timedelta(seconds=10)
    # Less frequent cleanup in development
    beat_schedule['cleanup-old-checkpoints']['schedule'] = crontab(hour=0, minute=0)
elif settings.environment == "production":
    # Add production-specific tasks
    beat_schedule['audit-log-export'] = {
        'task': 'worker.tasks.export_audit_logs',
        'schedule': crontab(hour=4, minute=0),  # Daily at 4 AM
        'options': {
            'queue': 'maintenance',
            'priority': 0
        },
        'args': (),
        'kwargs': {},
        'description': 'Export audit logs for compliance'
    }

# Dynamic schedule configuration based on settings
def configure_dynamic_schedules():
    """Configure schedules based on runtime settings."""
    
    # Adjust metric collection frequency
    if hasattr(settings, 'metrics_collection_interval'):
        interval = settings.metrics_collection_interval
        beat_schedule['collect-system-metrics']['schedule'] = timedelta(seconds=interval)
    
    # Configure backup frequency
    if hasattr(settings, 'backup_frequency_hours'):
        hours = settings.backup_frequency_hours
        beat_schedule['backup-checkpoints']['schedule'] = crontab(
            hour=f'*/{hours}', 
            minute=0
        )
    
    # Configure report generation
    if hasattr(settings, 'disable_reporting') and settings.disable_reporting:
        # Remove reporting tasks
        for task_name in list(beat_schedule.keys()):
            if 'report' in task_name:
                del beat_schedule[task_name]
    
    # Add custom schedules from configuration
    if hasattr(settings, 'custom_schedules'):
        for task_name, schedule_config in settings.custom_schedules.items():
            beat_schedule[task_name] = schedule_config

# Apply dynamic configuration
configure_dynamic_schedules()

# Update Celery app configuration
app.conf.beat_schedule = beat_schedule

# Beat configuration options
app.conf.beat_scheduler = 'celery.beat:PersistentScheduler'
app.conf.beat_schedule_filename = 'celerybeat-schedule.db'
app.conf.beat_max_loop_interval = 5  # Maximum number of seconds to sleep between re-checking schedules

# Timezone configuration
app.conf.timezone = 'UTC'

# Custom Beat class for advanced scheduling
from celery.beat import ScheduleEntry, Scheduler

class ReAgentScheduler(Scheduler):
    """Custom scheduler with additional features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        
    def setup_schedule(self):
        """Set up the schedule."""
        super().setup_schedule()
        self.logger.info(f"Loaded {len(self.schedule)} scheduled tasks")
        
        # Log schedule details
        for name, entry in self.schedule.items():
            self.logger.debug(f"Scheduled task: {name} - {entry.schedule}")
    
    def tick(self):
        """Run a tick of the scheduler."""
        # Could add custom logic here for dynamic schedule adjustments
        return super().tick()
    
    def reserve(self, entry):
        """Reserve an entry for execution."""
        # Could add custom logic for task reservation
        return super().reserve(entry)

# Function to run Beat scheduler
def run_beat():
    """Run the Celery Beat scheduler."""
    from celery.bin import beat
    
    beat = beat.beat(app=app)
    beat.run(
        scheduler=ReAgentScheduler,
        loglevel=settings.log_level,
        logfile=None,  # Log to stdout
        pidfile=None,  # Don't create pidfile
    )

# CLI entry point
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Celery Beat scheduler for ReAgent")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Loaded {len(beat_schedule)} scheduled tasks")
    
    # Run the beat scheduler
    run_beat()
