"""
Celery tasks for asynchronous processing in ReAgent.

This module defines all asynchronous tasks including question processing,
batch operations, maintenance tasks, and monitoring.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import traceback
from functools import wraps

from celery import Task, group, chain, chord
from celery.exceptions import SoftTimeLimitExceeded
import redis
import aioredis
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from worker.celery_app import app
from reagent.main import ReAgentSystem
from reagent.models import Message, MessageType
from reagent.monitoring import metrics_collector
from reagent.config import settings
from db.models import QuestionRecord, TaskRecord, CheckpointRecord
from db.session import get_async_session

logger = logging.getLogger(__name__)

# Redis client for pub/sub
redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=True
)

# Async Redis client for async tasks
async def get_async_redis():
    """Get async Redis client."""
    return await aioredis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
    )

# Decorator to run async functions in sync context
def async_task(func):
    """Decorator to run async tasks in Celery."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

# Task utilities
def publish_task_update(task_id: str, status: str, data: Dict[str, Any]):
    """Publish task status update to Redis."""
    try:
        update_data = {
            "task_id": task_id,
            "status": status,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        redis_client.publish(
            f"reagent:tasks:{task_id}",
            json.dumps(update_data)
        )
    except Exception as e:
        logger.error(f"Failed to publish task update: {e}")

def publish_question_update(question_id: str, update_data: Dict[str, Any]):
    """Publish question status update for WebSocket."""
    try:
        redis_client.publish(
            f"reagent:questions:{question_id}",
            json.dumps(update_data)
        )
    except Exception as e:
        logger.error(f"Failed to publish question update: {e}")

# Main tasks
@app.task(bind=True, name='worker.tasks.process_question')
@async_task
async def process_question(
    self: Task,
    question: str,
    question_id: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a single question through the ReAgent system.
    
    Args:
        question: The question to process
        question_id: Unique question identifier
        options: Processing options (timeout, max_backtrack_depth, etc.)
    
    Returns:
        Processing result with answer and metadata
    """
    start_time = time.time()
    task_id = self.request.id
    
    # Default options
    options = options or {}
    timeout = options.get('timeout', 60)
    max_backtrack_depth = options.get('max_backtrack_depth', 5)
    enable_caching = options.get('enable_caching', True)
    
    logger.info(f"Processing question {question_id}: {question[:50]}...")
    
    # Update task status
    publish_task_update(task_id, "processing", {
        "question_id": question_id,
        "progress": 0
    })
    
    # Publish initial question status
    publish_question_update(question_id, {
        "status": "received",
        "progress": 0,
        "current_phase": "initialization"
    })
    
    try:
        # Check cache if enabled
        if enable_caching:
            cached_result = await check_question_cache(question_id)
            if cached_result:
                logger.info(f"Using cached result for question {question_id}")
                return cached_result
        
        # Initialize ReAgent system
        system = ReAgentSystem()
        await system.start()
        
        # Record question in database
        async with get_async_session() as session:
            question_record = QuestionRecord(
                id=question_id,
                question_text=question,
                task_id=task_id,
                status="processing",
                created_at=datetime.now()
            )
            session.add(question_record)
            await session.commit()
        
        # Process question with progress updates
        result = None
        update_task = asyncio.create_task(
            periodic_progress_updates(question_id, system)
        )
        
        try:
            # Run question processing with timeout
            result = await asyncio.wait_for(
                system.process_question(question, timeout=timeout),
                timeout=timeout + 10  # Add buffer for system timeout
            )
            
            # Process successful
            if result['status'] == 'success':
                processing_time = time.time() - start_time
                
                # Update database
                async with get_async_session() as session:
                    await session.execute(
                        update(QuestionRecord)
                        .where(QuestionRecord.id == question_id)
                        .values(
                            status="completed",
                            answer=result['answer'],
                            confidence=result.get('confidence', 0.0),
                            processing_time=processing_time,
                            completed_at=datetime.now()
                        )
                    )
                    await session.commit()
                
                # Cache result if enabled
                if enable_caching:
                    await cache_question_result(question_id, result)
                
                # Publish final status
                publish_question_update(question_id, {
                    "status": "completed",
                    "progress": 100,
                    "final_answer": result['answer'],
                    "confidence": result.get('confidence', 0.0),
                    "supporting_facts": result.get('supporting_facts', [])
                })
                
                # Update metrics
                metrics_collector.record_question_processed(True, processing_time)
                
                return {
                    "status": "success",
                    "question_id": question_id,
                    "answer": result['answer'],
                    "confidence": result.get('confidence', 0.0),
                    "supporting_facts": result.get('supporting_facts', []),
                    "reasoning_trace": result.get('reasoning_trace', []),
                    "processing_time": processing_time,
                    "backtracking_events": len([
                        step for step in result.get('reasoning_trace', [])
                        if 'backtrack' in step.get('action', '').lower()
                    ])
                }
            else:
                # Processing failed
                raise Exception(result.get('error', 'Unknown error'))
                
        except asyncio.TimeoutError:
            raise SoftTimeLimitExceeded("Question processing timeout")
        finally:
            update_task.cancel()
            await system.stop()
            
    except SoftTimeLimitExceeded:
        logger.warning(f"Question {question_id} processing timeout")
        
        # Update status
        async with get_async_session() as session:
            await session.execute(
                update(QuestionRecord)
                .where(QuestionRecord.id == question_id)
                .values(
                    status="timeout",
                    error="Processing timeout exceeded",
                    completed_at=datetime.now()
                )
            )
            await session.commit()
        
        publish_question_update(question_id, {
            "status": "failed",
            "error": "Processing timeout"
        })
        
        metrics_collector.record_question_processed(False, time.time() - start_time)
        
        return {
            "status": "timeout",
            "question_id": question_id,
            "error": "Processing timeout exceeded",
            "partial_results": {}
        }
        
    except Exception as e:
        logger.error(f"Error processing question {question_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Update status
        async with get_async_session() as session:
            await session.execute(
                update(QuestionRecord)
                .where(QuestionRecord.id == question_id)
                .values(
                    status="failed",
                    error=str(e),
                    completed_at=datetime.now()
                )
            )
            await session.commit()
        
        publish_question_update(question_id, {
            "status": "failed",
            "error": str(e)
        })
        
        metrics_collector.record_question_processed(False, time.time() - start_time)
        
        return {
            "status": "failed",
            "question_id": question_id,
            "error": str(e)
        }

async def periodic_progress_updates(question_id: str, system: ReAgentSystem):
    """Send periodic progress updates for a question."""
    try:
        progress_map = {
            "decomposer": 20,
            "retriever": 40,
            "verifier": 60,
            "assembler": 80
        }
        
        while True:
            await asyncio.sleep(2)  # Update every 2 seconds
            
            # Get current system status
            status = system.get_system_status()
            
            # Estimate progress based on active agents
            active_agents = status.get('agents', {})
            max_progress = 0
            current_phase = "initializing"
            
            for agent, agent_status in active_agents.items():
                if agent_status.get('assertions', 0) > 0:
                    agent_type = agent.split('_')[0].lower()
                    if agent_type in progress_map:
                        max_progress = max(max_progress, progress_map[agent_type])
                        current_phase = f"{agent_type}_processing"
            
            publish_question_update(question_id, {
                "status": "processing",
                "progress": max_progress,
                "current_phase": current_phase
            })
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error in progress updates: {e}")

@app.task(bind=True, name='worker.tasks.process_batch_questions')
async def process_batch_questions(
    self: Task,
    questions: List[str],
    batch_id: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process multiple questions in batch.
    
    Args:
        questions: List of questions to process
        batch_id: Unique batch identifier
        options: Batch processing options
    
    Returns:
        Batch processing results
    """
    start_time = time.time()
    options = options or {}
    
    # Batch settings
    parallel_processing = options.get('parallel_processing', True)
    max_concurrent = options.get('max_concurrent', 5)
    fail_fast = options.get('fail_fast', False)
    
    logger.info(f"Processing batch {batch_id} with {len(questions)} questions")
    
    results = []
    
    if parallel_processing:
        # Process questions in parallel using Celery group
        job = group(
            process_question.s(
                question=q,
                question_id=f"{batch_id}_q{i}",
                options=options
            )
            for i, q in enumerate(questions)
        )
        
        # Execute and collect results
        async_result = job.apply_async()
        
        # Monitor progress
        while not async_result.ready():
            completed = sum(1 for r in async_result.results if r.ready())
            publish_task_update(self.request.id, "processing", {
                "batch_id": batch_id,
                "progress": (completed / len(questions)) * 100,
                "completed": completed,
                "total": len(questions)
            })
            
            # Check for fail-fast condition
            if fail_fast:
                for r in async_result.results:
                    if r.ready() and r.failed():
                        logger.warning(f"Batch {batch_id} failing fast due to error")
                        async_result.revoke(terminate=True)
                        break
            
            await asyncio.sleep(1)
        
        # Collect results
        for i, result in enumerate(async_result.results):
            try:
                results.append(result.get())
            except Exception as e:
                results.append({
                    "status": "failed",
                    "question_id": f"{batch_id}_q{i}",
                    "error": str(e)
                })
    else:
        # Process questions sequentially
        for i, question in enumerate(questions):
            try:
                result = await process_question(
                    question=question,
                    question_id=f"{batch_id}_q{i}",
                    options=options
                )
                results.append(result)
                
                # Update progress
                publish_task_update(self.request.id, "processing", {
                    "batch_id": batch_id,
                    "progress": ((i + 1) / len(questions)) * 100,
                    "completed": i + 1,
                    "total": len(questions)
                })
                
                # Check fail-fast
                if fail_fast and result['status'] == 'failed':
                    logger.warning(f"Batch {batch_id} stopping due to failure")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing question {i} in batch: {e}")
                results.append({
                    "status": "failed",
                    "question_id": f"{batch_id}_q{i}",
                    "error": str(e)
                })
                
                if fail_fast:
                    break
    
    # Calculate summary
    summary = {
        "completed": sum(1 for r in results if r['status'] == 'success'),
        "failed": sum(1 for r in results if r['status'] == 'failed'),
        "timeout": sum(1 for r in results if r['status'] == 'timeout')
    }
    
    processing_time = time.time() - start_time
    
    return {
        "batch_id": batch_id,
        "total_questions": len(questions),
        "results": results,
        "summary": summary,
        "processing_time": processing_time,
        "average_time_per_question": processing_time / len(questions) if questions else 0
    }

@app.task(bind=True, name='worker.tasks.execute_backtracking')
@async_task
async def execute_backtracking(
    self: Task,
    question_id: str,
    scope: str,
    trigger: str,
    checkpoint_id: Optional[str] = None,
    target_agent: Optional[str] = None,
    reason: str = "Manual trigger"
) -> Dict[str, Any]:
    """
    Execute backtracking operation for a question.
    
    Args:
        question_id: Question to backtrack
        scope: 'local' or 'global'
        trigger: What triggered the backtracking
        checkpoint_id: Specific checkpoint to rollback to
        target_agent: Target agent for local backtracking
        reason: Reason for backtracking
    
    Returns:
        Backtracking result
    """
    start_time = time.time()
    backtracking_id = f"bt_{self.request.id}"
    
    logger.info(f"Executing {scope} backtracking for question {question_id}")
    
    # Publish backtracking event
    redis_client.publish(
        f"reagent:backtracking:{question_id}",
        json.dumps({
            "event_id": backtracking_id,
            "type": scope,
            "trigger": trigger,
            "reason": reason,
            "agent_id": target_agent,
            "question_id": question_id,
            "checkpoint_id": checkpoint_id
        })
    )
    
    try:
        # Get system instance for the question
        # In production, this would retrieve the running system instance
        system = ReAgentSystem()
        await system.start()
        
        affected_agents = []
        
        if scope == "local":
            if not target_agent:
                raise ValueError("target_agent required for local backtracking")
            
            # Execute local backtracking
            agent = system.agents.get(target_agent.lower())
            if not agent:
                raise ValueError(f"Agent {target_agent} not found")
            
            if checkpoint_id:
                success = agent.local_knowledge.rollback(checkpoint_id)
            else:
                # Rollback to latest checkpoint
                if agent.local_knowledge.checkpoints:
                    checkpoint_id = agent.local_knowledge.checkpoints[-1].id
                    success = agent.local_knowledge.rollback(checkpoint_id)
                else:
                    success = False
            
            if success:
                affected_agents = [target_agent]
                logger.info(f"Local backtracking successful for {target_agent}")
            else:
                raise Exception("Local backtracking failed")
                
        else:  # global
            # Execute global backtracking
            supervisor = system.agents.get('supervisor')
            if not supervisor:
                raise ValueError("Supervisor agent not found")
            
            # Trigger global rollback
            await supervisor.initiate_global_rollback(
                {"reason": reason, "checkpoint_id": checkpoint_id}
            )
            
            affected_agents = list(system.agents.keys())
            logger.info("Global backtracking initiated")
        
        # Record backtracking in database
        async with get_async_session() as session:
            session.add(CheckpointRecord(
                id=backtracking_id,
                question_id=question_id,
                agent_id=target_agent or "system",
                checkpoint_type=scope,
                checkpoint_data=json.dumps({
                    "trigger": trigger,
                    "reason": reason,
                    "affected_agents": affected_agents
                }),
                created_at=datetime.now()
            ))
            await session.commit()
        
        processing_time = time.time() - start_time
        
        # Update metrics
        metrics_collector.record_backtracking(
            target_agent or "system",
            scope,
            len(affected_agents)
        )
        
        return {
            "backtracking_id": backtracking_id,
            "question_id": question_id,
            "scope": scope,
            "status": "completed",
            "affected_agents": affected_agents,
            "checkpoint_id": checkpoint_id,
            "processing_time": processing_time,
            "outcome": "Backtracking completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Backtracking failed: {e}")
        
        return {
            "backtracking_id": backtracking_id,
            "question_id": question_id,
            "scope": scope,
            "status": "failed",
            "error": str(e),
            "processing_time": time.time() - start_time
        }
    finally:
        await system.stop()

@app.task(name='worker.tasks.cleanup_old_checkpoints')
@async_task
async def cleanup_old_checkpoints() -> Dict[str, Any]:
    """
    Clean up old checkpoints based on retention policy.
    
    Returns:
        Cleanup statistics
    """
    start_time = time.time()
    retention_hours = settings.checkpoint_retention_hours
    cutoff_time = datetime.now() - timedelta(hours=retention_hours)
    
    logger.info(f"Cleaning up checkpoints older than {retention_hours} hours")
    
    try:
        async with get_async_session() as session:
            # Count checkpoints to delete
            result = await session.execute(
                select(CheckpointRecord)
                .where(CheckpointRecord.created_at < cutoff_time)
            )
            checkpoints_to_delete = result.scalars().all()
            count = len(checkpoints_to_delete)
            
            # Delete old checkpoints
            await session.execute(
                delete(CheckpointRecord)
                .where(CheckpointRecord.created_at < cutoff_time)
            )
            
            # Also clean up old completed questions
            await session.execute(
                delete(QuestionRecord)
                .where(QuestionRecord.completed_at < cutoff_time)
                .where(QuestionRecord.status.in_(['completed', 'failed', 'timeout']))
            )
            
            await session.commit()
        
        # Clean up Redis cache
        redis_keys = redis_client.keys("reagent:cache:*")
        expired_keys = 0
        
        for key in redis_keys:
            ttl = redis_client.ttl(key)
            if ttl == -1:  # No expiration set
                # Set expiration for old keys
                redis_client.expire(key, retention_hours * 3600)
                expired_keys += 1
        
        processing_time = time.time() - start_time
        
        logger.info(f"Cleaned up {count} checkpoints and set expiration for {expired_keys} cache keys")
        
        return {
            "checkpoints_deleted": count,
            "cache_keys_expired": expired_keys,
            "processing_time": processing_time,
            "cutoff_time": cutoff_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Checkpoint cleanup failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "processing_time": time.time() - start_time
        }

@app.task(name='worker.tasks.collect_system_metrics')
async def collect_system_metrics() -> Dict[str, Any]:
    """
    Collect and publish system metrics.
    
    Returns:
        Current metrics snapshot
    """
    try:
        # Get metrics summary
        metrics_summary = metrics_collector.get_metrics_summary()
        
        # Get Celery stats
        inspect = app.control.inspect()
        active_tasks = inspect.active()
        
        # Calculate additional metrics
        total_active = sum(len(tasks) for tasks in (active_tasks or {}).values())
        
        # Get Redis info
        redis_info = redis_client.info()
        
        metrics_data = {
            "active_questions": metrics_summary['questions']['active'],
            "total_processed": metrics_summary['questions']['total_processed'],
            "active_agents": metrics_summary['agents']['active'],
            "backtracking_count": metrics_summary['performance']['total_backtracking_events'],
            "celery_active_tasks": total_active,
            "redis_memory_mb": redis_info.get('used_memory', 0) / (1024 * 1024),
            "redis_connected_clients": redis_info.get('connected_clients', 0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Publish to Redis for WebSocket distribution
        redis_client.publish("reagent:metrics", json.dumps(metrics_data))
        
        # Update Prometheus metrics
        metrics_collector.update_memory_usage()
        metrics_collector.update_message_queue_size(total_active)
        
        return {
            "status": "success",
            "metrics": metrics_data,
            "collection_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@app.task(name='worker.tasks.generate_performance_report')
@async_task
async def generate_performance_report(report_type: str = "hourly") -> Dict[str, Any]:
    """
    Generate performance reports.
    
    Args:
        report_type: 'hourly', 'daily', or 'weekly'
    
    Returns:
        Performance report data
    """
    try:
        # Determine time range
        end_time = datetime.now()
        if report_type == "hourly":
            start_time = end_time - timedelta(hours=1)
        elif report_type == "daily":
            start_time = end_time - timedelta(days=1)
        elif report_type == "weekly":
            start_time = end_time - timedelta(weeks=1)
        else:
            raise ValueError(f"Invalid report type: {report_type}")
        
        async with get_async_session() as session:
            # Get question statistics
            result = await session.execute(
                select(QuestionRecord)
                .where(QuestionRecord.created_at >= start_time)
            )
            questions = result.scalars().all()
            
            # Calculate statistics
            total_questions = len(questions)
            completed = sum(1 for q in questions if q.status == 'completed')
            failed = sum(1 for q in questions if q.status == 'failed')
            timeout = sum(1 for q in questions if q.status == 'timeout')
            
            avg_processing_time = 0
            avg_confidence = 0
            
            if completed > 0:
                processing_times = [q.processing_time for q in questions 
                                  if q.status == 'completed' and q.processing_time]
                if processing_times:
                    avg_processing_time = sum(processing_times) / len(processing_times)
                
                confidences = [q.confidence for q in questions 
                             if q.status == 'completed' and q.confidence]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
        
        report = {
            "report_type": report_type,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": {
                "total_questions": total_questions,
                "completed": completed,
                "failed": failed,
                "timeout": timeout,
                "success_rate": (completed / total_questions * 100) if total_questions > 0 else 0
            },
            "performance": {
                "avg_processing_time_seconds": avg_processing_time,
                "avg_confidence_score": avg_confidence
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Store report in Redis with expiration
        redis_key = f"reagent:reports:{report_type}:{end_time.strftime('%Y%m%d%H')}"
        redis_client.setex(
            redis_key,
            timedelta(days=7),  # Keep reports for 7 days
            json.dumps(report)
        )
        
        logger.info(f"Generated {report_type} performance report")
        
        return report
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "report_type": report_type
        }

@app.task(name='worker.tasks.health_check_task')
async def health_check_task() -> Dict[str, Any]:
    """
    Perform system health check.
    
    Returns:
        Health check results
    """
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # Check Redis
    try:
        redis_client.ping()
        health_status["checks"]["redis"] = {"status": "healthy", "latency_ms": 1}
    except Exception as e:
        health_status["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}
    
    # Check database
    try:
        async with get_async_session() as session:
            await session.execute(select(1))
        health_status["checks"]["database"] = {"status": "healthy", "latency_ms": 5}
    except Exception as e:
        health_status["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
    
    # Check Celery workers
    try:
        inspect = app.control.inspect()
        stats = inspect.stats()
        if stats:
            health_status["checks"]["celery"] = {
                "status": "healthy",
                "workers": len(stats)
            }
        else:
            health_status["checks"]["celery"] = {"status": "unhealthy", "error": "No workers"}
    except Exception as e:
        health_status["checks"]["celery"] = {"status": "unhealthy", "error": str(e)}
    
    # Overall status
    all_healthy = all(
        check.get("status") == "healthy" 
        for check in health_status["checks"].values()
    )
    health_status["overall_status"] = "healthy" if all_healthy else "unhealthy"
    
    # Publish health status
    redis_client.setex(
        "reagent:health:latest",
        60,  # 1 minute expiration
        json.dumps(health_status)
    )
    
    return health_status

# Cache utilities
async def check_question_cache(question_id: str) -> Optional[Dict[str, Any]]:
    """Check if question result is cached."""
    try:
        cached = redis_client.get(f"reagent:cache:question:{question_id}")
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.error(f"Cache check failed: {e}")
    return None

async def cache_question_result(question_id: str, result: Dict[str, Any]):
    """Cache question result."""
    try:
        redis_client.setex(
            f"reagent:cache:question:{question_id}",
            timedelta(hours=24),  # 24 hour cache
            json.dumps(result)
        )
    except Exception as e:
        logger.error(f"Cache write failed: {e}")
