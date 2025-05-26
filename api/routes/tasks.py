"""
Task Management Routes

Handles task tracking and status monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, status
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from api.schemas.responses import (
    TaskStatus,
    TaskListResponse,
    TaskStatsResponse,
    TaskDetailResponse
)
from api.routes.questions import get_tasks_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks")

# ============== Routes ==============

@router.get("/{task_id}", response_model=TaskStatus)
async def get_task_status(
    task_id: str,
    tasks_store = Depends(get_tasks_store)
):
    """
    Get status of a processing task.
    
    Returns current status, progress, and results if available.
    """
    if task_id not in tasks_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    task = tasks_store[task_id]
    
    return TaskStatus(
        task_id=task_id,
        status=task['status'],
        progress=task['progress'],
        result=task.get('result'),
        error=task.get('error'),
        created_at=task['created_at'],
        updated_at=task['updated_at'],
        metadata=task.get('metadata', {})
    )

@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    batch_id: Optional[str] = Query(None, description="Filter by batch ID"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of tasks to return"),
    offset: int = Query(0, ge=0, description="Number of tasks to skip"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    tasks_store = Depends(get_tasks_store)
):
    """
    List all tasks with optional filtering and pagination.
    
    Supports filtering by status and batch ID, with configurable sorting.
    """
    # Filter tasks
    tasks = list(tasks_store.values())
    
    # Apply filters
    if status:
        valid_statuses = ['pending', 'processing', 'completed', 'failed', 'cancelled']
        if status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )
        tasks = [t for t in tasks if t['status'] == status]
    
    if batch_id:
        tasks = [t for t in tasks if t.get('batch_id') == batch_id]
    
    # Sort tasks
    reverse = (sort_order == 'desc')
    try:
        tasks.sort(key=lambda t: t.get(sort_by, ''), reverse=reverse)
    except:
        # Fallback to created_at if sort fails
        tasks.sort(key=lambda t: t['created_at'], reverse=reverse)
    
    # Get total before pagination
    total = len(tasks)
    
    # Apply pagination
    tasks = tasks[offset:offset + limit]
    
    # Convert to response format
    task_list = []
    for t in tasks:
        task_status = TaskStatus(
            task_id=t['task_id'],
            status=t['status'],
            progress=t['progress'],
            result=t.get('result'),
            error=t.get('error'),
            created_at=t['created_at'],
            updated_at=t['updated_at'],
            metadata=t.get('metadata', {})
        )
        task_list.append(task_status)
    
    return TaskListResponse(
        tasks=task_list,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + limit) < total
    )

@router.get("/stats/summary", response_model=TaskStatsResponse)
async def get_task_statistics(
    time_window: int = Query(3600, description="Time window in seconds"),
    tasks_store = Depends(get_tasks_store)
):
    """
    Get task processing statistics.
    
    Returns aggregated statistics for the specified time window.
    """
    cutoff_time = datetime.now() - timedelta(seconds=time_window)
    
    # Filter tasks within time window
    recent_tasks = [
        t for t in tasks_store.values()
        if t['created_at'] >= cutoff_time
    ]
    
    # Calculate statistics
    stats = {
        'total_tasks': len(recent_tasks),
        'by_status': {},
        'avg_processing_time': 0,
        'success_rate': 0,
        'active_tasks': 0
    }
    
    # Count by status
    for task in recent_tasks:
        status = task['status']
        stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
    
    # Calculate active tasks
    stats['active_tasks'] = stats['by_status'].get('processing', 0) + stats['by_status'].get('pending', 0)
    
    # Calculate average processing time for completed tasks
    completed_tasks = [
        t for t in recent_tasks 
        if t['status'] in ['completed', 'failed']
    ]
    
    if completed_tasks:
        processing_times = [
            (t['updated_at'] - t['created_at']).total_seconds()
            for t in completed_tasks
        ]
        stats['avg_processing_time'] = sum(processing_times) / len(processing_times)
    
    # Calculate success rate
    completed_count = stats['by_status'].get('completed', 0)
    failed_count = stats['by_status'].get('failed', 0)
    total_finished = completed_count + failed_count
    
    if total_finished > 0:
        stats['success_rate'] = completed_count / total_finished
    
    return TaskStatsResponse(
        time_window_seconds=time_window,
        total_tasks=stats['total_tasks'],
        tasks_by_status=stats['by_status'],
        average_processing_time=stats['avg_processing_time'],
        success_rate=stats['success_rate'],
        active_tasks=stats['active_tasks'],
        peak_time=max((t['created_at'] for t in recent_tasks), default=datetime.now()).isoformat() if recent_tasks else None
    )

@router.delete("/{task_id}")
async def delete_task(
    task_id: str,
    tasks_store = Depends(get_tasks_store)
):
    """
    Delete a task from the system.
    
    Only completed, failed, or cancelled tasks can be deleted.
    """
    if task_id not in tasks_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    task = tasks_store[task_id]
    
    if task['status'] not in ['completed', 'failed', 'cancelled']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete task in {task['status']} state"
        )
    
    # Delete the task
    del tasks_store[task_id]
    
    logger.info(f"Task {task_id} deleted")
    
    return {
        "message": f"Task {task_id} deleted successfully",
        "status": "deleted"
    }

@router.post("/cleanup")
async def cleanup_old_tasks(
    older_than_hours: int = Query(24, description="Delete tasks older than N hours"),
    status: Optional[str] = Query(None, description="Only delete tasks with this status"),
    dry_run: bool = Query(True, description="If true, only show what would be deleted"),
    tasks_store = Depends(get_tasks_store)
):
    """
    Clean up old tasks from the system.
    
    Removes tasks older than the specified time period.
    """
    cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
    
    # Find tasks to delete
    tasks_to_delete = []
    for task_id, task in tasks_store.items():
        # Check age
        if task['created_at'] >= cutoff_time:
            continue
        
        # Check status filter
        if status and task['status'] != status:
            continue
        
        # Don't delete active tasks
        if task['status'] in ['pending', 'processing']:
            continue
        
        tasks_to_delete.append(task_id)
    
    # Perform deletion if not dry run
    if not dry_run:
        for task_id in tasks_to_delete:
            del tasks_store[task_id]
        
        logger.info(f"Cleaned up {len(tasks_to_delete)} old tasks")
    
    return {
        "tasks_to_delete": len(tasks_to_delete),
        "task_ids": tasks_to_delete[:10],  # Show first 10
        "dry_run": dry_run,
        "message": f"{'Would delete' if dry_run else 'Deleted'} {len(tasks_to_delete)} tasks"
    }

@router.get("/{task_id}/detail", response_model=TaskDetailResponse)
async def get_task_detail(
    task_id: str,
    include_trace: bool = Query(True, description="Include reasoning trace"),
    tasks_store = Depends(get_tasks_store)
):
    """
    Get detailed information about a specific task.
    
    Includes full results, reasoning trace, and metadata.
    """
    if task_id not in tasks_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    task = tasks_store[task_id]
    
    # Extract detailed information
    detail = TaskDetailResponse(
        task_id=task_id,
        question=task.get('question', ''),
        status=task['status'],
        progress=task['progress'],
        result=task.get('result'),
        error=task.get('error'),
        created_at=task['created_at'],
        updated_at=task['updated_at'],
        metadata=task.get('metadata', {}),
        options=task.get('options', {}),
        batch_id=task.get('batch_id'),
        processing_time=(task['updated_at'] - task['created_at']).total_seconds() 
            if task['status'] in ['completed', 'failed'] else None
    )
    
    # Add reasoning trace if requested and available
    if include_trace and task.get('result') and isinstance(task['result'], dict):
        detail.reasoning_trace = task['result'].get('reasoning_trace', [])
    
    # Add performance metrics if available
    if task.get('result') and isinstance(task['result'], dict):
        detail.performance_metrics = {
            'backtracking_count': task['result'].get('backtracking_count', 0),
            'token_usage': task['result'].get('token_usage', {}),
            'agent_calls': task['result'].get('agent_calls', 0)
        }
    
    return detail

@router.post("/batch/{batch_id}/cancel")
async def cancel_batch(
    batch_id: str,
    tasks_store = Depends(get_tasks_store)
):
    """
    Cancel all tasks in a batch.
    
    Cancels all pending or processing tasks in the specified batch.
    """
    # Find all tasks in the batch
    batch_tasks = [
        (task_id, task) 
        for task_id, task in tasks_store.items() 
        if task.get('batch_id') == batch_id
    ]
    
    if not batch_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No tasks found for batch {batch_id}"
        )
    
    cancelled_count = 0
    for task_id, task in batch_tasks:
        if task['status'] in ['pending', 'processing']:
            task['status'] = 'cancelled'
            task['updated_at'] = datetime.now()
            task['error'] = 'Batch cancelled'
            cancelled_count += 1
    
    return {
        "batch_id": batch_id,
        "total_tasks": len(batch_tasks),
        "cancelled_tasks": cancelled_count,
        "message": f"Cancelled {cancelled_count} tasks in batch {batch_id}"
    }

@router.get("/batch/{batch_id}/status")
async def get_batch_status(
    batch_id: str,
    tasks_store = Depends(get_tasks_store)
):
    """
    Get status summary for all tasks in a batch.
    
    Returns aggregated status information for the batch.
    """
    # Find all tasks in the batch
    batch_tasks = [
        task for task in tasks_store.values() 
        if task.get('batch_id') == batch_id
    ]
    
    if not batch_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No tasks found for batch {batch_id}"
        )
    
    # Calculate batch statistics
    status_counts = {}
    total_progress = 0
    
    for task in batch_tasks:
        status = task['status']
        status_counts[status] = status_counts.get(status, 0) + 1
        total_progress += task['progress']
    
    # Determine overall batch status
    if status_counts.get('failed', 0) > 0:
        batch_status = 'partial_failure'
    elif status_counts.get('completed', 0) == len(batch_tasks):
        batch_status = 'completed'
    elif status_counts.get('processing', 0) > 0 or status_counts.get('pending', 0) > 0:
        batch_status = 'processing'
    else:
        batch_status = 'unknown'
    
    return {
        "batch_id": batch_id,
        "batch_status": batch_status,
        "total_tasks": len(batch_tasks),
        "status_breakdown": status_counts,
        "overall_progress": total_progress / len(batch_tasks) if batch_tasks else 0,
        "created_at": min(t['created_at'] for t in batch_tasks).isoformat(),
        "updated_at": max(t['updated_at'] for t in batch_tasks).isoformat()
    }