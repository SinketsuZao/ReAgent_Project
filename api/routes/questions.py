"""
Question Processing Routes

Handles submission and processing of multi-hop questions.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request, status
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
import logging
import asyncio

from api.schemas.requests import QuestionRequest, BatchQuestionRequest
from api.schemas.responses import (
    QuestionResponse, 
    QuestionDetailResponse,
    BatchQuestionResponse,
    QuestionListResponse
)
from reagent.models import ProcessingResult, Answer
from reagent.monitoring import metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/questions")

# ============== Dependencies ==============

async def get_reagent_system(request: Request):
    """Get ReAgent system from app state."""
    if not hasattr(request.app.state, 'reagent_system'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ReAgent system not initialized"
        )
    return request.app.state.reagent_system

async def get_tasks_store(request: Request) -> Dict[str, Any]:
    """Get tasks storage from app state."""
    if not hasattr(request.app.state, 'tasks'):
        request.app.state.tasks = {}
    return request.app.state.tasks

# ============== Routes ==============

@router.post("/", response_model=QuestionResponse)
async def submit_question(
    request: QuestionRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    reagent_system = Depends(get_reagent_system),
    tasks_store = Depends(get_tasks_store)
):
    """
    Submit a question for processing.
    
    This endpoint accepts a multi-hop question and queues it for asynchronous
    processing. Returns immediately with a task ID for tracking.
    """
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Validate question
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    if len(request.question) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question too long (max 1000 characters)"
        )
    
    # Check system capacity
    active_questions = sum(
        1 for task in tasks_store.values() 
        if task.get('status') == 'processing'
    )
    
    if active_questions >= reagent_system.performance.max_concurrent_agents:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System at capacity, please try again later"
        )
    
    # Create task entry
    task = {
        'task_id': task_id,
        'question': request.question,
        'status': 'pending',
        'progress': 0.0,
        'result': None,
        'error': None,
        'created_at': datetime.now(),
        'updated_at': datetime.now(),
        'metadata': request.metadata or {},
        'options': {
            'max_hops': request.max_hops,
            'temperature': request.temperature,
            'timeout': request.timeout
        }
    }
    
    tasks_store[task_id] = task
    
    # Record metrics
    metrics_collector.record_question_started(task_id)
    
    # Queue for processing
    background_tasks.add_task(
        process_question_background,
        reagent_system,
        task_id,
        request.question,
        task,
        tasks_store
    )
    
    logger.info(f"Question submitted: task_id={task_id}, question_length={len(request.question)}")
    
    return QuestionResponse(
        task_id=task_id,
        status="accepted",
        message=f"Question submitted for processing. Track at /api/v1/tasks/{task_id}",
        estimated_time=estimate_processing_time(request.question)
    )

@router.post("/batch", response_model=BatchQuestionResponse)
async def submit_batch_questions(
    request: BatchQuestionRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    reagent_system = Depends(get_reagent_system),
    tasks_store = Depends(get_tasks_store)
):
    """
    Submit multiple questions for batch processing.
    
    Processes questions in parallel up to system capacity.
    """
    if len(request.questions) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many questions in batch (max 10)"
        )
    
    batch_id = str(uuid.uuid4())
    task_ids = []
    
    for i, question in enumerate(request.questions):
        task_id = f"{batch_id}-{i}"
        
        task = {
            'task_id': task_id,
            'batch_id': batch_id,
            'question': question,
            'status': 'pending',
            'progress': 0.0,
            'result': None,
            'error': None,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'metadata': request.metadata or {},
            'options': {
                'batch_mode': True,
                'priority': request.priority
            }
        }
        
        tasks_store[task_id] = task
        task_ids.append(task_id)
        
        # Queue for processing
        background_tasks.add_task(
            process_question_background,
            reagent_system,
            task_id,
            question,
            task,
            tasks_store
        )
    
    return BatchQuestionResponse(
        batch_id=batch_id,
        task_ids=task_ids,
        status="accepted",
        message=f"Batch of {len(request.questions)} questions submitted"
    )

@router.get("/{task_id}", response_model=QuestionDetailResponse)
async def get_question_detail(
    task_id: str,
    tasks_store = Depends(get_tasks_store)
):
    """Get detailed information about a question processing task."""
    if task_id not in tasks_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    task = tasks_store[task_id]
    
    return QuestionDetailResponse(
        task_id=task_id,
        question=task['question'],
        status=task['status'],
        progress=task['progress'],
        result=task.get('result'),
        error=task.get('error'),
        created_at=task['created_at'],
        updated_at=task['updated_at'],
        metadata=task.get('metadata', {}),
        processing_time=(task['updated_at'] - task['created_at']).total_seconds() if task['status'] in ['completed', 'failed'] else None,
        reasoning_trace=task.get('result', {}).get('reasoning_trace', []) if task.get('result') else []
    )

@router.get("/", response_model=QuestionListResponse)
async def list_questions(
    status: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    tasks_store = Depends(get_tasks_store)
):
    """
    List questions with optional filtering.
    
    Supports pagination and filtering by status.
    """
    # Filter tasks
    tasks = list(tasks_store.values())
    
    if status:
        tasks = [t for t in tasks if t['status'] == status]
    
    # Sort by created_at descending
    tasks.sort(key=lambda t: t['created_at'], reverse=True)
    
    # Apply pagination
    total = len(tasks)
    tasks = tasks[offset:offset + limit]
    
    # Convert to response format
    questions = [
        {
            'task_id': t['task_id'],
            'question': t['question'][:100] + '...' if len(t['question']) > 100 else t['question'],
            'status': t['status'],
            'created_at': t['created_at'],
            'updated_at': t['updated_at']
        }
        for t in tasks
    ]
    
    return QuestionListResponse(
        questions=questions,
        total=total,
        limit=limit,
        offset=offset
    )

@router.delete("/{task_id}")
async def cancel_question(
    task_id: str,
    tasks_store = Depends(get_tasks_store)
):
    """Cancel a pending or processing question."""
    if task_id not in tasks_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    task = tasks_store[task_id]
    
    if task['status'] in ['completed', 'failed', 'cancelled']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel task in {task['status']} state"
        )
    
    # Update task status
    task['status'] = 'cancelled'
    task['updated_at'] = datetime.now()
    task['error'] = 'Cancelled by user'
    
    logger.info(f"Task {task_id} cancelled")
    
    return {
        "message": f"Task {task_id} cancelled",
        "status": "cancelled"
    }

@router.post("/{task_id}/retry")
async def retry_question(
    task_id: str,
    background_tasks: BackgroundTasks,
    req: Request,
    reagent_system = Depends(get_reagent_system),
    tasks_store = Depends(get_tasks_store)
):
    """Retry a failed question."""
    if task_id not in tasks_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    task = tasks_store[task_id]
    
    if task['status'] not in ['failed', 'cancelled']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Can only retry failed or cancelled tasks"
        )
    
    # Reset task for retry
    task['status'] = 'pending'
    task['progress'] = 0.0
    task['result'] = None
    task['error'] = None
    task['updated_at'] = datetime.now()
    task['metadata']['retry_count'] = task['metadata'].get('retry_count', 0) + 1
    
    # Queue for processing
    background_tasks.add_task(
        process_question_background,
        reagent_system,
        task_id,
        task['question'],
        task,
        tasks_store
    )
    
    return {
        "message": f"Task {task_id} queued for retry",
        "status": "pending",
        "retry_count": task['metadata']['retry_count']
    }

# ============== Background Processing ==============

async def process_question_background(
    system,
    task_id: str,
    question: str,
    task: Dict[str, Any],
    tasks_store: Dict[str, Any]
):
    """Process question in background and update task status."""
    try:
        # Update status to processing
        task['status'] = 'processing'
        task['updated_at'] = datetime.now()
        
        logger.info(f"Processing question: task_id={task_id}")
        
        # Notify WebSocket clients
        await notify_task_update(task_id, task)
        
        # Process the question
        options = task.get('options', {})
        result = await system.process_question(
            question,
            timeout=options.get('timeout', 60)
        )
        
        # Update task with result
        task['status'] = 'completed' if result.get('status') == 'success' else 'failed'
        task['progress'] = 1.0
        task['result'] = result
        task['updated_at'] = datetime.now()
        
        if result.get('status') != 'success':
            task['error'] = result.get('error', 'Unknown error')
        
        # Record metrics
        duration = (task['updated_at'] - task['created_at']).total_seconds()
        metrics_collector.record_question_processed(
            success=result.get('status') == 'success',
            duration=duration,
            question_id=task_id
        )
        
        logger.info(f"Question processed: task_id={task_id}, status={task['status']}, duration={duration:.2f}s")
        
        # Notify WebSocket clients
        await notify_task_update(task_id, task)
        
    except asyncio.CancelledError:
        # Task was cancelled
        task['status'] = 'cancelled'
        task['error'] = 'Processing cancelled'
        task['updated_at'] = datetime.now()
        logger.info(f"Task {task_id} cancelled")
        
    except Exception as e:
        # Update task with error
        task['status'] = 'failed'
        task['error'] = str(e)
        task['updated_at'] = datetime.now()
        
        # Record metrics
        duration = (task['updated_at'] - task['created_at']).total_seconds()
        metrics_collector.record_question_processed(
            success=False,
            duration=duration,
            question_id=task_id
        )
        
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        
        # Notify WebSocket clients
        await notify_task_update(task_id, task)

async def notify_task_update(task_id: str, task: Dict[str, Any]):
    """Notify WebSocket clients of task update."""
    # This would integrate with the WebSocket manager
    # For now, just log
    logger.debug(f"Task update notification: {task_id} -> {task['status']}")

def estimate_processing_time(question: str) -> float:
    """Estimate processing time based on question complexity."""
    # Simple heuristic based on question length and complexity
    base_time = 10.0  # Base time in seconds
    
    # Add time based on length
    length_factor = len(question) / 100  # 1 second per 100 characters
    
    # Add time for complexity indicators
    complexity_indicators = [
        'compare', 'difference', 'both', 'multiple',
        'before', 'after', 'when', 'given that'
    ]
    complexity_factor = sum(
        2 for indicator in complexity_indicators 
        if indicator in question.lower()
    )
    
    estimated_time = base_time + length_factor + complexity_factor
    
    return min(estimated_time, 60.0)  # Cap at 60 seconds