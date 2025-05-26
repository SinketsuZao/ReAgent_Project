"""
ReAgent Monitoring Module

This module provides comprehensive metrics collection and monitoring
capabilities for the ReAgent system using Prometheus client.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, Summary
from datetime import datetime
import logging
from typing import Dict, Any, Optional
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Centralized metrics collection for the ReAgent system.
    
    Collects and exposes metrics in Prometheus format for monitoring
    system performance, agent behavior, and resource usage.
    """
    
    def __init__(self):
        # ============== Counters ==============
        # Total number of questions processed
        self.questions_processed = Counter(
            'reagent_questions_processed_total',
            'Total number of questions processed by the system',
            ['status']  # success, failure, timeout
        )
        
        # Agent message counts
        self.agent_messages = Counter(
            'reagent_agent_messages_total',
            'Total number of messages sent between agents',
            ['sender', 'type']  # sender: agent_id, type: message_type
        )
        
        # Backtracking events
        self.backtracking_events = Counter(
            'reagent_backtracking_total',
            'Total number of backtracking events',
            ['agent', 'scope']  # scope: local or global
        )
        
        # Conflicts detected
        self.conflicts_detected = Counter(
            'reagent_conflicts_detected_total',
            'Total number of conflicts detected by agents',
            ['agent']
        )
        
        # LLM API calls
        self.llm_api_calls = Counter(
            'reagent_llm_api_calls_total',
            'Total number of LLM API calls made',
            ['agent', 'status']  # status: success, rate_limit, timeout, error
        )
        
        # Task completions (for Celery workers)
        self.task_completions = Counter(
            'reagent_task_completions_total',
            'Total number of completed tasks',
            ['task_name', 'status']
        )
        
        # ============== Histograms ==============
        # Question processing duration
        self.question_duration = Histogram(
            'reagent_question_duration_seconds',
            'Time taken to process questions',
            buckets=(1, 5, 10, 30, 60, 120, 300, 600)  # Up to 10 minutes
        )
        
        # Agent processing time
        self.agent_processing_time = Histogram(
            'reagent_agent_processing_seconds',
            'Time taken by each agent to process',
            ['agent'],
            buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30)
        )
        
        # LLM token usage
        self.token_usage = Histogram(
            'reagent_llm_tokens_used',
            'Number of tokens used in LLM calls',
            ['agent', 'type'],  # type: input or output
            buckets=(100, 500, 1000, 2000, 5000, 10000, 20000)
        )
        
        # Backtracking depth
        self.backtrack_depth = Histogram(
            'reagent_backtrack_depth',
            'Depth of backtracking operations',
            ['agent'],
            buckets=(1, 2, 3, 5, 10, 15, 20)
        )
        
        # Message queue latency
        self.message_latency = Histogram(
            'reagent_message_latency_seconds',
            'Latency of message processing',
            ['message_type'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5)
        )
        
        # ============== Summaries ==============
        # LLM response time
        self.llm_response_time = Summary(
            'reagent_llm_response_time_seconds',
            'Response time for LLM API calls',
            ['agent', 'model']
        )
        
        # Evidence retrieval time
        self.retrieval_time = Summary(
            'reagent_retrieval_time_seconds',
            'Time taken to retrieve evidence',
            ['source']  # elasticsearch, kg, llm
        )
        
        # ============== Gauges ==============
        # Active agents
        self.active_agents = Gauge(
            'reagent_active_agents',
            'Number of currently active agents'
        )
        
        # Knowledge assertions per agent
        self.knowledge_assertions = Gauge(
            'reagent_knowledge_assertions',
            'Current number of knowledge assertions',
            ['agent']
        )
        
        # Checkpoint count per agent
        self.checkpoint_count = Gauge(
            'reagent_checkpoints_total',
            'Total number of checkpoints stored',
            ['agent']
        )
        
        # Message queue size
        self.message_queue_size = Gauge(
            'reagent_message_queue_size',
            'Current size of the message queue'
        )
        
        # System memory usage
        self.memory_usage = Gauge(
            'reagent_memory_usage_bytes',
            'Memory usage of the ReAgent system',
            ['type']  # rss, vms, shared
        )
        
        # Active question count
        self.active_questions = Gauge(
            'reagent_active_questions',
            'Number of questions currently being processed'
        )
        
        # Agent reliability scores
        self.agent_reliability = Gauge(
            'reagent_agent_reliability_score',
            'Reliability score for each agent',
            ['agent']
        )
        
        # ============== Info ==============
        # System information
        self.system_info = Info(
            'reagent_system',
            'ReAgent system information'
        )
        self.system_info.info({
            'version': '1.0.0',
            'start_time': datetime.now().isoformat(),
            'environment': 'production'
        })
        
        # ============== Internal State ==============
        self._active_questions = set()
        self._agent_start_times = defaultdict(dict)
        self._llm_start_times = defaultdict(dict)
        self._retrieval_start_times = {}
        
    # ============== Recording Methods ==============
    
    def record_question_started(self, question_id: str):
        """Record that a question has started processing."""
        self._active_questions.add(question_id)
        self.active_questions.set(len(self._active_questions))
    
    def record_question_processed(self, success: bool, duration: float, question_id: str = None):
        """Record question processing completion."""
        status = 'success' if success else 'failure'
        self.questions_processed.labels(status=status).inc()
        self.question_duration.observe(duration)
        
        if question_id and question_id in self._active_questions:
            self._active_questions.remove(question_id)
            self.active_questions.set(len(self._active_questions))
        
        logger.info(f"Question processed: status={status}, duration={duration:.2f}s")
    
    def record_agent_message(self, sender: str, message_type: str):
        """Record an agent message."""
        self.agent_messages.labels(sender=sender, type=message_type).inc()
    
    def record_backtracking(self, agent: str, scope: str, depth: int = 1):
        """Record a backtracking event."""
        self.backtracking_events.labels(agent=agent, scope=scope).inc()
        if scope == "local":
            self.backtrack_depth.labels(agent=agent).observe(depth)
        
        logger.info(f"Backtracking recorded: agent={agent}, scope={scope}, depth={depth}")
    
    def record_conflict(self, agent: str):
        """Record conflict detection."""
        self.conflicts_detected.labels(agent=agent).inc()
    
    def record_agent_processing_start(self, agent: str, operation_id: str):
        """Record start of agent processing."""
        self._agent_start_times[agent][operation_id] = time.time()
    
    def record_agent_processing_end(self, agent: str, operation_id: str):
        """Record end of agent processing."""
        if agent in self._agent_start_times and operation_id in self._agent_start_times[agent]:
            duration = time.time() - self._agent_start_times[agent][operation_id]
            self.agent_processing_time.labels(agent=agent).observe(duration)
            del self._agent_start_times[agent][operation_id]
    
    def record_agent_processing(self, agent: str, duration: float):
        """Record agent processing time directly."""
        self.agent_processing_time.labels(agent=agent).observe(duration)
    
    def record_token_usage(self, agent: str, input_tokens: int, output_tokens: int):
        """Record LLM token usage."""
        self.token_usage.labels(agent=agent, type='input').observe(input_tokens)
        self.token_usage.labels(agent=agent, type='output').observe(output_tokens)
        
        # Log high token usage
        total_tokens = input_tokens + output_tokens
        if total_tokens > 5000:
            logger.warning(
                f"High token usage by {agent}: "
                f"input={input_tokens}, output={output_tokens}, total={total_tokens}"
            )
    
    def record_llm_call_start(self, agent: str, call_id: str, model: str = "gpt-4"):
        """Record start of LLM API call."""
        self._llm_start_times[agent][call_id] = {
            'start_time': time.time(),
            'model': model
        }
    
    def record_llm_call_end(self, agent: str, call_id: str, status: str = "success"):
        """Record end of LLM API call."""
        if agent in self._llm_start_times and call_id in self._llm_start_times[agent]:
            start_info = self._llm_start_times[agent][call_id]
            duration = time.time() - start_info['start_time']
            
            self.llm_api_calls.labels(agent=agent, status=status).inc()
            self.llm_response_time.labels(
                agent=agent, 
                model=start_info['model']
            ).observe(duration)
            
            del self._llm_start_times[agent][call_id]
    
    def record_retrieval_start(self, retrieval_id: str, source: str):
        """Record start of evidence retrieval."""
        self._retrieval_start_times[retrieval_id] = {
            'start_time': time.time(),
            'source': source
        }
    
    def record_retrieval_end(self, retrieval_id: str):
        """Record end of evidence retrieval."""
        if retrieval_id in self._retrieval_start_times:
            start_info = self._retrieval_start_times[retrieval_id]
            duration = time.time() - start_info['start_time']
            
            self.retrieval_time.labels(source=start_info['source']).observe(duration)
            del self._retrieval_start_times[retrieval_id]
    
    def record_message_latency(self, message_type: str, latency: float):
        """Record message processing latency."""
        self.message_latency.labels(message_type=message_type).observe(latency)
    
    def record_task_completion(self, task_name: str, status: str):
        """Record Celery task completion."""
        self.task_completions.labels(task_name=task_name, status=status).inc()
    
    # ============== Update Methods ==============
    
    def update_active_agents(self, count: int):
        """Update the number of active agents."""
        self.active_agents.set(count)
    
    def update_knowledge_assertions(self, agent: str, count: int):
        """Update knowledge assertions gauge for an agent."""
        self.knowledge_assertions.labels(agent=agent).set(count)
    
    def update_checkpoint_count(self, agent: str, count: int):
        """Update checkpoint count for an agent."""
        self.checkpoint_count.labels(agent=agent).set(count)
    
    def update_message_queue_size(self, size: int):
        """Update message queue size."""
        self.message_queue_size.set(size)
    
    def update_memory_usage(self):
        """Update memory usage metrics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.memory_usage.labels(type='rss').set(memory_info.rss)
            self.memory_usage.labels(type='vms').set(memory_info.vms)
            
            if hasattr(memory_info, 'shared'):
                self.memory_usage.labels(type='shared').set(memory_info.shared)
        except ImportError:
            logger.warning("psutil not available, memory metrics disabled")
        except Exception as e:
            logger.error(f"Failed to update memory metrics: {e}")
    
    def update_agent_reliability(self, agent: str, score: float):
        """Update agent reliability score."""
        self.agent_reliability.labels(agent=agent).set(score)
    
    # ============== Utility Methods ==============
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        summary = {
            'questions': {
                'active': len(self._active_questions),
                'total_processed': sum(
                    self.questions_processed.labels(status=s)._value._value
                    for s in ['success', 'failure']
                )
            },
            'agents': {
                'active': self.active_agents._value._value
            },
            'performance': {
                'avg_question_duration': self._calculate_average_duration(),
                'total_backtracking_events': self._count_total_backtracking()
            }
        }
        
        return summary
    
    def _calculate_average_duration(self) -> float:
        """Calculate average question processing duration."""
        # This is simplified - in production, use Prometheus queries
        try:
            if hasattr(self.question_duration, '_sum') and hasattr(self.question_duration, '_count'):
                if self.question_duration._count._value > 0:
                    return self.question_duration._sum._value / self.question_duration._count._value
        except:
            pass
        return 0.0
    
    def _count_total_backtracking(self) -> int:
        """Count total backtracking events."""
        # This is simplified - in production, use Prometheus queries
        total = 0
        try:
            for agent in ['A_Q', 'A_R', 'A_V', 'A_A', 'A_S', 'A_C']:
                for scope in ['local', 'global']:
                    total += self.backtracking_events.labels(agent=agent, scope=scope)._value._value
        except:
            pass
        return total
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        logger.warning("Resetting all metrics")
        
        # Reset counters
        self.questions_processed._metrics.clear()
        self.agent_messages._metrics.clear()
        self.backtracking_events._metrics.clear()
        self.conflicts_detected._metrics.clear()
        self.llm_api_calls._metrics.clear()
        self.task_completions._metrics.clear()
        
        # Reset internal state
        self._active_questions.clear()
        self._agent_start_times.clear()
        self._llm_start_times.clear()
        self._retrieval_start_times.clear()
        
        # Reset gauges
        self.active_questions.set(0)
        self.active_agents.set(0)
        self.message_queue_size.set(0)

# ============== Custom Metrics Classes ==============

class AgentPerformanceTracker:
    """Track detailed performance metrics for individual agents."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.operation_times = defaultdict(list)
        self.success_count = 0
        self.failure_count = 0
        self.conflict_count = 0
        self.backtrack_count = 0
        
    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record an operation performed by the agent."""
        self.operation_times[operation].append(duration)
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def record_conflict(self):
        """Record a conflict detected by the agent."""
        self.conflict_count += 1
    
    def record_backtrack(self):
        """Record a backtracking event."""
        self.backtrack_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'agent_id': self.agent_id,
            'success_rate': self.success_count / max(self.success_count + self.failure_count, 1),
            'conflict_rate': self.conflict_count / max(self.success_count + self.failure_count, 1),
            'backtrack_rate': self.backtrack_count / max(self.success_count + self.failure_count, 1),
            'operation_stats': {}
        }
        
        for operation, times in self.operation_times.items():
            if times:
                stats['operation_stats'][operation] = {
                    'count': len(times),
                    'avg_duration': sum(times) / len(times),
                    'min_duration': min(times),
                    'max_duration': max(times)
                }
        
        return stats

class SystemHealthMonitor:
    """Monitor overall system health."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks = []
        self.last_check_time = None
        self.health_history = []
        
    def add_health_check(self, name: str, check_func: callable, critical: bool = False):
        """Add a health check function."""
        self.health_checks.append({
            'name': name,
            'check': check_func,
            'critical': critical
        })
    
    def check_health(self) -> Dict[str, Any]:
        """Run all health checks and return status."""
        self.last_check_time = datetime.now()
        results = {
            'status': 'healthy',
            'timestamp': self.last_check_time.isoformat(),
            'checks': {}
        }
        
        for check in self.health_checks:
            try:
                check_result = check['check']()
                results['checks'][check['name']] = {
                    'status': 'pass' if check_result else 'fail',
                    'critical': check['critical']
                }
                
                if not check_result and check['critical']:
                    results['status'] = 'critical'
                elif not check_result and results['status'] != 'critical':
                    results['status'] = 'degraded'
                    
            except Exception as e:
                results['checks'][check['name']] = {
                    'status': 'error',
                    'error': str(e),
                    'critical': check['critical']
                }
                if check['critical']:
                    results['status'] = 'critical'
        
        # Update metrics
        self.metrics.update_memory_usage()
        
        # Store in history
        self.health_history.append(results)
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        return results
    
    def get_health_trend(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get health trend over specified duration."""
        cutoff_time = datetime.now().timestamp() - (duration_minutes * 60)
        
        recent_checks = [
            h for h in self.health_history
            if datetime.fromisoformat(h['timestamp']).timestamp() > cutoff_time
        ]
        
        if not recent_checks:
            return {'status': 'no_data'}
        
        # Calculate health statistics
        status_counts = defaultdict(int)
        for check in recent_checks:
            status_counts[check['status']] += 1
        
        total = len(recent_checks)
        return {
            'duration_minutes': duration_minutes,
            'total_checks': total,
            'health_percentage': (status_counts['healthy'] / total) * 100,
            'status_breakdown': dict(status_counts),
            'latest_status': recent_checks[-1]['status'] if recent_checks else 'unknown'
        }

# ============== Global Metrics Instance ==============

# Create global metrics collector instance
metrics_collector = MetricsCollector()

# Create system health monitor
health_monitor = SystemHealthMonitor(metrics_collector)

# Add default health checks
health_monitor.add_health_check(
    'memory_usage',
    lambda: metrics_collector.memory_usage.labels(type='rss')._value._value < 2 * 1024 * 1024 * 1024,  # 2GB
    critical=True
)

health_monitor.add_health_check(
    'active_questions',
    lambda: len(metrics_collector._active_questions) < 100,
    critical=False
)

# ============== Decorators for Easy Metric Collection ==============

def track_processing_time(agent_name: str):
    """Decorator to track agent processing time."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_agent_processing(agent_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_agent_processing(agent_name, duration)
                raise
        return wrapper
    return decorator

def track_token_usage(agent_name: str):
    """Decorator to track LLM token usage."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would need to be integrated with actual token counting
            result = await func(*args, **kwargs)
            # Token tracking would happen here based on the result
            return result
        return wrapper
    return decorator

# ============== Prometheus Registry Configuration ==============

# This is handled automatically by prometheus_client
# The metrics are automatically registered and will be exposed
# when the /metrics endpoint is accessed