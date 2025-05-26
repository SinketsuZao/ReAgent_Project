"""
Health Check Routes

Provides system health status and diagnostics.
"""

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import logging
import os
import psutil
import redis
import asyncpg
from elasticsearch import AsyncElasticsearch

from reagent.config import settings
from reagent.monitoring import health_monitor
from api.schemas.responses import HealthResponse, HealthDetailResponse, ComponentHealth

logger = logging.getLogger(__name__)

router = APIRouter()

# ============== Dependencies ==============

async def get_app_state(request: Request):
    """Get application state."""
    return request.app.state

# ============== Health Check Functions ==============

async def check_database_health() -> ComponentHealth:
    """Check PostgreSQL database health."""
    try:
        conn = await asyncpg.connect(
            host=settings.database.postgres_host,
            port=settings.database.postgres_port,
            user=settings.database.postgres_user,
            password=settings.database.postgres_password,
            database=settings.database.postgres_db,
            timeout=5
        )
        
        # Run a simple query
        result = await conn.fetchval('SELECT 1')
        await conn.close()
        
        return ComponentHealth(
            name="postgresql",
            status="healthy",
            response_time=0.1,  # Would measure actual time
            details={"connection": "ok", "query_result": result}
        )
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return ComponentHealth(
            name="postgresql",
            status="unhealthy",
            response_time=5.0,
            error=str(e),
            details={"connection": "failed"}
        )

async def check_redis_health() -> ComponentHealth:
    """Check Redis health."""
    try:
        client = redis.Redis(
            host=settings.redis.redis_host,
            port=settings.redis.redis_port,
            db=settings.redis.redis_db,
            password=settings.redis.redis_password,
            socket_connect_timeout=5
        )
        
        # Ping Redis
        start_time = datetime.now()
        ping_result = client.ping()
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Get some stats
        info = client.info()
        client.close()
        
        return ComponentHealth(
            name="redis",
            status="healthy",
            response_time=response_time,
            details={
                "ping": ping_result,
                "version": info.get('redis_version', 'unknown'),
                "connected_clients": info.get('connected_clients', 0),
                "used_memory_human": info.get('used_memory_human', 'unknown')
            }
        )
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return ComponentHealth(
            name="redis",
            status="unhealthy",
            response_time=5.0,
            error=str(e),
            details={"connection": "failed"}
        )

async def check_elasticsearch_health() -> ComponentHealth:
    """Check Elasticsearch health."""
    try:
        es = AsyncElasticsearch(
            [f"http://{settings.elasticsearch.elasticsearch_host}:{settings.elasticsearch.elasticsearch_port}"],
            basic_auth=(settings.elasticsearch.elasticsearch_username, settings.elasticsearch.elasticsearch_password)
            if settings.elasticsearch.elasticsearch_username else None,
            verify_certs=False,
            request_timeout=5
        )
        
        # Check cluster health
        start_time = datetime.now()
        health = await es.cluster.health()
        response_time = (datetime.now() - start_time).total_seconds()
        
        await es.close()
        
        return ComponentHealth(
            name="elasticsearch",
            status="healthy" if health['status'] in ['green', 'yellow'] else "unhealthy",
            response_time=response_time,
            details={
                "cluster_name": health.get('cluster_name'),
                "status": health.get('status'),
                "number_of_nodes": health.get('number_of_nodes'),
                "active_shards": health.get('active_shards')
            }
        )
    except Exception as e:
        logger.error(f"Elasticsearch health check failed: {e}")
        return ComponentHealth(
            name="elasticsearch",
            status="unhealthy",
            response_time=5.0,
            error=str(e),
            details={"connection": "failed"}
        )

async def check_reagent_system_health(app_state) -> ComponentHealth:
    """Check ReAgent system health."""
    try:
        if not hasattr(app_state, 'reagent_system'):
            return ComponentHealth(
                name="reagent_system",
                status="unhealthy",
                error="ReAgent system not initialized"
            )
        
        system_status = app_state.reagent_system.get_system_status()
        
        # Determine health based on system status
        if not system_status.get('is_running'):
            status = "unhealthy"
        elif system_status.get('constraint_violations', 0) > 0:
            status = "degraded"
        else:
            status = "healthy"
        
        return ComponentHealth(
            name="reagent_system",
            status=status,
            details={
                "is_running": system_status.get('is_running'),
                "active_agents": len(system_status.get('agents', {})),
                "constraint_violations": system_status.get('constraint_violations'),
                "recent_events": system_status.get('recent_events')
            }
        )
    except Exception as e:
        logger.error(f"ReAgent system health check failed: {e}")
        return ComponentHealth(
            name="reagent_system",
            status="unhealthy",
            error=str(e)
        )

def check_system_resources() -> ComponentHealth:
    """Check system resource usage."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Determine health based on resource usage
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = "degraded"
        else:
            status = "healthy"
        
        return ComponentHealth(
            name="system_resources",
            status=status,
            details={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
        )
    except Exception as e:
        logger.error(f"System resource check failed: {e}")
        return ComponentHealth(
            name="system_resources",
            status="unknown",
            error=str(e)
        )

# ============== Routes ==============

@router.get("/health", response_model=HealthResponse)
async def health_check(app_state = Depends(get_app_state)):
    """
    Basic health check endpoint.
    
    Returns simple health status for load balancer checks.
    """
    try:
        # Quick check - just verify the app is running
        is_healthy = hasattr(app_state, 'reagent_system') and app_state.reagent_system.is_running
        
        return HealthResponse(
            status="healthy" if is_healthy else "unhealthy",
            timestamp=datetime.now(),
            version=os.getenv('APP_VERSION', '1.0.0'),
            uptime=time.time() - app_state.start_time if hasattr(app_state, 'start_time') else 0
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@router.get("/health/detail", response_model=HealthDetailResponse)
async def detailed_health_check(
    app_state = Depends(get_app_state),
    check_external: bool = True
):
    """
    Detailed health check endpoint.
    
    Performs comprehensive health checks on all components.
    """
    start_time = datetime.now()
    
    # Collect component health
    components = []
    
    # Always check internal components
    components.append(check_system_resources())
    components.append(await check_reagent_system_health(app_state))
    
    # Check external components if requested
    if check_external:
        # Run checks in parallel
        external_checks = await asyncio.gather(
            check_database_health(),
            check_redis_health(),
            check_elasticsearch_health(),
            return_exceptions=True
        )
        
        for check in external_checks:
            if isinstance(check, Exception):
                components.append(ComponentHealth(
                    name="unknown",
                    status="error",
                    error=str(check)
                ))
            else:
                components.append(check)
    
    # Determine overall status
    statuses = [c.status for c in components]
    if any(s == "unhealthy" for s in statuses):
        overall_status = "unhealthy"
    elif any(s == "degraded" for s in statuses):
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    # Get system metrics from health monitor
    health_trend = health_monitor.get_health_trend(duration_minutes=60)
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    return HealthDetailResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version=os.getenv('APP_VERSION', '1.0.0'),
        uptime=time.time() - app_state.start_time if hasattr(app_state, 'start_time') else 0,
        components=components,
        total_check_time=total_time,
        metrics={
            "health_trend": health_trend,
            "active_tasks": len(app_state.tasks) if hasattr(app_state, 'tasks') else 0,
            "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024)
        }
    )

@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 if the service is alive, 503 otherwise.
    """
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness_probe(app_state = Depends(get_app_state)):
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 if the service is ready to accept traffic, 503 otherwise.
    """
    try:
        # Check if critical components are ready
        if not hasattr(app_state, 'reagent_system'):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "ReAgent system not initialized"}
            )
        
        if not app_state.reagent_system.is_running:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "ReAgent system not running"}
            )
        
        # Quick Redis check
        try:
            client = redis.Redis(
                host=settings.redis.redis_host,
                port=settings.redis.redis_port,
                socket_connect_timeout=2
            )
            client.ping()
            client.close()
        except:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "Redis not available"}
            )
        
        return {"status": "ready"}
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "error": str(e)}
        )

@router.post("/health/check")
async def run_health_check():
    """
    Manually trigger a comprehensive health check.
    
    Useful for debugging and monitoring.
    """
    # Run health check through health monitor
    result = health_monitor.check_health()
    
    return {
        "status": result['status'],
        "timestamp": result['timestamp'],
        "checks": result['checks'],
        "issues": len(result['checks']) - sum(1 for c in result['checks'].values() if c.get('status') == 'pass')
    }

@router.get("/health/history")
async def get_health_history(
    duration_minutes: int = 60,
    limit: int = 100
):
    """
    Get health check history.
    
    Returns recent health check results for trend analysis.
    """
    # Get health trend from monitor
    trend = health_monitor.get_health_trend(duration_minutes=duration_minutes)
    
    # Get recent health checks
    recent_checks = health_monitor.health_history[-limit:]
    
    return {
        "trend": trend,
        "history": recent_checks,
        "duration_minutes": duration_minutes
    }

# ============== Utility Functions ==============

import time  # Add this import at the top

async def measure_response_time(func, *args, **kwargs):
    """Measure response time of an async function."""
    start = time.time()
    try:
        result = await func(*args, **kwargs)
        response_time = time.time() - start
        return result, response_time
    except Exception as e:
        response_time = time.time() - start
        raise e