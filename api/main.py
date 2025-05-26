"""
ReAgent FastAPI Application

Main entry point for the ReAgent REST API service.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
from prometheus_client import make_asgi_app
import logging
import time
from typing import Dict, Any
import asyncio

from reagent.main import ReAgentSystem
from reagent.config import settings
from reagent.monitoring import metrics_collector
from api import API_TITLE, API_DESCRIPTION, API_VERSION, API_CONTACT, API_LICENSE
from api.routes import questions, tasks, health
from api.routes.websocket import websocket_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for background tasks
background_tasks = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    Handles startup and shutdown operations.
    """
    # Startup
    logger.info("Starting ReAgent API service...")
    
    # Initialize ReAgent system
    try:
        app.state.reagent_system = ReAgentSystem()
        await app.state.reagent_system.start()
        logger.info("ReAgent system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ReAgent system: {e}")
        raise
    
    # Initialize task storage
    app.state.tasks = {}
    app.state.active_connections = set()
    
    # Record startup metrics
    metrics_collector.system_info.info({
        'version': API_VERSION,
        'environment': settings.environment,
        'debug': str(settings.debug)
    })
    
    logger.info(f"ReAgent API started: environment={settings.environment}, debug={settings.debug}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ReAgent API service...")
    
    # Cancel background tasks
    for task in background_tasks:
        task.cancel()
    
    # Wait for background tasks to complete
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)
    
    # Stop ReAgent system
    if hasattr(app.state, 'reagent_system'):
        await app.state.reagent_system.stop()
    
    logger.info("ReAgent API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    license_info=API_LICENSE,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ============== Middleware ==============

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

# Trusted host middleware
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.reagent.ai", "localhost"]
    )

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    
    # Record request metrics
    metrics_collector.record_message_latency(
        request.method + " " + request.url.path,
        process_time
    )
    
    return response

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response

# ============== Exception Handlers ==============

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": exc.errors(),
            "body": exc.body,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# ============== Routes ==============

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(questions.router, prefix="/api/v1", tags=["questions"])
app.include_router(tasks.router, prefix="/api/v1", tags=["tasks"])
app.include_router(websocket_router, prefix="/ws", tags=["websocket"])

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ReAgent API",
        "version": API_VERSION,
        "status": "operational",
        "environment": settings.environment,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/health",
            "metrics": "/metrics",
            "questions": "/api/v1/questions",
            "tasks": "/api/v1/tasks",
            "websocket": "/ws"
        },
        "links": {
            "documentation": "https://docs.reagent.ai",
            "github": "https://github.com/your-org/reagent",
            "support": "support@reagent.ai"
        }
    }

# API info endpoint
@app.get("/api/v1/info", tags=["info"])
async def api_info():
    """Get detailed API information."""
    return {
        "api": {
            "title": API_TITLE,
            "version": API_VERSION,
            "description": API_DESCRIPTION.split('\n')[1].strip()  # First line
        },
        "system": {
            "status": app.state.reagent_system.get_system_status() if hasattr(app.state, 'reagent_system') else {},
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        },
        "configuration": {
            "environment": settings.environment,
            "debug": settings.debug,
            "max_concurrent_questions": settings.performance.max_concurrent_agents,
            "backtracking_enabled": settings.features.enable_global_backtracking
        },
        "metrics": metrics_collector.get_metrics_summary()
    }

# ============== Utility Functions ==============

def run():
    """Run the API server (for development)."""
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host if hasattr(settings, 'api_host') else "0.0.0.0",
        port=settings.api_port if hasattr(settings, 'api_port') else 8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )

# ============== Background Task Management ==============

def create_background_task(coro):
    """Create and track a background task."""
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task

# ============== Startup Tasks ==============

@app.on_event("startup")
async def startup_event():
    """Additional startup tasks."""
    app.state.start_time = time.time()
    
    # Log configuration summary
    logger.info(f"Configuration loaded: {settings.export_config(include_secrets=False)}")
    
    # Warm up the system
    if settings.environment == "production":
        logger.info("Warming up ReAgent system...")
        # Could add warmup logic here

@app.on_event("shutdown")
async def shutdown_event():
    """Additional shutdown tasks."""
    logger.info("Performing cleanup tasks...")
    
    # Save any pending data
    # Clean up temporary files
    # etc.

if __name__ == "__main__":
    run()