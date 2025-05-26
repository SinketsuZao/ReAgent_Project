"""
ReAgent API Package

This package contains the FastAPI-based REST API for the ReAgent system.
"""

from .main import app, lifespan
from .routes import questions, tasks, health
from .schemas import requests, responses

__version__ = "1.0.0"

__all__ = [
    "app",
    "lifespan",
    "questions",
    "tasks", 
    "health",
    "requests",
    "responses",
]

# API metadata
API_TITLE = "ReAgent Multi-Agent QA System"
API_DESCRIPTION = """
Reversible multi-agent reasoning for knowledge-enhanced multi-hop question answering.

## Features
- Multi-hop question processing
- Real-time status tracking
- WebSocket support for live updates
- Comprehensive metrics and monitoring

## Endpoints
- **Questions**: Submit and process multi-hop questions
- **Tasks**: Track processing status
- **Health**: System health checks
- **Metrics**: Prometheus metrics
"""
API_VERSION = __version__
API_CONTACT = {
    "name": "ReAgent Team",
    "url": "https://github.com/your-org/reagent",
    "email": "support@reagent.ai",
}
API_LICENSE = {
    "name": "MIT",
    "url": "https://opensource.org/licenses/MIT",
}