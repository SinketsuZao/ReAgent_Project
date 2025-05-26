"""
ReAgent: Reversible Multi-Agent Reasoning System

A production-ready implementation of reversible multi-agent reasoning for
knowledge-enhanced multi-hop question answering with explicit backtracking mechanisms.
"""

# Version information
__version__ = "1.0.0"
__author__ = "ReAgent Team"
__email__ = "team@reagent.ai"
__license__ = "MIT"

# Import main components for easier access
from .main import (
    ReAgentSystem,
    QuestionDecomposerAgent,
    RetrieverAgent,
    VerifierAgent,
    AnswerAssemblerAgent,
    SupervisorAgent,
    ControllerAgent,
    BaseAgent,
    LocalKnowledge,
    MessageBus,
    PersistentLog,
    TemporalTracker,
)

from .models import (
    Message,
    MessageType,
    KnowledgeAssertion,
    BacktrackingNode,
)

from .monitoring import metrics_collector

from .config import settings, prompts, load_prompts

# Define what should be imported with "from reagent import *"
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Main system
    "ReAgentSystem",
    
    # Agents
    "QuestionDecomposerAgent",
    "RetrieverAgent",
    "VerifierAgent",
    "AnswerAssemblerAgent",
    "SupervisorAgent",
    "ControllerAgent",
    "BaseAgent",
    
    # Core components
    "LocalKnowledge",
    "MessageBus",
    "PersistentLog",
    "TemporalTracker",
    
    # Models
    "Message",
    "MessageType",
    "KnowledgeAssertion",
    "BacktrackingNode",
    
    # Monitoring
    "metrics_collector",
    
    # Configuration
    "settings",
    "prompts",
    "load_prompts",
]

# Package metadata
PACKAGE_NAME = "reagent"
PACKAGE_DESCRIPTION = "Reversible Multi-Agent Reasoning System"

# Logging configuration
import logging

# Create package logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Set default logging format
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_version():
    """Return the current version of ReAgent."""
    return __version__

def get_agent_names():
    """Return list of available agent names."""
    return [
        "QuestionDecomposerAgent",
        "RetrieverAgent",
        "VerifierAgent",
        "AnswerAssemblerAgent",
        "SupervisorAgent",
        "ControllerAgent",
    ]

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = {
        'openai': 'openai',
        'redis': 'redis',
        'fastapi': 'fastapi',
        'celery': 'celery',
        'sqlalchemy': 'sqlalchemy',
        'prometheus_client': 'prometheus-client',
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        logger.warning(
            f"Missing required packages: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )
        return False
    return True

# Check dependencies on import
check_dependencies()

# Initialize the system on import if in development mode
import os
if os.getenv('ENVIRONMENT') == 'development':
    logger.info(f"ReAgent {__version__} initialized in development mode")