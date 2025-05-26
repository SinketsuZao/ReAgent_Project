"""
ReAgent Configuration Management

This module handles all configuration aspects of the ReAgent system,
including environment variables, YAML configs, and runtime settings.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseSettings, Field, validator
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# ============== Settings Classes ==============

class OpenAISettings(BaseSettings):
    """OpenAI API configuration."""
    openai_api_key: str = Field(..., env='OPENAI_API_KEY')
    llm_model: str = Field('gpt-4o', env='LLM_MODEL')
    llm_temp_decomposer: float = Field(0.8, env='LLM_TEMP_DECOMPOSER')
    llm_temp_default: float = Field(0.6, env='LLM_TEMP_DEFAULT')
    max_tokens_per_call: int = Field(4000, env='MAX_TOKENS_PER_CALL')
    llm_request_timeout: int = Field(60, env='LLM_REQUEST_TIMEOUT')
    
    @validator('openai_api_key')
    def validate_api_key(cls, v):
        if not v or v == 'sk-your-actual-openai-api-key-here':
            raise ValueError('Valid OpenAI API key required')
        return v
    
    class Config:
        env_prefix = ''

class DatabaseSettings(BaseSettings):
    """Database configuration."""
    postgres_host: str = Field('postgres', env='POSTGRES_HOST')
    postgres_port: int = Field(5432, env='POSTGRES_PORT')
    postgres_db: str = Field('reagent_db', env='POSTGRES_DB')
    postgres_user: str = Field('reagent_user', env='POSTGRES_USER')
    postgres_password: str = Field(..., env='POSTGRES_PASSWORD')
    postgres_pool_size: int = Field(20, env='POSTGRES_POOL_SIZE')
    postgres_max_overflow: int = Field(40, env='POSTGRES_MAX_OVERFLOW')
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    class Config:
        env_prefix = ''

class RedisSettings(BaseSettings):
    """Redis configuration."""
    redis_host: str = Field('redis', env='REDIS_HOST')
    redis_port: int = Field(6379, env='REDIS_PORT')
    redis_db: int = Field(0, env='REDIS_DB')
    redis_password: Optional[str] = Field(None, env='REDIS_PASSWORD')
    redis_max_connections: int = Field(50, env='REDIS_MAX_CONNECTIONS')
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    class Config:
        env_prefix = ''

class ElasticsearchSettings(BaseSettings):
    """Elasticsearch configuration."""
    elasticsearch_host: str = Field('elasticsearch', env='ELASTICSEARCH_HOST')
    elasticsearch_port: int = Field(9200, env='ELASTICSEARCH_PORT')
    elasticsearch_username: Optional[str] = Field(None, env='ELASTICSEARCH_USERNAME')
    elasticsearch_password: Optional[str] = Field(None, env='ELASTICSEARCH_PASSWORD')
    elasticsearch_index_prefix: str = Field('reagent', env='ELASTICSEARCH_INDEX_PREFIX')
    
    @property
    def elasticsearch_url(self) -> str:
        """Get Elasticsearch URL."""
        if self.elasticsearch_username and self.elasticsearch_password:
            return f"http://{self.elasticsearch_username}:{self.elasticsearch_password}@{self.elasticsearch_host}:{self.elasticsearch_port}"
        return f"http://{self.elasticsearch_host}:{self.elasticsearch_port}"
    
    class Config:
        env_prefix = ''

class BacktrackingSettings(BaseSettings):
    """Backtracking configuration."""
    max_backtrack_depth: int = Field(5, env='MAX_BACKTRACK_DEPTH')
    local_backtrack_threshold: int = Field(3, env='LOCAL_BACKTRACK_THRESHOLD')
    global_backtrack_timeout: int = Field(300, env='GLOBAL_BACKTRACK_TIMEOUT')
    checkpoint_retention_hours: int = Field(24, env='CHECKPOINT_RETENTION_HOURS')
    
    class Config:
        env_prefix = ''

class PerformanceSettings(BaseSettings):
    """Performance tuning configuration."""
    max_concurrent_agents: int = Field(10, env='MAX_CONCURRENT_AGENTS')
    message_queue_size: int = Field(1000, env='MESSAGE_QUEUE_SIZE')
    token_budget_per_question: int = Field(20000, env='TOKEN_BUDGET_PER_QUESTION')
    question_timeout: int = Field(300, env='QUESTION_TIMEOUT')
    api_workers: int = Field(4, env='API_WORKERS')
    celery_worker_concurrency: int = Field(4, env='CELERY_WORKER_CONCURRENCY')
    
    class Config:
        env_prefix = ''

class LoggingSettings(BaseSettings):
    """Logging configuration."""
    log_level: str = Field('INFO', env='LOG_LEVEL')
    log_format: str = Field('json', env='LOG_FORMAT')
    log_file_path: str = Field('/app/logs/reagent.log', env='LOG_FILE_PATH')
    log_file_max_size: str = Field('100MB', env='LOG_FILE_MAX_SIZE')
    log_file_backup_count: int = Field(5, env='LOG_FILE_BACKUP_COUNT')
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Invalid log level: {v}')
        return v.upper()
    
    class Config:
        env_prefix = ''

class SecuritySettings(BaseSettings):
    """Security configuration."""
    jwt_secret_key: str = Field('change_this_secret', env='JWT_SECRET_KEY')
    jwt_algorithm: str = Field('HS256', env='JWT_ALGORITHM')
    jwt_expiration_hours: int = Field(24, env='JWT_EXPIRATION_HOURS')
    internal_api_key: str = Field('internal_key', env='INTERNAL_API_KEY')
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8000"],
        env='CORS_ORIGINS'
    )
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v
    
    class Config:
        env_prefix = ''

class FeatureFlags(BaseSettings):
    """Feature flags configuration."""
    enable_caching: bool = Field(True, env='ENABLE_CACHING')
    enable_metrics: bool = Field(True, env='ENABLE_METRICS')
    enable_tracing: bool = Field(True, env='ENABLE_TRACING')
    enable_global_backtracking: bool = Field(True, env='ENABLE_GLOBAL_BACKTRACKING')
    enable_conflict_detection: bool = Field(True, env='ENABLE_CONFLICT_DETECTION')
    
    class Config:
        env_prefix = ''

class Settings(BaseSettings):
    """Main settings class that combines all settings."""
    
    # Sub-settings
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    backtracking: BacktrackingSettings = Field(default_factory=BacktrackingSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    
    # Environment
    environment: str = Field('production', env='ENVIRONMENT')
    debug: bool = Field(False, env='DEBUG')
    
    # Direct access properties for backward compatibility
    @property
    def openai_api_key(self) -> str:
        return self.openai.openai_api_key
    
    @property
    def llm_model(self) -> str:
        return self.openai.llm_model
    
    @property
    def postgres_host(self) -> str:
        return self.database.postgres_host
    
    @property
    def postgres_port(self) -> int:
        return self.database.postgres_port
    
    @property
    def postgres_db(self) -> str:
        return self.database.postgres_db
    
    @property
    def postgres_user(self) -> str:
        return self.database.postgres_user
    
    @property
    def postgres_password(self) -> str:
        return self.database.postgres_password
    
    @property
    def redis_host(self) -> str:
        return self.redis.redis_host
    
    @property
    def redis_port(self) -> int:
        return self.redis.redis_port
    
    @property
    def redis_db(self) -> int:
        return self.redis.redis_db
    
    @property
    def redis_password(self) -> Optional[str]:
        return self.redis.redis_password
    
    @property
    def redis_max_connections(self) -> int:
        return self.redis.redis_max_connections
    
    @property
    def elasticsearch_host(self) -> str:
        return self.elasticsearch.elasticsearch_host
    
    @property
    def elasticsearch_port(self) -> int:
        return self.elasticsearch.elasticsearch_port
    
    @property
    def max_backtrack_depth(self) -> int:
        return self.backtracking.max_backtrack_depth
    
    @property
    def local_backtrack_threshold(self) -> int:
        return self.backtracking.local_backtrack_threshold
    
    @property
    def checkpoint_retention_hours(self) -> int:
        return self.backtracking.checkpoint_retention_hours
    
    @property
    def max_concurrent_agents(self) -> int:
        return self.performance.max_concurrent_agents
    
    @property
    def message_queue_size(self) -> int:
        return self.performance.message_queue_size
    
    @property
    def max_tokens_per_call(self) -> int:
        return self.openai.max_tokens_per_call
    
    @property
    def llm_request_timeout(self) -> int:
        return self.openai.llm_request_timeout
    
    @property
    def log_level(self) -> str:
        return self.logging.log_level
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False

# ============== Prompt Configuration ==============

def load_prompts(path: str = "configs/prompts.yaml") -> Dict[str, Any]:
    """
    Load agent prompts from YAML file.
    
    Args:
        path: Path to the prompts YAML file
        
    Returns:
        Dictionary containing agent prompts
    """
    prompts_path = Path(path)
    
    # Try multiple locations
    if not prompts_path.exists():
        # Try relative to current file
        prompts_path = Path(__file__).parent.parent / "configs" / "prompts.yaml"
    
    if not prompts_path.exists():
        # Try in /app/configs (Docker environment)
        prompts_path = Path("/app/configs/prompts.yaml")
    
    if prompts_path.exists():
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
                logger.info(f"Loaded prompts from {prompts_path}")
                return prompts
        except Exception as e:
            logger.error(f"Failed to load prompts from {prompts_path}: {e}")
    else:
        logger.warning(f"Prompts file not found at {path}, using defaults")
    
    # Return default prompts if file not found
    return get_default_prompts()

def get_default_prompts() -> Dict[str, Any]:
    """Get default agent prompts."""
    return {
        'question_decomposer': {
            'system_prompt': 'You are the Question-Decomposer Agent...',
            'temperature': 0.8,
            'max_tokens': 1000
        },
        'retriever': {
            'system_prompt': 'You are the Retriever Agent...',
            'temperature': 0.6,
            'max_tokens': 2000
        },
        'verifier': {
            'system_prompt': 'You are the Verifier Agent...',
            'temperature': 0.6,
            'max_tokens': 1500
        },
        'answer_assembler': {
            'system_prompt': 'You are the Answer-Assembler Agent...',
            'temperature': 0.6,
            'max_tokens': 2000
        },
        'supervisor': {
            'system_prompt': 'You are the Supervisor Agent...',
            'temperature': 0.6,
            'max_tokens': 1500
        },
        'controller': {
            'system_prompt': 'You are the Controller Agent...',
            'temperature': 0.6,
            'max_tokens': 1000
        }
    }

# ============== Configuration Validation ==============

def validate_config(settings: Settings) -> List[str]:
    """
    Validate configuration and return list of issues.
    
    Args:
        settings: Settings instance to validate
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    # Check OpenAI API key
    if not settings.openai_api_key or settings.openai_api_key.startswith('sk-your'):
        issues.append("Valid OpenAI API key required")
    
    # Check database connection
    if settings.environment == 'production':
        if settings.database.postgres_password == 'changeme':
            issues.append("Default PostgreSQL password should not be used in production")
        
        if settings.security.jwt_secret_key == 'change_this_secret':
            issues.append("Default JWT secret should not be used in production")
    
    # Check Redis connection
    if settings.redis.redis_max_connections < 10:
        issues.append("Redis max connections should be at least 10")
    
    # Check performance settings
    if settings.performance.max_concurrent_agents < 1:
        issues.append("Max concurrent agents must be at least 1")
    
    if settings.performance.message_queue_size < 100:
        issues.append("Message queue size should be at least 100")
    
    # Check backtracking settings
    if settings.backtracking.max_backtrack_depth < 1:
        issues.append("Max backtrack depth must be at least 1")
    
    return issues

# ============== Singleton Instances ==============

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

@lru_cache()
def get_prompts() -> Dict[str, Any]:
    """Get cached prompts."""
    return load_prompts()

# Create global instances
settings = get_settings()
prompts = get_prompts()

# Validate configuration on import
validation_issues = validate_config(settings)
if validation_issues:
    for issue in validation_issues:
        logger.warning(f"Configuration issue: {issue}")

# ============== Configuration Utilities ==============

def update_settings(**kwargs):
    """
    Update settings at runtime.
    
    Note: This should be used sparingly and primarily for testing.
    """
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            logger.warning(f"Unknown setting: {key}")

def reload_prompts(path: Optional[str] = None):
    """Reload prompts from file."""
    global prompts
    if path:
        prompts = load_prompts(path)
    else:
        prompts = load_prompts()
    get_prompts.cache_clear()

def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """Get configuration for a specific agent."""
    agent_prompt = prompts.get(agent_name, {})
    
    return {
        'system_prompt': agent_prompt.get('system_prompt', ''),
        'temperature': agent_prompt.get('temperature', settings.openai.llm_temp_default),
        'max_tokens': agent_prompt.get('max_tokens', settings.openai.max_tokens_per_call),
        'model': settings.openai.llm_model,
        'timeout': settings.openai.llm_request_timeout
    }

def export_config(include_secrets: bool = False) -> Dict[str, Any]:
    """
    Export current configuration as dictionary.
    
    Args:
        include_secrets: Whether to include sensitive values
        
    Returns:
        Configuration dictionary
    """
    config = settings.dict()
    
    if not include_secrets:
        # Mask sensitive values
        config['openai']['openai_api_key'] = '***'
        config['database']['postgres_password'] = '***'
        config['redis']['redis_password'] = '***' if config['redis']['redis_password'] else None
        config['security']['jwt_secret_key'] = '***'
        config['security']['internal_api_key'] = '***'
    
    return config

# Log configuration summary on import
logger.info(f"ReAgent configuration loaded: environment={settings.environment}, debug={settings.debug}")