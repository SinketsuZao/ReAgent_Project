# Multi-stage Dockerfile for ReAgent system

# Stage 1: Base image with common dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    postgresql-client \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Create app user and directories
RUN groupadd -r reagent && useradd -r -g reagent reagent \
    && mkdir -p /app/logs /app/data \
    && chown -R reagent:reagent /app

WORKDIR /app

# Stage 2: Python dependencies
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Stage 3: Application code
FROM dependencies as app

# Copy application code
COPY --chown=reagent:reagent . /app

# Set Python path
ENV PYTHONPATH=/app

# Create directories for configs and logs
RUN mkdir -p /app/configs /app/logs /app/data \
    && chown -R reagent:reagent /app

# Switch to non-root user
USER reagent

# Stage 4: API service
FROM app as api

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 5: Worker service
FROM app as worker

# No exposed ports for worker

# Default command for worker
CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info"]

# Stage 6: Development image with additional tools
FROM app as development

# Switch back to root for installing dev tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    less \
    htop \
    iputils-ping \
    telnet \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dev tools
RUN pip install \
    ipython \
    ipdb \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit

# Switch back to reagent user
USER reagent

# Use bash as default shell in development
SHELL ["/bin/bash", "-c"]

# Default command for development
CMD ["/bin/bash"]