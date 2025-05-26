#!/usr/bin/env python
"""
ReAgent: Reversible Multi-Agent Reasoning System
Setup configuration for pip installation
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from __init__.py
version_file = this_directory / "reagent" / "__init__.py"
version = "1.0.0"
if version_file.exists():
    with open(version_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

# Core dependencies (minimal set for the library)
install_requires = [
    "fastapi>=0.104.1",
    "pydantic>=2.5.0",
    "openai>=1.3.0",
    "redis>=5.0.1",
    "asyncpg>=0.29.0",
    "sqlalchemy>=2.0.23",
    "celery>=5.3.4",
    "numpy>=1.26.2",
    "pyyaml>=6.0.1",
    "python-dotenv>=1.0.0",
    "prometheus-client>=0.19.0",
    "elasticsearch>=8.11.0",
    "structlog>=23.2.0",
]

# Optional dependencies for different use cases
extras_require = {
    "dev": [
        "pytest>=7.4.3",
        "pytest-asyncio>=0.21.1",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.12.0",
        "black>=23.12.1",
        "flake8>=6.1.0",
        "mypy>=1.7.1",
        "pre-commit>=3.6.0",
        "ipython>=8.19.0",
        "ipdb>=0.13.13",
    ],
    "ml": [
        "transformers>=4.36.2",
        "torch>=2.1.2",
        "sentence-transformers>=2.2.2",
        "spacy>=3.7.2",
        "langchain>=0.0.350",
    ],
    "docs": [
        "mkdocs>=1.5.3",
        "mkdocs-material>=9.5.3",
        "mkdocstrings[python]>=0.24.0",
    ],
    "monitoring": [
        "opentelemetry-api>=1.21.0",
        "opentelemetry-sdk>=1.21.0",
        "opentelemetry-instrumentation-fastapi>=0.42b0",
        "flower>=2.0.1",
    ],
    "vector": [
        "chromadb>=0.4.22",
        "qdrant-client>=1.7.0",
        "pinecone-client>=2.2.4",
        "weaviate-client>=4.4.1",
        "faiss-cpu>=1.7.4",
    ],
    "graph": [
        "neo4j>=5.15.0",
        "py2neo>=2021.2.4",
    ],
}

# All extras combined
extras_require["all"] = list(set(sum(extras_require.values(), [])))

# Python version requirement
python_requires = ">=3.11"

# Package metadata
setup(
    name="reagent",
    version=version,
    author="ReAgent Team",
    author_email="team@reagent.ai",
    description="Reversible Multi-Agent Reasoning System for Knowledge-Enhanced Multi-Hop QA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/reagent",
    project_urls={
        "Documentation": "https://docs.reagent.ai",
        "Source": "https://github.com/your-org/reagent",
        "Issues": "https://github.com/your-org/reagent/issues",
        "Changelog": "https://github.com/your-org/reagent/blob/main/CHANGELOG.md",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "multi-agent",
        "reasoning",
        "question-answering",
        "nlp",
        "ai",
        "backtracking",
        "knowledge-graph",
        "llm",
        "gpt",
        "openai",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "docs", "scripts"]),
    package_data={
        "reagent": [
            "configs/*.yaml",
            "configs/*.yml",
            "prompts/*.yaml",
            "prompts/*.yml",
        ],
    },
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=python_requires,
    entry_points={
        "console_scripts": [
            "reagent=reagent.cli:main",
            "reagent-api=api.main:run",
            "reagent-worker=worker.celery_app:main",
        ],
    },
    zip_safe=False,
    platforms="any",
)

# Development installation message
if "develop" in sys.argv or "install" in sys.argv and "--editable" in sys.argv:
    print("\n" + "="*60)
    print("ReAgent installed in development mode!")
    print("="*60)
    print("\nNext steps:")
    print("1. Copy .env.example to .env and configure")
    print("2. Run 'docker-compose up' to start services")
    print("3. Access API at http://localhost:8000")
    print("\nFor development tools, install with:")
    print("  pip install -e '.[dev]'")
    print("\nHappy coding! 🚀")
    print("="*60 + "\n")