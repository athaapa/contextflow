"""
setup.py - Defines how to install contextflow as a package
"""

from setuptools import setup, find_packages

setup(
    name="contextflow",  # Package name
    version="0.1.0",  # Your version
    author="Your Name",  # Optional
    author_email="you@example.com",  # Optional
    description="Intelligent context optimization for LLM applications",
    # This is the magic: find_packages() discovers src/ automatically
    packages=find_packages(),
    # Python version requirement
    python_requires=">=3.9",
    # Dependencies (same as requirements.txt)
    install_requires=[
        # Web Framework
        "fastapi==0.115.0",
        "uvicorn[standard]==0.32.0",
        "pydantic==2.9.2",
        "pydantic-settings==2.6.0",
        # Database - USE SQLITE FOR MVP (no psycopg2 needed!)
        "sqlalchemy==2.0.36",
        "alembic==1.14.0",
        "aiosqlite==0.20.0",  # Async SQLite driver
        # ML/AI
        "sentence-transformers==3.2.1",
        "scikit-learn==1.5.2",
        # LLM APIs
        "together==1.3.3",
        # Token Counting
        "tiktoken==0.8.0",
        # CLI
        "click==8.1.7",
        "rich==13.9.4",
        # Utilities
        "python-dotenv==1.0.1",
        "httpx==0.28.1",
    ],
    # Development/testing dependencies
    extras_require={"dev": ["pytest==8.3.3", "pytest-asyncio==0.24.0"]},
)
