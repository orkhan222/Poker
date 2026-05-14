#!/usr/bin/env python3
"""
Setup script for Poker Agent package
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# ============================================
# Safe read of README (handle missing file)
# ============================================
readme_file = Path(__file__).parent / "README.md"
long_description = "A competitive Texas Hold'em poker agent with RL and LLM support"

if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    print(f"Warning: {readme_file} not found, using default description")

# ============================================
# Core dependencies
# ============================================
core_requirements = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "gymnasium>=0.29.0",
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "pydantic>=2.0.0",
    "python-multipart>=0.0.6",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "python-dotenv>=1.0.0",
    "click>=8.1.0",
    "psutil>=5.9.0",
    "httpx>=0.24.0",
    "aiofiles>=23.0.0",
    "cachetools>=5.3.0",
]

# ============================================
# Optional dependencies
# ============================================
dev_requirements = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-timeout>=2.1.0",
    "pytest-xdist>=3.3.0",
    "pytest-mock>=3.11.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "mypy>=1.5.0",
    "ruff>=0.0.280",
    "pre-commit>=3.3.0",
    "ipython>=8.14.0",
    "jupyter>=1.0.0",
]

gpu_requirements = ["nvidia-cuda-runtime-cu12>=12.1.0"]
llm_requirements = ["openai>=1.0.0", "anthropic>=0.7.0"]
monitoring_requirements = ["tensorboard>=2.13.0", "wandb>=0.15.0"]

# ============================================
# Setup configuration
# ============================================
setup(
    name="poker-agent",
    version="1.0.0",
    author="Orkhan",
    author_email="orkhanmustafayev44@gmail.com",
    description="A competitive Texas Hold'em poker agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    
    packages=find_packages(where=".", exclude=["tests", "tests.*", "experiments", "data", "docker"]),
    package_dir={"": "."},
    
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require={
        "dev": dev_requirements,
        "gpu": gpu_requirements,
        "llm": llm_requirements,
        "monitoring": monitoring_requirements,
        "all": dev_requirements + gpu_requirements + llm_requirements + monitoring_requirements,
    },
    
    entry_points={
        "console_scripts": [
            "poker-api=api.main:app",
            "poker-train=scripts.train_supervised:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    include_package_data=True,
    zip_safe=False,
)

print("\n✅ Poker Agent package configured successfully!")