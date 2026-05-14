# ============================================
# Poker Agent Makefile
# ============================================

.PHONY: help install install-dev clean test lint format precommit
.PHONY: run-api run-train run-eval docker-build docker-up docker-down
.PHONY: data-prep data-clean export-model notebook check

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy

# Default target
help:
	@echo "Poker Agent - Available Commands"
	@echo "================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install production dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make clean            - Clean temporary files"
	@echo ""
	@echo "Data Processing:"
	@echo "  make data-prep        - Parse JSONL to CSV"
	@echo "  make preprocess       - Preprocess data"
	@echo "  make data-clean       - Clean data directories"
	@echo ""
	@echo "Training:"
	@echo "  make train-supervised - Run supervised training"
	@echo "  make train-rl         - Run RL training"
	@echo "  make self-play        - Run self-play training"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate         - Evaluate model"
	@echo "  make compare          - Compare agents"
	@echo "  make eda              - Run EDA"
	@echo ""
	@echo "API & Deployment:"
	@echo "  make run-api          - Start API server"
	@echo "  make run-dev          - Start dev API with reload"
	@echo "  make export-model     - Export model to ONNX/TorchScript"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker images"
	@echo "  make docker-up        - Start Docker services"
	@echo "  make docker-down      - Stop Docker services"
	@echo "  make docker-logs      - View Docker logs"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format           - Format code with black"
	@echo "  make lint             - Run linters"
	@echo "  make test             - Run tests"
	@echo "  make check            - Run all checks"
	@echo ""
	@echo "Notebooks:"
	@echo "  make notebook         - Start Jupyter notebook"
	@echo ""
	@echo "Utilities:"
	@echo "  make info             - Show system info"
	@echo "  make requirements     - Generate requirements.txt"

# ============================================
# Installation
# ============================================
install:
	@echo "Installing production dependencies..."
	$(PIP) install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	@echo "Installing pre-commit hooks..."
	pre-commit install

# ============================================
# Data Processing
# ============================================
data-prep:
	@echo "Parsing JSONL to CSV..."
	$(PYTHON) scripts/parse_jsonl_to_csv.py

preprocess:
	@echo "Preprocessing data..."
	$(PYTHON) scripts/preprocess_data.py

data-clean:
	@echo "Cleaning data directories..."
	rm -rf data/processed/*.csv
	rm -rf data/datasets/*.pt
	rm -rf data/raw/*.jsonl

# ============================================
# Training
# ============================================
train-supervised:
	@echo "Starting supervised training..."
	$(PYTHON) scripts/train_supervised.py --config configs/config.yaml

train-rl:
	@echo "Starting RL training..."
	$(PYTHON) scripts/train_rl.py --episodes 5000 --num-opponents 5

self-play:
	@echo "Starting self-play training..."
	$(PYTHON) scripts/run_self_play.py --iterations 50 --hands-per-iter 500

# ============================================
# Evaluation
# ============================================
evaluate:
	@echo "Evaluating model..."
	$(PYTHON) scripts/evaluate_model.py --num-hands 1000

compare:
	@echo "Comparing agents..."
	$(PYTHON) scripts/compare_agents.py --num-hands 500

eda:
	@echo "Running EDA..."
	$(PYTHON) scripts/eda.py

# ============================================
# API
# ============================================
run-api:
	@echo "Starting API server..."
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

run-dev:
	@echo "Starting development API server..."
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

export-model:
	@echo "Exporting model..."
	$(PYTHON) scripts/export_model.py --format all

# ============================================
# Docker
# ============================================
docker-build:
	@echo "Building Docker images..."
	docker-compose -f docker/docker-compose.yml build

docker-up:
	@echo "Starting Docker services..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo "Services started at http://localhost:8000"

docker-down:
	@echo "Stopping Docker services..."
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	@echo "Showing Docker logs..."
	docker-compose -f docker/docker-compose.yml logs -f

# ============================================
# Code Quality
# ============================================
format:
	@echo "Formatting code..."
	$(BLACK) src/ scripts/ api/ tests/
	$(ISORT) src/ scripts/ api/ tests/

lint:
	@echo "Running linters..."
	$(FLAKE8) src/ scripts/ api/ tests/
	$(MYPY) src/ --ignore-missing-imports

test:
	@echo "Running tests..."
	$(PYTEST) tests/ -v --cov=src --cov-report=html

check: format lint test
	@echo "All checks passed!"

precommit:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files

# ============================================
# Notebooks
# ============================================
notebook:
	@echo "Starting Jupyter notebook..."
	jupyter notebook experiments/notebooks/

# ============================================
# Utilities
# ============================================
info:
	@echo "System Information:"
	@echo "=================="
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo ""
	@echo "CUDA available: $(shell $(PYTHON) -c "import torch; print(torch.cuda.is_available())")"
	@echo "GPU count: $(shell $(PYTHON) -c "import torch; print(torch.cuda.device_count())")"

requirements:
	@echo "Generating requirements.txt..."
	pip freeze > requirements.txt
	pip freeze > requirements-dev.txt

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	@echo "Clean complete!"