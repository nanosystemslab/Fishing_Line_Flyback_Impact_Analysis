# Makefile for Fishing Line Flyback Impact Analysis

.PHONY: help install test test-fast test-all test-integration test-performance coverage coverage-html lint docs clean dev

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and dependencies
	poetry install

test: ## Run fast tests (excluding slow, GUI, visualization)
	nox --session=tests-fast

test-fast: ## Run only fast unit tests
	nox --session=tests-fast

test-all: ## Run all tests including slow ones
	nox --session=tests-all

test-integration: ## Run integration tests
	nox --session=tests-integration

test-performance: ## Run performance tests
	nox --session=tests-performance

test-matrix: ## Run the full test matrix
	nox --session=test-matrix

coverage: ## Generate coverage report
	nox --session=coverage

coverage-html: ## Generate HTML coverage report
	nox --session=coverage-html
	@echo "Open htmlcov/index.html to view the report"

lint: ## Run all linting tools
	nox --session=pre-commit

safety: ## Run security checks
	nox --session=safety

mypy: ## Run type checking
	nox --session=mypy

docs: ## Build and serve documentation
	nox --session=docs

docs-build: ## Build documentation
	nox --session=docs-build

clean: ## Clean build artifacts
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf .coverage.*
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

dev: ## Development setup and fast test
	nox --session=dev

# Convenience targets for common workflows
check: lint test ## Run linting and fast tests

ci: lint test test-integration coverage ## Run CI pipeline locally

all: lint test-matrix coverage-html docs-build ## Run everything

# Individual test files (examples)
test-shared: ## Test shared components only
	pytest tests/test_shared_*.py -v

test-impulse: ## Test impulse analysis only
	pytest tests/test_impulse_analysis.py -v

test-viz: ## Test visualization (requires display)
	pytest tests/test_visualization.py -v -m "not slow"

test-gui: ## Test GUI components
	pytest tests/test_gui.py -v

# Coverage with specific targets
coverage-shared: ## Coverage for shared components
	pytest tests/test_shared_*.py --cov=src/Fishing_Line_Flyback_Impact_Analysis/shared --cov-report=term-missing

coverage-impulse: ## Coverage for impulse analysis
	pytest tests/test_impulse_analysis.py --cov=src/Fishing_Line_Flyback_Impact_Analysis/impulse_analysis --cov-report=term-missing

# Development helpers
install-dev: ## Install development dependencies
	poetry install --with dev

update-deps: ## Update dependencies
	poetry update

lock: ## Update lock file
	poetry lock

build: ## Build package
	poetry build

publish-test: ## Publish to test PyPI
	poetry publish --repository testpypi

publish: ## Publish to PyPI
	poetry publish

# Environment info
info: ## Show environment information
	@echo "Python version:"
	@python --version
	@echo "\nPoetry version:"
	@poetry --version
	@echo "\nNox version:"
	@nox --version
	@echo "\nPackage info:"
	@poetry show fishing-line-flyback-impact-analysis 2>/dev/null || echo "Package not installed"

# Quick development workflow
quick: ## Quick development check (lint + fast tests)
	nox --session=pre-commit --session=tests-fast

# Performance profiling
profile: ## Profile the test suite
	pytest tests/ --profile-svg -m "not slow and not gui"

# Database/analysis helpers
analyze-results: ## Analyze test results from last run
	@echo "Coverage summary:"
	@coverage report --show-missing 2>/dev/null || echo "No coverage data found"
	@echo "\nTest summary from last run:"
	@tail -20 .pytest_cache/lastfailed 2>/dev/null || echo "No test results found"
