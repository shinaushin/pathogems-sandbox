# Developer convenience targets. Run `make help` for a summary.
#
# These assume the pathogems conda env is active and the package is
# installed in editable mode (`pip install -e stage3_experiments`).

.PHONY: help lint format typecheck test test-fast install clean

help:  ## Show this help
	@grep -E '^[a-z_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint:  ## Run ruff lint (auto-fix safe issues)
	ruff check --fix --config stage3_experiments/pyproject.toml stage3_experiments/

format:  ## Run ruff format (in-place)
	ruff format --config stage3_experiments/pyproject.toml stage3_experiments/

typecheck:  ## Run mypy strict type checking
	mypy --config-file stage3_experiments/pyproject.toml stage3_experiments/src/

test:  ## Run full test suite (including slow smoke test)
	pytest stage3_experiments/tests -v --tb=short

test-fast:  ## Run fast tests only (skip slow smoke test)
	pytest stage3_experiments/tests -v --tb=short -m "not slow"

install:  ## Install the Stage 3 package in editable mode
	cd stage3_experiments && pip install -e .

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache stage3_experiments/.mypy_cache
