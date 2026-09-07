PYTHON ?= python

.PHONY: install test lint format typecheck benchmark figures notebooks verify build
install:
	$(PYTHON) -m pip install -e ".[dev]"
test:
	$(PYTHON) -m pytest --cov=ml_from_scratch --cov-report=term-missing --cov-report=xml
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
typecheck:
	$(PYTHON) -m mypy
benchmark:
	$(PYTHON) -m benchmarks.run_all
figures:
	$(PYTHON) scripts/generate_figures.py
notebooks:
	$(PYTHON) scripts/verify_artifacts.py --notebooks
verify:
	$(PYTHON) scripts/verify_artifacts.py
build:
	$(PYTHON) -m build
