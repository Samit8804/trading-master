# CrowdWisdomQuant Makefile
.PHONY: install test run clean lint

SHELL := powershell.exe
PYTHON := python
PACKAGE := crowdwisdom_quant

# ── Installation ──────────────────────────────────────────────────────────────
install:
	pip install -e .
	pip install -r requirements.txt

install-dev: install
	pip install pytest pytest-cov mypy black

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests\ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests\ -v --cov=$(PACKAGE) --cov-report=term --cov-report=html

# ── Pipeline ──────────────────────────────────────────────────────────────────
run:
	$(PYTHON) main.py run_all

run-scrape:
	$(PYTHON) main.py scrape

run-preprocess:
	$(PYTHON) main.py preprocess

run-feature:
	$(PYTHON) main.py feature

run-validate:
	$(PYTHON) main.py validate

run-visualize:
	$(PYTHON) main.py visualize

run-report:
	$(PYTHON) main.py report

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check $(PACKAGE)\
		--ignore E501,W503

typecheck:
	$(PYTHON) -m mypy $(PACKAGE) --ignore-missing-imports

clean:
	Remove-Item -Recurse -Force build\ -ErrorAction SilentlyContinue
	Remove-Item -Recurse -Force dist\ -ErrorAction SilentlyContinue
	Remove-Item -Recurse -Force .pytest_cache\ -ErrorAction SilentlyContinue
	Remove-Item -Recurse -Force __pycache__\ -ErrorAction SilentlyContinue
	Get-ChildItem -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t crowdwisdom-quant .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data crowdwisdom-quant

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	Write-Host "Targets: install, test, run, lint, clean, docker-build"
