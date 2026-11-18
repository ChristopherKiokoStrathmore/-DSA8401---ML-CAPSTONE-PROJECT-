PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install lint test smoke data

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

test:
	$(BIN)/pytest

smoke:
	$(BIN)/python -m capstone.generate_sample --seed 42 --out data/sample/messages.csv
	$(BIN)/python -m capstone.prepare_data data/sample/messages.csv \
		--out data/private/smoke \
		--audit reports/data_audit.md \
		--audit-kind harness

data:
	@test -n "$(DATA)" || (echo "usage: make data DATA=path/to/workbook.xlsx" && exit 1)
	$(BIN)/python -m capstone.prepare_data "$(DATA)" \
		--out data/private \
		--audit reports/data_audit.md \
		--audit-kind data
