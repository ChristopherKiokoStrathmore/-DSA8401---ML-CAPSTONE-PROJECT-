PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install lint test smoke data labels

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
	$(BIN)/python -m capstone.generate_sample --seed 42 --out data/sample/messages.csv \
		--labels data/sample/labels
	$(BIN)/python -m capstone.prepare_data data/sample/messages.csv \
		--out data/private/smoke \
		--audit reports/data_audit.md \
		--audit-kind harness
	$(BIN)/python -m capstone.splits --labels data/sample/labels \
		--out reports/manifests --kind harness
	$(BIN)/python -m capstone.agreement --labels data/sample/labels \
		--out reports/agreement.json --kind harness
	$(BIN)/python -m capstone.leakage --manifests reports/manifests

data:
	@test -n "$(DATA)" || (echo "usage: make data DATA=path/to/workbook.xlsx" && exit 1)
	$(BIN)/python -m capstone.prepare_data "$(DATA)" \
		--out data/private \
		--audit reports/data_audit.md \
		--audit-kind data

labels:
	@test -n "$(LABELS)" || (echo "usage: make labels LABELS=path/to/label_dir" && exit 1)
	$(BIN)/python -m capstone.splits --labels "$(LABELS)" \
		--out data/private/manifests --kind data
	$(BIN)/python -m capstone.agreement --labels "$(LABELS)" \
		--out data/private/agreement.json --kind data
	$(BIN)/python -m capstone.leakage --manifests data/private/manifests
