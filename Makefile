PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install lint test smoke data labels train-baselines train-transformer evaluate real serve-smoke hf-space

CONFIG ?= configs/afroxlmr.yaml
MANIFESTS ?= data/private/manifests
KIND ?= data
SERVE_PORT ?= 8765

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts

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
	$(MAKE) train-baselines LABELS=data/sample/labels MANIFESTS=reports/manifests KIND=harness
	$(MAKE) train-transformer LABELS=data/sample/labels MANIFESTS=reports/manifests \
		KIND=harness CONFIG=configs/tiny.yaml TINY=1
	$(MAKE) evaluate LABELS=data/sample/labels MANIFESTS=reports/manifests KIND=harness

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

train-baselines:
	@test -n "$(LABELS)" || (echo "usage: make train-baselines LABELS=... MANIFESTS=..." && exit 1)
	$(BIN)/python -m capstone.models.baselines \
		--labels "$(LABELS)" --manifests "$(MANIFESTS)" \
		--out artifacts/tfidf --kind "$(KIND)" --seed 42

train-transformer:
	@test -n "$(LABELS)" || (echo "usage: make train-transformer LABELS=... MANIFESTS=..." && exit 1)
	$(BIN)/python -m capstone.models.train \
		--config $(CONFIG) --labels "$(LABELS)" --manifests "$(MANIFESTS)" \
		--out artifacts/afroxlmr --kind "$(KIND)" $(if $(TINY),--tiny,)

evaluate:
	@test -n "$(LABELS)" || (echo "usage: make evaluate LABELS=... MANIFESTS=..." && exit 1)
	$(BIN)/python -m capstone.evaluate \
		--labels "$(LABELS)" --manifests "$(MANIFESTS)" \
		--tfidf artifacts/tfidf --transformer artifacts/afroxlmr \
		--baselines artifacts/baselines --out reports \
		--errors data/private/errors.csv --kind "$(KIND)"

real:
	@test -n "$(DATA)" && test -n "$(LABELS)" || (echo "usage: make real DATA=workbook LABELS=label_dir" && exit 1)
	$(MAKE) data DATA="$(DATA)"
	$(MAKE) labels LABELS="$(LABELS)"
	$(MAKE) train-baselines LABELS="$(LABELS)" MANIFESTS=data/private/manifests KIND=data
	$(MAKE) train-transformer LABELS="$(LABELS)" MANIFESTS=data/private/manifests KIND=data CONFIG=configs/afroxlmr.yaml
	$(MAKE) evaluate LABELS="$(LABELS)" MANIFESTS=data/private/manifests KIND=data

serve-smoke:
	SERVE_PORT=$(SERVE_PORT) BIN=$(BIN) bash scripts/serve_smoke.sh

hf-space:
	$(PYTHON) scripts/push_hf_space.py
