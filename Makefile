PY     ?= python
DATA   ?= data
CONFIG ?= configs/default.yaml

.PHONY: install data test lint run ablation figure clean

install:  ## editable install with test and lint tools
	$(PY) -m pip install -e ".[dev]"

data:  ## download train.csv into $(DATA)/ (needs `pip install kaggle` and Kaggle API credentials)
	bash scripts/download_data.sh $(DATA)

test:
	$(PY) -m pytest

lint:
	ruff check . && ruff format --check .

run:  ## walk-forward train + evaluate, writes results/
	hull-timing run --config $(CONFIG) --data $(DATA) --out results

ablation:  ## same run without lagged-target features, writes results/no_lags/
	hull-timing run --config configs/no_lags.yaml --data $(DATA) --out results/no_lags

figure:
	hull-timing plot --results results --out assets/walk_forward.png

clean:
	rm -rf .pytest_cache .ruff_cache build src/*.egg-info
	find . -name __pycache__ -prune -exec rm -rf {} +
