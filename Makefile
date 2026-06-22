# Convenience targets. `make reproduce` runs the full data prep + evaluation.
PY ?= python

.PHONY: install preprocess evaluate train test reproduce clean

install:
	pip install -r requirements.txt

preprocess:
	$(PY) scripts/data_preprocessing.py

evaluate:
	$(PY) scripts/evaluate.py

train:
	$(PY) scripts/train.py

test:
	pytest

reproduce: preprocess evaluate

clean:
	rm -rf artifacts/* runs/* **/__pycache__ .pytest_cache
