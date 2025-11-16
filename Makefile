.PHONY: help venv install test lint format

help:
	@echo "Available targets: venv install test lint format"

venv:
	python3 -m venv .venv

install: venv
	. .venv/bin/activate && pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && pip install flake8 && flake8

format:
	. .venv/bin/activate && pip install black && black .
