.DEFAULT_GOAL := help

PYTHON ?= python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
QUESTION ?= What James Brown songs do you have?
DATABASE ?= chinook.db

.PHONY: help setup install run evaluate test clean

help:
	@printf "Available targets:\n"
	@printf "  setup     Create the virtual environment and install dependencies\n"
	@printf "  install   Install dependencies into the existing virtual environment\n"
	@printf "  run       Ask one question (QUESTION=...)\n"
	@printf "  evaluate  Run the evaluation suite\n"
	@printf "  test      Run all tests\n"
	@printf "  clean     Remove the virtual environment and pytest cache\n"

setup: $(VENV_PYTHON)

$(VENV_PYTHON): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

install: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install -r requirements.txt

run: setup
	$(VENV_PYTHON) -m app.main --question "$(QUESTION)" --database "$(DATABASE)"

evaluate: setup
	$(VENV_PYTHON) -m app.main --evaluate --database "$(DATABASE)"

test: setup
	$(VENV_PYTHON) -m pytest tests -q

clean:
	rm -rf $(VENV) .pytest_cache
