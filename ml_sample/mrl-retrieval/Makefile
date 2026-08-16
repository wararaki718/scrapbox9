PYTHON ?= python
ARGS ?=

.PHONY: run test

run:
	$(PYTHON) -m app.main $(ARGS)

test:
	$(PYTHON) -m unittest discover -s tests -v