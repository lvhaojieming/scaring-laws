PYTHON ?= python3

.PHONY: install run test clean

install:
	$(PYTHON) -m pip install -e .

run:
	PYTHONPATH=src $(PYTHON) -m scaring_laws

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf .venv .pytest_cache .coverage build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
