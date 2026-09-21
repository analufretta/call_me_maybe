.PHONY: install run debug clean lint lint-strict

# Default input/output paths (overridable on the command line)
FUNCTIONS ?= data/input/functions_definition.json
INPUT     ?= data/input/function_calling_tests.json
OUTPUT    ?= data/output/function_calling_results.json

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
             --disallow-untyped-defs --check-untyped-defs

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	uv run python -m pdb -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .mypy_cache .pytest_cache
	find . -type f -name '*.pyc' -delete

lint:
	flake8 .
	mypy . $(MYPY_FLAGS)

lint-strict:
	flake8 .
	mypy . --strict
