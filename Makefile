.PHONY: install run debug clean fclean re lint lint-strict

# Default input/output paths, overridable on the command line:
#   make run INPUT=other.json OUTPUT=out.json
FUNCTIONS ?= data/input/functions_definition.json
INPUT     ?= data/input/function_calling_tests.json
OUTPUT    ?= data/output/function_calling_results.json

MYPY_FLAGS =--warn-return-any --warn-unused-ignores --ignore-missing-imports \
             --disallow-untyped-defs --check-untyped-defs

install:
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	command -v uv >/dev/null 2>&1 || \
	{ echo "uv missing. Installing..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }; \
	uv sync --locked --link-mode=copy

# `|| true`: the program prints its own error message, so make's extra
# "make: *** [run] Error 1" line is suppressed.
run:
	@uv run python -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT) || true

debug:
	@uv run python -m pdb -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

clean:
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf .mypy_cache .pytest_cache
	@find . -type f -name '*.pyc' -delete

fclean: clean
	@rm -rf .venv

re: fclean install

lint:
	@uv run flake8 . --extend-exclude .venv/,llm_sdk/
	@uv run mypy . --exclude '(\.venv|llm_sdk)/' --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

lint-strict:
	@uv run flake8 . --extend-exclude .venv/,llm_sdk/
	@uv run mypy . --exclude '(\.venv|llm_sdk)/' --strict
