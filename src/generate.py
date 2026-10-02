"""Constrained generation: pick a function and build its call object."""

import json
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.models import FunctionDef, TestPrompt
from src.vocabulary import Vocabulary


def pick_function(
    prompt: TestPrompt, functions: list[FunctionDef], model: Small_LLM_Model, vocab: Vocabulary
) -> FunctionDef:
    """Choose which function the prompt is asking for.

    Runs the LLM constrained to the valid function names only, so the model
    can pick *which* function but never invent a name that is not in
    ``functions``.
    """
    pass


def generate_call(
    prompt: TestPrompt, function: FunctionDef, model: Small_LLM_Model, vocab: Vocabulary
) -> dict:
    """Build one output object for ``prompt`` calling ``function``.

    Walks the JSON state machine: forced scaffold (keys, punctuation)
    interleaved with constrained model steps for each parameter value.
    Returns the object ``{"prompt", "name", "parameters"}`` (V.4.1).
    """
    pass


def write_output(path: Path, results: list[dict]) -> None:
    """Write the result objects to ``path`` as a single JSON array.

    Creates the parent directory if missing, then dumps the list with
    indentation. Each object's numbers are already floats (V.4.1); this just
    serialises what it is given.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
