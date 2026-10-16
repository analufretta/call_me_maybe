"""Constrained generation: pick a function and build its call object."""

import json
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.models import FunctionDef, TestPrompt
from src.vocabulary import Vocabulary


def pick_function(
    prompt: TestPrompt, functions: list[FunctionDef],
    model: Small_LLM_Model, vocab: Vocabulary
) -> str:
    """Choose the function by constrained greedy decoding over the names.

    Builds one valid token path per function name and walks it token by
    token. At each step only tokens that continue a still-matching name are
    allowed; the model's argmax among those is taken. The walk ends when the
    chosen tokens equal a complete name, so the result is always a real
    function -- the model only steers *between* valid names, never off them.
    """
    base_call = build_prompt(prompt, functions)
    legal_paths = allowed_tokens(functions, model)
    base_ids = model.encode(base_call)[0].tolist()
    choosen: list[int] = []

    while True:
        depth = len(choosen)
        matches: list[list[int]] = [
            fn_tokens for fn_tokens in legal_paths.values()
            if fn_tokens[:depth] == choosen
            ]
        allowed = {toks[depth] for toks in matches if len(toks) > depth}
        if not allowed:
            break
        logits = model.get_logits_from_input_ids(base_ids + choosen)
        choosen.append(max(allowed, key=lambda t: logits[t]))

    for f, toks in legal_paths.items():
        if toks == choosen:
            return f
    raise AssertionError("constrained walk ended off every valid name")


def build_prompt(prompt: TestPrompt, functions: list[FunctionDef]) -> str:
    """Build the model input for choosing a function."""
    names = ", ".join(f.name for f in functions)
    return (
        f"Available functions: {names}\n"
        f"User request: {prompt.prompt}\n"
        f'{{"name": "'
    )


def allowed_tokens(
    functions: list[FunctionDef], model: Small_LLM_Model
) -> dict[str, list[int]]:
    legal_paths: dict[str, list[int]] | None = {}
    for f in functions:
        encoded = model.encode(f.name)[0].tolist()
        legal_paths[f.name] = encoded
    assert legal_paths is not None
    return legal_paths


def generate_call(
    prompt: TestPrompt, function: FunctionDef,
    model: Small_LLM_Model, vocab: Vocabulary
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
