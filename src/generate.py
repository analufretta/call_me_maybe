"""Constrained generation: pick a function and build its call object."""

import json
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.models import FunctionDef, TestPrompt
from src.vocabulary import Vocabulary


def constrained_max(
    ids: list[int], allowed: set[int], model: Small_LLM_Model
) -> int:
    """Pick the max-logit token within ``allowed`` for context ``ids``.

    The one masked-argmax primitive every value fill builds on: score the
    next token with the model, then take the highest-scoring token that the
    mask permits. ``allowed`` must be non-empty.
    """
    logits = model.get_logits_from_input_ids(ids)
    return max(allowed, key=lambda t: logits[t])


def prompt_funtion(prompt: TestPrompt, functions: list[FunctionDef]) -> str:
    """Build the model input for choosing a function."""
    names = ", ".join(f.name for f in functions)
    return (
        f"Available functions: {names}\n"
        f"User request: {prompt.prompt}\n"
        f'{{"name": "'
    )


def prompt_output(function: FunctionDef, prompt: TestPrompt) -> str:
    param_list = ", ".join(
        f"{param} ({spec.type})" for param, spec in function.parameters.items()
    )
    return (
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters: {param_list}\n"
        f"User request: {prompt.prompt}\n"
        f'{{"name": "{function.name}", "parameters": {{'
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


def pick_function(
    prompt: TestPrompt, functions: list[FunctionDef],
    model: Small_LLM_Model, vocab: Vocabulary
) -> FunctionDef:
    """Choose the function by constrained greedy decoding over the names.

    Builds one valid token path per function name and walks it token by
    token. At each step only tokens that continue a still-matching name are
    allowed; the model's argmax among those is taken. The walk ends when the
    chosen tokens equal a complete name, so the result is always a real
    function -- the model only steers *between* valid names, never off them.
    """
    base_call = prompt_funtion(prompt, functions)
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
        choosen.append(constrained_max(base_ids + choosen, allowed, model))
    by_name = {fn.name: fn for fn in functions}
    for name, toks in legal_paths.items():
        if toks == choosen:
            return by_name[name]
    raise AssertionError("constrained walk ended off every valid name")


def generate_output(
    prompt: TestPrompt, function: FunctionDef,
    model: Small_LLM_Model, vocab: Vocabulary
) -> dict:
    """Build one output object for ``prompt`` calling ``function``.

    Walks the JSON state machine: forced scaffold (keys, punctuation)
    interleaved with constrained model steps for each parameter value.
    Returns the object ``{"prompt", "name", "parameters"}`` (V.4.1).
    """
    from src.format import force, fill_boolean, fill_number, fill_string

    fill_by_type = {
        "number": fill_number, "string": fill_string, "boolean": fill_boolean,
    }

    ids = model.encode(prompt_output(function, prompt))[0].tolist()
    params: dict[str, object] = {}
    for i, (param, ptype) in enumerate(function.parameters.items()):
        if i:
            force(ids, ", ", model)
        opener = '"' if ptype.type == "string" else ""
        force(ids, f'"{param}": {opener}', model)
        fill = fill_by_type.get(ptype.type, fill_string)
        params[param] = fill(ids, model, vocab)
    force(ids, "}}", model)

    return {"prompt": prompt.prompt, "name": function.name, "parameters": params}


def write_output(path: Path, results: list[dict]) -> None:
    """Write the result objects to ``path`` as a single JSON array.

    Creates the parent directory if missing, then dumps the list with
    indentation. Each object's numbers are already floats (V.4.1); this just
    serialises what it is given.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
