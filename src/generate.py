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

    Scores each candidate name by summing the model's logits along its tokens
    and returns the highest. Because only real names are ever scored, the
    result is always a valid function (never an invented name).
    """
    base_call = build_prompt(prompt, functions)
    legal_paths = [allowed_tokens(functions)]
    base_ids = model.encode(base_call)[0].tolist()

    chosen_tokens: list[int] = []
    while True:
        depth = len(chosen_tokens)
        matches: list[list[int]] = [fn_tokens for _, fn_tokes if fn_tokens[:depth] == choosen]
        allowed: list[int] = []
        for tokens in matches:
            if len(tokes) > depth:
                alloewed.append(tokens[depth])
        if not allowed:
            break
        logits = model.get_logits_from_input_ids(base_ids + choosen)
        chosen.append(max(allowed, key=lambda t: logits[t]))
    
    result: FunctionDef | None = None
    for f, tokens in paths:
        if tokens == chosen:
            result = function
    if result is None:
        raise AssertionError("constrained walk ended off every valid name")
    return result

def build_prompt(prompt: TestPrompt, functions: list[FunctionDef]) -> str:
    """Build the minimal model input for choosing a function.

    For now just the user request plus the opening JSON scaffold, so the
    model's next tokens are the function name. Prompt wording and accuracy
    tuning (chat format, instructions, function catalogue) are a later phase.
    """
    names = ", ".join(f.name for f in functions)
    return (
        f"Available functions: {names}\n"
        f"User request: {prompt}\n"
        f"what is the most probable available function to return the right result?"
    )

def allowed_tokens(functions: list[FunctionDef]) -> dict[FunctionDef, list[int]]:
    legal_paths: dict[FunctionDef, list[int]] | None = {} 
    for f in functions:
        encoded = model.encode(f.name)[0].tolist()
        legal_paths[f] = encoded
    assert legal_paths is not None
    return legal_paths

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
