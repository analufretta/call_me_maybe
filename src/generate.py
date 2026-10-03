"""Constrained generation: pick a function and build its call object."""

import json
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.models import FunctionDef, TestPrompt
from src.vocabulary import Vocabulary


def build_prompt(prompt: TestPrompt, functions: list[FunctionDef]) -> str:
    """Build the minimal model input for choosing a function.

    For now just the user request plus the opening JSON scaffold, so the
    model's next tokens are the function name. Prompt wording and accuracy
    tuning (chat format, instructions, function catalogue) are a later phase.
    """
    return f'{prompt.prompt}\n{{"name": "'


def pick_function(
    prompt: TestPrompt, functions: list[FunctionDef], model: Small_LLM_Model, vocab: Vocabulary
) -> FunctionDef:
    """Choose which function the prompt is asking for.

    Scores each candidate name by summing the model's logits along its tokens
    and returns the highest. Because only real names are ever scored, the
    result is always a valid function (never an invented name).
    """
    base = build_prompt(prompt, functions)
    base_ids = model.encode(base)[0].tolist()

    best_function: FunctionDef | None = None
    best_score = float("-inf")
    for f in functions:
        fn_tokens = model.encode(f.name)[0].tolist()
        ids = base_ids[:]
        score = 0.0
        for fn_token in fn_tokens:
            logits = model.get_logits_from_input_ids(ids)
            score += logits[fn_token]
            ids.append(fn_token)
        if score > best_score:
            best_function, best_score = f, score

    assert best_function is not None  # functions is never empty
    return best_function


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
