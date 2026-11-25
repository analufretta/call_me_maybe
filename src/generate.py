"""Constrained generation: pick a function and build its call object."""

import json

from llm_sdk import Small_LLM_Model as Model

from src.decoding import constrained_max, fill, force
from src.models import FunctionDef, TestPrompt
from src.prompts import prompt_function, prompt_output
from src.vocabulary import Vocabulary


def allowed_fntokens(
    functions: list[FunctionDef], model: Model
) -> dict[str, list[int]]:
    """Map each function name to its token path, closed by a quote.

    The closing quote makes "stop here" a token the model can choose, so a
    name that is a token prefix of another (fn_get_user / fn_get_user_by_id)
    stays selectable.
    """
    quote = model.encode('"')[0].tolist()
    return {
        f.name: model.encode(f.name)[0].tolist() + quote for f in functions
    }


def pick_function(
    prompt: TestPrompt, functions: list[FunctionDef], model: Model
) -> FunctionDef:
    """Choose the function by constrained greedy decoding over the names.

    Builds one valid token path per function name and walks it token by
    token. At each step only tokens that continue a still-matching name are
    allowed; the model's argmax among those is taken. The walk ends when the
    chosen tokens equal a complete name, so the result is always a real
    function -- the model only steers *between* valid names, never off them.
    """
    base_call = prompt_function(prompt, functions)
    legal_paths = allowed_fntokens(functions, model)
    base_ids = model.encode(base_call)[0].tolist()
    chosen: list[int] = []

    while True:
        depth = len(chosen)
        matches: list[list[int]] = [
            fn_tokens for fn_tokens in legal_paths.values()
            if fn_tokens[:depth] == chosen
            ]
        allowed = {toks[depth] for toks in matches if len(toks) > depth}
        if not allowed:
            break
        chosen.append(constrained_max(base_ids + chosen, allowed, model))
    by_name = {fn.name: fn for fn in functions}
    for name, toks in legal_paths.items():
        if toks == chosen:
            return by_name[name]
    raise AssertionError("constrained walk ended off every valid name")


def generate_output(
    prompt: TestPrompt, function: FunctionDef,
    model: Model, vocab: Vocabulary
) -> dict[str, object]:
    """Build one output object for ``prompt`` calling ``function``.

    Walks the JSON state machine: forced scaffold (keys, punctuation)
    interleaved with constrained model steps for each parameter value.
    Returns the object ``{"prompt", "name", "parameters"}`` (V.4.1).
    """

    ids = model.encode(prompt_output(function, prompt))[0].tolist()
    params: dict[str, object] = {}
    for i, (param, ptype) in enumerate(function.parameters.items()):
        if i:
            force(ids, ", ", model)
        force(ids, f"{json.dumps(param)}: ", model)
        params[param] = fill(ids, model, vocab, ptype.type, prompt.prompt)
    return {
        "prompt": prompt.prompt, "name": function.name, "parameters": params
    }
