"""Low-level generation moves: forced scaffold and masked value fills.

FORCE emits structure we already know (keys, braces, quotes, commas) with no
model call and no choice. FILL asks the model for a value but masks its
choices to the tokens legal for the target type, so the emitted JSON is valid
by construction. Every model step goes through ``constrained_max``.
"""

import json
import re
from typing import Callable

import numpy as np
from llm_sdk import Small_LLM_Model as Model

from src.vocabulary import Vocabulary

def constrained_max(
    ids: list[int], allowed: set[int], model: Model
) -> int:
    """Pick the max-logit token within ``allowed`` for context ``ids``.

    The one masked-argmax primitive every model step builds on: score the
    next token with the model, then take the highest-scoring token that the
    mask permits. ``allowed`` must be non-empty.

    When the mask leaves a single token there is no choice to make, so it is
    returned without a model call -- e.g. the tail of a function name once
    its prefix is unique.
    """
    if len(allowed) == 1:
        return next(iter(allowed))
    logits = model.get_logits_from_input_ids(ids)
    return max(allowed, key=lambda t: logits[t])

def force(ids: list[int], text: str, model: Model) -> None:
    """Append the tokens of a known scaffold string to ``ids`` in place.

    Structural text -- keys, braces, quotes, commas -- is fixed, so it is
    encoded and appended with no logits call and no choice. Mutates ``ids``
    so the caller's growing buffer carries both forced and filled tokens.
    """
    ids.extend(model.encode(text)[0].tolist())

def find_bool(
    ids: list[int], model: Model, vocab: Vocabulary
    ) -> bool:
    pass

def find_num(
    ids: list[int], model: Model, vocab: Vocabulary
    ) -> int | float:
    pass

def find_str(
    ids: list[int], model: Model, vocab: Vocabulary
    ) -> str:
    pass

FILL_BY_TYPE: dict[str, Callable[[list[int], Model, Vocabulary], object]] = {
    "number": find_num, "integer": find_num,
    "boolean": find_bool, "string": find_str,
}

def fill(
    ids: list[int], model: Model, vocab: Vocabulary, ptype: str
    ) -> object:
    """Fill one parameter value of schema type ``ptype``."""
    return FILL_BY_TYPE[ptype](ids, model, vocab)
