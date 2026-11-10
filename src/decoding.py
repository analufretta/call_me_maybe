"""Low-level generation moves: forced scaffold and masked value fills.

FORCE emits structure we already know (keys, braces, quotes, commas) with no
model call and no choice. FILL asks the model for a value but masks its
choices to the tokens legal for the target type, so the emitted JSON is valid
by construction. Every model step goes through ``constrained_max``.
"""

from typing import Callable

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


def fill_bool(
    ids: list[int], model: Model, vocab: Vocabulary
) -> bool:
    """Choose ``true`` or ``false`` and append its tokens to ``ids``.

    The two literals start with different tokens, so one masked model step
    over those first tokens decides the value. The rest of the chosen
    literal is forced. Returns the Python bool, not the text, so that
    ``json.dump`` writes a real JSON boolean.
    """
    true_ids = model.encode("true")[0].tolist()
    false_ids = model.encode("false")[0].tolist()
    allowed = {true_ids[0], false_ids[0]}
    first_tok = constrained_max(ids, allowed, model)
    result = first_tok == true_ids[0]
    ids.extend(true_ids if result else false_ids)
    return result


def fill_num(
    ids: list[int], model: Model, vocab: Vocabulary,
    ptype: str
) -> int | float:
    """Generate a number digit by digit and append its tokens to ``ids``.

    Each step masks the model to the characters that keep the text a valid
    number: ``-`` only first, ``.`` only once and only for floats. Stop
    tokens become legal once the text ends in a digit; the chosen stop is
    not appended, the forced scaffold that follows writes the real one.
    Returns an ``int`` for ``integer`` and a ``float`` otherwise.
    """
    base = vocab.num_tokens()
    stops = vocab.stop_tokens()
    minus = vocab.token_to_id("-")
    dot = vocab.token_to_id(".")
    text = ""
    while True:
        allowed = set(base)
        if text:
            allowed.discard(minus)
        if (ptype == "integer") or ("." in text) or (text in ("", "-")):
            allowed.discard(dot)
        if text and text[-1].isdigit():
            allowed |= stops
        tok = constrained_max(ids, allowed, model)
        if tok in stops:
            break
        ids.append(tok)
        text += model.decode([tok])
    return int(text) if ptype == "integer" else float(text)


def fill_str(
    ids: list[int], model: Model, vocab: Vocabulary
    ) -> str:
    pass


def fill(
    ids: list[int], model: Model, vocab: Vocabulary, ptype: str
    ) -> object:
    """Fill one parameter value of schema type ``ptype``."""
    if ptype == "boolean":
        return fill_bool(ids, model, vocab)
    elif ptype == "string":
        return fill_str(ids, model, vocab)
    elif ptype == "integer":
        return fill_num(ids, model, vocab, ptype)
    else:
        return fill_num(ids, model, vocab, ptype)
