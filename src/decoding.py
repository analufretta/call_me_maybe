"""Low-level generation moves: forced scaffold and masked value fills.

FORCE emits structure we already know (keys, braces, quotes, commas) with no
model call and no choice. FILL asks the model for a value but masks its
choices to the tokens legal for the target type, so the emitted JSON is valid
by construction. Every model step goes through ``constrained_max``.
"""

import json

from llm_sdk import Small_LLM_Model as Model

from src.nums import MAX_NUM_TOKENS, choose_sign, free_mask, to_number
from src.strings import double_check, generate_string_bytes
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
    """Choose ``true`` or ``false`` and append its token to ``ids``.

    Each literal is a single vocab token, so one masked model step over
    those two ids decides the value. Returns the Python bool, not the text,
    so that ``json.dump`` writes a real JSON boolean.
    """
    true_id, false_id = vocab.bool_tokens
    result = constrained_max(ids, {true_id, false_id}, model) == true_id
    ids.append(true_id if result else false_id)
    return result


def fill_num(
    ids: list[int], model: Model, vocab: Vocabulary, ptype: str
) -> int | float:
    """Generate a number digit by digit and append its tokens to ``ids``.

    Each step is masked to number syntax only, so the model can convert
    words like ``minus five``. The sign is decided first by
    ``choose_sign``. The chosen stop token is not appended; the forced
    scaffold that follows writes the real one. Returns an ``int`` for
    ``integer`` and a ``float`` otherwise; ``1/3`` is divided by
    ``to_number``.
    """
    text = choose_sign(ids, model, vocab)
    for _ in range(MAX_NUM_TOKENS):
        allowed = free_mask(text, ptype, vocab)
        tok = constrained_max(ids, allowed, model)
        if tok in vocab.stop_tokens:
            break
        ids.append(tok)
        text += model.decode([tok])
    else:
        raise ValueError(
            f"Generated number is bigger than {MAX_NUM_TOKENS} tokens"
        )
    return to_number(text, ptype)


def fill_str(
    ids: list[int], model: Model, vocab: Vocabulary, request: str
) -> str:
    """Write a JSON string value into ``ids`` and return its Python value."""
    force(ids, '"', model)
    content_start = len(ids)

    content_bytes = generate_string_bytes(ids, model, vocab)
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Model generated invalid UTF-8") from exc

    value = double_check(content, request)
    if value != content:
        content = json.dumps(value, ensure_ascii=False)[1:-1]

    # Replace the generated tokens with the exact content and delimiter.
    ids[content_start:] = model.encode(content)[0].tolist()
    force(ids, '"', model)

    return value


def fill(
    ids: list[int], model: Model, vocab: Vocabulary, ptype: str,
    request: str
) -> object:
    """Fill one parameter value of schema type ``ptype``."""
    if ptype == "boolean":
        return fill_bool(ids, model, vocab)
    elif ptype == "string":
        return fill_str(ids, model, vocab, request)
    else:
        return fill_num(ids, model, vocab, ptype)
