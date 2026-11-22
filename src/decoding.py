"""Low-level generation moves: forced scaffold and masked value fills.

FORCE emits structure we already know (keys, braces, quotes, commas) with no
model call and no choice. FILL asks the model for a value but masks its
choices to the tokens legal for the target type, so the emitted JSON is valid
by construction. Every model step goes through ``constrained_max``.
"""

import json

import numpy as np

from llm_sdk import Small_LLM_Model as Model

from src.vocabulary import Vocabulary


MAX_STR_TOKENS = 100
MAX_NUM_TOKENS = 20


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


# TODO: copy-only mask. Take number candidates from the request with a regex
# (-?\d+(\.\d+)?), allow only next chars that keep the text a prefix of a
# candidate, allow a stop only on a full candidate. Fall back to the free
# numeric mask below when there are no candidates. Fixes digit loops (1/3 ->
# 1.3333...) and the 30-digit copy error; needs `request` passed to fill().
# TODO: words-to-number. Turn "minus five", "twenty-one" into digit candidates
# and merge them with the regex ones, ordered by position in the request.
# Do after fill_str.
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
    base = vocab.num_tokens
    stops = vocab.stop_tokens
    minus = vocab.token_to_id("-")
    dot = vocab.token_to_id(".")
    text = ""
    for _ in range(MAX_NUM_TOKENS):
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
    else:
        raise ValueError(
            f"Generated number is bigger than {MAX_NUM_TOKENS} tokens"
        )
    return int(text) if ptype == "integer" else float(text)


def generate_string_bytes(
    ids: list[int], model: Model, vocab: Vocabulary
) -> bytes:
    """Generate tokens until an unescaped closing quote appears.

    Accepted tokens are appended to ``ids``. The closing quote itself
    is not.

    Returns:
        The raw bytes of the string content, without the quotes.

    Raises:
        ValueError: If the string does not close within MAX_STR_TOKENS.
    """
    content_bytes = bytearray()
    escaped = False

    for _ in range(MAX_STR_TOKENS):
        logits = np.asarray(model.get_logits_from_input_ids(ids))
        tok = int(np.argmax(logits[:vocab.size]))
        token_bytes = vocab.token_bytes(tok)

        closing_at = None
        for i, byte in enumerate(token_bytes):
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                closing_at = i
                break

        if closing_at is not None:
            content_bytes.extend(token_bytes[:closing_at])
            return bytes(content_bytes)

        ids.append(tok)
        content_bytes.extend(token_bytes)
    else:
        raise ValueError(
            f"Generated string did not close within {MAX_STR_TOKENS} tokens"
        )


def fill_str(ids: list[int], model: Model, vocab: Vocabulary) -> str:
    """Write a JSON string value into ``ids`` and return its Python value."""
    force(ids, '"', model)
    content_start = len(ids)

    content_bytes = generate_string_bytes(ids, model, vocab)

    try:
        content = content_bytes.decode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Model generated an invalid JSON string") from exc

    # Replace the generated tokens with the exact content and delimiter.
    ids[content_start:] = model.encode(content)[0].tolist()
    force(ids, '"', model)

    return value


def fill(
    ids: list[int], model: Model, vocab: Vocabulary, ptype: str
) -> object:
    """Fill one parameter value of schema type ``ptype``."""
    if ptype == "boolean":
        return fill_bool(ids, model, vocab)
    elif ptype == "string":
        return fill_str(ids, model, vocab)
    else:
        return fill_num(ids, model, vocab, ptype)
