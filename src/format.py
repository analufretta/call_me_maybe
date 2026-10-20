"""Low-level generation moves: forced scaffold and masked value fills.

FORCE emits structure we already know (keys, braces, quotes, commas) with no
model call and no choice. FILL asks the model for a value but masks its
choices to the tokens legal for the target type, so the emitted JSON is valid
by construction. The fills mask with ``constrained_max`` from ``generate``.
"""

import json

from llm_sdk import Small_LLM_Model

from src.generate import constrained_max
from src.vocabulary import Vocabulary

# Safety cap: a weak model may never emit a value's stop token (e.g. a regex
# string that never reaches a closing quote), looping forever. After this many
# tokens the fill stops and closes cleanly -- the JSON stays valid.
MAX_VALUE_TOKENS = 64

# Repetition guard: a weak model can loop a chunk (`\d+\s+\d+\s+...`) instead
# of closing. A chunk of at least this many tokens repeated back-to-back means
# the value is looping -- keep one copy and close. 2, not 1, so a legit
# doubled token (`**`) is not mistaken for a loop.
MIN_REPEAT_TOKENS = 2


def _repeat_len(emitted: list[int]) -> int:
    """Length of a chunk repeated twice at the end of ``emitted``, else 0.

    Checks every chunk size k from MIN_REPEAT_TOKENS up to half the length:
    if the last k tokens equal the k before them, the value is looping.
    """
    for k in range(MIN_REPEAT_TOKENS, len(emitted) // 2 + 1):
        if emitted[-k:] == emitted[-2 * k:-k]:
            return k
    return 0


def force(ids: list[int], text: str, model: Small_LLM_Model) -> None:
    """Append the tokens of a known scaffold string to ``ids`` in place.

    Structural text -- keys, braces, quotes, commas -- is fixed, so it is
    encoded and appended with no logits call and no choice. Mutates ``ids``
    so the caller's growing buffer carries both forced and filled tokens.
    """
    ids.extend(model.encode(text)[0].tolist())

# TODO: ANALYZES THESE FUNCTIONS

def fill_boolean(
    ids: list[int], model: Small_LLM_Model, vocab: Vocabulary
) -> bool:
    """Fill an unquoted boolean by trie-walking ``true`` / ``false``.

    Same constrained walk as ``pick_function``, but over two fixed paths. At
    each step only tokens that still continue ``true`` or ``false`` are
    allowed, so the result is always one of the two words.
    """
    paths = {True: model.encode("true")[0].tolist(),
             False: model.encode("false")[0].tolist()}
    chosen: list[int] = []
    while True:
        depth = len(chosen)
        matching = [toks for toks in paths.values() if toks[:depth] == chosen]
        allowed = {toks[depth] for toks in matching if len(toks) > depth}
        if not allowed:
            break
        chosen.append(constrained_max(ids + chosen, allowed, model))
    for value, toks in paths.items():
        if toks == chosen:
            ids.extend(chosen)
            return value
    raise AssertionError("boolean walk ended off true/false")


def fill_number(
    ids: list[int], model: Small_LLM_Model, vocab: Vocabulary
) -> float:
    """Fill a JSON number: emit digit/'.'/'-' tokens until a ',' or '}'.

    The mask is ``number_pieces | value_stops``. When the model reaches for a
    stop token the number is done; it is NOT appended (the orchestrator forces
    the real separator next). Returns the decoded tokens as ``float``.
    """
    allowed = vocab.number_pieces | vocab.value_stops
    emitted: list[int] = []
    while len(emitted) < MAX_VALUE_TOKENS:
        tok = constrained_max(ids, allowed, model)
        if tok in vocab.value_stops:
            break
        ids.append(tok)
        emitted.append(tok)
    return float(vocab.decode(emitted))


def fill_string(
    ids: list[int], model: Small_LLM_Model, vocab: Vocabulary
) -> str:
    """Fill a quoted string: FORCE '"', emit safe tokens, FORCE closing '"'.

    The mask is ``string_pieces | string_stops``. Content tokens come from
    string_pieces (never contain a '"'); the string ends when the model
    picks any ``string_stops`` token (a quote-first ender like ``",`` or a
    bare ``"``). The ender is NOT appended -- we force our own closing '"'
    and return the text between the quotes.
    """
    force(ids, '"', model)
    allowed = vocab.string_pieces | vocab.string_stops
    emitted: list[int] = []
    while len(emitted) < MAX_VALUE_TOKENS:
        tok = constrained_max(ids, allowed, model)
        if tok in vocab.string_stops:      # model chose to close the string
            break
        ids.append(tok)
        emitted.append(tok)
        k = _repeat_len(emitted)
        if k:                              # looping: drop the copy, close
            del emitted[-k:]
            del ids[-k:]
            break
    force(ids, '"', model)
    text = vocab.decode(emitted)
    # The model writes JSON-escaped text (`\\d` for `\d`); unescape it so the
    # value is the real string. A cap-truncated value can end mid-escape
    # (lone `\`), which json rejects -- keep the raw text then.
    try:
        return json.loads(f'"{text}"')
    except json.JSONDecodeError:
        return text
