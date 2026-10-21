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
from llm_sdk import Small_LLM_Model

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

# JSON number grammar. PREFIX: text that can still grow into a number
# ("", "-", "12", "12."); no leading zeros, no bare ".5". COMPLETE: text
# that may end here ("12", "-0.5"); "12." and "-" may not.
NUMBER_PREFIX = re.compile(r"-?((0|[1-9]\d*)(\.\d*)?)?")
NUMBER_COMPLETE = re.compile(r"-?(0|[1-9]\d*)(\.\d+)?")


def _repeat_len(emitted: list[int]) -> int:
    """Length of a chunk repeated twice at the end of ``emitted``, else 0.

    Checks every chunk size k from MIN_REPEAT_TOKENS up to half the length:
    if the last k tokens equal the k before them, the value is looping.
    """
    for k in range(MIN_REPEAT_TOKENS, len(emitted) // 2 + 1):
        if emitted[-k:] == emitted[-2 * k:-k]:
            return k
    return 0


def constrained_max(
    ids: list[int], allowed: set[int], model: Small_LLM_Model
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


def ranked_pick(
    ids: list[int], model: Small_LLM_Model, accept: Callable[[int], bool]
) -> int:
    """Pick the highest-logit token that ``accept`` approves.

    For constraints that depend on what was already emitted (a number may
    not take a second '.'), a full mask would test every vocab token each
    step. Instead walk tokens best-first and stop at the first accepted
    one -- usually within the top few, so only a handful get checked.
    """
    logits = np.asarray(model.get_logits_from_input_ids(ids))
    for tok in np.argsort(-logits):
        if accept(int(tok)):
            return int(tok)
    raise AssertionError("no token satisfies the constraint")


def force(ids: list[int], text: str, model: Small_LLM_Model) -> None:
    """Append the tokens of a known scaffold string to ``ids`` in place.

    Structural text -- keys, braces, quotes, commas -- is fixed, so it is
    encoded and appended with no logits call and no choice. Mutates ``ids``
    so the caller's growing buffer carries both forced and filled tokens.
    """
    ids.extend(model.encode(text)[0].tolist())


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

    Each step takes the best token that keeps the text on the JSON number
    grammar: a piece only if the result is still a ``NUMBER_PREFIX``, a stop
    only once the text is a ``NUMBER_COMPLETE`` number. So ``--``, ``1.2.3``
    and an empty value can never come out. The stop is NOT appended (the
    orchestrator forces the real separator next). Returns a ``float``.
    """
    text = ""
    emitted: list[int] = []

    def accept(tok: int) -> bool:
        if tok in vocab.value_stops:
            return NUMBER_COMPLETE.fullmatch(text) is not None
        if tok in vocab.number_pieces:
            piece = vocab.decode([tok])
            return NUMBER_PREFIX.fullmatch(text + piece) is not None
        return False

    while len(emitted) < MAX_VALUE_TOKENS:
        tok = ranked_pick(ids, model, accept)
        if tok in vocab.value_stops:
            break
        ids.append(tok)
        emitted.append(tok)
        text += vocab.decode([tok])
    # Cap hit mid-number ("12.", "-"): drop the dangling part.
    if not NUMBER_COMPLETE.fullmatch(text):
        text = text.rstrip(".-") or "0"
    return float(text)


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
        unescaped: str = json.loads(f'"{text}"')
        return unescaped
    except json.JSONDecodeError:
        return text
