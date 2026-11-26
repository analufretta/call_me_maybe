"""Number values: masks that keep the generated text a valid number.

Only number syntax is enforced, so the model can also convert numbers
that are not written in digits (``minus five``, ``pi``, ``1/3``).
"""

import re

from llm_sdk import Small_LLM_Model as Model

from src.vocabulary import Vocabulary


MAX_NUM_TOKENS = 50


def choose_sign(ids: list[int], model: Model, vocab: Vocabulary) -> str:
    """Decide the sign of a number in the tokenizer's natural form.

    The scaffold ``"a": `` ends in a lone space token, but the tokenizer
    writes a negative number as ``Ġ-`` (space and minus in one token), so
    a bare ``-`` there looks unnatural and the model avoids it. The space
    is dropped and the model chooses between ``Ġ-`` and ``Ġ``.
    Returns ``"-"`` or ``""``.
    """
    space = vocab.token_to_id("Ġ")
    space_minus = vocab.token_to_id("Ġ-")
    if space_minus is None or not ids or ids[-1] != space:
        return ""
    logits = model.get_logits_from_input_ids(ids[:-1])
    if logits[space] >= logits[space_minus]:
        return ""
    ids[-1] = space_minus
    return "-"


def free_mask(text: str, ptype: str, vocab: Vocabulary) -> set[int]:
    """Return the token ids that keep ``text`` a valid number.

    ``-`` only first, ``.`` only once and only for floats, after a digit.
    ``/`` once, for floats, after a whole number, so a fraction can be
    copied from the request; its denominator is whole and not zero.
    Stop tokens become legal once the text ends in a digit.
    """
    allowed = set(vocab.num_tokens)
    last_char: bool = text[-1:].isdigit()
    whole: bool = "." not in text and "/" not in text
    if text:
        allowed.discard(vocab.token_to_id("-"))
    if ptype == "integer" or not whole or not last_char:
        allowed.discard(vocab.token_to_id("."))
        allowed.discard(vocab.token_to_id("/"))
    if text.endswith("/"):
        allowed.discard(vocab.token_to_id("0"))
    if last_char:
        allowed |= vocab.stop_tokens
    return allowed


def continues_fraction(text: str, request: str) -> bool:
    """Return whether ``request`` writes ``text`` as a fraction's numerator.

    The model copies ``5/12`` but tends to stop after ``5``: a JSON number
    never holds ``/``. When the digits so far are followed by ``/`` and a
    non-zero digit in the request, the number must go on as a fraction.
    """
    digits = text.lstrip("-")
    if not digits.isdigit():
        return False
    return re.search(rf"(?<![\d.]){digits}\/[1-9]", request) is not None


def ends_at_decimal(text: str, request: str) -> bool:
    """Return whether ``request`` writes ``text`` followed by a decimal part.

    An integer mask forbids ``.``, so for ``Is 4.5 prime?`` the model keeps
    adding digits after ``4`` and never stops. When the digits so far are
    followed by ``.`` and a digit in the request, the integer ends here.
    """
    digits = text.lstrip("-")
    if not digits.isdigit():
        return False
    return re.search(rf"(?<![\d.]){digits}\.\d", request) is not None


def to_number(text: str, ptype: str) -> int | float:
    """Return the value of the generated ``text``; a fraction is divided."""
    if "/" in text:
        num, den = text.split("/")
        return int(num) / int(den)
    return int(text) if ptype == "integer" else float(text)
