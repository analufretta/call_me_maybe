"""Number values: masks that keep the generated text a valid number.

Two modes. COPY restricts the model to the numbers written in the request
(integers, decimals, fractions), so it picks which number goes where but
cannot miscopy a digit or loop. FREE only enforces number syntax, for
requests whose numbers are not all written in digits (``minus five``).
"""

import re

from llm_sdk import Small_LLM_Model as Model

from src.vocabulary import Vocabulary


MAX_NUM_TOKENS = 50

# A fraction is tried first, so "1/3" is one number and not "1" and "3".
NUMBER = re.compile(r"-?\d+/\d+|-?\d+(?:\.\d+)?")


def number_candidates(request: str, ptype: str) -> list[str]:
    """Return the numbers written in ``request``, in order, as digit text.

    Fractions become their decimal value. ``integer`` parameters only get
    whole numbers. Duplicates are kept, so the count matches the request.
    """
    found: list[str] = []
    for match in NUMBER.findall(request):
        if "/" in match:
            num, den = match.split("/")
            if int(den) == 0:
                continue
            match = repr(int(num) / int(den))
            if "e" in match:
                continue  # exponent form, e.g. 1e-05: not typeable here
        if ptype == "integer" and "." in match:
            continue
        found.append(match)
    return found


def choose_sign(
    ids: list[int], model: Model, vocab: Vocabulary,
    negative: bool, positive: bool
) -> str:
    """Decide the sign of a number in the tokenizer's natural form.

    The scaffold ``"a": `` ends in a lone space token, but the tokenizer
    writes a negative number as ``Ġ-`` (space and minus in one token), so
    a bare ``-`` there looks unnatural and the model avoids it. The space
    is dropped and the model chooses between ``Ġ-`` and ``Ġ``; when only
    one sign is possible it is forced. Returns ``"-"`` or ``""``.
    """
    space = vocab.token_to_id("Ġ")
    space_minus = vocab.token_to_id("Ġ-")
    if not negative or space_minus is None or not ids or ids[-1] != space:
        return ""
    if positive:
        logits = model.get_logits_from_input_ids(ids[:-1])
        if logits[space] >= logits[space_minus]:
            return ""
    ids[-1] = space_minus
    return "-"


def copy_mask(
    text: str, candidates: list[str], vocab: Vocabulary
) -> set[int]:
    """Return the token ids that keep ``text`` a prefix of a candidate.

    Stop tokens are allowed once ``text`` is a full candidate, so the
    number can only end on a value written in the request.
    """
    allowed: set[int] = set()
    for candidate in candidates:
        if len(candidate) > len(text) and candidate.startswith(text):
            token_id = vocab.token_to_id(candidate[len(text)])
            if token_id is not None:
                allowed.add(token_id)
    if text in candidates:
        allowed |= vocab.stop_tokens
    return allowed


def free_mask(text: str, ptype: str, vocab: Vocabulary) -> set[int]:
    """Return the token ids that keep ``text`` a valid number.

    ``-`` only first, ``.`` only once and only for floats, after a digit.
    Stop tokens become legal once the text ends in a digit.
    """
    allowed = set(vocab.num_tokens)
    if text:
        allowed.discard(vocab.token_to_id("-"))
    if ptype == "integer" or "." in text or text in ("", "-"):
        allowed.discard(vocab.token_to_id("."))
    if text and text[-1].isdigit():
        allowed |= vocab.stop_tokens
    return allowed
