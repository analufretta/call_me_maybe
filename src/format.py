"""Low-level generation moves: forced scaffold and masked value fills.

FORCE emits structure we already know (keys, braces, quotes, commas) with no
model call and no choice. FILL asks the model for a value but masks its
choices to the tokens legal for the target type, so the emitted JSON is valid
by construction. The fills mask with ``constrained_max`` from ``generate``.
"""

from llm_sdk import Small_LLM_Model

from src.vocabulary import Vocabulary


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
    """Fill an unquoted boolean value (``true``/``false``). STUB (step 3)."""
    return False


def fill_number(
    ids: list[int], model: Small_LLM_Model, vocab: Vocabulary
) -> float:
    """Fill a JSON number value, decoded to ``float``. STUB (step 3)."""
    return 0.0


def fill_string(
    ids: list[int], model: Small_LLM_Model, vocab: Vocabulary
) -> str:
    """Fill a quoted string value, quotes stripped. STUB (step 3)."""
    return ""
