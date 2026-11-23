"""String values: free generation, then completion from the request.

The model writes the value freely until it closes the quote. Because a
0.6B model often closes too early inside a quoted span of the request
(e.g. before a ``]`` glued to the quote token), the result is snapped to
the full span when it is a prefix of one.
"""

import json
import re

import numpy as np

from llm_sdk import Small_LLM_Model as Model

from src.vocabulary import Vocabulary


MAX_STR_TOKENS = 100

# Text inside '...' or "...". A quote glued to a letter on its inner side,
# like the apostrophe in "I'm", is not a delimiter.
QUOTEDTXT = re.compile(r"(?<!\w)'(.*?)'(?!\w)|(?<!\w)\"(.*?)\"(?!\w)")


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

    raise ValueError(
        f"Generated string did not close within {MAX_STR_TOKENS} tokens"
    )


def quoted_texts(request: str) -> list[str]:
    """Return the texts written inside '...' or "..." in the request."""
    return [single or double for single, double in QUOTEDTXT.findall(request)]


def double_check(content: str, request: str) -> str:
    """Complete a value the model cut short inside a quoted span.

    The content is compared as written and JSON-unescaped, since the
    model escapes inconsistently. A value equal to a quoted text is
    already complete; otherwise it is completed to the first quoted text
    it starts. Other values are kept as written when the request contains
    them that way, and JSON-unescaped otherwise.
    """
    try:
        parsed: str = json.loads(f'"{content}"')
    except json.JSONDecodeError:
        parsed = content
    texts = quoted_texts(request)
    for block in (content, parsed):
        if block in texts:
            return block
    for text in texts:
        for block in (content, parsed):
            if block and text.startswith(block):
                return text
    # Not from a quoted span: keep a bare copy of the request text,
    # otherwise trust the model's JSON escaping.
    if content in request:
        return content
    return parsed
