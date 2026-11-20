"""Vocabulary: token<->id lookups and
byte-level decoding for the model vocab."""


from pydantic import BaseModel


def _build_byte_table() -> dict[str, int]:
    """GPT-2 / Qwen byte-level BPE table, reversed: costume char -> raw byte.

    Each of the 256 byte values appears in vocab.json as a printable
    "costume" character (space -> 'Ġ', newline -> 'Ċ', etc.). Reverse that
    map so a token's costume string can become real bytes again.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_TABLE = _build_byte_table()


class Vocabulary(BaseModel):
    """Token<->id lookups loaded from the model's vocab.json.

    The strings are still in byte-level "costume" form (``Ġ`` for a leading
    space, ``Ċ`` for newline, etc.) exactly as they appear in vocab.json;
    decoding them to real text is a later step. This skeleton only holds the
    two maps so later tasks (byte decoding, precomputed helper sets) can build
    on it.

    Attributes
    ----------
    token_id:
        The raw file contents: costume string -> id.
    id_token:
        The inverse: id -> costume string.
    """

    token_id: dict[str, int]
    id_token: dict[int, str]

    @property
    def size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.token_id)

    def id_to_token(self, id_num: int) -> str | None:
        """Find string by giving the id"""
        return self.id_token.get(id_num)

    def token_to_id(self, token: str) -> int | None:
        """Find the id by giving the string"""
        return self.token_id.get(token)

    def token_bytes(self, id_num: int) -> bytes:
        """Turn one token id into its raw bytes (costume string -> bytes)."""
        token = self.id_token[id_num]
        return bytes(_BYTE_TABLE[char] for char in token)

    def num_tokens(self) -> set[int]:
        """Ids of the single-character tokens legal inside a number."""
        chars = "-.0123456789"
        num_ids = set()

        for char in chars:
            token_id = self.token_to_id(char)
            if token_id is not None:
                num_ids.add(token_id)
        return num_ids
    
    def stop_tokens(self) -> set[int]:
        """Ids of the tokens that end number value."""
        end_values = [",", "}", "]", "Ġ", "Ċ", "}}"]
        end_ids = set()
        for end_value in end_values:
            token_id = self.token_to_id(end_value)
            if token_id is not None:
                end_ids.add(token_id)
        return end_ids
