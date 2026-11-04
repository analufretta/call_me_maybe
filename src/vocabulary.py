"""Vocabulary: token<->id lookups and
byte-level decoding for the model vocab."""


from functools import cached_property

from pydantic import BaseModel

# TODO: UNDERSTAND IT BETTER
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

    # TODO: WHY DO I HAVE MY OWN DECODE?
    def decode(self, ids: list[int]) -> str:
        """Turn a list of token ids into real text.

        Collects every token's raw bytes FIRST, then decodes the whole run as
        UTF-8 once -- so a token holding only part of a multi-byte character
        never crashes on its own.
        """
        buffer = bytearray()

        for id_num in ids:
            buffer += self.token_bytes(id_num)
        return buffer.decode("utf-8")

