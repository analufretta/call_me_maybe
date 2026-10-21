"""Vocabulary: token<->id lookups and
byte-level decoding for the model vocab."""


from functools import cached_property

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

    @cached_property
    def number_pieces(self) -> set[int]:
        """Token ids whose text is only digits, '.', or '-'.

        The mask for ``fill_number``: a token passes only if every one of its
        characters is a legal number character, so the model can never emit a
        letter or space inside a number. Computed once over the vocab.
        """
        legal = set("0123456789.-")
        out: set[int] = set()
        for id_num in self.id_token:
            text = self.token_bytes(id_num).decode("utf-8", "ignore")
            if text and all(char in legal for char in text):
                out.add(id_num)
        return out

    @cached_property
    def string_pieces(self) -> set[int]:
        """Token ids safe inside a JSON string: no '"', no control chars.

        The mask for ``fill_string``. A partial multi-byte piece (half of
        ``é`` or ``😀``) can never hold a '"' (0x22 is ASCII, never a UTF-8
        continuation byte), so it is allowed. The standalone '"' token is
        excluded on purpose -- it is how the string closes.
        """
        out: set[int] = set()
        for id_num in self.id_token:
            try:
                text = self.token_bytes(id_num).decode("utf-8")
            except UnicodeDecodeError:
                out.add(id_num)
                continue
            if '"' not in text and not any(ord(c) < 0x20 for c in text):
                out.add(id_num)
        return out

    @cached_property
    def string_stops(self) -> set[int]:
        """Token ids that close a JSON string: any token starting with '"'.

        A string's stop is the closing quote. The model's trained ender is
        usually a MERGED token (`",`  `":`  `","`), all excluded from
        string_pieces -- so without this set the model can never pick its
        natural close and rambles to the cap. Content tokens never contain a
        quote, so every quote-first token here is a legit terminator.
        """
        out: set[int] = set()
        for id_num in self.id_token:
            text = self.token_bytes(id_num).decode("utf-8", "ignore")
            if text.startswith('"'):
                out.add(id_num)
        return out

    @cached_property
    def value_stops(self) -> set[int]:
        """Token ids that end a number: any token starting with ',' or '}'.

        Like ``string_stops``: the model's natural ender is often a MERGED
        token (``,"``  ``}}``  ``},``), so every comma- or brace-first token
        counts. When ``fill_number`` picks one, the number is done.
        """
        out: set[int] = set()
        for id_num in self.id_token:
            text = self.token_bytes(id_num).decode("utf-8", "ignore")
            if text.startswith((",", "}")):
                out.add(id_num)
        return out
