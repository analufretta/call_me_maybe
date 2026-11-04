"""File I/O: load and validate the JSON inputs, write the results file."""

import json
from pathlib import Path

from pydantic import ValidationError

from src.models import FunctionDef, TestPrompt
from src.vocabulary import Vocabulary


def load_functions(path: Path) -> list[FunctionDef]:
    """Read, parse and validate the functions-definition file.

    Runs three steps, each with its own error:
      1. read the file to text          (FileNotFoundError if missing)
      2. parse the text as JSON         (JSONDecodeError if malformed)
      3. validate each object           (ValidationError if wrong shape)

    Step 3 uses ``FunctionDef(**f)``: the ``**`` unpacks each dict ``f``
    into keyword arguments, so ``{"name": ..., "description": ...}`` becomes
    ``FunctionDef(name=..., description=...)``. Pydantic then checks the
    types; an unexpected or missing key raises ValidationError.
    """
    try:
        raw = path.read_text()
    except FileNotFoundError:
        raise SystemExit(f"error: file not found: {path}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: invalid JSON in {path}: {e}")
    try:
        return [FunctionDef(**f) for f in data]
    except ValidationError as e:
        raise SystemExit(f"error: bad structure in {path}: {e}")


def load_vocab(path: Path) -> Vocabulary:
    """Read, parse and build the token maps from the model vocab file.

    Same three guards as load_functions:
      1. read the file to text    (FileNotFoundError if missing)
      2. parse the text as JSON   (JSONDecodeError if malformed)
      3. build the maps           (ValidationError if wrong shape)

    vocab.json shape is ``{"token_string": id, ...}``. We keep that as
    token_to_id and invert it to id_to_token.
    """
    try:
        raw = path.read_text()
    except FileNotFoundError:
        raise SystemExit(f"error: file not found: {path}")
    try:
        token_id = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: invalid JSON in {path}: {e}")
    try:
        id_token = {i: tok for tok, i in token_id.items()}
        return Vocabulary(token_id=token_id, id_token=id_token)
    except ValidationError as e:
        raise SystemExit(f"error: bad structure in {path}: {e}")


def load_prompts(path: Path) -> list[TestPrompt]:
    """Read, parse and validate the prompts file.

    Runs the same three steps as load_functions:
      1. read the file to text          (FileNotFoundError if missing)
      2. parse the text as JSON         (JSONDecodeError if malformed)
      3. validate each object           (ValidationError if wrong shape)

    Step 3 uses ``TestPrompt(**p)``: the ``**`` unpacks each dict ``p``
    into keyword arguments, so ``{"prompt": ...}`` becomes
    ``TestPrompt(prompt=...)``. Pydantic then checks the types; an
    unexpected or missing key raises ValidationError.
    """
    try:
        raw = path.read_text()
    except FileNotFoundError:
        raise SystemExit(f"error: file not found: {path}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: invalid JSON in {path}: {e}")
    try:
        return [TestPrompt(**p) for p in data]
    except ValidationError as e:
        raise SystemExit(f"error: bad structure in {path}: {e}")


def write_output(path: Path, results: list[dict]) -> None:
    """Write the result objects to ``path`` as a single JSON array.

    Creates the parent directory if missing, then dumps the list with
    indentation. Each object's numbers are already floats (V.4.1); this just
    serialises what it is given.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2) # TODO: WHY IDENT=2?
