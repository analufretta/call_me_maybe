"""File I/O: load and validate the JSON inputs, write the results file."""

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from src.models import FunctionDef, TestPrompt
from src.vocabulary import Vocabulary

# Validators for the whole file shape, not just each item: a dict where a
# list is expected, or a string where a dict is expected, raises
# ValidationError instead of a TypeError/AttributeError crash.
_FUNCTIONS = TypeAdapter(list[FunctionDef])
_PROMPTS = TypeAdapter(list[TestPrompt])
_VOCAB = TypeAdapter(dict[str, int])


def load_functions(path: Path) -> list[FunctionDef]:
    """Read, parse and validate the functions-definition file.

    Runs three steps, each with its own error:
      1. read the file to text          (OSError if unreadable)
      2. parse the text as JSON         (JSONDecodeError if malformed)
      3. validate the whole list        (ValidationError if wrong shape)

    Step 3 checks that the data is a list and that every item fits
    ``FunctionDef``; a non-list, a non-dict item or a missing key all raise
    ValidationError.
    """
    try:
        raw = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        raise SystemExit(f"error: file {path} is invalid: {e}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: invalid JSON in {path}: {e}")
    try:
        functions = _FUNCTIONS.validate_python(data)
    except ValidationError as e:
        raise SystemExit(f"error: bad structure in {path}: {e}")
    if not functions:
        raise SystemExit(f"error: no functions defined in {path}")
    return functions


def load_vocab(path: Path) -> Vocabulary:
    """Read, parse and build the token maps from the model vocab file.

    Same three guards as load_functions:
      1. read the file to text    (OSError if unreadable)
      2. parse the text as JSON   (JSONDecodeError if malformed)
      3. build the maps           (ValidationError if wrong shape)

    vocab.json shape is ``{"token_string": id, ...}``. We keep that as
    token_to_id and invert it to id_to_token.
    """
    try:
        raw = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        raise SystemExit(f"error: file {path} is invalid: {e}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: invalid JSON in {path}: {e}")
    try:
        token_id = _VOCAB.validate_python(data)
        id_token = {i: tok for tok, i in token_id.items()}
        return Vocabulary(token_id=token_id, id_token=id_token)
    except ValidationError as e:
        raise SystemExit(f"error: bad structure in {path}: {e}")


def load_prompts(path: Path) -> list[TestPrompt]:
    """Read, parse and validate the prompts file.

    Runs the same three steps as load_functions:
      1. read the file to text          (OSError if unreadable)
      2. parse the text as JSON         (JSONDecodeError if malformed)
      3. validate the whole list        (ValidationError if wrong shape)

    Step 3 checks that the data is a list and that every item fits
    ``TestPrompt``; a non-list, a non-dict item or a missing key all raise
    ValidationError.
    """
    try:
        raw = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        raise SystemExit(f"error: file {path} is invalid: {e}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: invalid JSON in {path}: {e}")
    try:
        return _PROMPTS.validate_python(data)
    except ValidationError as e:
        raise SystemExit(f"error: bad structure in {path}: {e}")


def write_output(path: Path, results: list[dict[str, object]]) -> None:
    """Write the result objects to ``path`` as a single JSON array.

    Creates the parent directory if missing, then dumps the list with
    indentation. Each object's numbers are already floats (V.4.1); this just
    serialises what it is given. Exits with a clear message if the file
    cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
    except OSError as e:
        raise SystemExit(f"error: cannot write output to {path}: {e}")
