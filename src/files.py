"""File I/O: load and validate the JSON inputs, write the results file."""

import json
import os
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
    except (OSError, UnicodeDecodeError):
        raise SystemExit(f"ERROR: file ``{path}`` is invalid.\n\n")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit(f"ERROR: invalid JSON in ``{path}``.\n\n")
    try:
        functions = _FUNCTIONS.validate_python(data)
    except ValidationError:
        raise SystemExit(f"ERROR: bad structure in ``{path}``.\n\n")
    if not functions:
        raise SystemExit(f"ERROR: no functions defined in ``{path}``")
    names = [f.name for f in functions]
    duplicates = sorted(({n for n in names if names.count(n) > 1}))
    if duplicates:
        raise SystemExit("ERROR: duplicate function names in"
                         f" ``{path}``: {duplicates}")
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
    except (OSError, UnicodeDecodeError):
        raise SystemExit(f"ERROR: file ``{path}`` is invalid.\n\n")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit(f"ERROR: invalid JSON in ``{path}``.\n\n")
    try:
        token_id = _VOCAB.validate_python(data)
        id_token = {i: tok for tok, i in token_id.items()}
        return Vocabulary(token_id=token_id, id_token=id_token)
    except ValidationError:
        raise SystemExit(f"ERROR: bad structure in ``{path}``.\n\n")


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
    except (OSError, UnicodeDecodeError):
        raise SystemExit(f"ERROR: file ``{path}`` is invalid.\n\n")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit(f"ERROR: invalid JSON in ``{path}``.\n\n")
    try:
        return _PROMPTS.validate_python(data)
    except ValidationError:
        raise SystemExit(f"ERROR: bad structure in ``{path}``.\n\n")


def check_output(path: Path) -> None:
    """Exit early if ``path`` cannot be written.

    Runs before the model loads, so a bad output path does not throw away
    minutes of generation at the very end. Creates the parent directory if
    missing, as ``write_output`` would.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise SystemExit(f"ERROR: cannot write output to ``{path}``.\n\n")
    target = path if path.exists() else path.parent
    if path.is_dir() or not os.access(target, os.W_OK):
        raise SystemExit(f"ERROR: cannot write output to ``{path}``.\n\n")


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
    except OSError:
        raise SystemExit(f"ERROR: cannot write output to ``{path}``.\n\n")
