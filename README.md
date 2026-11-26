*This project has been created as part of the 42 curriculum by afretta-.*

# call_me_maybe

A function-calling engine for the small **Qwen/Qwen3-0.6B** LLM. It turns a
natural-language request into a structured function call, using token-level
constrained decoding (logit masking driven by the model's vocabulary) so the
output is always valid, schema-compliant JSON.

## Description

LLMs are good at producing human language and bad at producing exact,
machine-readable output, especially small ones. Asked to "answer in JSON", a
0.6B model gets it right only some of the time. Function calling fixes this:
the model does not answer the question, it names the **tool** that answers it
and fills in the arguments with the right types.

```
Request:  "What is the sum of 40 and 2?"
Output:   {"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

The program reads two JSON files:

- **function definitions**: name, description, typed parameters and return
  type of every available function;
- **prompts**: a list of natural-language requests.

For each prompt it lets the LLM pick a function and extract its arguments,
then writes one JSON array of `{"prompt", "name", "parameters"}` objects.

The model never generates free-form JSON. The program writes the structure
itself and only asks the model to choose between legal options: which
function name, which digit, `true` or `false`. The function is always chosen
by the LLM, never by keyword matching.

Constraints from the subject: Python 3.10+, `flake8` and `mypy` clean,
`pydantic` for all classes, only `numpy`, `json` and `pydantic` as direct
dependencies (no torch, transformers, outlines, …), and only the public
methods of the provided `llm_sdk`.

## Instructions

### Requirements

- Python 3.12 or 3.13 (`.python-version` selects 3.13.1)
- [`uv`](https://docs.astral.sh/uv/); `make install` installs it if missing
- About 2 GB of free disk space for the model weights, downloaded on the
  first run

### Install

```bash
make install        # installs uv if needed, then runs `uv sync --locked`
```

Running `uv sync` alone also works. That is what the reviewer and the
moulinette do.

### Run

```bash
make run            # default input/output paths
```

or call the program directly:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

All three arguments are optional. The defaults are the paths shown above.
The Makefile accepts the same paths as variables:

```bash
make run FUNCTIONS=test/input/test_function_definitions.json \
         INPUT=test/input/test_function_calling.json \
         OUTPUT=test/output/results.json
```

### Other Makefile rules

| Rule | What it does |
|---|---|
| `make debug` | runs the program under `pdb` |
| `make lint` | `flake8` + `mypy` with the flags required by the subject |
| `make lint-strict` | `flake8` + `mypy --strict` |
| `make clean` | removes `__pycache__`, `.mypy_cache`, `*.pyc` |
| `make fclean` | `clean` + removes `.venv` |
| `make re` | `fclean` + `install` |

## Example usage

Input `data/input/functions_definition.json` (excerpt):

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {"s": {"type": "string"}},
    "returns": {"type": "string"}
  }
]
```

Input `data/input/function_calling_tests.json` (excerpt):

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"},
  {"prompt": "Replace all vowels in 'Programming is fun, HELLO' with asterisks"}
]
```

Run:

```bash
make run
```

Output `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  },
  {
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "Hello 34 I'm 233 years old",
      "regex": "\\d+",
      "replacement": "NUMBERS"
    }
  }
]
```

The terminal also prints one `RESULT OVERVIEW:` line per prompt while it runs.

Numbers in the prompt do not have to be written as digits:

```text
"What is the sum of minus five and ten?"  ->  {"a": -5.0, "b": 10.0}
"What is the sum of 5/12 and 3/4?"        ->  {"a": 0.4166666666666667, "b": 0.75}
```

Bad input never produces a traceback. The program prints one clear line and
exits. The input files and the output path are all checked before the model
loads, so a typo in `--output` does not waste a full run:

```text
$ make run FUNCTIONS=does/not/exist.json
ERROR: file ``does/not/exist.json`` is invalid.

$ make run FUNCTIONS=test/test_edge_cases/fn_duplicate_names.json
ERROR: duplicate function names in ``test/test_edge_cases/fn_duplicate_names.json``: ['fn_add_numbers']

$ make run OUTPUT=data/input
ERROR: cannot write output to ``data/input``.
```

If one prompt fails during generation, it is skipped with an `ERROR:` line
on stderr, the rest are still processed, and a summary says how many were
left out.

## Algorithm explanation

Every model step goes through one primitive, `constrained_max`
(`src/decoding.py`): get the logits for the current token ids, then take the
highest-scoring token **within an allowed set**. Every other token is ignored,
which is the same as setting its logit to −∞. If the allowed set has a single
token there is nothing to choose, so it is returned without calling the model.

Decoding is greedy (argmax), so the output is deterministic.

### 1. Choosing the function

`pick_function` (`src/generate.py`):

1. The prompt lists every function with its description, then ends with
   `Function name: "`.
2. Each function name is encoded into its token path, **followed by the
   closing-quote token**.
3. The walk goes one token at a time. The allowed set holds the next token of
   every name that still matches the tokens chosen so far. The model picks
   among them.
4. The walk ends when no path continues. The chosen tokens are then exactly
   one full name.

The model can never produce a name that does not exist. It only steers
between valid names. Once the prefix is unique, the rest of the name is
filled in without any model call.

```
names:  fn_add_numbers, fn_greet          (shown as text, really tokens)
step 1  allowed {fn}            -> 1 option, no model call
step 2  allowed {_add, _greet}  -> model picks _add
step 3  allowed {_numbers}      -> 1 option, no model call
step 4  allowed {"}             -> done: fn_add_numbers
```

### 2. Filling the arguments

`generate_output` builds a second prompt: rules, three few-shot examples, the
chosen function and its typed parameters, then the beginning of the answer
`{"name": "<fn>", "parameters": {`. After that it alternates:

- **force**: structural text that is already known (`"key": `, `, `) is
  encoded and appended with no model call;
- **fill**: the model generates one value, constrained by the parameter's
  schema type.

| Type | How the value is generated |
|---|---|
| `boolean` | The allowed set is exactly the `true` and `false` tokens. One model step. |
| `integer` / `number` | First the sign: the tokenizer writes negatives as the single token `Ġ-`, so the model chooses between `Ġ` and `Ġ-`. Then digit by digit, masked by `free_mask`: digits always; `-` never after the start; `.` once, only for `number`, only after a digit; `/` once, only for `number`, only after a whole number (to copy fractions like `1/3`), never followed by `0`; a stop token (`,` `}` space newline) only after a digit. For `integer`, the number also ends when the request writes the same digits followed by a decimal part (`4.5` → `4`). Capped at 50 tokens. |
| `string` | The key is forced as `"key":` without the space. The first token is masked to the tokens that start with ` "` (space + quote), because the tokenizer usually glues the opening quote to the first word (` "hello`). After that the model generates greedily until a token contains an unescaped `"`, which is detected on the token's raw bytes. Capped at 100 tokens. The result is then checked against the quoted texts in the request (see *Challenges*). |

Each value is converted to a Python value (`int`, `float`, `bool`, `str`) and
stored in a dict. The output file is written with `json.dump`, so it is valid
JSON by construction, and each value has the type declared in the schema.

### Working with the vocabulary

`src/vocabulary.py` loads `vocab.json` (through `get_path_to_vocab_file()`)
into two maps, token→id and id→token. Qwen uses byte-level BPE: every byte is
shown as a printable "costume" character (`Ġ` = space, `Ċ` = newline). The
reversed GPT-2 byte table turns any token back into its raw bytes. The masks
(digit tokens, stop tokens, `true`/`false`) are looked up once and cached.

## Design decisions

- **The program writes the structure, the model fills the values.** Keys,
  braces and commas are never left to the model, so the model cannot break
  the JSON shape.
- **The LLM picks the function through a constrained walk over name tokens.**
  No heuristics and no keyword matching, as the subject requires. Restricting
  the walk to real names makes an invalid name impossible.
- **Closing quote at the end of each name path.** Without it, a name whose
  tokens are a prefix of another (`fn_get_user` / `fn_get_user_by_id`) could
  never be chosen, because the walk always went on to the longer name.
- **Numbers: syntax masks, not copying.** The mask only enforces number
  syntax, so the model can still convert `minus five` → `-5`, or `pi` →
  `3.14…`. `number` values become `float` (`2.0`, as in the subject's
  example) and `integer` values become `int`.
- **Fractions are allowed in `number` values.** The model naturally writes
  `1/3`, which is not a JSON number. The mask allows one `/`, and the
  program divides before writing the output.
- **Strings are not masked token by token.** Almost any token can appear in
  a string, and masking would hurt the model's copying. Only the first token
  is masked, so that the string opens with a quote. The string then ends at
  the first unescaped `"` in the generated bytes, and the final value
  is serialised again with `json.dumps`, so escaping is always correct.
- **Pydantic validates whole files.** `TypeAdapter(list[FunctionDef])`
  checks the shape as well as each item, so a dict where a list is expected
  gives a clean error, not an `AttributeError`. Unsupported types, empty
  names and duplicate function names are rejected at load time.
- **Fail early, and only once per prompt.** Bad input files or an
  unwritable output path stop the program with `ERROR: …` before the model
  loads. An exception on a single prompt only skips that prompt.
- **Only the public SDK.** The program uses `encode`, `decode`,
  `get_logits_from_input_ids` and `get_path_to_vocab_file`, and nothing
  private.
- **Dependencies.** `pyproject.toml` lists only what `src/` imports
  (`llm-sdk`, `numpy`, `pydantic`). The torch and transformers versions
  needed by `llm_sdk` are pinned as uv constraints so the lock resolves the
  same way on every machine. On Linux, the CPU-only torch build is used,
  which avoids about 2.5 GB of CUDA libraries the program does not need.

## Performance analysis

Measured on an Intel Core i7-9750H laptop (6 cores, CPU only, no GPU),
before the string-opening and integer-decimal changes described in
*Challenges faced*:

| Input | Prompts | Right function | Right arguments | Time |
|---|---|---|---|---|
| `function_calling_tests.json` (subject sample) | 11 | 11/11 | 10/11 | 3 min 34 s |
| `test_num.json` (hard numbers) | 15 | 15/15 | 14/15 | 8 min |

- **Accuracy.** Function selection was correct on every prompt. The two
  argument misses are typical 0.6B errors, not format errors:
  - *"Replace all vowels in 'Programming is fun, HELLO' with asterisks"*
    gave `"regex": "aeiouAEIOU"` (no `[...]`) and `"replacement": "**"`;
  - *"sum of apples and oranges"* has no real numbers, so the model fell
    back on its few-shot habits and wrote `2.0` and `3.0`.

  Word numbers (`minus five`), constants (`pi`, `e`), fractions (`1/7`,
  `-2/5`), long decimals and distractor numbers (*"In 2024, …"*) were all
  extracted correctly.

- **Valid JSON: 100%.** The file is written by `json.dump` from Python
  values whose types come from the schema, so it always parses. Function
  names come from the constrained walk, so they always exist.
- **Speed.** About 20–30 s per prompt on this laptop, which is slower than
  the subject's 5-minute target for larger sets. Each model step costs one
  full forward pass over the whole prompt (the public SDK has no KV cache).
  Long numbers and strings mean more steps, which is why `test_num.json` is
  the slowest set. Several things already cut the number of steps: forced
  scaffold, single-option shortcuts (unique name tails), and booleans
  decided in one step. A faster CPU, or caching the shared prompt prefix,
  would be the next things to try.
- **Reliability.** Invalid input files exit cleanly with a message. Runaway
  values are capped (50 tokens for numbers, 100 for strings) and only skip
  their prompt.
- **Known limits.** Argument accuracy is bounded by the 0.6B model's
  understanding: a request with no real value (`"sum of apples and
  oranges"`) still gets *some* valid number, and an unrelated request is
  still mapped to the closest function. A string longer than 100 tokens is
  rejected.

## Challenges faced

| Problem | Fix |
|---|---|
| **Prefix names could not be chosen.** `fn_get_user`'s tokens are the start of `fn_get_user_by_id`, so the walk always went on to the longer name. | Each name path ends with the closing-quote token, so "stop here" is a choice the model can make. |
| **Negative numbers came out positive.** The scaffold ends in a space token, but the tokenizer writes `-5` as `Ġ-` + `5`, so a bare `-` after a space looked unnatural and the model avoided it. | `choose_sign` removes the trailing space and asks the model to choose between `Ġ` and `Ġ-`. |
| **Fractions were cut short.** For `5/12` the model often stopped after `5`, because a JSON number never contains `/`. | `/` was added to the number mask, and `continues_fraction` forces the `/` when the request writes those digits as a fraction. |
| **Integers never stopped on decimals.** The integer mask forbids `.`, so for *"Is 4.5 prime?"* the model kept adding digits after `4`. | `ends_at_decimal` ends the integer when the request writes those digits followed by `.` and a digit. |
| **A forced lone `"` looked unnatural.** The tokenizer writes `: "hello` as `:` + ` "hello`, with the space, quote and first word in one token. A separately forced `"` pushed the model off its usual token path. | Strings are forced as `"key":` without the space, and the first token is chosen among the tokens that start with ` "`. The bytes after the quote count as the first part of the content. |
| **Strings closed too early.** In `'Programming is fun]'`, the tokenizer glues `]` and the quote into one token, so the model closed the string before the `]`. | `double_check` compares the value with the quoted texts in the request. If it is the start of one, it is completed to the full quoted text. |
| **Inconsistent escaping.** The model sometimes escapes characters (`\\d`) and sometimes not. | The value is compared both as written and JSON-unescaped, and the form that matches the request is kept. |
| **Reading token bytes.** Tokens like `Ġ` or multi-byte UTF-8 pieces are not plain text. | A reversed GPT-2 byte table decodes tokens to raw bytes. The closing quote is searched in bytes, and the content is decoded as UTF-8 once, at the end. |
| **Heavy install on Linux.** The default PyPI torch pulls in about 2.5 GB of CUDA libraries, though the model runs on CPU. | uv constraints pin the torch and transformers versions, and on Linux torch comes from the CPU-only PyTorch index. |
| **mypy and `llm_sdk`.** The top-level `llm_sdk/` folder shadowed the installed package, it has no `py.typed` file, and following it made mypy load all of torch and transformers (minutes per run). | `namespace_packages = false`, `llm_sdk/` excluded, and `ignore_missing_imports` for `llm_sdk`, so it is treated as untyped. |

## Testing strategy

The full guide with every command is in [`test/TESTING.md`](test/TESTING.md).
In short:

1. **Lint.** `make lint` and `make lint-strict` (flake8 + mypy) must pass.
2. **Sample data.** `make run` on the subject's prompts, plus
   `data/input/test_num.json` (negatives, word numbers, fractions, huge
   numbers, distractor numbers like *"In 2024, …"*) and
   `data/input/test_str.json` (quotes, regex, escapes). Each run is timed
   with `time` to check the 5-minute limit.
3. **A different function set.** `test/input/` has new functions with
   `integer` and `boolean` parameters and look-alike names
   (`fn_count_words` / `fn_count_characters`), so nothing depends on the provided
   examples.
4. **Loader edge cases.** `test/test_edge_cases/` holds over 30 broken or
   unusual files: missing file, directory, empty file, malformed JSON,
   invalid UTF-8, wrong shapes, missing keys, unsupported or wrongly cased
   types, duplicate or empty names. Each must exit with a clean `ERROR:`
   line and no traceback. The expected outcome of each file is listed in
   [`test/test_edge_cases/README.md`](test/test_edge_cases/README.md).
5. **Pipeline edge cases.** Functions with no parameters or eight
   parameters, names with quotes/braces/spaces, prefix names, prompt
   injection, empty prompts, ~20k-character prompts, other languages.
   These must run and produce schema-valid output.
6. **Output check.** A short script confirms that every output file is a
   JSON list of objects with exactly the keys `prompt`, `name` and
   `parameters`, and that no prompt was dropped.
7. **Fresh clone.** `git clone` → `make install` → `make run` →
   `make lint`, to check that the lock file installs from scratch.

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): the model used.
- [Hugging Face: Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary): BPE and byte-level BPE.
- [OpenAI GPT-2 `encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py): the original `bytes_to_unicode` table, reversed in `src/vocabulary.py`.
- [Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023)](https://arxiv.org/abs/2307.09702): constrained decoding as a state machine over the vocabulary.
- [Hugging Face blog: How to generate text](https://huggingface.co/blog/how-to-generate): greedy search, sampling and logits.
- [OpenAI: Function calling guide](https://platform.openai.com/docs/guides/function-calling): what function calling looks like in production.
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259): what counts as a valid JSON number and string.
- [Pydantic documentation](https://docs.pydantic.dev/latest/): models, validators and `TypeAdapter`.
- [uv documentation](https://docs.astral.sh/uv/): workspaces, constraint dependencies, custom indexes.

### AI usage

AI (Claude, through Claude Code) was used as an assistant, never as a
replacement for understanding. Everything it produced was reviewed, tested
and is understood by the author.

- **This README.** It was drafted with AI from the subject and the code, then
  reviewed and corrected by the author.
- **Concepts and explanations.** Learning how tokenization, byte-level BPE
  (the `Ġ`/`Ċ` costume characters), logits and constrained decoding work,
  before designing the decoder.
- **Tests and edge cases.** Generating the edge-case input files in
  `test/test_edge_cases/`, the new function set in `test/input/`, and the
  `test/TESTING.md` guide.
- **Debugging and review.** Finding the cause of specific bugs (for example
  the prefix-name selection problem and the early-closing strings), reviewing
  diffs, and setting up the `flake8`/`mypy`/`uv` configuration
  (`pyproject.toml`, `.flake8`).
