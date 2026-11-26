# How to test

Run everything from the repo root.

All results go to `test/output/`. The folder is created on the first run.

## 0. Setup and lint

```sh
make install
make lint
```

## 1. Baseline: sample data

```sh
time make run
time make run INPUT=data/input/test_num.json OUTPUT=test/output/test_num_results.json
time make run INPUT=data/input/test_str.json OUTPUT=test/output/test_str_results.json
```

Check: `time` stays under 5 minutes, and every prompt has a correct function and arguments.

## 2. New function set

```sh
time make run FUNCTIONS=test/input/test_function_definitions.json \
              INPUT=test/input/test_function_calling.json \
              OUTPUT=test/output/test_function_calling_results.json
```

Check: right function chosen (`count_words` vs `count_characters`),
`integer` params are ints (no `.0`), `boolean` params are `true`/`false`,
and the unrelated "weather" prompt doesn't crash.

## 3. Loader edge cases: each must print `error: ...` and exit, no traceback

Missing file and wrong-type paths:

```sh
make run FUNCTIONS=does/not/exist.json
make run INPUT=does/not/exist.json
make run FUNCTIONS=test/test_edge_cases/a_directory.json
make run INPUT=test/test_edge_cases/a_directory.json
```

Broken files (each one, as both FUNCTIONS and INPUT):

```sh
for f in empty_file not_json malformed_trailing_comma malformed_unclosed \
         malformed_single_quotes invalid_utf8 utf8_bom; do
  echo "== FUNCTIONS=$f"; make run FUNCTIONS=test/test_edge_cases/$f.json
  echo "== INPUT=$f";     make run INPUT=test/test_edge_cases/$f.json
done
```

Bad function definitions:

```sh
for f in fn_empty_list fn_not_a_list fn_item_not_object fn_missing_name \
         fn_missing_description fn_missing_parameters fn_missing_returns \
         fn_param_missing_type fn_parameters_as_list fn_wrong_field_types \
         fn_unsupported_type fn_type_wrong_case fn_duplicate_names \
         fn_empty_names; do
  echo "== $f"; make run FUNCTIONS=test/test_edge_cases/$f.json
done
```

Bad prompt files:

```sh
for f in prompts_not_a_list prompts_plain_strings prompts_missing_key \
         prompts_wrong_value_type; do
  echo "== $f"; make run INPUT=test/test_edge_cases/$f.json
done
```

## 4. Pipeline edge cases: must run and give valid output

```sh
make run INPUT=test/test_edge_cases/prompts_empty_list.json      OUTPUT=test/output/empty_list.json   # expect []
make run INPUT=test/test_edge_cases/prompts_extra_keys.json      OUTPUT=test/output/extra_keys.json
make run INPUT=test/test_edge_cases/prompts_tricky_content.json  OUTPUT=test/output/tricky.json
time make run INPUT=test/test_edge_cases/prompts_very_long.json  OUTPUT=test/output/very_long.json

make run FUNCTIONS=test/test_edge_cases/fn_no_parameters.json  INPUT=test/test_edge_cases/prompts_special_functions.json OUTPUT=test/output/no_params.json
make run FUNCTIONS=test/test_edge_cases/fn_many_params.json    INPUT=test/test_edge_cases/prompts_special_functions.json OUTPUT=test/output/many_params.json
make run FUNCTIONS=test/test_edge_cases/fn_weird_names.json    INPUT=test/test_edge_cases/prompts_special_functions.json OUTPUT=test/output/weird_names.json
make run FUNCTIONS=test/test_edge_cases/fn_single_function.json                                        OUTPUT=test/output/single_fn.json
make run FUNCTIONS=test/test_edge_cases/fn_extra_fields.json                                           OUTPUT=test/output/extra_fields.json
make run FUNCTIONS=test/test_edge_cases/fn_prefix_names.json   INPUT=test/test_edge_cases/prompts_prefix_names.json      OUTPUT=test/output/prefix_names.json
```

Check `prefix_names.json`: each prompt gets its own function. Before the
closing-quote fix in `allowed_fntokens`, `fn_get_user`, `fn_is_prime` and
`fn_add` could never be chosen (their tokens are the first tokens of the
longer name).

## 5. Output path edge cases

```sh
make run OUTPUT=test/output/new/nested/dir/results.json   # parent dirs created
make run OUTPUT=test/output                               # output is a directory: expect error
make run OUTPUT=/results.json                    # no permission: expect error
```

## 6. Check every output file is valid JSON with the right keys

```sh
for f in test/output/*.json data/output/*.json; do
  python3 -c "
import json, sys
d = json.load(open('$f'))
assert isinstance(d, list), 'not a list'
for o in d:
    assert set(o) == {'prompt', 'name', 'parameters'}, o
print('ok  ', '$f', len(d), 'results')
" || echo "FAIL $f"
done
```

Then compare the number of results with the number of prompts. If any are
missing, prompts were skipped (look for `ERROR! skipped` in the run's stderr).

## 7. Fresh clone

```sh
cd "$(mktemp -d)" && git clone <repo-url> cmm && cd cmm
make install && make run && make lint
```
