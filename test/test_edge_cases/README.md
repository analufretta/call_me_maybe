# Edge cases

Run one case by overriding the Makefile paths, e.g.:

```sh
make run FUNCTIONS=test/test_edge_cases/fn_missing_name.json
make run INPUT=test/test_edge_cases/prompts_tricky_content.json
make run FUNCTIONS=test/test_edge_cases/fn_no_parameters.json \
         INPUT=test/test_edge_cases/prompts_special_functions.json
make run FUNCTIONS=does/not/exist.json      # missing file
```

`fn_*` files go to `FUNCTIONS`, `prompts_*` files to `INPUT`, the rest work for either.

## Rejected at load (clean `error:` exit)

| File | Case |
|---|---|
| *(nonexistent path)* | missing file |
| `a_directory.json` | path is a directory |
| `empty_file.json` | 0-byte file |
| `not_json.json` | plain text |
| `malformed_trailing_comma.json` | trailing comma |
| `malformed_unclosed.json` | truncated JSON |
| `malformed_single_quotes.json` | single-quoted keys |
| `invalid_utf8.json` | latin-1 byte, not UTF-8 |
| `utf8_bom.json` | UTF-8 BOM prefix |
| `fn_empty_list.json` | `[]` — no functions |
| `fn_not_a_list.json` | single object instead of list |
| `fn_item_not_object.json` | list items are string/int |
| `fn_missing_name.json` | no `name` |
| `fn_missing_description.json` | no `description` |
| `fn_missing_parameters.json` | no `parameters` |
| `fn_missing_returns.json` | no `returns` |
| `fn_param_missing_type.json` | param `{}` without `type` |
| `fn_parameters_as_list.json` | `parameters` is a list |
| `fn_wrong_field_types.json` | `name` int, `description` null, `returns` string |
| `fn_unsupported_type.json` | `type: "array"` |
| `fn_type_wrong_case.json` | `"Number"` / `"NUMBER"` |
| `fn_duplicate_names.json` | two functions share a name |
| `fn_empty_names.json` | empty / blank function name, empty param name |
| `prompts_not_a_list.json` | single object instead of list |
| `prompts_plain_strings.json` | list of strings, not `{"prompt": ...}` |
| `prompts_missing_key.json` | `text` instead of `prompt` |
| `prompts_wrong_value_type.json` | `prompt` is int / null / list |

## Accepted at load — check pipeline behaviour

| File | Case | What to check |
|---|---|---|
| `fn_no_parameters.json` | function with `parameters: {}` | output `"parameters": {}` is valid JSON |
| `fn_single_function.json` | only one function | unrelated prompts still forced to it |
| `fn_many_params.json` | 8 mixed-type params | all keys present, types correct |
| `fn_extra_fields.json` | unknown keys (`default`, `version`) | ignored, no crash |
| `fn_weird_names.json` | quotes/braces/spaces in names | output JSON still escaped and valid |
| `fn_prefix_names.json` | name whose tokens start another name (`fn_get_user` / `fn_get_user_by_id`) | with `prompts_prefix_names.json`: short names must be selectable |
| `prompts_empty_list.json` | `[]` | output file is `[]` |
| `prompts_extra_keys.json` | extra keys per prompt | ignored |
| `prompts_tricky_content.json` | empty/blank prompt, injection, unknown function, unrelated request, two requests in one, newlines/tabs, JSON in prompt, NaN/Infinity, 1e308, hex/binary, locale separators, too few/many args, other language | no crash; every output is schema-valid |
| `prompts_very_long.json` | ~20k-char prompt | context length / speed |
| `prompts_prefix_names.json` | one prompt per short and long name | pair with `fn_prefix_names.json` |
| `prompts_special_functions.json` | prompts for `fn_no_parameters.json`, `fn_many_params.json`, `fn_weird_names.json` | pair with those definition files |
