#Original version:
def fill_str(
    ids: list[int], model: Model, vocab: Vocabulary
) -> str:
    closing_quotes = vocab.quote_tokens
    force(ids, '"', model)
    start = len(ids)
    text = ""
    for _ in range(MAX_STR_TOKENS):
        logits = model.get_logits_from_input_ids(ids)
        tok = max(range(len(logits)), key=lambda t: logits[t])
        if tok in closing_quotes:
            before, _, _ = model.decode([tok]).partition('"')
            if before:
                force(ids, before, model)
            text += before
            break
        ids.append(tok)
    text += model.decode(ids[start:])
    force(ids, '"', model)
    try:
        text = json.loads('"' + text + '"')
    except json.JSONDecodeError:
        pass
    return text

#version 2:
def fill_str(
    ids: list[int], model: Model, vocab: Vocabulary
) -> str:
    force(ids, '"', model)
    content_start = len(ids)
    content_bytes = bytearray()
    escaped = False

    for _ in range(MAX_STR_TOKENS):
        logits = model.get_logits_from_input_ids(ids)
        tok = max(range(len(logits)), key=lambda t: logits[t])
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
            break

        ids.append(tok)
        content_bytes.extend(token_bytes)
    else:
        raise ValueError(
            f"Generated string did not close within {MAX_STR_TOKENS} tokens"
        )

    try:
        content = content_bytes.decode("utf-8")
        value = json.loads(f'"{content}"')
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Model generated an invalid JSON string") from exc

    if not isinstance(value, str):
        raise ValueError("Decoded JSON value is not a string")

    # Replace the generated token sequence with the exact content and delimiter.
    ids[content_start:] = model.encode(content)[0].tolist()
    force(ids, '"', model)

    return value


#version 3:
def extract_quoted_values(text: str) -> list[str]:
    values: list[str] = []
    i = 0

    while i < len(text):
        quote = text[i]
        if quote not in {"'", '"'}:
            i += 1
            continue

        start = i
        i += 1
        chars: list[str] = []

        while i < len(text):
            char = text[i]

            # An escaped matching quote is part of the value.
            if char == "\\" and i + 1 < len(text) and text[i + 1] == quote:
                chars.append(quote)
                i += 2
            elif char == quote:
                values.append("".join(chars))
                i += 1
                break
            else:
                # Preserve other backslashes, e.g. in C:\Users\ana.
                chars.append(char)
                i += 1
        else:
            # Ignore an unmatched quote and keep scanning.
            i = start + 1

    return values

def fill_str(
    ids: list[int],
    model: Model,
    vocab: Vocabulary,
    request: str,
    *,
    copy_if_unambiguous: bool,
) -> str:
    candidates = extract_quoted_values(request)
    if copy_if_unambiguous and len(candidates) == 1:
        value = candidates[0]
        force(ids, json.dumps(value, ensure_ascii=False), model)
        return value

    # Existing model-generated fallback continues here.


string_param_count = sum(
    spec.type == "string" for spec in function.parameters.values()
)


# Pass the request and the “one string parameter” condition through fill from src/generate.py:
# Inside the parameter loop:
params[param] = fill(
    ids,
    model,
    vocab,
    ptype.type,
    request=prompt.prompt,
    copy_if_unambiguous=(string_param_count == 1),
)