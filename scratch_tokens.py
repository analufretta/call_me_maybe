"""Scratch: watch the tokenizer + model turn text into IDs and logits.

Throwaway learning script. Not part of src/. Run:
    uv run python scratch_tokens.py
"""

import json
import numpy as np
import time

from llm_sdk import Small_LLM_Model

MAX_TOKENS = 30


def main() -> None:
    model = Small_LLM_Model()          # first run downloads Qwen3-0.6B (~1.2GB)

    text = (
        "<|im_start|>user\n"
        "You have a function: add(a: int, b: int). "
        'Respond ONLY with JSON like {"name": "...", "arguments": {...}}. '
        "What is the sum of 2 and 3?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # 1. text -> token IDs
    ids = model.encode(text)
    print("text      :", text)
    print("token IDs :", ids[0].tolist())
    print("num tokens:", ids.shape[-1])
    print()

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocab = json.load(f)
    inv_vocab = {i: t for t, i in vocab.items()}

    for tok_id in ids[0].tolist():
        costumed = inv_vocab.get(tok_id)
        decoded = model.decode([tok_id])
        print(f"{tok_id:<7} {costumed!r:<10} {decoded!r}")
    print()

    # 2. greedy generation loop (argmax, feed each token back in)
    eos_token_id = int(model.encode("<|im_end|>")[0, -1])   # Qwen3 chat EOS = 151645
    generated = ids[0].tolist()                             # flat list[int], grows each step
        
    total = 0
    for step in range(MAX_TOKENS):
        t0 = time.perf_counter()
        logits = model.get_logits_from_input_ids(generated)
        nxt_id = int(np.argmax(logits))
        dt = time.perf_counter() - t0
        total += dt
        generated.append(nxt_id)
        print(f"step {step:>3} | {dt:5.2f}s | id {nxt_id:>6} | {model.decode([nxt_id])!r}")
        if nxt_id == eos_token_id:
            break

    print()
    new_ids = generated[ids.shape[-1]:]        # drop prompt, keep model output
    print("generated:", model.decode(new_ids))
    n_calls = len(new_ids)
    print(f"\ncalls: {n_calls} | total: {total:.1f}s | avg: {total/n_calls:.2f}s/call")
    print(f"projected 40-prompt eval: {total*40/60:.1f} min  (budget < 5 min)")
    

    # 3. peek inside the vocab file (the costumed dictionary)
    print("vocab file:", vocab_path)
    with open(vocab_path) as f:
        vocab = json.load(f)
    print("vocab size:", len(vocab))
    print("first 5 tokens:", list(vocab.items())[:5])   # spot the Ġ costumes


if __name__ == "__main__":
    main()
