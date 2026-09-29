"""Scratch: watch the tokenizer + model turn text into IDs and logits.

Throwaway learning script. Not part of src/. Run:
    uv run python scratch_tokens.py
"""

import json
import numpy as np

from llm_sdk import Small_LLM_Model

MAX_TOKENS = 100


def main() -> None:
    model = Small_LLM_Model()          # first run downloads Qwen3-0.6B (~1.2GB)

    text = (
    "<|im_start|>user\n"
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


    # 2. token IDs -> logits (one score per vocab token)
    for _ in range(MAX_TOKENS):
        logits = model.get_logits_from_input_ids(ids[0].tolist())
        nxt_id = int(np.argmax(logits))
        ids.append(nxt_id)
        if nxt_id == eos_token_id:
            break 

    # 3. peek inside the vocab file (the costumed dictionary)
    print("vocab file:", vocab_path)
    with open(vocab_path) as f:
        vocab = json.load(f)
    print("vocab size:", len(vocab))
    print("first 5 tokens:", list(vocab.items())[:5])   # spot the Ġ costumes


if __name__ == "__main__":
    main()
