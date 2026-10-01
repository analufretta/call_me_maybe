"""Scratch: watch the tokenizer + model turn text into IDs and logits.

Throwaway learning script. Not part of src/. Run:
    uv run python scratch_tokens.py
"""

import json
import numpy as np
import time

from llm_sdk import Small_LLM_Model

MAX_TOKENS = 30

def bytes_to_unicode() -> dict[str, int]:
    """GPT-2 / Qwen byte-level BPE table, reversed: costume-char -> raw byte.

    Training never stores raw bytes 0-255 directly (some are control chars /
    spaces that break text files). Each byte is dressed in a printable 'costume'
    char -- that's what you see inside vocab.json (the Ġ etc.). To recover the
    real bytes a token stands for, undo this mapping.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:          # byte with no printable glyph -> give it a costume
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}   # costume char -> byte


def demo_multibyte(model) -> None:
    """Show one character split across several token ids (the multi-byte trap)."""
    char2byte = bytes_to_unicode()

    with open(model.get_path_to_vocab_file()) as f:
        vocab = json.load(f)                       # token_string -> id
    inv_vocab = {i: t for t, i in vocab.items()}   # id -> token_string (costume)

    text = "café 😀 日本"      # accent (2B), emoji (4B), CJK (3B each)
    ids = model.encode(text)[0].tolist()
    print("\n=== multi-byte demo ===")
    print("text:", text, "\n")
    print("ids :", ids, "\n")

    run = bytearray()
    for tid in ids:
        costume = inv_vocab[tid]
        print(costume)
        raw = bytes(char2byte[ch] for ch in costume)   # costume -> real bytes
        print(raw)
        run += raw
        print(run)
        try:
            solo = repr(raw.decode("utf-8"))           # decode THIS token alone
        except UnicodeDecodeError:
            solo = "<<partial char — invalid UTF-8 alone>>"
        print(f"id {tid:<7} costume {costume!r:<8} bytes {list(raw)!s:<18} solo={solo}")

    print("\njoined bytes decoded:", run.decode("utf-8"))

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
    print()

    demo_multibyte(model)


if __name__ == "__main__":
    main()
