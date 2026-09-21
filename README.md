# call_me_maybe
A function-calling engine for a small 0.6B LLM. Uses token-level constrained decoding (logit masking driven by the model's vocabulary) to guarantee 100% valid, schema-compliant JSON output, and lets the LLM itself choose the function and extract typed arguments.
