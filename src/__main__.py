"""CLI entrypoint: parse arguments, load inputs, run the pipeline."""

import sys

from argparse import ArgumentParser, Namespace
from pathlib import Path

from llm_sdk import Small_LLM_Model as Model

from src.files import load_functions, load_prompts, load_vocab, write_output
from src.generate import pick_function, generate_output


def parse() -> Namespace:
    """Parse command-line arguments and return them as a Namespace."""
    parser = ArgumentParser()
    parser.add_argument("-f", "--functions_definition",
                        type=Path,
                        default=Path("data/input/functions_definition.json"))
    parser.add_argument("-i", "--input", type=Path,
                        default=Path("data/input/function_calling_tests.json"))
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("data/output/function_calling_results.json"))
    return (parser.parse_args())


def main() -> None:
    """Parse CLI arguments and run the function-calling pipeline."""
    print("\n\n===== CALL ME MAYBE initiated =====\n\n")

    results: list[dict[str, object]] = []

    args = parse()
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    try:
        model = Model(device="cpu")
    except Exception as e:
        raise SystemExit(f"error: cannot load the model: {e}")
    vocab = load_vocab(Path(model.get_path_to_vocab_file()))

    failed = 0
    for prompt in prompts:
        try:
            function = pick_function(prompt, functions, model)
            obj = generate_output(prompt, function, model, vocab)
            results.append(obj)
            print(f'\nRESULT OVERVIEW: {obj}\n')
        except Exception as e:
            failed += 1
            print(f"\nERROR: skipped {prompt.prompt!r}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
    write_output(args.output, results)
    if failed:
        print(f"\n{failed}/{len(prompts)} prompts failed and were left "
              f"out of {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
