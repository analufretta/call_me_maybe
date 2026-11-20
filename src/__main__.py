"""CLI entrypoint: parse arguments, load inputs, run the pipeline."""

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
    results: list[dict] = []

    args = parse()
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    model = Model(device="cpu")
    vocab = load_vocab(Path(model.get_path_to_vocab_file()))

    for prompt in prompts:
        try:
            function = pick_function(prompt, functions, model)
            obj = generate_output(prompt, function, model, vocab) # TODO: finish up
            results.append(obj)
            print(f'\nCHECK RESULT OBJECT: {obj}\n')
        except Exception as e:
            print(f"ERROR! skipped {prompt.prompt!r}: {e}")
    write_output(args.output, results)


if __name__ == "__main__":
    main()
