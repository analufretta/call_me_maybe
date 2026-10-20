"""CLI entrypoint: parse arguments, load inputs, run the pipeline."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.load import load_functions, load_prompts, load_vocab
from src.generate import pick_function, generate_output, write_output


def parse() -> Namespace:
    """Parse command-line arguments and return them as a Namespace."""
    parser = ArgumentParser()
    parser.add_argument("-f", "--functions_definition",
                        type=Path,
                        default=Path("data/input/functions_definition.json"))
    parser.add_argument("-i", "--input", type=Path,
                        default=Path("data/input/function_calling_tests.json"))
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("data/output/function_calling_results.json"))
    return (parser.parse_args())


def main() -> None:
    """Parse CLI arguments and run the function-calling pipeline."""
    results: list[dict] = []


    args = parse()
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    model = Small_LLM_Model()
    vocab = load_vocab(Path(model.get_path_to_vocab_file()))

    for prompt in prompts:
        try:
            function = pick_function(prompt, functions, model, vocab)
            obj = generate_output(prompt, function, model, vocab)
            results.append(obj)
            print(f"\n{obj}\n")
        except Exception as e:
            print(f"skip {prompt.prompt!r}: {e}")
    write_output(args.output, results)


if __name__ == "__main__":
    main()
