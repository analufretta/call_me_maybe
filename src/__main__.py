"""CLI entrypoint: parse arguments, load inputs, run the pipeline."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.load import load_functions, load_prompts


def parse() -> Namespace:
    """Parse command-line arguments and return them as a Namespace."""
    parser = ArgumentParser()
    parser.add_argument("-f", "--functions_definition",
                        type=Path,
                        default=Path("data/input/functions_definition.json"))
    parser.add_argument("-i", "--input", type=Path,
                        default=Path("data/input/function_calling_tests.json"))
    return (parser.parse_args())


def main() -> None:
    """Parse CLI arguments and run the function-calling pipeline."""
    args = parse()
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    print(args)
    print(args.functions_definition)
    for f in functions:
        print(f)
        print()

    for p in prompts:
        print(p)
        print()

    # TODO: for each prompt pick a funcion + extract params
    # TODO: results = results = run_pipeline(functions, prompts)
    # TODO: write_output(args.output, results)


if __name__ == "__main__":
    main()
