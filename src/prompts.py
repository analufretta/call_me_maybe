"""Prompt text fed to the model before each constrained decoding walk."""

from src.models import FunctionDef, TestPrompt


def prompt_function(prompt: TestPrompt, functions: list[FunctionDef]) -> str:
    """Build the model input for choosing a function."""
    listing = "\n".join(f"- {f.name}: {f.description}" for f in functions)
    return (
        f"Available functions:\n{listing}\n"
        f"User request: {prompt.prompt}\n"
        f'{{"name": "'
    )


RULES = (
    "Fill in the parameters of the function from the user request.\n"
    "Copy values exactly as written in the request. "
    "Do not compute the answer.\n\n"
)

EXAMPLE = (
    "Function: fn_demo\n"
    "Description: Demo function.\n"
    "Parameters: count (integer), price (number), label (string), "
    "flag (boolean)\n"
    'User request: Add 3 boxes called "red box" at 4.5 each, enabled\n'
    '{"name": "fn_demo", "parameters": {"count": 3, "price": 4.5, '
    '"label": "red box", "flag": true}}\n\n'
)


def prompt_output(function: FunctionDef, prompt: TestPrompt) -> str:
    """Build the model input for filling the chosen function's arguments."""
    param_list = ", ".join(
        f"{param} ({spec.type})" for param, spec in function.parameters.items()
    )
    return (
        f"{RULES}"
        f"{EXAMPLE}"
        f"Now the real request:\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters: {param_list}\n"
        f"User request: {prompt.prompt}\n"
        f'{{"name": "{function.name}", "parameters": {{'
    )
