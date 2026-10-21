"""Prompt text fed to the model before each constrained decoding walk."""

from src.models import FunctionDef, TestPrompt


# TODO: IMPROVE PROMPT?
def prompt_function(prompt: TestPrompt, functions: list[FunctionDef]) -> str:
    """Build the model input for choosing a function."""
    names = ", ".join(f.name for f in functions)
    return (
        f"Available functions: {names}\n"
        f"User request: {prompt.prompt}\n"
        f'{{"name": "'
    )


# TODO: IMPROVE PROMPT?
def prompt_output(function: FunctionDef, prompt: TestPrompt) -> str:
    """Build the model input for filling the chosen function's arguments."""
    param_list = ", ".join(
        f"{param} ({spec.type})" for param, spec in function.parameters.items()
    )
    return (
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters: {param_list}\n"
        f"User request: {prompt.prompt}\n"
        f'{{"name": "{function.name}", "parameters": {{'
    )
