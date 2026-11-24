"""Prompt text fed to the model before each constrained decoding walk."""

from src.models import FunctionDef, TestPrompt


def prompt_function(prompt: TestPrompt, functions: list[FunctionDef]) -> str:
    """Build the model input for choosing a function."""
    listing = "\n".join(f"- {f.name}: {f.description}" for f in functions)
    return (
        "Task: Choose the function that best matches the user request.\n"
        "Rules:\n"
        "1. Choose only a function from the available list.\n"
        "2. Do not invent function names or parameters.\n"
        "3. Output only the function name.\n"
        "4. Match the user's intent, not just individual keywords.\n\n"
        "Examples:\n"
        "User request: Add 12 and 8\n"
        'Function name: "fn_add_numbers"\n\n'
        "User request: Reverse the text 'hello'\n"
        'Function name: "fn_reverse_string"\n\n'
        "Available functions:\n"
        f"{listing}\n\n"
        f"User request: {prompt.prompt}\n"
        'Function name: "'
    )


RULES = (
    "Extract function parameters from the user request.\n"
    "Rules:\n"
    "1. Copy string values exactly.\n"
    "2. Convert written numbers to JSON number types.\n"
    "3. Copy regex patterns exactly, including escape characters.\n"
    "4. Do not calculate, transform, or explain the result.\n"
    "5. Return valid JSON in exactly the shown format.\n\n"
)

EXAMPLE = (
    "Example 1:\n"
    "Function: fn_add_numbers\n"
    "Description: Add two numbers together.\n"
    "Parameters: a (number), b (number)\n"
    "User request: What is the sum of minus seven and twelve?\n"
    '{"name": "fn_add_numbers", "parameters": {"a": -7.0, "b": 12.0}}\n\n'

    "Example 2:\n"
    "Function: fn_add_numbers\n"
    "Description: Add two numbers together.\n"
    "Parameters: a (number), b (number)\n"
    "User request: Add 265 and negative four\n"
    '{"name": "fn_add_numbers", "parameters": {"a": 265.0, "b": -4.0}}\n\n'

    "Example 3:\n"
    "Function: fn_substitute_string_with_regex\n"
    "Description: Replace regex matches in a string.\n"
    "Parameters: source_string (string), regex (string), "
    "replacement (string)\n"
    'User request: Replace all vowels in "Programming is fun]" with '
    "asterisks\n"
    '{"name": "fn_substitute_string_with_regex", '
    '"parameters": {"source_string": "Programming is fun]", '
    '"regex": "[aeiouAEIOU]", "replacement": "*"}}\n\n'
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
