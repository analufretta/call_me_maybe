"""Pydantic models describing the functions_definition.json input file."""

from pydantic import BaseModel


class ParamSpec(BaseModel):
    """Represent one typed slot, e.g. ``{"type": "number"}``.

    Shared by function parameters and the return declaration, since both
    have the same single-field shape.
    """

    type: str


class FunctionDef(BaseModel):
    """Describe one callable function from functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, ParamSpec]
    returns: ParamSpec


class TestPrompt(BaseModel):
    """Represent one prompt from function_calling_tests.json."""

    prompt: str
