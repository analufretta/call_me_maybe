"""Pydantic models describing the functions_definition.json input file."""

from pydantic import BaseModel, field_validator

SUPPORTED = {"number", "integer", "string", "boolean"}


class ParamSpec(BaseModel):
    """Represent one typed slot, e.g. ``{"type": "number"}``.

    Shared by function parameters and the return declaration, since both
    have the same single-field shape.
    """

    type: str

    @field_validator("type")
    @classmethod
    def known_type(cls, value: str) -> str:
        if value not in SUPPORTED:
            raise ValueError(
                f"unsupported type {value!r}, "
                f"expected one of {sorted(SUPPORTED)}"
            )
        return value


class FunctionDef(BaseModel):
    """Describe one callable function from functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, ParamSpec]
    returns: ParamSpec


class TestPrompt(BaseModel):
    """Represent one prompt from function_calling_tests.json."""

    prompt: str
