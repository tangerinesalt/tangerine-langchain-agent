from __future__ import annotations

from pydantic import ValidationError

from langchain_code_agent.action_registry.registry import get_action_spec


def validate_action_arguments(action: str, arguments: dict[str, object]) -> str | None:
    spec = get_action_spec(action)
    if spec is None:
        return f"Unsupported action: {action}"

    unknown_arguments = sorted(set(arguments) - spec.allowed_arguments)
    if unknown_arguments:
        return (
            f"Action '{action}' does not accept arguments: {', '.join(unknown_arguments)}"
        )

    missing_arguments = sorted(key for key in spec.required_arguments if key not in arguments)
    if missing_arguments:
        return (
            f"Action '{action}' is missing required arguments: {', '.join(missing_arguments)}"
        )
    if spec.langchain_args_schema is not None:
        try:
            spec.langchain_args_schema.model_validate(arguments)
        except ValidationError as exc:
            return f"Action '{action}' has invalid arguments: {exc}"
    return None
