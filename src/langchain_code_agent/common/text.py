from __future__ import annotations


def excerpt_text(
    value: object,
    *,
    max_chars: int,
    truncated_suffix: str = "\n...[truncated]",
) -> str | None:
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + truncated_suffix
