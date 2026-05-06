from __future__ import annotations

from pathlib import Path, PurePosixPath


def normalize_relative_path(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix().lstrip("./")


def normalize_workspace_relative_path(raw_path: str, workspace_root: Path) -> str:
    normalized = raw_path.strip().replace("\\", "/")
    if not normalized:
        return normalized

    try:
        candidate = Path(normalized)
        if candidate.is_absolute():
            return candidate.resolve().relative_to(workspace_root.resolve()).as_posix()
    except Exception:
        pass

    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if workspace_root.name in parts:
        anchor = parts.index(workspace_root.name)
        remainder = parts[anchor + 1 :]
        if remainder:
            return "/".join(remainder)
    return "/".join(parts)

