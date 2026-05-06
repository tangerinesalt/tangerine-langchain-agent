from __future__ import annotations

from langchain_code_agent.models.result import FileChange


def diff_file_states(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> list[FileChange]:
    changes: list[FileChange] = []
    for path in sorted(set(before) | set(after)):
        before_state = before.get(path)
        after_state = after.get(path)
        if before_state is None and after_state is not None:
            changes.append(FileChange(path=path, change_type="added", after=after_state))
        elif before_state is not None and after_state is None:
            changes.append(FileChange(path=path, change_type="deleted", before=before_state))
        elif before_state != after_state and before_state is not None and after_state is not None:
            changes.append(
                FileChange(
                    path=path,
                    change_type="modified",
                    before=before_state,
                    after=after_state,
                )
            )
    return changes

