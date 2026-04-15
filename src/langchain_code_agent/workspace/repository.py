from __future__ import annotations

import re
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path


class RepositoryError(RuntimeError):
    """Raised when repository access fails."""


class Repository:
    def __init__(self, root: Path, ignore_patterns: Iterable[str]) -> None:
        self.root = root.resolve()
        self.ignore_patterns = list(ignore_patterns)
        if not self.root.exists():
            raise RepositoryError(f"Workspace does not exist: {self.root}")
        if not self.root.is_dir():
            raise RepositoryError(f"Workspace is not a directory: {self.root}")

    def list_files(self, limit: int = 200) -> list[str]:
        files: list[str] = []
        for path in self._iter_files():
            files.append(path.relative_to(self.root).as_posix())
            if len(files) >= limit:
                break
        return files

    def glob_files(self, pattern: str, limit: int = 200) -> list[str]:
        matches: list[str] = []
        for path in self._iter_files():
            relative = path.relative_to(self.root).as_posix()
            if fnmatch(relative, pattern):
                matches.append(relative)
                if len(matches) >= limit:
                    break
        return matches

    def find_files_by_name(self, name: str, limit: int = 200) -> list[str]:
        lowered_name = name.lower()
        matches: list[str] = []
        for path in self._iter_files():
            if lowered_name in path.name.lower():
                matches.append(path.relative_to(self.root).as_posix())
                if len(matches) >= limit:
                    break
        return matches

    def read_text(self, relative_path: str, encoding: str = "utf-8") -> str:
        path = self._resolve_relative_path(relative_path)
        return path.read_text(encoding=encoding)

    def read_text_head(
        self,
        relative_path: str,
        *,
        start_line: int = 1,
        max_lines: int = 200,
        encoding: str = "utf-8",
    ) -> dict[str, object]:
        if start_line < 1:
            raise RepositoryError("start_line must be >= 1")
        if max_lines < 1:
            raise RepositoryError("max_lines must be >= 1")

        path = self._resolve_relative_path(relative_path)
        lines = path.read_text(encoding=encoding).splitlines()
        start_index = start_line - 1
        selected = lines[start_index : start_index + max_lines]
        return {
            "path": relative_path,
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1 if selected else start_line - 1,
            "content": "\n".join(selected),
        }

    def write_text(
        self,
        relative_path: str,
        content: str,
        *,
        overwrite: bool = False,
        encoding: str = "utf-8",
    ) -> dict[str, object]:
        path = self._resolve_writable_path(relative_path)
        existed = path.exists()
        if existed and not overwrite:
            raise RepositoryError(f"Path already exists and overwrite is disabled: {relative_path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return {
            "path": path.relative_to(self.root).as_posix(),
            "bytes_written": len(content.encode(encoding)),
            "created": not existed,
            "overwritten": existed,
        }

    def replace_in_file(
        self,
        relative_path: str,
        *,
        old_text: str,
        new_text: str,
        count: int = 1,
        encoding: str = "utf-8",
    ) -> dict[str, object]:
        if count < 1:
            raise RepositoryError("count must be >= 1")

        path = self._resolve_relative_path(relative_path)
        content = path.read_text(encoding=encoding)
        replacements = content.count(old_text)
        if replacements == 0:
            raise RepositoryError(f"Text to replace was not found in file: {relative_path}")

        replaced_content = content.replace(old_text, new_text, count)
        actual_replacements = min(replacements, count)
        path.write_text(replaced_content, encoding=encoding)
        return {
            "path": relative_path,
            "replacements": actual_replacements,
        }

    def insert_text(
        self,
        relative_path: str,
        *,
        anchor: str,
        text: str,
        position: str = "after",
        encoding: str = "utf-8",
    ) -> dict[str, object]:
        if position not in {"before", "after"}:
            raise RepositoryError("position must be 'before' or 'after'")

        path = self._resolve_relative_path(relative_path)
        content = path.read_text(encoding=encoding)
        anchor_index = content.find(anchor)
        if anchor_index == -1:
            raise RepositoryError(f"Anchor was not found in file: {relative_path}")

        insert_at = anchor_index if position == "before" else anchor_index + len(anchor)
        updated_content = content[:insert_at] + text + content[insert_at:]
        path.write_text(updated_content, encoding=encoding)
        return {
            "path": relative_path,
            "position": position,
            "anchor": anchor,
        }

    def delete_file(self, relative_path: str) -> dict[str, object]:
        path = self._resolve_relative_path(relative_path)
        path.unlink()
        return {"path": relative_path, "deleted": True}

    def move_file(self, source_path: str, destination_path: str) -> dict[str, object]:
        source = self._resolve_relative_path(source_path)
        destination = self._resolve_writable_path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        return {
            "source_path": source_path,
            "destination_path": destination.relative_to(self.root).as_posix(),
            "moved": True,
        }

    def search_text(self, query: str, max_results: int = 20) -> list[dict[str, object]]:
        return self.search_text_advanced(
            query,
            max_results=max_results,
            case_sensitive=False,
            use_regex=False,
            path_glob=None,
        )

    def search_text_advanced(
        self,
        query: str,
        *,
        max_results: int = 20,
        case_sensitive: bool = False,
        use_regex: bool = False,
        path_glob: str | None = None,
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        pattern = re.compile(query, 0 if case_sensitive else re.IGNORECASE) if use_regex else None
        query_lower = query.lower()
        for path in self._iter_files():
            relative_path = path.relative_to(self.root).as_posix()
            if path_glob is not None and not fnmatch(relative_path, path_glob):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            for line_number, line in enumerate(content.splitlines(), start=1):
                if _matches_query(
                    line,
                    query_lower=query_lower,
                    pattern=pattern,
                    case_sensitive=case_sensitive,
                    use_regex=use_regex,
                ):
                    matches.append(
                        {
                            "path": relative_path,
                            "line_number": line_number,
                            "line": line.strip(),
                        }
                    )
                    if len(matches) >= max_results:
                        return matches
        return matches

    def tree_view(self, relative_path: str = ".", *, depth: int = 2) -> list[str]:
        if depth < 0:
            raise RepositoryError("depth must be >= 0")

        base = self._resolve_tree_path(relative_path)
        base_relative = "." if base == self.root else base.relative_to(self.root).as_posix()
        lines = [base_relative]
        self._append_tree_lines(base, lines, current_depth=0, max_depth=depth)
        return lines

    def snapshot_file_state(self) -> dict[str, dict[str, int]]:
        state: dict[str, dict[str, int]] = {}
        for path in self._iter_files():
            stat = path.stat()
            state[path.relative_to(self.root).as_posix()] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        return state

    def _resolve_relative_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if self.root not in path.parents and path != self.root:
            raise RepositoryError(f"Path escapes workspace root: {relative_path}")
        if not path.exists():
            raise RepositoryError(f"Path does not exist: {relative_path}")
        if not path.is_file():
            raise RepositoryError(f"Path is not a file: {relative_path}")
        return path

    def _is_ignored(self, path: Path) -> bool:
        relative = path.relative_to(self.root)
        parts = set(relative.parts)
        relative_posix = relative.as_posix()
        return any(
            pattern in parts or relative_posix.startswith(f"{pattern}/")
            for pattern in self.ignore_patterns
        )

    def _resolve_writable_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if self.root not in path.parents and path != self.root:
            raise RepositoryError(f"Path escapes workspace root: {relative_path}")
        if path.exists() and path.is_dir():
            raise RepositoryError(f"Path is a directory: {relative_path}")
        return path

    def _resolve_tree_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if self.root not in path.parents and path != self.root:
            raise RepositoryError(f"Path escapes workspace root: {relative_path}")
        if not path.exists():
            raise RepositoryError(f"Path does not exist: {relative_path}")
        if not path.is_dir():
            raise RepositoryError(f"Path is not a directory: {relative_path}")
        return path

    def _append_tree_lines(
        self,
        directory: Path,
        lines: list[str],
        *,
        current_depth: int,
        max_depth: int,
    ) -> None:
        if current_depth >= max_depth:
            return

        ordered_children = sorted(
            directory.iterdir(),
            key=lambda item: (item.is_file(), item.name.lower()),
        )
        for child in ordered_children:
            if self._is_ignored(child):
                continue
            relative = child.relative_to(self.root).as_posix()
            indent = "  " * (current_depth + 1)
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{indent}{relative}{suffix}")
            if child.is_dir():
                self._append_tree_lines(
                    child,
                    lines,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                )

    def _iter_files(self) -> Iterable[Path]:
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and not self._is_ignored(path):
                yield path


def _matches_query(
    line: str,
    *,
    query_lower: str,
    pattern: re.Pattern[str] | None,
    case_sensitive: bool,
    use_regex: bool,
) -> bool:
    if use_regex:
        return pattern is not None and pattern.search(line) is not None
    if case_sensitive:
        return query_lower in line
    return query_lower in line.lower()
