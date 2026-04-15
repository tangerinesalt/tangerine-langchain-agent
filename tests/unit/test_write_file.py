from pathlib import Path

from langchain_code_agent.tools.write_file import write_file_tool
from langchain_code_agent.workspace.repository import Repository


def test_write_file_tool_creates_file(tmp_path: Path) -> None:
    repository = Repository(tmp_path, ignore_patterns=[])

    result = write_file_tool(repository, path="notes/out.txt", content="hello")

    assert result.ok is True
    assert (tmp_path / "notes" / "out.txt").read_text(encoding="utf-8") == "hello"
    assert result.data["created"] is True
    assert result.data["overwritten"] is False


def test_write_file_tool_rejects_escape_path(tmp_path: Path) -> None:
    repository = Repository(tmp_path, ignore_patterns=[])

    result = write_file_tool(repository, path="../escape.txt", content="blocked")

    assert result.ok is False
    assert "escapes workspace root" in str(result.error)


def test_write_file_tool_respects_overwrite_flag(tmp_path: Path) -> None:
    repository = Repository(tmp_path, ignore_patterns=[])
    (tmp_path / "note.txt").write_text("old", encoding="utf-8")

    blocked = write_file_tool(repository, path="note.txt", content="new")
    allowed = write_file_tool(repository, path="note.txt", content="new", overwrite=True)

    assert blocked.ok is False
    assert allowed.ok is True
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "new"
    assert allowed.data["overwritten"] is True
