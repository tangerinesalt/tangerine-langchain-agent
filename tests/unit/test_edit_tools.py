from pathlib import Path

from langchain_code_agent.tools.delete_file import delete_file_tool
from langchain_code_agent.tools.insert_text import insert_text_tool
from langchain_code_agent.tools.move_file import move_file_tool
from langchain_code_agent.tools.replace_in_file import replace_in_file_tool
from langchain_code_agent.workspace.repository import Repository


def test_replace_in_file_tool_replaces_text(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    repository = Repository(tmp_path, ignore_patterns=[])

    result = replace_in_file_tool(
        repository,
        path="main.py",
        old_text="value = 1",
        new_text="value = 2",
    )

    assert result.ok is True
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "value = 2\n"
    assert result.data["replacements"] == 1


def test_insert_text_tool_inserts_around_anchor(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("alpha\nbeta\n", encoding="utf-8")
    repository = Repository(tmp_path, ignore_patterns=[])

    result = insert_text_tool(
        repository,
        path="main.py",
        anchor="alpha\n",
        text="inserted\n",
        position="after",
    )

    assert result.ok is True
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "alpha\ninserted\nbeta\n"


def test_move_and_delete_file_tools_update_files(tmp_path: Path) -> None:
    (tmp_path / "old.txt").write_text("content", encoding="utf-8")
    repository = Repository(tmp_path, ignore_patterns=[])

    moved = move_file_tool(
        repository,
        source_path="old.txt",
        destination_path="nested/new.txt",
    )
    deleted = delete_file_tool(repository, path="nested/new.txt")

    assert moved.ok is True
    assert moved.data["destination_path"] == "nested/new.txt"
    assert deleted.ok is True
    assert not (tmp_path / "nested" / "new.txt").exists()
