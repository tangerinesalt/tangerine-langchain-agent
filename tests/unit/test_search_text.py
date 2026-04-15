from pathlib import Path

from langchain_code_agent.workspace.repository import Repository


def test_search_text_finds_matches(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hello world')\n", encoding="utf-8")
    repository = Repository(tmp_path, ignore_patterns=[])

    matches = repository.search_text("hello")

    assert len(matches) == 1
    assert matches[0]["path"] == "main.py"


def test_search_text_advanced_filters_by_glob_and_regex(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "main.py").write_text("Value = 1\nvalue = 2\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("value = 3\n", encoding="utf-8")
    repository = Repository(tmp_path, ignore_patterns=[])

    matches = repository.search_text_advanced(
        r"Value\s*=\s*\d",
        max_results=5,
        case_sensitive=True,
        use_regex=True,
        path_glob="pkg/*.py",
    )

    assert len(matches) == 1
    assert matches[0]["path"] == "pkg/main.py"
