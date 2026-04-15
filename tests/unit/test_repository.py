from pathlib import Path

from langchain_code_agent.workspace.repository import Repository


def test_repository_lists_and_reads_files(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "skip.pyc").write_bytes(b"123")
    repository = Repository(tmp_path, ignore_patterns=["__pycache__"])

    files = repository.list_files()
    content = repository.read_text("pkg/module.py")

    assert files == ["pkg/module.py"]
    assert content == "value = 1\n"


def test_repository_supports_phase1_discovery_methods(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("Value = 1\nvalue = 2\n", encoding="utf-8")
    (tmp_path / "pkg" / "other_test.py").write_text("assert True\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.txt").write_text("Alpha\nBeta\nGamma\n", encoding="utf-8")
    repository = Repository(tmp_path, ignore_patterns=[])

    globbed = repository.glob_files("**/*.py")
    named = repository.find_files_by_name("module")
    head = repository.read_text_head("docs/notes.txt", start_line=2, max_lines=2)
    tree = repository.tree_view(".", depth=2)
    searched = repository.search_text_advanced(
        r"Value\s*=\s*\d",
        max_results=5,
        use_regex=True,
        case_sensitive=True,
        path_glob="pkg/*.py",
    )

    assert globbed == ["pkg/module.py", "pkg/other_test.py"]
    assert named == ["pkg/module.py"]
    assert head["content"] == "Beta\nGamma"
    assert tree[0] == "."
    assert any(line.endswith("pkg/") for line in tree)
    assert searched[0]["path"] == "pkg/module.py"
