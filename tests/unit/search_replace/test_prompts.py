"""Unit tests for prompt helpers in search_replace."""

from llm_evolution.search_replace import (
    number_lines,
    search_replace_error_feedback,
    search_replace_format_example,
)


class TestNumberLines:
    def test_basic(self):
        result = number_lines("a\nb\nc\n", every=1)
        assert "1|" in result
        assert "2|" in result
        assert "3|" in result

    def test_every_10(self):
        lines = "\n".join(f"line{i}" for i in range(1, 21)) + "\n"
        result = number_lines(lines, every=10)
        assert " 1|" in result
        assert "10|" in result
        assert "20|" in result

    def test_first_line_always_numbered(self):
        result = number_lines("only-line\n", every=999)
        assert "1|" in result


class TestFormatExample:
    def test_contains_markers(self):
        ex = search_replace_format_example()
        assert "<<<<<<< SEARCH" in ex
        assert "=======" in ex
        assert ">>>>>>> REPLACE" in ex


class TestErrorFeedback:
    def test_includes_error_and_file(self):
        feedback = search_replace_error_feedback(
            "lib.py", "x = 0\n", "search text not found"
        )
        assert "search text not found" in feedback
        assert "lib.py" in feedback
        assert "<<<<<<< SEARCH" in feedback

    def test_extracts_line_context_when_present(self):
        code = "\n".join(f"line {i}" for i in range(1, 21)) + "\n"
        feedback = search_replace_error_feedback("file.txt", code, "broke at line 10")
        assert ">>>" in feedback
        assert "line 10" in feedback

    def test_uses_code_fence_lang(self):
        feedback = search_replace_error_feedback(
            "file.py", "x = 0\n", "boom", code_fence_lang="python"
        )
        assert "```python" in feedback
