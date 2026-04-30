"""Unit tests for search_replace block extraction."""

from llm_evolution.implementations.utils.search_replace import extract_search_replace


class TestExtractSearchReplace:
    def test_single_block(self):
        response = (
            "Summary line.\n\n"
            "<<<<<<< SEARCH\nold code\n=======\nnew code\n>>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 1
        assert blocks[0].search == "old code"
        assert blocks[0].replace == "new code"

    def test_multiple_blocks(self):
        response = (
            "<<<<<<< SEARCH\naaa\n=======\nbbb\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\nccc\n=======\nddd\n>>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 2
        assert blocks[0].search == "aaa"
        assert blocks[1].search == "ccc"

    def test_multiline_search(self):
        response = (
            "<<<<<<< SEARCH\nline1\nline2\nline3\n=======\nnew1\nnew2\n>>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert blocks[0].search == "line1\nline2\nline3"
        assert blocks[0].replace == "new1\nnew2"

    def test_no_blocks_returns_none(self):
        assert extract_search_replace("no blocks here") is None

    def test_empty_response_returns_none(self):
        assert extract_search_replace("") is None

    def test_block_with_surrounding_text(self):
        response = (
            "Here is the fix:\n\n"
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n\nDone."
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 1

    def test_empty_replace(self):
        response = "<<<<<<< SEARCH\ndelete me\n=======\n\n>>>>>>> REPLACE"
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert blocks[0].replace == ""

    def test_trailing_whitespace_on_markers(self):
        response = (
            "<<<<<<< SEARCH   \nold code\n=======  \nnew code\n>>>>>>> REPLACE \n"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert blocks[0].search == "old code"
        assert blocks[0].replace == "new code"

    def test_blocks_wrapped_in_markdown_code_fence(self):
        response = (
            "Here is the fix:\n\n"
            "```\n"
            "<<<<<<< SEARCH\nold line\n=======\nnew line\n>>>>>>> REPLACE\n"
            "```\n"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 1
        assert blocks[0].search == "old line"

    def test_blocks_wrapped_in_language_code_fence(self):
        response = (
            "```python\n"
            "<<<<<<< SEARCH\nimport os\n=======\nimport sys\n>>>>>>> REPLACE\n"
            "```\n"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert blocks[0].search == "import os"

    def test_multiple_blocks_in_separate_fences(self):
        response = (
            "```\n<<<<<<< SEARCH\naaa\n=======\nbbb\n>>>>>>> REPLACE\n```\n"
            "```\n<<<<<<< SEARCH\nccc\n=======\nddd\n>>>>>>> REPLACE\n```\n"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 2

    def test_block_with_indented_code(self):
        response = (
            "<<<<<<< SEARCH\n"
            "    if x > 0:\n"
            "        return x\n"
            "=======\n"
            "    if x > 0:\n"
            "        return x + 1\n"
            ">>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert "return x" in blocks[0].search
        assert "return x + 1" in blocks[0].replace

    def test_noop_blocks_filtered_out(self):
        response = (
            "<<<<<<< SEARCH\nold code\n=======\nnew code\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\nsame\n=======\nsame\n>>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 1
        assert blocks[0].search == "old code"

    def test_all_noop_blocks_returns_none(self):
        response = (
            "<<<<<<< SEARCH\nsame\n=======\nsame\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\nalso same\n=======\nalso same\n>>>>>>> REPLACE"
        )
        assert extract_search_replace(response) is None

    def test_block_with_blank_lines_in_code(self):
        response = (
            "<<<<<<< SEARCH\nint a;\n\nint b;\n=======\nint a;\nint b;\n>>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert "\n\n" in blocks[0].search
        assert "\n\n" not in blocks[0].replace

    def test_block_with_tabs(self):
        response = (
            "<<<<<<< SEARCH\n\tint x = 0;\n=======\n\tint x = 42;\n>>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert "\t" in blocks[0].search
        assert "42" in blocks[0].replace

    def test_block_with_special_chars(self):
        response = (
            "<<<<<<< SEARCH\n"
            "ptr->data[i] = (a + b) * c;\n"
            "=======\n"
            "ptr->data[i] = (a + b) * d;\n"
            ">>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert "ptr->data[i]" in blocks[0].search

    def test_block_with_backslashes(self):
        response = (
            "<<<<<<< SEARCH\n"
            'printf("hello\\n");\n'
            "=======\n"
            'printf("world\\n");\n'
            ">>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert "\\n" in blocks[0].search
