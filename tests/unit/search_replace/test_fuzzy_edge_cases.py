"""Aggressive fuzzy-matching edge cases for search/replace.

These cases stress the matcher with the kinds of whitespace, encoding,
and line-ending mismatches LLMs commonly produce.  The matcher should
accept all of them; unmatchable noise (different identifiers, missing
content) must still raise.
"""

import pytest

from llm_evolution.search_replace import (
    SearchReplaceBlock,
    apply_search_replace,
)


class TestLineEndings:
    def test_crlf_in_file_lf_in_search(self):
        original = "line1\r\nline2\r\nline3\r\n"
        blocks = [SearchReplaceBlock(search="line1\nline2", replace="LINE1\nLINE2")]
        result = apply_search_replace(original, blocks)
        assert "LINE1" in result
        assert "LINE2" in result

    def test_lf_in_file_crlf_in_search(self):
        original = "line1\nline2\nline3\n"
        blocks = [SearchReplaceBlock(search="line1\r\nline2", replace="LINE1\nLINE2")]
        result = apply_search_replace(original, blocks)
        assert "LINE1" in result

    def test_old_mac_cr_in_file(self):
        """Lone \\r line endings should still match LF search."""
        original = "line1\rline2\rline3\r"
        blocks = [SearchReplaceBlock(search="line2", replace="LINE2")]
        result = apply_search_replace(original, blocks)
        assert "LINE2" in result

    def test_mixed_line_endings_in_search_block(self):
        original = "alpha\nbeta\ngamma\n"
        blocks = [SearchReplaceBlock(search="alpha\r\nbeta\ngamma", replace="x\ny\nz")]
        result = apply_search_replace(original, blocks)
        assert "x\ny\nz" in result


class TestBlankLineTolerance:
    def test_search_has_leading_blank_lines(self):
        original = "alpha\nbeta\ngamma\n"
        blocks = [SearchReplaceBlock(search="\n\nalpha\nbeta", replace="A\nB")]
        result = apply_search_replace(original, blocks)
        assert "A\nB" in result

    def test_search_has_trailing_blank_lines(self):
        original = "alpha\nbeta\ngamma\n"
        blocks = [SearchReplaceBlock(search="alpha\nbeta\n\n", replace="A\nB")]
        result = apply_search_replace(original, blocks)
        assert "A\nB" in result

    def test_search_has_both_padding(self):
        original = "alpha\nbeta\ngamma\n"
        blocks = [SearchReplaceBlock(search="\nalpha\nbeta\n\n", replace="A\nB")]
        result = apply_search_replace(original, blocks)
        assert "A\nB" in result


class TestInternalWhitespaceCollapse:
    def test_double_space_in_file_single_in_search(self):
        original = "int  x = 0;\nint y = 1;\n"
        blocks = [SearchReplaceBlock(search="int x = 0;", replace="int x = 99;")]
        result = apply_search_replace(original, blocks)
        assert "99" in result

    def test_tab_between_tokens(self):
        original = "int\tx = 0;\nint y = 1;\n"
        blocks = [SearchReplaceBlock(search="int x = 0;", replace="int x = 99;")]
        result = apply_search_replace(original, blocks)
        assert "99" in result

    def test_multiple_tabs_and_spaces_mixed(self):
        original = "if  ( x  ==\t 0 ) {\n    return 1;\n}\n"
        blocks = [
            SearchReplaceBlock(
                search="if (x == 0) {\n    return 1;\n}",
                replace="if (x == 0) {\n    return 2;\n}",
            )
        ]
        result = apply_search_replace(original, blocks)
        assert "return 2;" in result


class TestUnicodeWhitespace:
    def test_nbsp_in_file_regular_space_in_search(self):
        original = "alpha = 1\nbeta = 2\n"
        blocks = [SearchReplaceBlock(search="alpha = 1", replace="alpha = 99")]
        result = apply_search_replace(original, blocks)
        assert "99" in result

    def test_bom_at_start_of_file(self):
        original = "﻿alpha\nbeta\n"
        blocks = [SearchReplaceBlock(search="alpha", replace="ALPHA")]
        result = apply_search_replace(original, blocks)
        assert "ALPHA" in result


class TestCombined:
    def test_crlf_and_internal_whitespace(self):
        original = "if  (x)\r\n    return 1;\r\n"
        blocks = [
            SearchReplaceBlock(
                search="if (x)\n    return 1;",
                replace="if (x)\n    return 2;",
            )
        ]
        result = apply_search_replace(original, blocks)
        assert "return 2;" in result

    def test_blank_padding_plus_crlf(self):
        original = "alpha\r\nbeta\r\ngamma\r\n"
        blocks = [
            SearchReplaceBlock(search="\nalpha\nbeta\n\n", replace="A\nB"),
        ]
        result = apply_search_replace(original, blocks)
        assert "A\nB" in result

    def test_tabs_indent_plus_internal_whitespace(self):
        original = "\tif  (x)  {\n\t\treturn 1;\n\t}\n"
        blocks = [
            SearchReplaceBlock(
                search="    if (x) {\n        return 1;\n    }",
                replace="    if (x) {\n        return 2;\n    }",
            )
        ]
        result = apply_search_replace(original, blocks)
        assert "return 2;" in result


class TestStillRejectsTrueMismatches:
    """Aggressive fuzz must not silently accept genuinely different code."""

    def test_different_identifier_raises(self):
        original = "alpha = 1\nbeta = 2\n"
        blocks = [SearchReplaceBlock(search="zeta = 1", replace="zeta = 99")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace(original, blocks)

    def test_extra_content_in_search_raises(self):
        original = "a\nb\n"
        blocks = [SearchReplaceBlock(search="a\nMISSING\nb", replace="X")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace(original, blocks)

    def test_different_numeric_literal_raises(self):
        original = "x = 100\n"
        blocks = [SearchReplaceBlock(search="x = 999", replace="x = 1")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace(original, blocks)


class TestDuplicatesUnderFuzz:
    def test_crlf_duplicates_replaced_all(self):
        original = "x = 1\r\ny = 1\r\nx = 1\r\n"
        blocks = [SearchReplaceBlock(search="x = 1", replace="x = 99")]
        result = apply_search_replace(original, blocks)
        assert result.count("x = 99") == 2

    def test_internal_ws_duplicates_replaced_all(self):
        original = "int  a = 0;\nint  a = 0;\n"
        blocks = [SearchReplaceBlock(search="int a = 0;", replace="int a = 1;")]
        result = apply_search_replace(original, blocks)
        assert result.count("int a = 1;") == 2
