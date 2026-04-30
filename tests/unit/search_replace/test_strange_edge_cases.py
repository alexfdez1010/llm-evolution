"""Stranger fuzzy edge cases: pathological inputs, weird Unicode, big files.

Doubles the coverage of ``test_fuzzy_edge_cases.py``.  Anything here that
fails would represent a real LLM-output failure mode in the wild.
"""

import pytest

from llm_evolution.search_replace import (
    SearchReplaceBlock,
    apply_search_replace,
)


class TestPathologicalSearchInputs:
    def test_search_only_blank_lines_raises(self):
        original = "alpha\nbeta\n"
        blocks = [SearchReplaceBlock(search="\n\n  \n", replace="X")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace(original, blocks)

    def test_search_single_newline_raises(self):
        original = "alpha\nbeta\n"
        blocks = [SearchReplaceBlock(search="\n", replace="X")]
        result = apply_search_replace(original, blocks)
        # "\n" appears as a substring; replace via exact-match path.
        assert "X" in result

    def test_search_only_whitespace_chars_raises(self):
        original = "alpha\nbeta\n"
        blocks = [SearchReplaceBlock(search=" \t \t ", replace="X")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace(original, blocks)

    def test_single_punctuation_line_match(self):
        """A lone closing brace at varying indents must find the right one."""
        original = "if x:\n    return 1\n}\n"
        blocks = [SearchReplaceBlock(search="}", replace="# end")]
        result = apply_search_replace(original, blocks)
        assert "# end" in result


class TestStrangeUnicode:
    def test_emspace_collapses_to_space(self):
        """U+2003 EM SPACE between tokens should still match a regular space."""
        original = "alpha = 1\n"
        blocks = [SearchReplaceBlock(search="alpha = 1", replace="alpha = 99")]
        result = apply_search_replace(original, blocks)
        assert "99" in result

    def test_enspace_collapses_to_space(self):
        original = "alpha = 1\n"
        blocks = [SearchReplaceBlock(search="alpha = 1", replace="alpha = 99")]
        result = apply_search_replace(original, blocks)
        assert "99" in result

    def test_zero_width_space_in_middle(self):
        """U+200B ZERO WIDTH SPACE — not real whitespace; identifier mismatch
        must still raise."""
        original = "alp​ha = 1\n"
        blocks = [SearchReplaceBlock(search="alpha = 1", replace="alpha = 99")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace(original, blocks)

    def test_bom_only_stripped_when_at_start(self):
        """A BOM in the middle of the file must not be treated as a start BOM."""
        original = "alpha\n﻿beta\n"
        blocks = [SearchReplaceBlock(search="beta", replace="BETA")]
        result = apply_search_replace(original, blocks)
        assert "BETA" in result

    def test_multiple_leading_boms_all_stripped(self):
        original = "﻿﻿alpha\nbeta\n"
        blocks = [SearchReplaceBlock(search="alpha", replace="ALPHA")]
        result = apply_search_replace(original, blocks)
        assert "ALPHA" in result

    def test_form_feed_treated_as_whitespace(self):
        original = "alpha\f=\f1\n"
        blocks = [SearchReplaceBlock(search="alpha = 1", replace="alpha = 99")]
        result = apply_search_replace(original, blocks)
        assert "99" in result

    def test_vertical_tab_treated_as_whitespace(self):
        original = "alpha\v=\v1\n"
        blocks = [SearchReplaceBlock(search="alpha = 1", replace="alpha = 99")]
        result = apply_search_replace(original, blocks)
        assert "99" in result


class TestBoundaryAndStructure:
    def test_match_at_file_end_without_trailing_newline(self):
        original = "alpha\nbeta"
        blocks = [SearchReplaceBlock(search="beta", replace="BETA")]
        result = apply_search_replace(original, blocks)
        assert result.endswith("BETA")

    def test_replace_with_crlf_preserved(self):
        original = "alpha\nbeta\n"
        blocks = [SearchReplaceBlock(search="alpha", replace="X\r\nY")]
        result = apply_search_replace(original, blocks)
        assert "X\r\nY" in result

    def test_search_spans_entire_file_with_mixed_endings(self):
        original = "line1\r\nline2\r\nline3"
        blocks = [SearchReplaceBlock(search="line1\nline2\nline3", replace="ONLY")]
        result = apply_search_replace(original, blocks)
        assert result == "ONLY"

    def test_back_to_back_identical_matches(self):
        original = "a\na\na\n"
        blocks = [SearchReplaceBlock(search="a\n", replace="b\n")]
        result = apply_search_replace(original, blocks)
        assert result == "b\nb\nb\n"

    def test_overlapping_match_does_not_double_replace(self):
        """Search 'aa' in 'aaa' — three substrings, but Python str.replace is
        non-overlapping (left-to-right, count==2 here)."""
        original = "aaaa\n"
        blocks = [SearchReplaceBlock(search="aa", replace="X")]
        result = apply_search_replace(original, blocks)
        # str.replace is non-overlapping: "aaaa" -> "XX"
        assert result == "XX\n"

    def test_match_in_middle_of_long_line(self):
        original = "x" * 500 + "TARGET" + "y" * 500 + "\n"
        blocks = [SearchReplaceBlock(search="TARGET", replace="HIT")]
        result = apply_search_replace(original, blocks)
        assert "HIT" in result
        assert "TARGET" not in result


class TestIndentMisalignment:
    def test_three_space_indent_vs_four_space_search(self):
        """Token strategy rescues mismatched indent depth."""
        original = "def f():\n   return 1\n"
        blocks = [
            SearchReplaceBlock(
                search="def f():\n    return 1",
                replace="def f():\n    return 2",
            )
        ]
        result = apply_search_replace(original, blocks)
        assert "return 2" in result

    def test_two_space_vs_tab_indent(self):
        original = "def f():\n  return 1\n"
        blocks = [
            SearchReplaceBlock(
                search="def f():\n\treturn 1",
                replace="def f():\n\treturn 2",
            )
        ]
        result = apply_search_replace(original, blocks)
        assert "return 2" in result

    def test_search_with_blank_line_in_middle_kept(self):
        """Internal blank lines inside the search must still be matched."""
        original = "alpha\n\nbeta\n"
        blocks = [
            SearchReplaceBlock(search="alpha\n\nbeta", replace="X\n\nY"),
        ]
        result = apply_search_replace(original, blocks)
        assert result == "X\n\nY\n"

    def test_search_blank_line_present_only_in_file(self):
        """File has extra blank line in middle — token-level rescue not safe;
        must still raise (we don't drop blanks from the file)."""
        original = "alpha\n\nbeta\n"
        blocks = [SearchReplaceBlock(search="alpha\nbeta", replace="X\nY")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace(original, blocks)


class TestSemanticSafety:
    def test_hex_vs_decimal_literal_does_not_match(self):
        original = "x = 0x10\n"
        blocks = [SearchReplaceBlock(search="x = 16", replace="x = 0")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace(original, blocks)

    def test_string_literal_internal_spaces_collapse_under_tokens(self):
        """Documented aggressive behaviour: tokenised match equates double
        spaces inside string literals.  Acceptable for code-evolution use."""
        original = 'x = "a  b"\n'
        blocks = [SearchReplaceBlock(search='x = "a b"', replace='x = "z"')]
        result = apply_search_replace(original, blocks)
        assert '"z"' in result

    def test_different_operator_does_not_match(self):
        """+ and - are distinct tokens; must not be conflated."""
        original = "y = a + b\n"
        blocks = [SearchReplaceBlock(search="y = a - b", replace="y = 0")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace(original, blocks)

    def test_compound_operator_conflated_by_token_strategy(self):
        """Documented limitation: punctuation tokenises to single chars, so
        ``==`` and ``= =`` look identical under the most aggressive strategy.
        Acceptable trade-off — operator splits are rare in valid LLM output."""
        original = "if x = = 0:\n    pass\n"
        blocks = [
            SearchReplaceBlock(
                search="if x == 0:\n    pass",
                replace="if x == 0:\n    return",
            )
        ]
        result = apply_search_replace(original, blocks)
        assert "return" in result

    def test_swapped_identifier_order_does_not_match(self):
        original = "result = a + b\n"
        blocks = [SearchReplaceBlock(search="result = b + a", replace="X")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace(original, blocks)


class TestStrangeMultipleBlocks:
    def test_each_block_uses_a_different_strategy(self):
        """One block exact-matches, another needs token-level fuzz."""
        original = "alpha = 1\nbeta\t=\t2\n"
        blocks = [
            SearchReplaceBlock(search="alpha = 1", replace="alpha = 11"),
            SearchReplaceBlock(search="beta = 2", replace="beta = 22"),
        ]
        result = apply_search_replace(original, blocks)
        assert "alpha = 11" in result
        assert "beta = 22" in result

    def test_fuzzy_then_already_applied_skip(self):
        """Block 1 fuzzy-matches and replaces; block 2 with same fuzzy search
        skips silently because the search text is gone."""
        original = "int  a = 0;\n"
        blocks = [
            SearchReplaceBlock(search="int a = 0;", replace="int a = 1;"),
            SearchReplaceBlock(search="int a = 0;", replace="int a = 2;"),
        ]
        result = apply_search_replace(original, blocks)
        assert "int a = 1;" in result
        assert "int a = 2;" not in result
