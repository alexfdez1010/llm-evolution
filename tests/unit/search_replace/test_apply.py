"""Unit tests for apply_search_replace."""

import pytest

from llm_evolution.search_replace import (
    SearchReplaceBlock,
    apply_search_replace,
)


class TestApplySearchReplace:
    def test_simple_replacement(self):
        original = "aaa\nbbb\nccc\n"
        blocks = [SearchReplaceBlock(search="bbb", replace="xxx")]
        assert apply_search_replace(original, blocks) == "aaa\nxxx\nccc\n"

    def test_multiline_replacement(self):
        original = "aaa\nbbb\nccc\nddd\n"
        blocks = [SearchReplaceBlock(search="bbb\nccc", replace="xxx\nyyy\nzzz")]
        assert apply_search_replace(original, blocks) == "aaa\nxxx\nyyy\nzzz\nddd\n"

    def test_multiple_blocks_applied_sequentially(self):
        original = "aaa\nbbb\nccc\n"
        blocks = [
            SearchReplaceBlock(search="aaa", replace="AAA"),
            SearchReplaceBlock(search="ccc", replace="CCC"),
        ]
        assert apply_search_replace(original, blocks) == "AAA\nbbb\nCCC\n"

    def test_search_not_found_raises(self):
        original = "aaa\nbbb\n"
        blocks = [SearchReplaceBlock(search="zzz", replace="xxx")]
        with pytest.raises(ValueError, match="search text not found"):
            apply_search_replace(original, blocks)

    def test_deletion(self):
        original = "aaa\nbbb\nccc\n"
        blocks = [SearchReplaceBlock(search="bbb\n", replace="")]
        assert apply_search_replace(original, blocks) == "aaa\nccc\n"

    def test_insertion_via_context(self):
        original = "aaa\nccc\n"
        blocks = [SearchReplaceBlock(search="aaa\nccc", replace="aaa\nbbb\nccc")]
        assert apply_search_replace(original, blocks) == "aaa\nbbb\nccc\n"

    def test_second_block_sees_result_of_first(self):
        original = "old1\nold2\n"
        blocks = [
            SearchReplaceBlock(search="old1", replace="new1"),
            SearchReplaceBlock(search="new1", replace="final"),
        ]
        assert apply_search_replace(original, blocks) == "final\nold2\n"

    def test_empty_blocks_list(self):
        original = "hello\n"
        assert apply_search_replace(original, []) == "hello\n"

    def test_search_with_special_regex_chars(self):
        original = "if (x > 0 && y < 10) { return (a + b); }\n"
        blocks = [
            SearchReplaceBlock(
                search="if (x > 0 && y < 10) { return (a + b); }",
                replace="if (x > 0 && y < 10) { return (a * b); }",
            )
        ]
        assert "(a * b)" in apply_search_replace(original, blocks)

    def test_strict_false_no_blocks_applied_raises(self):
        """Lenient mode still raises when nothing applied at all."""
        original = "hello\n"
        blocks = [SearchReplaceBlock(search="zzz", replace="xxx")]
        with pytest.raises(ValueError, match="No search/replace blocks"):
            apply_search_replace(original, blocks, strict=False)

    def test_strict_false_partial_apply_succeeds(self):
        """Lenient mode keeps applying after a missing block."""
        original = "aaa\nbbb\n"
        blocks = [
            SearchReplaceBlock(search="zzz", replace="xxx"),
            SearchReplaceBlock(search="aaa", replace="AAA"),
        ]
        result = apply_search_replace(original, blocks, strict=False)
        assert "AAA" in result


class TestApplySearchReplaceDuplicates:
    def test_duplicate_search_replaces_all(self):
        original = "aaa\naaa\n"
        blocks = [SearchReplaceBlock(search="aaa", replace="bbb")]
        assert apply_search_replace(original, blocks) == "bbb\nbbb\n"

    def test_duplicate_search_three_occurrences(self):
        original = "x = sizeof(T);\ny = sizeof(T);\nz = sizeof(T);\n"
        blocks = [SearchReplaceBlock(search="sizeof(T)", replace="16")]
        result = apply_search_replace(original, blocks)
        assert result.count("16") == 3
        assert "sizeof(T)" not in result

    def test_duplicate_multiline_search_replaces_all(self):
        original = "// block 1\na = 1;\nb = 2;\n// middle\n// block 2\na = 1;\nb = 2;\n"
        blocks = [
            SearchReplaceBlock(
                search="a = 1;\nb = 2;",
                replace="a = 10;\nb = 20;",
            )
        ]
        result = apply_search_replace(original, blocks)
        assert result.count("a = 10;") == 2
        assert result.count("b = 20;") == 2

    def test_two_blocks_same_search_text_second_skipped(self):
        original = "aaa\nbbb\naaa\n"
        blocks = [
            SearchReplaceBlock(search="aaa", replace="xxx"),
            SearchReplaceBlock(search="aaa", replace="xxx"),
        ]
        assert apply_search_replace(original, blocks) == "xxx\nbbb\nxxx\n"

    def test_overlapping_blocks_second_search_consumed(self):
        original = "aaa\nbbb\nccc\n"
        blocks = [
            SearchReplaceBlock(search="aaa\nbbb", replace="xxx\nyyy"),
            SearchReplaceBlock(search="aaa", replace="zzz"),
        ]
        result = apply_search_replace(original, blocks)
        assert result == "xxx\nyyy\nccc\n"

    def test_replacement_already_present_skipped(self):
        original = "old_val\n"
        blocks = [
            SearchReplaceBlock(search="old_val", replace="new_val"),
            SearchReplaceBlock(search="old_val", replace="new_val"),
        ]
        assert apply_search_replace(original, blocks) == "new_val\n"

    def test_same_replacement_in_different_functions_not_skipped(self):
        original = (
            "def func_a():\n"
            "    x = compute_fast(j)  # already fixed\n"
            "def func_b():\n"
            "    x = compute(j)  # needs fixing\n"
        )
        blocks = [
            SearchReplaceBlock(
                search="    x = compute(j)  # needs fixing",
                replace="    x = compute_fast(j)  # already fixed",
            )
        ]
        result = apply_search_replace(original, blocks)
        assert result.count("compute_fast(j)") == 2
        assert "compute(j)" not in result


class TestApplySearchReplaceFuzzy:
    def test_trailing_whitespace_tolerance(self):
        original = "aaa  \nbbb\n"
        blocks = [SearchReplaceBlock(search="aaa", replace="xxx")]
        assert "xxx" in apply_search_replace(original, blocks)

    def test_tab_vs_spaces_leading_whitespace(self):
        original = "\tint x = 0;\n\tint y = 1;\n"
        blocks = [
            SearchReplaceBlock(
                search="    int x = 0;\n    int y = 1;",
                replace="    int x = 42;\n    int y = 99;",
            )
        ]
        result = apply_search_replace(original, blocks)
        assert "42" in result
        assert "99" in result

    def test_spaces_in_file_tabs_in_search(self):
        original = "    int x = 0;\n    int y = 1;\n"
        blocks = [
            SearchReplaceBlock(
                search="\tint x = 0;\n\tint y = 1;",
                replace="\tint x = 42;\n\tint y = 99;",
            )
        ]
        result = apply_search_replace(original, blocks)
        assert "42" in result

    def test_mixed_indent_tabs_spaces(self):
        original = "void foo() {\t\n\treturn 0;\n}\n"
        blocks = [
            SearchReplaceBlock(
                search="void foo() {\n\treturn 0;",
                replace="void foo() {\n\treturn 1;",
            )
        ]
        assert "return 1;" in apply_search_replace(original, blocks)

    def test_fuzzy_match_with_duplicates_replaces_all(self):
        original = "  x = sizeof(T);  \n  y = sizeof(T);  \n"
        blocks = [
            SearchReplaceBlock(
                search="  x = sizeof(T);",
                replace="  x = 16;",
            )
        ]
        assert "16" in apply_search_replace(original, blocks)

    def test_fuzzy_multiline_with_trailing_spaces(self):
        original = "void f() {   \n    return 0;   \n}   \n"
        blocks = [
            SearchReplaceBlock(
                search="void f() {\n    return 0;\n}",
                replace="void f() {\n    return 1;\n}",
            )
        ]
        assert "return 1;" in apply_search_replace(original, blocks)


class TestApplySearchReplaceSpecialChars:
    def test_backslash_in_code(self):
        original = 'printf("hello\\nworld");\n'
        blocks = [
            SearchReplaceBlock(
                search='printf("hello\\nworld");',
                replace='printf("goodbye\\nworld");',
            )
        ]
        assert "goodbye" in apply_search_replace(original, blocks)

    def test_dollar_sign(self):
        original = "cost = $100;\n"
        blocks = [SearchReplaceBlock(search="$100", replace="$200")]
        assert "$200" in apply_search_replace(original, blocks)

    def test_curly_braces(self):
        original = "if (x) { y(); }\n"
        blocks = [
            SearchReplaceBlock(
                search="if (x) { y(); }",
                replace="if (x) { z(); }",
            )
        ]
        assert "z();" in apply_search_replace(original, blocks)

    def test_square_brackets(self):
        original = "arr[0] = arr[1];\n"
        blocks = [
            SearchReplaceBlock(
                search="arr[0] = arr[1];",
                replace="arr[0] = arr[2];",
            )
        ]
        assert "arr[2]" in apply_search_replace(original, blocks)

    def test_parentheses_and_asterisks(self):
        original = "int* p = (int*)malloc(sizeof(int));\n"
        blocks = [
            SearchReplaceBlock(
                search="int* p = (int*)malloc(sizeof(int));",
                replace="int* p = (int*)malloc(4);",
            )
        ]
        assert "malloc(4)" in apply_search_replace(original, blocks)

    def test_angle_brackets(self):
        original = "#include <stdio.h>\n"
        blocks = [
            SearchReplaceBlock(
                search="#include <stdio.h>",
                replace='#include "stdio.h"',
            )
        ]
        assert '"stdio.h"' in apply_search_replace(original, blocks)

    def test_pipe_and_ampersand(self):
        original = "if (a & b || c | d) {}\n"
        blocks = [
            SearchReplaceBlock(
                search="if (a & b || c | d) {}",
                replace="if (a & b && c | d) {}",
            )
        ]
        assert "&&" in apply_search_replace(original, blocks)

    def test_quotes_single_and_double(self):
        original = "char c = 'x'; char* s = \"hello\";\n"
        blocks = [
            SearchReplaceBlock(
                search="char c = 'x'; char* s = \"hello\";",
                replace="char c = 'y'; char* s = \"world\";",
            )
        ]
        result = apply_search_replace(original, blocks)
        assert "'y'" in result
        assert '"world"' in result


class TestApplySearchReplaceEdgeCases:
    def test_empty_original(self):
        blocks = [SearchReplaceBlock(search="x", replace="y")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace("", blocks)

    def test_empty_search_still_matches(self):
        original = "hello\n"
        blocks = [SearchReplaceBlock(search="", replace="world")]
        result = apply_search_replace(original, blocks)
        assert "world" in result

    def test_replace_with_longer_text(self):
        original = "a\n"
        blocks = [SearchReplaceBlock(search="a", replace="aaa\nbbb\nccc")]
        result = apply_search_replace(original, blocks)
        assert "aaa\nbbb\nccc" in result

    def test_replace_with_shorter_text(self):
        original = "aaa\nbbb\nccc\n"
        blocks = [SearchReplaceBlock(search="aaa\nbbb\nccc", replace="x")]
        result = apply_search_replace(original, blocks)
        assert result.startswith("x")

    def test_many_blocks(self):
        original = "\n".join(f"line{i}" for i in range(10)) + "\n"
        blocks = [
            SearchReplaceBlock(search=f"line{i}", replace=f"LINE{i}") for i in range(10)
        ]
        result = apply_search_replace(original, blocks)
        for i in range(10):
            assert f"LINE{i}" in result

    def test_search_at_beginning_of_file(self):
        original = "first line\nsecond line\n"
        blocks = [SearchReplaceBlock(search="first line", replace="new first")]
        assert apply_search_replace(original, blocks).startswith("new first")

    def test_search_at_end_of_file(self):
        original = "first line\nlast line\n"
        blocks = [SearchReplaceBlock(search="last line", replace="new last")]
        assert "new last" in apply_search_replace(original, blocks)

    def test_search_spanning_entire_file(self):
        original = "entire file content"
        blocks = [SearchReplaceBlock(search="entire file content", replace="new")]
        assert apply_search_replace(original, blocks) == "new"

    def test_unicode_content(self):
        original = "# Comment: café résumé\nx = 0\n"
        blocks = [
            SearchReplaceBlock(
                search="# Comment: café résumé",
                replace="# Comment: updated",
            )
        ]
        assert "updated" in apply_search_replace(original, blocks)

    def test_block_that_creates_duplicate_then_next_block_hits_it(self):
        original = "unique\nother\n"
        blocks = [
            SearchReplaceBlock(search="other", replace="unique"),
            SearchReplaceBlock(search="unique", replace="final"),
        ]
        result = apply_search_replace(original, blocks)
        assert result.count("final") == 2
