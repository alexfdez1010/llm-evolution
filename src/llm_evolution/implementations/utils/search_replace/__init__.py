"""Search/replace block parsing and application for LLM-driven edits.

Reusable, dependency-free utilities for applying LLM edits expressed as
search/replace blocks.  Useful for evolving any text artefact (code,
config, prose) where exact-string edits are easier than diffs.
"""

from llm_evolution.implementations.utils.search_replace.apply import (
    apply_search_replace,
)
from llm_evolution.implementations.utils.search_replace.blocks import (
    SearchReplaceBlock,
    extract_search_replace,
)
from llm_evolution.implementations.utils.search_replace.prompts import (
    number_lines,
    search_replace_error_feedback,
    search_replace_format_example,
)

__all__ = [
    "SearchReplaceBlock",
    "apply_search_replace",
    "extract_search_replace",
    "number_lines",
    "search_replace_error_feedback",
    "search_replace_format_example",
]
