"""Prompt helpers: format example, line numbering, error feedback."""

from __future__ import annotations

import re


def number_lines(code: str, every: int = 10) -> str:
    """Add line numbers every *every*-th line and on the first line."""
    lines = code.splitlines(keepends=True)
    width = len(str(len(lines)))
    out: list[str] = []
    for i, line in enumerate(lines, 1):
        if i == 1 or i % every == 0:
            out.append(f"{i:>{width}}| {line}")
        else:
            out.append(f"{' ' * width}| {line}")
    return "".join(out)


def _extract_error_line(error_message: str) -> int | None:
    m = re.search(r"at line (\d+)", error_message)
    return int(m.group(1)) if m else None


def _source_context_snippet(code: str, line_no: int, radius: int = 5) -> str:
    lines = code.splitlines()
    start = max(0, line_no - 1 - radius)
    end = min(len(lines), line_no + radius)
    width = len(str(end))
    out: list[str] = []
    for i in range(start, end):
        marker = ">>>" if i == line_no - 1 else "   "
        out.append(f"{marker} {i + 1:>{width}}| {lines[i]}")
    return "\n".join(out)


def search_replace_format_example() -> str:
    """Return a concrete search/replace example for the LLM."""
    return """\
## Search/replace block format

Each block MUST have the SEARCH section (old code) and a DIFFERENT REPLACE
section (new code).  The REPLACE section must NOT be identical to SEARCH —
that would be a no-op.  Every block must change something.

Example — single-line change:

<<<<<<< SEARCH
old line
=======
new line
>>>>>>> REPLACE

Example — multi-line change (include enough context to be unique):

<<<<<<< SEARCH
    foo = compute(x)
    bar = compute(y)
=======
    foo = compute_fast(x)
    bar = compute_fast(y)
>>>>>>> REPLACE

IMPORTANT:
- SEARCH must match the file EXACTLY (same whitespace, same indentation).
- If the same text appears in multiple places and you want to change ALL
  of them, a single block is enough — all occurrences will be replaced.
- REPLACE must be DIFFERENT from SEARCH — every block must make a change.
"""


def search_replace_error_feedback(
    file_name: str,
    code: str,
    error_message: str,
    code_fence_lang: str = "",
) -> str:
    """Build feedback for the LLM when search/replace blocks failed.

    Args:
        file_name: Name of the artefact (purely for prompt readability).
        code: Current file content (will be shown with line numbers).
        error_message: Error string from `apply_search_replace` or similar.
        code_fence_lang: Optional language tag for the code fence.
    """
    context = ""
    line_no = _extract_error_line(error_message)
    if line_no:
        snippet = _source_context_snippet(code, line_no)
        context = (
            f"\nLines around the error location (line {line_no}):\n"
            f"```\n{snippet}\n```\n"
        )

    return (
        f"Your previous edit could NOT be applied.  Read the error carefully "
        f"and try again.\n\n"
        f"Error: {error_message}\n{context}\n"
        f"RULES — read these before responding:\n"
        f"1. The SEARCH text must be copied EXACTLY from the current file "
        f"(same indentation, same whitespace, character for character).\n"
        f"2. If the same text appears multiple times, a single search/replace "
        f"block is enough — all occurrences will be replaced.\n"
        f"3. The REPLACE text must be DIFFERENT from the SEARCH text.  "
        f"Every block must change something — do not emit no-op blocks.\n\n"
        f"Current {file_name} (with line numbers):\n"
        f"```{code_fence_lang}\n{number_lines(code)}\n```\n\n"
        f"{search_replace_format_example()}"
    )
