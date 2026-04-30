"""Search/replace block dataclass and extraction from LLM responses."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_SR_BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH[ \t]*\n(.*?)\n=======[ \t]*\n(.*?)\n>>>>>>> REPLACE[ \t]*",
    re.DOTALL,
)


@dataclass(slots=True)
class SearchReplaceBlock:
    """A single search->replace edit."""

    search: str
    replace: str


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that some LLMs wrap around the blocks."""
    return (
        re.sub(r"^```[a-zA-Z]*\s*\n", "", text, flags=re.MULTILINE)
        .replace("\n```\n", "\n")
        .replace("\n```", "\n")
    )


def extract_search_replace(response: str) -> list[SearchReplaceBlock] | None:
    """Extract search/replace blocks from an LLM response.

    Format::

        <<<<<<< SEARCH
        old code exactly as it appears
        =======
        new replacement code
        >>>>>>> REPLACE

    Tolerates trailing whitespace on marker lines and markdown code fences
    wrapping the blocks.  Returns ``None`` if no blocks are found or every
    block is a no-op (search == replace).
    """
    matches = _SR_BLOCK_RE.findall(response)
    if not matches:
        cleaned = _strip_markdown_fences(response)
        matches = _SR_BLOCK_RE.findall(cleaned)
    if not matches:
        return None
    blocks = [SearchReplaceBlock(search=s, replace=r) for s, r in matches if s != r]
    if not blocks:
        logger.warning("All search/replace blocks were no-ops (search == replace)")
        return None
    return blocks
