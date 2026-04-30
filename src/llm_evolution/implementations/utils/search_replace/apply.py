"""Apply search/replace blocks to a text instance."""

from __future__ import annotations

import logging

from llm_evolution.implementations.utils.search_replace._fuzzy import (
    fuzzy_search_replace,
)
from llm_evolution.implementations.utils.search_replace.blocks import (
    SearchReplaceBlock,
)

logger = logging.getLogger(__name__)


def apply_search_replace(
    original: str,
    blocks: list[SearchReplaceBlock],
    strict: bool = True,
) -> str:
    """Apply search/replace blocks sequentially to *original*.

    Strategy per block:
    1. Exact unique match -> single replacement.
    2. Multiple exact matches -> replace ALL occurrences.
    3. No exact match -> fuzzy (whitespace-tolerant) match.
    4. Already-applied detection -> skip silently.
    5. Otherwise -> raise ``ValueError`` (strict) or warn-and-skip.
    """
    result = original
    applied: list[tuple[str, str]] = []
    applied_count = 0

    for i, block in enumerate(blocks, 1):
        count = result.count(block.search)

        if count == 1:
            result = result.replace(block.search, block.replace, 1)
            applied.append((block.search, block.replace))
            applied_count += 1
            continue

        if count > 1:
            logger.info("Block %d: search appears %d times; replacing all", i, count)
            result = result.replace(block.search, block.replace)
            applied.append((block.search, block.replace))
            applied_count += 1
            continue

        fuzzy = fuzzy_search_replace(result, block.search, block.replace)
        if fuzzy is not None:
            result = fuzzy
            applied.append((block.search, block.replace))
            applied_count += 1
            continue

        if _was_already_applied(block, applied):
            logger.info("Block %d: already handled by prior block; skipping", i)
            continue

        preview = block.search[:120].replace("\n", "\\n")
        message = (
            f"Search/replace block {i}: search text not found in file. "
            f"Searched for: {preview!r}"
        )
        if strict:
            raise ValueError(message)
        logger.warning("%s — skipping block", message)

    if not strict and applied_count == 0 and blocks:
        raise ValueError("No search/replace blocks could be applied to the file.")
    return result


def _was_already_applied(
    block: SearchReplaceBlock, applied: list[tuple[str, str]]
) -> bool:
    """Check if a block's search text was already consumed by a prior block."""
    for prev_search, prev_replace in applied:
        if prev_search == block.search:
            return True
        if block.search in prev_search:
            return True
        if block.search in prev_replace:
            return True
    return False
