"""Whitespace-tolerant fuzzy matching for search/replace blocks.

Strategies, applied in order of increasing aggressiveness:

1. Trailing-whitespace normalisation (rstrip per line).
2. Leading tab <-> 4 spaces normalisation.
3. Aggressive whitespace collapse: NBSP -> space, internal runs of any
   whitespace -> single space, leading tabs -> 4 spaces.
4. Token-level normalisation: identifiers + punctuation only, joined by
   single spaces.  Matches code with arbitrary spacing around tokens.

In addition the matcher first normalises CRLF / CR to LF, strips a
leading BOM, and trims leading/trailing blank lines from the search
text.  These transforms preserve token order and identifier spelling,
so semantically distinct code still fails to match.
"""

from __future__ import annotations

import re

_WS_RUN = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_NBSP = " "
_BOM = "﻿"


def _normalize_line_endings(text: str) -> str:
    """Collapse CRLF/CR to LF, strip leading BOM, neutralise \\f and \\v.

    Form-feed and vertical-tab are split as line separators by Python's
    ``splitlines``, which would shred otherwise-matchable lines.  We
    replace them with spaces so they fall under the in-line whitespace
    normalisers instead.
    """
    if text.startswith(_BOM):
        text = text.lstrip(_BOM)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\f", " ").replace("\v", " ")


def _strip_blank_lines(lines: list[str]) -> list[str]:
    """Trim leading and trailing blank lines from a search-line list."""
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _norm_line_trailing(line: str) -> str:
    """Strategy 1: drop trailing whitespace only."""
    return line.rstrip()


def _norm_line_leading_tabs(line: str) -> str:
    """Strategy 2: drop trailing ws, expand leading tabs to 4 spaces."""
    stripped = line.rstrip()
    leading = len(stripped) - len(stripped.lstrip())
    prefix = stripped[:leading].replace("\t", "    ")
    return prefix + stripped[leading:]


def _norm_line_aggressive(line: str) -> str:
    """Strategy 3: collapse NBSP and internal whitespace runs to one space."""
    line = line.replace(_NBSP, " ").rstrip()
    stripped = line.lstrip()
    leading_len = len(line) - len(stripped)
    prefix = line[:leading_len].replace("\t", "    ")
    body = _WS_RUN.sub(" ", stripped)
    return prefix + body


def _norm_line_tokens(line: str) -> str:
    """Strategy 4: tokenise the line; identifiers and punctuation only."""
    return " ".join(_TOKEN_RE.findall(line))


def _find_all_line_matches(text_norm: list[str], search_norm: list[str]) -> list[int]:
    """Return start indices of all non-overlapping line-based matches."""
    slen = len(search_norm)
    if slen == 0:
        return []
    matches: list[int] = []
    i = 0
    while i <= len(text_norm) - slen:
        if text_norm[i : i + slen] == search_norm:
            matches.append(i)
            i += slen
        else:
            i += 1
    return matches


def _replace_line_matches(
    text_lines: list[str],
    match_starts: list[int],
    search_len: int,
    replace: str,
) -> str:
    """Replace all matched line regions in *text_lines*."""
    replace_lines = replace.splitlines(keepends=True)
    result_lines = list(text_lines)
    for start in reversed(match_starts):
        end = start + search_len
        patched = list(replace_lines)
        if patched and not patched[-1].endswith("\n"):
            has_trailing = end < len(result_lines) or (
                result_lines and result_lines[-1].endswith("\n")
            )
            if has_trailing:
                patched[-1] += "\n"
        result_lines[start:end] = patched
    return "".join(result_lines)


def fuzzy_search_replace(text: str, search: str, replace: str) -> str | None:
    """Match *search* against *text* with whitespace tolerance; replace all.

    Returns the modified text or ``None`` if nothing matched even under
    the most aggressive normalisation.
    """
    text_norm = _normalize_line_endings(text)
    search_norm = _normalize_line_endings(search)
    text_lines = text_norm.splitlines(keepends=True)
    search_lines = _strip_blank_lines(search_norm.splitlines())
    if not search_lines:
        return None
    slen = len(search_lines)

    for normalizer in (
        _norm_line_trailing,
        _norm_line_leading_tabs,
        _norm_line_aggressive,
        _norm_line_tokens,
    ):
        s_norm = [normalizer(ln) for ln in search_lines]
        t_norm = [normalizer(ln) for ln in text_lines]
        if not any(s.strip() for s in s_norm):
            continue
        matches = _find_all_line_matches(t_norm, s_norm)
        if matches:
            return _replace_line_matches(text_lines, matches, slen, replace)

    return None
