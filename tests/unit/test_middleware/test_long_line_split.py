from __future__ import annotations

import pytest

from homie_core.config import HomieConfig, ContextConfig
from homie_core.middleware.long_line_split import LongLineSplitMiddleware


def make_mw(threshold: int = 10) -> LongLineSplitMiddleware:
    cfg = HomieConfig(context=ContextConfig(long_line_threshold=threshold))
    return LongLineSplitMiddleware(cfg)


def test_short_lines_pass_through_unchanged():
    mw = make_mw(threshold=10)
    result = mw.wrap_tool_result("tool", "short\nlines\nok")
    assert result == "short\nlines\nok"


def test_long_line_is_split_with_markers():
    mw = make_mw(threshold=5)
    # 10-char line -> 2 chunks of 5
    result = mw.wrap_tool_result("tool", "0123456789")
    assert result == "[line 1.1] 01234\n[line 1.2] 56789"


def test_long_line_three_chunks():
    mw = make_mw(threshold=4)
    # 10-char line -> chunks: 0123, 4567, 89
    result = mw.wrap_tool_result("tool", "0123456789")
    assert result == "[line 1.1] 0123\n[line 1.2] 4567\n[line 1.3] 89"


def test_multiple_long_lines_handled_independently():
    mw = make_mw(threshold=5)
    result = mw.wrap_tool_result("tool", "AAAAAAAAAA\nBBBBBBBBBB")
    lines = result.split("\n")
    # First long line: 2 chunks labelled line 1
    assert lines[0] == "[line 1.1] AAAAA"
    assert lines[1] == "[line 1.2] AAAAA"
    # Second long line: 2 chunks labelled line 2
    assert lines[2] == "[line 2.1] BBBBB"
    assert lines[3] == "[line 2.2] BBBBB"


def test_mixed_short_and_long_lines():
    mw = make_mw(threshold=5)
    result = mw.wrap_tool_result("tool", "short\nAAAAAAAAAAAA\nok")
    lines = result.split("\n")
    assert lines[0] == "short"
    assert lines[1] == "[line 2.1] AAAAA"
    assert lines[2] == "[line 2.2] AAAAA"
    assert lines[3] == "[line 2.3] AA"
    assert lines[4] == "ok"


def test_threshold_boundary_exactly_at_threshold_no_split():
    mw = make_mw(threshold=10)
    line = "a" * 10  # exactly threshold
    result = mw.wrap_tool_result("tool", line)
    assert result == line  # no markers


def test_threshold_boundary_one_over_splits():
    mw = make_mw(threshold=10)
    line = "a" * 11  # threshold + 1
    result = mw.wrap_tool_result("tool", line)
    assert "[line 1.1]" in result
    assert "[line 1.2]" in result


def test_empty_result():
    mw = make_mw(threshold=5)
    assert mw.wrap_tool_result("tool", "") == ""


def test_single_empty_line():
    # "\n".splitlines() yields [""] — one empty-string element — so the
    # result after the round-trip through splitlines/join is "".
    mw = make_mw(threshold=5)
    assert mw.wrap_tool_result("tool", "\n") == ""


def test_name_and_order():
    mw = make_mw()
    assert mw.name == "long_line_split"
    assert mw.order == 85


def test_reads_threshold_from_config():
    cfg = HomieConfig(context=ContextConfig(long_line_threshold=3))
    mw = LongLineSplitMiddleware(cfg)
    # 6-char line -> 2 chunks at threshold=3
    result = mw.wrap_tool_result("tool", "abcdef")
    assert "[line 1.1] abc" in result
    assert "[line 1.2] def" in result
