from __future__ import annotations

import pytest

from homie_core.middleware.token_utils import estimate_tokens, estimate_conversation_tokens


def test_estimate_tokens_known_string():
    # 400 chars -> 100 tokens
    text = "x" * 400
    assert estimate_tokens(text) == 100


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0


def test_estimate_tokens_four_chars():
    assert estimate_tokens("abcd") == 1


def test_estimate_tokens_three_chars_truncates():
    # 3 // 4 == 0
    assert estimate_tokens("abc") == 0


def test_estimate_tokens_eight_chars():
    assert estimate_tokens("abcdefgh") == 2


def test_estimate_tokens_longer_text():
    text = "a" * 1000
    assert estimate_tokens(text) == 250


def test_estimate_conversation_tokens_basic():
    conversation = [
        {"role": "user", "content": "a" * 400},     # 100 tokens
        {"role": "assistant", "content": "b" * 200}, # 50 tokens
    ]
    assert estimate_conversation_tokens(conversation) == 150


def test_estimate_conversation_tokens_empty_list():
    assert estimate_conversation_tokens([]) == 0


def test_estimate_conversation_tokens_missing_content():
    # Messages without 'content' key should be treated as empty
    conversation = [
        {"role": "user"},
        {"role": "assistant", "content": "a" * 80},  # 20 tokens
    ]
    assert estimate_conversation_tokens(conversation) == 20


def test_estimate_conversation_tokens_empty_content():
    conversation = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "x" * 40},  # 10 tokens
    ]
    assert estimate_conversation_tokens(conversation) == 10


def test_estimate_conversation_tokens_sum_of_messages():
    conversation = [
        {"role": "user", "content": "a" * 4},    # 1 token
        {"role": "assistant", "content": "b" * 8},  # 2 tokens
        {"role": "user", "content": "c" * 12},   # 3 tokens
    ]
    assert estimate_conversation_tokens(conversation) == 6
