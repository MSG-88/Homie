"""Tests for news service."""
import pytest
from unittest.mock import patch, MagicMock
from homie_core.intelligence.news import NewsService


def test_get_headlines():
    service = NewsService(api_key="test_key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "ok",
        "articles": [
            {"title": "Test headline", "source": {"name": "Test Source"}, "url": "http://example.com", "description": "desc"},
        ],
    }
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        result = service.get_headlines(country="in")

    assert len(result["articles"]) == 1
    assert result["articles"][0]["title"] == "Test headline"


def test_get_headlines_no_api_key():
    service = NewsService(api_key="")
    result = service.get_headlines()
    assert "error" in result


def test_format_headlines():
    service = NewsService(api_key="test")
    data = {"articles": [{"title": "Big news", "source": "BBC", "url": "http://bbc.com"}]}
    text = service.format_headlines(data)
    assert "Big news" in text


def test_format_empty_headlines():
    service = NewsService(api_key="test")
    data = {"articles": []}
    text = service.format_headlines(data)
    assert "No news" in text
