"""Tests for weather service."""
import pytest
from unittest.mock import patch, MagicMock
from homie_core.intelligence.weather import WeatherService


def test_get_current_weather():
    service = WeatherService(api_key="test_key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "main": {"temp": 32, "humidity": 65, "feels_like": 35},
        "weather": [{"description": "clear sky"}],
        "wind": {"speed": 3.5},
        "name": "Chennai",
    }
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        result = service.get_current("Chennai")

    assert result["city"] == "Chennai"
    assert result["temp"] == 32
    assert result["description"] == "clear sky"


def test_get_weather_no_api_key():
    service = WeatherService(api_key="")
    result = service.get_current("Chennai")
    assert "error" in result


def test_format_weather():
    service = WeatherService(api_key="test")
    data = {"city": "Chennai", "temp": 32, "humidity": 65, "description": "clear sky", "wind_speed": 3.5, "feels_like": 35}
    text = service.format_current(data)
    assert "Chennai" in text
    assert "32" in text
    assert "clear sky" in text


def test_format_weather_error():
    service = WeatherService(api_key="test")
    data = {"error": "API failed"}
    text = service.format_current(data)
    assert "API failed" in text
