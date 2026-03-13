"""Weather data fetching and formatting."""
from __future__ import annotations


class WeatherService:
    """Fetches weather data from OpenWeatherMap API."""

    _BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: str):
        self._api_key = api_key

    def get_current(self, city: str) -> dict:
        if not self._api_key:
            return {"error": "No weather API key configured. Use /connect weather to set one up."}
        try:
            import requests
            resp = requests.get(
                f"{self._BASE_URL}/weather",
                params={"q": city, "appid": self._api_key, "units": "metric"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "city": data.get("name", city),
                "temp": data["main"]["temp"],
                "feels_like": data["main"].get("feels_like"),
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"] if data.get("weather") else "unknown",
                "wind_speed": data.get("wind", {}).get("speed", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_forecast(self, city: str, days: int = 3) -> dict:
        if not self._api_key:
            return {"error": "No weather API key configured."}
        try:
            import requests
            resp = requests.get(
                f"{self._BASE_URL}/forecast",
                params={"q": city, "appid": self._api_key, "units": "metric", "cnt": days * 8},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            forecasts = []
            for item in data.get("list", [])[:days * 8:8]:
                forecasts.append({
                    "dt_txt": item.get("dt_txt", ""),
                    "temp": item["main"]["temp"],
                    "description": item["weather"][0]["description"] if item.get("weather") else "",
                })
            return {"city": data.get("city", {}).get("name", city), "forecasts": forecasts}
        except Exception as e:
            return {"error": str(e)}

    def format_current(self, data: dict) -> str:
        if "error" in data:
            return data["error"]
        lines = [f"**Weather in {data['city']}:**"]
        lines.append(f"  Temperature: {data['temp']}\u00b0C")
        if data.get("feels_like"):
            lines.append(f"  Feels like: {data['feels_like']}\u00b0C")
        lines.append(f"  Conditions: {data['description']}")
        lines.append(f"  Humidity: {data['humidity']}%")
        lines.append(f"  Wind: {data['wind_speed']} m/s")
        return "\n".join(lines)

    def format_forecast(self, data: dict) -> str:
        if "error" in data:
            return data["error"]
        lines = [f"**Forecast for {data['city']}:**"]
        for f in data.get("forecasts", []):
            lines.append(f"  {f['dt_txt']}: {f['temp']}\u00b0C, {f['description']}")
        return "\n".join(lines)
