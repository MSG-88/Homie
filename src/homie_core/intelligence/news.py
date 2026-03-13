"""News data fetching and formatting."""
from __future__ import annotations


class NewsService:
    """Fetches news from NewsAPI."""

    _BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str):
        self._api_key = api_key

    def get_headlines(self, country: str = "us", query: str = "", category: str = "") -> dict:
        if not self._api_key:
            return {"error": "No news API key configured. Use /connect news to set one up.", "articles": []}
        try:
            import requests
            params = {"apiKey": self._api_key, "country": country, "pageSize": 10}
            if query:
                params["q"] = query
            if category:
                params["category"] = category
            resp = requests.get(f"{self._BASE_URL}/top-headlines", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for a in data.get("articles", []):
                articles.append({
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "url": a.get("url", ""),
                    "description": a.get("description", ""),
                })
            return {"articles": articles}
        except Exception as e:
            return {"error": str(e), "articles": []}

    def format_headlines(self, data: dict) -> str:
        if "error" in data and not data.get("articles"):
            return data["error"]
        if not data.get("articles"):
            return "No news articles found."
        lines = ["**Top Headlines:**"]
        for i, a in enumerate(data["articles"][:10], 1):
            source = f" ({a['source']})" if a.get("source") else ""
            lines.append(f"  {i}. {a['title']}{source}")
        return "\n".join(lines)
