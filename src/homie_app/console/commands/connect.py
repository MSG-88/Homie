"""Handlers for /connect, /disconnect, /connections."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter

_OAUTH_PROVIDERS = {"gmail", "slack", "twitter", "reddit", "linkedin", "facebook", "instagram"}
_APIKEY_PROVIDERS = {"weather", "news"}
_ALL_PROVIDERS = _OAUTH_PROVIDERS | _APIKEY_PROVIDERS | {"blog"}

_SM_PLATFORMS = {
    "twitter": ("https://twitter.com/i/oauth2/authorize", "https://api.twitter.com/2/oauth2/token", ["tweet.read", "users.read"], 8551),
    "reddit": ("https://www.reddit.com/api/v1/authorize", "https://www.reddit.com/api/v1/access_token", ["identity", "read"], 8552),
    "linkedin": ("https://www.linkedin.com/oauth/v2/authorization", "https://www.linkedin.com/oauth/v2/accessToken", ["r_liteprofile", "r_emailaddress"], 8553),
    "facebook": ("https://www.facebook.com/v18.0/dialog/oauth", "https://graph.facebook.com/v18.0/oauth/access_token", ["public_profile", "email"], 8554),
    "instagram": ("https://api.instagram.com/oauth/authorize", "https://api.instagram.com/oauth/access_token", ["user_profile", "user_media"], 8555),
}


def _handle_connect(args: str, **ctx) -> str:
    provider = args.strip().lower()
    if not provider:
        providers = ", ".join(sorted(_ALL_PROVIDERS))
        return f"Usage: /connect <provider>\nAvailable: {providers}"

    if provider not in _ALL_PROVIDERS:
        return f"Unknown provider: {provider}. Available: {', '.join(sorted(_ALL_PROVIDERS))}"

    if provider == "gmail":
        return _connect_gmail(**ctx)
    elif provider in _APIKEY_PROVIDERS:
        return _connect_apikey(provider, **ctx)
    elif provider == "blog":
        return _connect_blog(**ctx)
    else:
        return _connect_social_oauth(provider, **ctx)


def _connect_gmail(**ctx) -> str:
    """Run Gmail OAuth flow inline in the console."""
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        existing = vault.get_credential("gmail_oauth_client", "default")
        if existing:
            client_id = existing.access_token
            client_secret = existing.refresh_token
        else:
            print("  Gmail requires OAuth credentials from Google Cloud Console.")
            print("  See: https://console.cloud.google.com/apis/credentials")
            client_id = input("  Client ID: ").strip()
            client_secret = input("  Client Secret: ").strip()
            if not client_id or not client_secret:
                return "Cancelled — client ID and secret are required."
            vault.store_credential(
                provider="gmail_oauth_client", account_id="default",
                token_type="oauth_client",
                access_token=client_id, refresh_token=client_secret,
                scopes=[], expires_at=None,
            )

        from homie_core.email.oauth import GmailOAuth
        oauth = GmailOAuth(client_id=client_id, client_secret=client_secret)
        auth_url = oauth.get_auth_url(use_local_server=True)

        import webbrowser
        print(f"\n  Opening browser for Google sign-in...")
        print(f"  If it doesn't open, visit: {auth_url}\n")
        webbrowser.open(auth_url)

        print("  Waiting for authorization (up to 120s)...")
        code = oauth.wait_for_redirect(timeout=120)
        if not code:
            return "Authorization timed out. Try again with /connect gmail"

        tokens = oauth.exchange(code, client_id=client_id, client_secret=client_secret)
        profile = oauth.get_profile(tokens["access_token"])
        email_addr = profile.get("emailAddress", "unknown")

        vault.store_credential(
            provider="gmail", account_id=email_addr,
            token_type="oauth2",
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token", ""),
            scopes=tokens.get("scope", "").split(),
            expires_at=tokens.get("expires_at"),
        )
        vault.set_connection_status("gmail", connected=True, label=email_addr)
        vault.log_consent("gmail", "connected", scopes=tokens.get("scope", "").split())

        return f"Gmail connected: {email_addr}"
    except Exception as e:
        return f"Gmail connection failed: {e}"


def _connect_social_oauth(provider: str, **ctx) -> str:
    """Run social media OAuth flow."""
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        print(f"  {provider.title()} OAuth setup...")
        client_id = input(f"  {provider.title()} Client ID: ").strip()
        client_secret = input(f"  {provider.title()} Client Secret: ").strip()
        if not client_id or not client_secret:
            return "Cancelled."

        if provider not in _SM_PLATFORMS:
            return f"OAuth not configured for {provider} yet."

        auth_url_tpl, token_url, scopes, port = _SM_PLATFORMS[provider]
        import secrets
        state = secrets.token_urlsafe(16)

        from urllib.parse import urlencode
        params = {
            "client_id": client_id,
            "redirect_uri": f"http://localhost:{port}/callback",
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
        }
        auth_url = f"{auth_url_tpl}?{urlencode(params)}"

        import webbrowser
        print(f"\n  Opening browser for {provider.title()} sign-in...")
        print(f"  If it doesn't open, visit: {auth_url}\n")
        webbrowser.open(auth_url)

        from homie_core.email.oauth import _wait_for_oauth_redirect
        code = _wait_for_oauth_redirect(port=port, timeout=120)
        if not code:
            return f"Authorization timed out for {provider}."

        import requests
        resp = requests.post(token_url, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": f"http://localhost:{port}/callback",
            "client_id": client_id,
            "client_secret": client_secret,
        })
        resp.raise_for_status()
        tokens = resp.json()

        vault.store_credential(
            provider=provider, account_id="default",
            token_type="oauth2",
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token", ""),
            scopes=scopes,
            expires_at=tokens.get("expires_in"),
        )
        vault.set_connection_status(provider, connected=True)
        vault.log_consent(provider, "connected", scopes=scopes)

        return f"{provider.title()} connected!"
    except Exception as e:
        return f"{provider.title()} connection failed: {e}"


def _connect_apikey(provider: str, **ctx) -> str:
    """Connect an API-key-based provider (weather, news)."""
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        hints = {
            "weather": "Get a free API key from https://openweathermap.org/api",
            "news": "Get a free API key from https://newsapi.org/register",
        }
        print(f"  {hints.get(provider, '')}")
        api_key = input(f"  {provider.title()} API Key: ").strip()
        if not api_key:
            return "Cancelled."

        # Validate API key with a test call
        import requests
        if provider == "weather":
            resp = requests.get("https://api.openweathermap.org/data/2.5/weather",
                                params={"q": "London", "appid": api_key}, timeout=10)
            if resp.status_code == 401:
                return "Invalid API key. Please check and try again."
        elif provider == "news":
            resp = requests.get("https://newsapi.org/v2/top-headlines",
                                params={"country": "us", "apiKey": api_key, "pageSize": 1}, timeout=10)
            if resp.status_code == 401:
                return "Invalid API key. Please check and try again."

        vault.store_credential(
            provider=provider, account_id="default",
            token_type="api_key",
            access_token=api_key, refresh_token="",
            scopes=[], expires_at=None,
        )
        vault.set_connection_status(provider, connected=True)
        return f"{provider.title()} connected!"
    except Exception as e:
        return f"{provider.title()} connection failed: {e}"


def _connect_blog(**ctx) -> str:
    """Connect a blog via RSS feed URL."""
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        feed_url = input("  Blog RSS/Atom feed URL: ").strip()
        if not feed_url:
            return "Cancelled."

        vault.store_credential(
            provider="blog", account_id="default",
            token_type="api_key",
            access_token=feed_url, refresh_token="",
            scopes=[], expires_at=None,
        )
        vault.set_connection_status("blog", connected=True, label=feed_url)
        return f"Blog connected: {feed_url}"
    except Exception as e:
        return f"Blog connection failed: {e}"


def _handle_disconnect(args: str, **ctx) -> str:
    provider = args.strip().lower()
    if not provider:
        return "Usage: /disconnect <provider>"
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        vault.set_connection_status(provider, connected=False)
        vault.log_consent(provider, "disconnected")
        return f"Disconnected: {provider}"
    except Exception as e:
        return f"Could not disconnect {provider}: {e}"


def _handle_connections(args: str, **ctx) -> str:
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        connections = vault.get_all_connections()
        if not connections:
            return "No connections configured. Use /connect <provider> to set one up."
        lines = ["**Connections:**"]
        for c in connections:
            icon = "+" if c.connected else "-"
            lines.append(f"  [{icon}] {c.provider}: {c.display_label or 'no label'}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not check connections: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(name="connect", description="Connect a provider (gmail, linkedin, weather, etc.)", args_spec="<provider>", handler_fn=_handle_connect))
    router.register(SlashCommand(name="disconnect", description="Disconnect a provider", args_spec="<provider>", handler_fn=_handle_disconnect))
    router.register(SlashCommand(name="connections", description="Show all provider connections", handler_fn=_handle_connections))
