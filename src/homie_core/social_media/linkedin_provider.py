"""LinkedIn social media provider — profile via OpenID, feed/publish via Community API.

Current scopes: openid, profile, email, w_member_social.
- OpenID userinfo (/v2/userinfo) — always works for profile data.
- REST API (/rest/*) — requires Community Management API product enabled
  in the LinkedIn Developer Portal. Without it, feed/publish return empty.
- v2 unversioned (/v2/me, /v2/ugcPosts) — deprecated, requires legacy products.

To enable full functionality, add the "Community Management API" product
at https://www.linkedin.com/developers/apps → Products tab.
"""
from __future__ import annotations

import logging
import time

import requests

from homie_core.social_media.models import ProfileInfo, ProfileStats, SocialPost
from homie_core.social_media.provider import (
    FeedProvider,
    ProfileProvider,
    PublishProvider,
    SocialMediaProviderBase,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.linkedin.com/v2"
REST_URL = "https://api.linkedin.com/rest"
USERINFO_URL = "https://api.linkedin.com/v2/userinfo"
# LinkedIn API version for /rest endpoints (YYYYMM format)
API_VERSION = "202503"


class LinkedInProvider(SocialMediaProviderBase, FeedProvider, ProfileProvider, PublishProvider):
    """LinkedIn REST API provider.

    Implements Feed, Profile, and Publish capabilities.
    Does **not** implement DirectMessageProvider — the LinkedIn API does not
    expose messaging endpoints to third-party applications.
    """

    platform_name: str = "linkedin"

    def __init__(self) -> None:
        super().__init__()
        self._person_id: str | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, credential) -> bool:
        """Store token, call userinfo to verify, and cache person URN."""
        try:
            self._token = credential.access_token
            # Try OpenID userinfo first (works with openid+profile scopes)
            try:
                info = self._userinfo()
                self._person_id = info.get("sub")
                self._userinfo_cache = info
            except Exception:
                # Fall back to legacy /me endpoint
                resp = self._call("GET", "/me")
                self._person_id = resp.get("id")
                self._userinfo_cache = None
            self._connected = True
            return True
        except Exception:
            logger.exception("Failed to connect LinkedIn")
            self._connected = False
            return False

    def _userinfo(self) -> dict:
        """Fetch OpenID Connect userinfo (name, email, picture, sub)."""
        resp = requests.get(
            USERINFO_URL,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    def _call(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
        retries: int = 2,
        versioned: bool = False,
    ) -> dict:
        """Issue an authenticated request to the LinkedIn API.

        Automatically retries on 429 (rate-limit) responses up to *retries*
        times with a 1-second back-off.

        If *versioned* is True, uses the /rest base URL with LinkedIn-Version header.
        Otherwise uses the legacy /v2 base URL with X-Restli-Protocol-Version.
        """
        if versioned:
            url = f"{REST_URL}{path}"
            headers = {
                "Authorization": f"Bearer {self._token}",
                "LinkedIn-Version": API_VERSION,
            }
        else:
            url = f"{BASE_URL}{path}"
            headers = {
                "Authorization": f"Bearer {self._token}",
                "X-Restli-Protocol-Version": "2.0.0",
            }

        for attempt in range(retries + 1):
            resp = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
            )

            if resp.status_code == 429 and attempt < retries:
                time.sleep(1)
                continue

            resp.raise_for_status()
            return resp.json()

        # Should not reach here, but satisfy type-checkers.
        resp.raise_for_status()  # type: ignore[possibly-undefined]
        return resp.json()  # type: ignore[possibly-undefined]

    # ------------------------------------------------------------------
    # FeedProvider
    # ------------------------------------------------------------------

    def get_feed(self, limit: int = 20) -> list[SocialPost]:
        """Fetch the user's feed posts.

        Requires Community Management API product. Returns empty list
        if the product is not enabled.
        """
        try:
            # Try versioned API first (Community Management)
            data = self._call(
                "GET", "/posts",
                params={
                    "q": "author",
                    "author": f"urn:li:person:{self._person_id}",
                    "count": limit,
                },
                versioned=True,
            )
        except Exception:
            try:
                # Fall back to legacy v2
                data = self._call("GET", "/ugcPosts", params={
                    "q": "authors",
                    "authors": f"List(urn:li:person:{self._person_id})",
                    "count": limit,
                })
            except Exception:
                logger.warning("LinkedIn feed unavailable — Community Management API not enabled")
                return []

        posts: list[SocialPost] = []
        for item in data.get("elements", []):
            content = item.get("commentary", "")
            if not content:
                # Legacy UGC format
                content = (item.get("specificContent", {})
                          .get("com.linkedin.ugc.ShareContent", {})
                          .get("shareCommentary", {}).get("text", ""))
            posts.append(
                SocialPost(
                    id=item.get("id", ""),
                    platform="linkedin",
                    author=item.get("author", ""),
                    content=content,
                    timestamp=item.get("createdAt", item.get("created", {}).get("time", 0.0)),
                    likes=item.get("likeCount", 0),
                    comments=item.get("commentCount", 0),
                    shares=item.get("shareCount", 0),
                )
            )
        return posts

    def search_posts(self, query: str, limit: int = 10) -> list[SocialPost]:
        """Search posts — requires Community Management API product."""
        logger.info("LinkedIn post search requires Community Management API")
        return []

    # ------------------------------------------------------------------
    # ProfileProvider
    # ------------------------------------------------------------------

    def get_profile(self, username: str | None = None) -> ProfileInfo:
        """Fetch the authenticated user's profile.

        Uses cached OpenID userinfo when available, falls back to the
        legacy ``/me`` endpoint for richer fields.
        """
        # Try OpenID userinfo (available with openid+profile+email scopes)
        info = getattr(self, "_userinfo_cache", None)
        if info is None:
            try:
                info = self._userinfo()
            except Exception:
                info = None

        if info:
            display_name = info.get("name", "")
            email = info.get("email", "")
            return ProfileInfo(
                platform="linkedin",
                username=info.get("sub", ""),
                display_name=display_name,
                bio=email,
                avatar_url=info.get("picture"),
                profile_url=f"https://www.linkedin.com/in/{username}" if username else None,
            )

        # Legacy v2 /me fallback
        data = self._call(
            "GET",
            "/me",
            params={
                "projection": "(id,firstName,lastName,headline,profilePicture,vanityName)"
            },
        )

        first = _localized_field(data.get("firstName", {}))
        last = _localized_field(data.get("lastName", {}))
        display_name = f"{first} {last}".strip()

        return ProfileInfo(
            platform="linkedin",
            username=data.get("vanityName", data.get("id", "")),
            display_name=display_name,
            bio=data.get("headline", {}).get("localized", {}).get("en_US", "")
            if isinstance(data.get("headline"), dict)
            else str(data.get("headline", "")),
            avatar_url=data.get("profilePicture", {})
            .get("displayImage~", {})
            .get("elements", [{}])[0]
            .get("identifiers", [{}])[0]
            .get("identifier"),
            profile_url=f"https://www.linkedin.com/in/{data.get('vanityName', '')}",
        )

    def get_stats(self) -> ProfileStats:
        data = self._call("GET", "/me", params={"projection": "(numConnections)"})
        return ProfileStats(
            platform="linkedin",
            followers=data.get("numConnections", 0),
        )

    # ------------------------------------------------------------------
    # PublishProvider
    # ------------------------------------------------------------------

    def publish(self, content: str, media_urls: list[str] | None = None) -> dict:
        person_urn = f"urn:li:person:{self._person_id}"
        payload: dict = {
            "author": person_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": content},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }

        if media_urls:
            media_items = [
                {"status": "READY", "originalUrl": url} for url in media_urls
            ]
            share = payload["specificContent"]["com.linkedin.ugc.ShareContent"]
            share["shareMediaCategory"] = "ARTICLE"
            share["media"] = media_items

        data = self._call("POST", "/ugcPosts", json_body=payload)
        return {"id": data.get("id", ""), "status": "published"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _localized_field(field_obj: dict) -> str:
    """Extract the first localized value from a LinkedIn multi-locale field."""
    localized = field_obj.get("localized", {})
    if not localized:
        return ""
    # Return the first available locale value.
    return next(iter(localized.values()), "")
