"""Gmail API implementation of EmailProvider.

Uses google-api-python-client to interact with Gmail.
All API calls go through self._service (a googleapiclient Resource).
"""
from __future__ import annotations

import base64
import email.mime.text
import time
from typing import Any

from homie_core.email.models import EmailMessage, HistoryChange, Label
from homie_core.email.provider import EmailProvider


def _header(headers: list[dict], name: str) -> str:
    """Extract a header value from Gmail API headers list."""
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def _decode_body(payload: dict) -> str:
    """Decode message body from Gmail API payload."""
    # Direct body
    body_data = payload.get("body", {}).get("data", "")
    if body_data:
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")

    # Multipart — find text/plain
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        # Nested multipart
        if part.get("parts"):
            result = _decode_body(part)
            if result:
                return result

    return ""


class GmailProvider(EmailProvider):
    """Gmail API implementation."""

    def __init__(self, account_id: str):
        self._account_id = account_id
        self._service = None  # Set by authenticate() or test injection
        self._creds = None
        self._vault = None
        self._credential_id = None

    def authenticate(self, credential, vault=None, client_id: str = "", client_secret: str = "") -> None:
        """Build Gmail API service from stored Credential dataclass.

        Args:
            credential: vault Credential dataclass (attribute access: .access_token, etc.)
            vault: SecureVault instance for token refresh and revocation handling
            client_id: OAuth client ID (for token refresh)
            client_secret: OAuth client secret (for token refresh)
        """
        self._vault = vault
        self._credential_id = f"{credential.provider}:{credential.account_id}"
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            creds = Credentials(
                token=credential.access_token,
                refresh_token=credential.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret,
            )
            self._creds = creds
            self._service = build("gmail", "v1", credentials=creds)
        except ImportError:
            raise ImportError(
                "Gmail provider requires google-api-python-client. "
                "Install with: pip install 'homie-ai[email]'"
            )

    def _check_token_freshness(self) -> None:
        """Check and refresh token if expired. Handles revocation."""
        if not self._creds or not self._creds.expired:
            return
        try:
            from google.auth.transport.requests import Request
            self._creds.refresh(Request())
            if self._vault:
                self._vault.refresh_credential(
                    self._credential_id,
                    new_access_token=self._creds.token,
                    new_expires_at=self._creds.expiry.timestamp() if self._creds.expiry else None,
                )
        except Exception:
            if self._vault:
                self._vault.set_connection_status("gmail", connected=False)
                self._vault.log_consent("gmail", "token_revoked", reason="refresh_failed")
            raise

    def fetch_messages(self, since: float, max_results: int = 100) -> list[EmailMessage]:
        """Fetch messages newer than `since` timestamp."""
        self._check_token_freshness()
        query = f"newer_than:7d" if since == 0.0 else f"after:{int(since)}"
        response = (
            self._service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        msg_ids = [m["id"] for m in response.get("messages", [])]
        return [self._fetch_and_parse(mid) for mid in msg_ids]

    def fetch_message_body(self, message_id: str) -> str:
        """Fetch full body text of a specific message."""
        self._check_token_freshness()
        raw = (
            self._service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        return _decode_body(raw.get("payload", {}))

    def get_history(self, start_history_id: str) -> tuple[list[HistoryChange], str]:
        """Get changes since history_id."""
        self._check_token_freshness()
        changes = []
        response = (
            self._service.users()
            .history()
            .list(userId="me", startHistoryId=start_history_id)
            .execute()
        )
        new_history_id = response.get("historyId", start_history_id)

        for record in response.get("history", []):
            for added in record.get("messagesAdded", []):
                changes.append(HistoryChange(
                    message_id=added["message"]["id"],
                    change_type="added",
                    labels=added["message"].get("labelIds", []),
                ))
            for deleted in record.get("messagesDeleted", []):
                changes.append(HistoryChange(
                    message_id=deleted["message"]["id"],
                    change_type="deleted",
                ))
            for label_added in record.get("labelsAdded", []):
                changes.append(HistoryChange(
                    message_id=label_added["message"]["id"],
                    change_type="labelAdded",
                    labels=label_added.get("labelIds", []),
                ))
            for label_removed in record.get("labelsRemoved", []):
                changes.append(HistoryChange(
                    message_id=label_removed["message"]["id"],
                    change_type="labelRemoved",
                    labels=label_removed.get("labelIds", []),
                ))

        return changes, new_history_id

    def search(self, query: str, max_results: int = 20) -> list[EmailMessage]:
        """Search using Gmail query syntax."""
        self._check_token_freshness()
        response = (
            self._service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        msg_ids = [m["id"] for m in response.get("messages", [])]
        return [self._fetch_and_parse(mid) for mid in msg_ids]

    def apply_label(self, message_id: str, label_id: str) -> None:
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"addLabelIds": [label_id]},
        ).execute()

    def remove_label(self, message_id: str, label_id: str) -> None:
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"removeLabelIds": [label_id]},
        ).execute()

    def trash(self, message_id: str) -> None:
        self._check_token_freshness()
        self._service.users().messages().trash(userId="me", id=message_id).execute()

    def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        reply_to: str | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> str:
        """Create a draft. Returns draft ID."""
        self._check_token_freshness()
        mime_msg = email.mime.text.MIMEText(body)
        mime_msg["To"] = to
        mime_msg["Subject"] = subject
        if cc:
            mime_msg["Cc"] = ", ".join(cc)
        if bcc:
            mime_msg["Bcc"] = ", ".join(bcc)
        if reply_to:
            mime_msg["In-Reply-To"] = reply_to
            mime_msg["References"] = reply_to

        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        draft_body: dict[str, Any] = {"message": {"raw": encoded}}

        result = (
            self._service.users()
            .drafts()
            .create(userId="me", body=draft_body)
            .execute()
        )
        return result["id"]

    def list_labels(self) -> list[Label]:
        self._check_token_freshness()
        response = self._service.users().labels().list(userId="me").execute()
        return [
            Label(id=l["id"], name=l["name"], type=l.get("type", "user"))
            for l in response.get("labels", [])
        ]

    def get_profile(self) -> dict:
        self._check_token_freshness()
        return self._service.users().getProfile(userId="me").execute()

    def mark_read(self, message_id: str) -> None:
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()

    def archive(self, message_id: str) -> None:
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"removeLabelIds": ["INBOX"]},
        ).execute()

    # --- Internal helpers ---

    def _fetch_and_parse(self, message_id: str) -> EmailMessage:
        """Fetch a single message by ID and parse it."""
        raw = (
            self._service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        return self._parse_message(raw)

    def _parse_message(self, raw: dict) -> EmailMessage:
        """Parse a Gmail API message dict into EmailMessage."""
        payload = raw.get("payload", {})
        headers = payload.get("headers", [])

        # Check for attachments
        attachment_names = []
        has_attachments = False
        for part in payload.get("parts", []):
            filename = part.get("filename", "")
            if filename:
                has_attachments = True
                attachment_names.append(filename)

        date_ms = int(raw.get("internalDate", "0"))
        label_ids = raw.get("labelIds", [])

        return EmailMessage(
            id=raw["id"],
            thread_id=raw.get("threadId", ""),
            account_id=self._account_id,
            provider="gmail",
            subject=_header(headers, "Subject"),
            sender=_header(headers, "From"),
            recipients=(
                [r.strip() for r in _header(headers, "To").split(",") if r.strip()]
                + [r.strip() for r in _header(headers, "Cc").split(",") if r.strip()]
            ),
            snippet=raw.get("snippet", ""),
            body=None,  # Body fetched separately on demand
            labels=label_ids,
            date=date_ms / 1000.0,
            is_read="UNREAD" not in label_ids,
            is_starred="STARRED" in label_ids,
            has_attachments=has_attachments,
            attachment_names=attachment_names,
        )
