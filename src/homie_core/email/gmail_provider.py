"""Gmail API implementation of EmailProvider.

Uses google-api-python-client to interact with Gmail.
All API calls go through self._service (a googleapiclient Resource).
"""
from __future__ import annotations

import base64
import email.mime.text
import email.mime.multipart
import email.mime.base
import mimetypes
import os
import time
from typing import Any

from homie_core.email.models import EmailAttachment, EmailDraft, EmailMessage, EmailThread, HistoryChange, Label
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
            if hasattr(credential, 'expires_at') and credential.expires_at:
                from datetime import datetime, timezone
                creds.expiry = datetime.fromtimestamp(credential.expires_at, tz=timezone.utc)
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
        new_history_id = start_history_id
        page_token = None

        while True:
            kwargs = {"userId": "me", "startHistoryId": start_history_id}
            if page_token:
                kwargs["pageToken"] = page_token
            response = (
                self._service.users()
                .history()
                .list(**kwargs)
                .execute()
            )
            new_history_id = response.get("historyId", new_history_id)

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

            page_token = response.get("nextPageToken")
            if not page_token:
                break

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

    def fetch_message(self, message_id: str) -> "EmailMessage":
        """Fetch and parse a single message by ID."""
        return self._fetch_and_parse(message_id)

    def create_label(self, name: str, visibility: str = "labelShow") -> "Label":
        """Create a new Gmail label."""
        self._check_token_freshness()
        result = self._service.users().labels().create(
            userId="me",
            body={"name": name, "labelListVisibility": visibility,
                  "messageListVisibility": "show"},
        ).execute()
        return Label(id=result["id"], name=result["name"], type=result.get("type", "user"))

    def get_thread(self, thread_id: str) -> EmailThread:
        self._check_token_freshness()
        raw = self._service.users().threads().get(userId="me", id=thread_id, format="full").execute()
        return self._parse_thread(raw)

    def list_threads(self, query: str, max_results: int = 20) -> list[EmailThread]:
        self._check_token_freshness()
        response = self._service.users().threads().list(userId="me", q=query, maxResults=max_results).execute()
        threads = []
        for t in response.get("threads", []):
            raw = self._service.users().threads().get(userId="me", id=t["id"], format="full").execute()
            threads.append(self._parse_thread(raw))
        return threads

    def get_inbox_threads(self, start=0, max_results=20):
        return self.list_threads("in:inbox", max_results=max_results)

    def get_starred_threads(self, start=0, max_results=20):
        return self.list_threads("is:starred", max_results=max_results)

    def get_spam_threads(self, start=0, max_results=20):
        return self.list_threads("in:spam", max_results=max_results)

    def get_trash_threads(self, start=0, max_results=20):
        return self.list_threads("in:trash", max_results=max_results)

    def get_unread_count(self, label="INBOX"):
        self._check_token_freshness()
        result = self._service.users().labels().get(userId="me", id=label).execute()
        return result.get("messagesUnread", 0)

    def archive_thread(self, thread_id):
        self._check_token_freshness()
        self._service.users().threads().modify(userId="me", id=thread_id, body={"removeLabelIds": ["INBOX"]}).execute()

    def trash_thread(self, thread_id):
        self._check_token_freshness()
        self._service.users().threads().trash(userId="me", id=thread_id).execute()

    def apply_label_to_thread(self, thread_id, label_id):
        self._check_token_freshness()
        self._service.users().threads().modify(userId="me", id=thread_id, body={"addLabelIds": [label_id]}).execute()

    def mark_thread_read(self, thread_id):
        self._check_token_freshness()
        self._service.users().threads().modify(userId="me", id=thread_id, body={"removeLabelIds": ["UNREAD"]}).execute()

    def mark_thread_unread(self, thread_id):
        self._check_token_freshness()
        self._service.users().threads().modify(userId="me", id=thread_id, body={"addLabelIds": ["UNREAD"]}).execute()

    def list_drafts(self, max_results=20):
        self._check_token_freshness()
        response = self._service.users().drafts().list(userId="me", maxResults=max_results).execute()
        drafts = []
        for d in response.get("drafts", []):
            raw = self._service.users().drafts().get(userId="me", id=d["id"], format="full").execute()
            drafts.append(self._parse_draft(raw))
        return drafts

    def get_draft(self, draft_id):
        self._check_token_freshness()
        raw = self._service.users().drafts().get(userId="me", id=draft_id, format="full").execute()
        return self._parse_draft(raw)

    def update_draft(self, draft_id, to, subject, body, cc=None, bcc=None):
        self._check_token_freshness()
        mime_msg = email.mime.text.MIMEText(body)
        mime_msg["To"] = to
        mime_msg["Subject"] = subject
        if cc:
            mime_msg["Cc"] = ", ".join(cc)
        if bcc:
            mime_msg["Bcc"] = ", ".join(bcc)
        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        result = self._service.users().drafts().update(
            userId="me", id=draft_id, body={"message": {"raw": encoded}}
        ).execute()
        return result["id"]

    def delete_draft(self, draft_id):
        self._check_token_freshness()
        self._service.users().drafts().delete(userId="me", id=draft_id).execute()

    def send_email(self, to, subject, body, cc=None, bcc=None, attachments=None, reply_to_message_id=None):
        self._check_token_freshness()
        if attachments:
            mime_msg = self._build_mime_with_attachments(to, subject, body, cc, bcc, attachments, reply_to_message_id)
        else:
            mime_msg = email.mime.text.MIMEText(body)
            mime_msg["To"] = to
            mime_msg["Subject"] = subject
            if cc:
                mime_msg["Cc"] = ", ".join(cc)
            if bcc:
                mime_msg["Bcc"] = ", ".join(bcc)
            if reply_to_message_id:
                mime_msg["In-Reply-To"] = reply_to_message_id
                mime_msg["References"] = reply_to_message_id
        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        result = self._service.users().messages().send(userId="me", body={"raw": encoded}).execute()
        return result["id"]

    def send_draft(self, draft_id):
        self._check_token_freshness()
        result = self._service.users().drafts().send(userId="me", body={"id": draft_id}).execute()
        return result["id"]

    def reply(self, message_id, body, send=False):
        self._check_token_freshness()
        original = self._fetch_and_parse(message_id)
        raw_msg = self._service.users().messages().get(
            userId="me", id=message_id, format="metadata",
            metadataHeaders=["Message-ID", "Subject", "From"]
        ).execute()
        headers = raw_msg.get("payload", {}).get("headers", [])
        message_id_header = _header(headers, "Message-ID")

        mime_msg = email.mime.text.MIMEText(body)
        mime_msg["To"] = original.sender
        mime_msg["Subject"] = f"Re: {original.subject}" if not original.subject.startswith("Re:") else original.subject
        mime_msg["In-Reply-To"] = message_id_header
        mime_msg["References"] = message_id_header

        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        msg_body = {"raw": encoded, "threadId": original.thread_id}

        if send:
            result = self._service.users().messages().send(userId="me", body=msg_body).execute()
        else:
            result = self._service.users().drafts().create(userId="me", body={"message": msg_body}).execute()
        return result["id"]

    def reply_all(self, message_id, body, send=False):
        self._check_token_freshness()
        original = self._fetch_and_parse(message_id)
        raw_msg = self._service.users().messages().get(
            userId="me", id=message_id, format="metadata",
            metadataHeaders=["Message-ID", "Subject", "From", "To", "Cc"]
        ).execute()
        headers = raw_msg.get("payload", {}).get("headers", [])
        message_id_header = _header(headers, "Message-ID")

        all_recipients = set()
        all_recipients.add(original.sender)
        for r in original.recipients:
            all_recipients.add(r)
        all_recipients.discard(self._account_id)

        mime_msg = email.mime.text.MIMEText(body)
        mime_msg["To"] = ", ".join(all_recipients)
        mime_msg["Subject"] = f"Re: {original.subject}" if not original.subject.startswith("Re:") else original.subject
        mime_msg["In-Reply-To"] = message_id_header
        mime_msg["References"] = message_id_header

        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        msg_body = {"raw": encoded, "threadId": original.thread_id}

        if send:
            result = self._service.users().messages().send(userId="me", body=msg_body).execute()
        else:
            result = self._service.users().drafts().create(userId="me", body={"message": msg_body}).execute()
        return result["id"]

    def forward(self, message_id, to, body, send=False):
        self._check_token_freshness()
        original = self._fetch_and_parse(message_id)
        original_body = self.fetch_message_body(message_id)

        forward_body = f"{body}\n\n---------- Forwarded message ----------\nFrom: {original.sender}\nSubject: {original.subject}\n\n{original_body}"

        mime_msg = email.mime.text.MIMEText(forward_body)
        mime_msg["To"] = to
        mime_msg["Subject"] = f"Fwd: {original.subject}" if not original.subject.startswith("Fwd:") else original.subject

        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        msg_body = {"raw": encoded}

        if send:
            result = self._service.users().messages().send(userId="me", body=msg_body).execute()
        else:
            result = self._service.users().drafts().create(userId="me", body={"message": msg_body}).execute()
        return result["id"]

    def get_attachments(self, message_id):
        self._check_token_freshness()
        raw = self._service.users().messages().get(userId="me", id=message_id, format="full").execute()
        attachments = []
        for part in raw.get("payload", {}).get("parts", []):
            filename = part.get("filename", "")
            if filename:
                body = part.get("body", {})
                attachments.append(EmailAttachment(
                    id=body.get("attachmentId", ""),
                    message_id=message_id,
                    filename=filename,
                    mime_type=part.get("mimeType", "application/octet-stream"),
                    size=body.get("size", 0),
                ))
        return attachments

    def download_attachment(self, message_id, attachment_id, save_path):
        self._check_token_freshness()
        result = self._service.users().messages().attachments().get(
            userId="me", messageId=message_id, id=attachment_id
        ).execute()
        data = base64.urlsafe_b64decode(result["data"])
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, attachment_id)
        with open(file_path, "wb") as f:
            f.write(data)
        return file_path

    def get_aliases(self):
        self._check_token_freshness()
        result = self._service.users().settings().sendAs().list(userId="me").execute()
        return [a["sendAsEmail"] for a in result.get("sendAs", [])]

    def star(self, message_id):
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id, body={"addLabelIds": ["STARRED"]},
        ).execute()

    def unstar(self, message_id):
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id, body={"removeLabelIds": ["STARRED"]},
        ).execute()

    def mark_unread(self, message_id):
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id, body={"addLabelIds": ["UNREAD"]},
        ).execute()

    def move_to_inbox(self, message_id):
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id, body={"addLabelIds": ["INBOX"]},
        ).execute()

    def delete_label(self, label_id):
        self._check_token_freshness()
        self._service.users().labels().delete(userId="me", id=label_id).execute()

    def update_label(self, label_id, new_name):
        self._check_token_freshness()
        result = self._service.users().labels().update(
            userId="me", id=label_id, body={"name": new_name},
        ).execute()
        return Label(id=result["id"], name=result["name"], type=result.get("type", "user"))

    def untrash(self, message_id):
        self._check_token_freshness()
        self._service.users().messages().untrash(userId="me", id=message_id).execute()

    def _build_mime_with_attachments(self, to, subject, body, cc, bcc, file_paths, reply_to_message_id=None):
        """Build multipart MIME message with file attachments."""
        msg = email.mime.multipart.MIMEMultipart()
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = ", ".join(cc)
        if bcc:
            msg["Bcc"] = ", ".join(bcc)
        if reply_to_message_id:
            msg["In-Reply-To"] = reply_to_message_id
            msg["References"] = reply_to_message_id
        msg.attach(email.mime.text.MIMEText(body))
        for path in file_paths:
            content_type, _ = mimetypes.guess_type(path)
            if content_type is None:
                content_type = "application/octet-stream"
            maintype, subtype = content_type.split("/", 1)
            with open(path, "rb") as f:
                attachment = email.mime.base.MIMEBase(maintype, subtype)
                attachment.set_payload(f.read())
            import email.encoders
            email.encoders.encode_base64(attachment)
            attachment.add_header("Content-Disposition", "attachment", filename=os.path.basename(path))
            msg.attach(attachment)
        return msg

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

    def _parse_thread(self, raw: dict) -> EmailThread:
        """Parse a Gmail API thread dict into EmailThread."""
        messages = [self._parse_message(m) for m in raw.get("messages", [])]
        participants = list({m.sender for m in messages} | {r for m in messages for r in m.recipients})
        last_msg = messages[-1] if messages else None
        return EmailThread(
            id=raw["id"],
            account_id=self._account_id,
            subject=messages[0].subject if messages else "",
            participants=participants,
            message_count=len(messages),
            last_message_date=last_msg.date if last_msg else 0.0,
            snippet=last_msg.snippet if last_msg else "",
            labels=list({lbl for m in messages for lbl in m.labels}),
            messages=messages,
        )

    def _parse_draft(self, raw: dict):
        """Parse a Gmail API draft dict into EmailDraft."""
        from homie_core.email.models import EmailDraft
        msg_raw = raw.get("message", {})
        msg = self._parse_message(msg_raw)
        return EmailDraft(id=raw["id"], message=msg, updated_at=time.time())
