# Gmail Service Full Integration Design

**Date:** 2026-03-23
**Status:** Approved
**Approach:** Extend existing classes (Approach 1)

## Overview

Fill the gaps in Homie's Gmail integration to match the full Google Apps Script Gmail Service API surface. After successful auth and sync, Homie processes all emails through a knowledge extraction pipeline, builds a contact/topic graph, and provides actionable insights to the user.

Multi-account support throughout. Send capability with draft-as-default and explicit `send=True` flag. Full attachment round-trip (download + compose).

## 1. Provider Layer — New GmailProvider Methods

Add to `EmailProvider` ABC and `GmailProvider` implementation.

### Threads

- `get_thread(thread_id) -> EmailThread` — fetch full thread with all messages from Gmail API
- `list_threads(query, max_results) -> list[EmailThread]` — search/list threads (uses `pageToken` internally, `start` is converted to page offset)
- `get_inbox_threads(start, max) -> list[EmailThread]` — threads labeled INBOX
- `get_starred_threads(start, max) -> list[EmailThread]` — threads with STARRED
- `get_spam_threads(start, max) -> list[EmailThread]` — threads in SPAM
- `get_trash_threads(start, max) -> list[EmailThread]` — threads in TRASH
- `get_unread_count(label) -> int` — unread count for any label

**Thread-level batch operations** (use Gmail `threads().modify()` / `threads().trash()` API, not per-message iteration):
- `archive_thread(thread_id) -> None` — remove INBOX label from thread
- `trash_thread(thread_id) -> None` — move entire thread to trash
- `apply_label_to_thread(thread_id, label_id) -> None` — label all messages in thread
- `mark_thread_read(thread_id) -> None` — mark entire thread as read
- `mark_thread_unread(thread_id) -> None` — mark entire thread as unread

### Drafts

- `list_drafts(max_results) -> list[EmailDraft]` — all drafts
- `get_draft(draft_id) -> EmailDraft` — single draft with message content
- `update_draft(draft_id, to, subject, body, cc, bcc) -> str` — update existing draft, returns draft ID
- `delete_draft(draft_id) -> None` — permanently delete a draft

### Send / Reply / Forward

- `send_email(to, subject, body, cc, bcc, attachments, reply_to_message_id) -> str` — send directly, returns message ID
- `send_draft(draft_id) -> str` — send an existing draft, returns message ID
- `reply(message_id, body, send=False) -> str` — reply; draft by default, returns draft/message ID. Internally fetches original message's `Message-ID` header to set `In-Reply-To` and `References`.
- `reply_all(message_id, body, send=False) -> str` — reply-all; draft by default. Resolves all To/Cc recipients from original message.
- `forward(message_id, to, body, send=False) -> str` — forward; draft by default. Includes original attachments.

### Attachments

- `get_attachments(message_id) -> list[EmailAttachment]` — list attachment metadata
- `download_attachment(message_id, attachment_id, save_path) -> str` — download to local storage, returns file path
- `_build_mime_with_attachments(to, subject, body, cc, bcc, file_paths, reply_headers) -> MIMEMultipart` — internal helper to build multipart MIME with attachments (private, not in ABC)

### Misc

- `get_aliases() -> list[str]` — list send-as aliases
- `star(message_id) -> None` — add STARRED label
- `unstar(message_id) -> None` — remove STARRED label
- `mark_unread(message_id) -> None` — add UNREAD label
- `move_to_inbox(message_id) -> None` — add INBOX label (undo archive)
- `delete_label(label_id) -> None` — delete a user label
- `update_label(label_id, new_name) -> Label` — rename a label
- `untrash(message_id) -> None` — restore from trash

## 2. Email Knowledge Extraction & Graph Integration

### New class: `EmailKnowledgeExtractor`

Location: `homie_core/email/knowledge_extractor.py`

Runs after classification in the sync pipeline. For each email (or batch), extracts:

- **Entities:** people (sender/recipients → contacts), organizations (from domains/signatures), projects/products mentioned
- **Relationships:** "works with", "reports to", "client of" — inferred from email patterns (who emails whom, CC patterns, reply chains)
- **Events:** meetings, deadlines, appointments from body text
- **Action items:** tasks, requests, commitments ("please send by Friday", "can you review")
- **Financial:** bills, invoices, payment confirmations (extends existing `extract_financial`)
- **Topics:** conversation topics/themes per thread for contextual grouping

### Knowledge Graph Integration

Feeds into existing `homie_core/knowledge/` using real API signatures:

```python
from homie_core.knowledge.models import Entity, Relationship

# Entities — use graph.merge_entity() for dedup by (name, entity_type)
person = Entity(name="Alice", entity_type="person", attributes={"email": "alice@corp.com"}, source="email_sync")
entity_id = graph.merge_entity(person)

# Relationships — use graph.add_relationship() with Relationship dataclass
rel = Relationship(subject_id=sender_id, relation="works_on", object_id=project_id, source="email_sync")
graph.add_relationship(rel)
```

- Events/deadlines → stored as `event` entities with `valid_from`/`valid_until` on relationships for temporal tracking
- Action items → stored in `email_action_items` cache table AND as `task` entities in knowledge graph, surfaced proactively via working memory + notifications
- Thread context → stored as topic clusters for "what's happening with X" queries

### Relationship to existing `EmailIndexer`

`EmailKnowledgeExtractor` **supersedes** `homie_core/knowledge/email_indexer.py`. The existing `EmailIndexer` creates basic `person` + `document` entities with `authored` relationships. The new extractor does everything `EmailIndexer` does plus deep LLM-powered extraction (contacts, orgs, events, actions, topics). `EmailIndexer` will be deprecated — its `index_recent()` callers will be redirected to `EmailKnowledgeExtractor.batch_extract()`.

### Processing Modes

- **Initial sync:** Batch-process all 7 days of emails through LLM extraction (chunked by `extraction_batch_size`, default 20). Runs **asynchronously in background** — does not block user interaction. Progress stored in `email_sync_state`.
- **Incremental sync:** Process new/changed messages as they arrive (inline, fast — typically 1-3 messages)
- **On-demand:** Deep extraction when user asks about a specific email/thread/contact
- **LLM unavailable fallback:** If LLM is down, use heuristic extraction only (sender/recipient parsing, regex-based deadline detection, domain-based org inference). Same pattern as existing `EmailClassifier.has_llm` check.

### Knowledge Graph Schema Extensions

New `entity_type` values to add to `Entity` docstring:
- `organization` — companies/domains (add to allowed types alongside existing person, project, concept, etc.)

New `relation` values to add to `Relationship` docstring:
- `works_with` — inferred from email exchange patterns
- `reports_to` — inferred from CC/forwarding patterns
- `client_of` — inferred from domain + email content
- `contacted` — direct email communication record

Existing types that map directly:
- `person` → contacts
- `project` → inferred thread topics
- `event` → meetings, deadlines
- `task` → action items extracted from emails
- `document` → attachments linked to threads

## 3. EmailService Facade — New Methods

All multi-account aware with `account` parameter (first-match default).

### Thread Operations

- `get_thread_messages(thread_id, account) -> EmailThread`
- `list_inbox_threads(account, start, max) -> list[EmailThread]`
- `list_starred_threads(account, start, max) -> list[EmailThread]`
- `list_spam_threads(account, start, max) -> list[EmailThread]`
- `get_unread_counts(account) -> dict` — `{inbox, spam, starred, total}`

### Draft Management

- `list_drafts(account) -> list[EmailDraft]`
- `get_draft(draft_id, account) -> EmailDraft`
- `update_draft(draft_id, to, subject, body, account) -> str`
- `delete_draft(draft_id, account) -> None`

### Send/Reply/Forward

- `send_email(to, subject, body, cc, bcc, attachments, account) -> str`
- `send_draft(draft_id, account) -> str`
- `reply(message_id, body, send=False, account) -> str`
- `reply_all(message_id, body, send=False, account) -> str`
- `forward(message_id, to, body, send=False, account) -> str`

### Attachments

- `get_attachments(message_id, account) -> list[EmailAttachment]`
- `download_attachment(message_id, attachment_id, save_path, account) -> str`

### Knowledge & Insights

- `get_contact_insights(email_or_name) -> ContactInsight`
- `get_topic_summary(topic) -> str`
- `get_pending_actions() -> list[ActionItem]`
- `get_email_insights(days) -> str` — intelligence briefing

### Sync Pipeline Changes

In `SyncEngine._classify_and_organize()`, after classification + organization:

```python
self._knowledge_extractor.process_message(msg)
```

In `SyncEngine.initial_sync()`, after all messages processed:

```python
self._knowledge_extractor.batch_extract(all_messages)
self._knowledge_extractor.build_contact_graph()
```

### Attachment Storage

- Path: `~/.homie/attachments/{account_id}/{message_id}/{filename}`
- Auto-download for emails classified as "bill", "order", "work" (configurable)
- Size limit from config (default 25MB per attachment)

## 4. New Brain Tools

### Thread Tools

- `email_inbox_threads` — list inbox threads with pagination (`account`, `start`, `max_results`)
- `email_thread_full` — fetch complete thread from API with all messages (`thread_id`)
- `email_unread_counts` — get unread counts by category (`account`)

### Draft Tools

- `email_list_drafts` — list all drafts (`account`)
- `email_get_draft` — read a specific draft (`draft_id`)
- `email_update_draft` — modify an existing draft (`draft_id`, `to`, `subject`, `body`, `cc`, `bcc`)
- `email_delete_draft` — delete a draft (`draft_id`)

### Send Tools (HITL gated)

- `email_send` — send email directly (`to`, `subject`, `body`, `cc`, `bcc`, `attachments`, `account`)
- `email_send_draft` — send an existing draft (`draft_id`)
- `email_reply` — reply to a message (`message_id`, `body`, `send` flag, defaults to draft)
- `email_forward` — forward a message (`message_id`, `to`, `body`, `send` flag)

### Attachment Tools

- `email_attachments` — list attachments for a message (`message_id`)
- `email_download_attachment` — download to local storage (`message_id`, `attachment_id`)

### Knowledge/Insight Tools

- `email_contact_insights` — relationship history, frequency, topics for a contact (`email_or_name`)
- `email_topic_summary` — cross-thread summary of a topic/project (`topic`)
- `email_pending_actions` — all action items with deadlines and urgency
- `email_briefing` — daily/weekly intelligence briefing (`days`)

### Safety Rules

- **HITL gate wiring:** Add these tool names to `HITLMiddleware._gated` set at startup: `{"email_send", "email_send_draft", "email_reply", "email_forward"}` (when `send=True`). The `HITLMiddleware.__init__(gated_tools=...)` receives these alongside existing `run_command` and `write_file`.
- `email_reply` and `email_forward` only hit HITL gate when `send` param is `"true"`. When `send` is omitted/false, they create drafts without approval.
- Attachment downloads respect size limit from config
- Attachment `save_path` is validated to stay within `~/.homie/attachments/` — path traversal (e.g. `../../etc/passwd`) is rejected
- Tools never expose raw tokens or credentials

### Error Handling Policy

- **Send failures must NOT be silent.** `send_email`, `send_draft`, `reply`, `forward` propagate exceptions to the tool layer, which returns `{"error": "..."}` to the brain. No bare `except: pass`.
- **Auth failures** (401/403): Set `vault.set_connection_status("gmail", connected=False)` and return clear error to user.
- **Rate limiting** (429): Return `{"error": "rate_limited", "retry_after": N}` — do not retry automatically in the tool.
- **Non-send operations** (list, search, fetch): Follow existing pattern — catch and return empty/error dict, log via `get_logger()`.

## 5. Models & Config

### New Dataclasses in `models.py`

```python
@dataclass
class EmailThread:  # ADDITIVE extension — keep all existing fields
    # Existing fields (unchanged):
    id: str
    account_id: str
    subject: str
    participants: list[str]
    message_count: int
    last_message_date: float
    snippet: str
    labels: list[str] = field(default_factory=list)
    # New field added:
    messages: list[EmailMessage] = field(default_factory=list)  # populated on full fetch

@dataclass
class EmailDraft:
    id: str
    message: EmailMessage
    updated_at: float

@dataclass
class EmailAttachment:
    id: str
    message_id: str
    filename: str
    mime_type: str
    size: int  # bytes
    data: bytes | None = None  # populated on download

@dataclass
class ContactInsight:
    email: str
    name: str
    organization: str
    relationship: str  # "colleague", "client", "vendor", etc.
    email_count: int
    last_contact: float
    topics: list[str]
    pending_actions: list[str]

@dataclass
class ActionItem:
    id: str
    message_id: str
    thread_id: str
    description: str
    assignee: str
    deadline: float | None
    urgency: str  # "high", "medium", "low"
    status: str  # "pending", "done", "expired"
    extracted_at: float
```

### Config Additions

New section in `homie.config.yaml` or under `EmailConfig` Pydantic model:

```yaml
email:
  auto_download_attachments: true
  auto_download_categories: ["bill", "order", "work"]
  max_attachment_size_mb: 25
  knowledge_extraction: true
  extraction_batch_size: 20
  insight_refresh_interval: 3600
  send_requires_confirmation: true
```

### Cache DB Schema Additions

```sql
CREATE TABLE IF NOT EXISTS email_threads (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    subject TEXT,
    participants TEXT,  -- JSON array
    message_count INTEGER DEFAULT 0,
    last_message_date REAL,
    snippet TEXT,
    labels TEXT,  -- JSON array
    fetched_at REAL
);

CREATE TABLE IF NOT EXISTS email_drafts (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    to_addr TEXT,
    subject TEXT,
    body TEXT,
    cc TEXT,  -- JSON array
    bcc TEXT,  -- JSON array
    reply_to_message_id TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS email_attachments (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    mime_type TEXT,
    size INTEGER DEFAULT 0,
    local_path TEXT,  -- NULL if not downloaded
    downloaded_at REAL,
    FOREIGN KEY (message_id) REFERENCES emails(id)
);

CREATE TABLE IF NOT EXISTS email_action_items (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    thread_id TEXT,
    account_id TEXT NOT NULL,
    description TEXT NOT NULL,
    assignee TEXT,
    deadline REAL,
    urgency TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending',
    extracted_at REAL,
    completed_at REAL
);

CREATE TABLE IF NOT EXISTS email_contacts (
    email TEXT PRIMARY KEY,
    name TEXT,
    organization TEXT,
    relationship TEXT,
    email_count INTEGER DEFAULT 0,
    last_contact REAL,
    topics TEXT,  -- JSON array
    entity_id TEXT,  -- FK to knowledge graph entity
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS email_topics (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    thread_ids TEXT,  -- JSON array
    entity_ids TEXT,  -- JSON array of related knowledge graph entities
    message_count INTEGER DEFAULT 0,
    last_activity REAL,
    updated_at REAL
);
```

## 6. Post-Auth Intelligence Flow

### On First Successful Authentication

1. `EmailService.initialize()` authenticates provider
2. `SyncEngine.initial_sync()` fetches 7 days of emails
3. Each email → classify (spam/priority/categories) → organize (labels/archive)
4. Batch all messages to `EmailKnowledgeExtractor.batch_extract()` — LLM processes in chunks of 20
5. `build_contact_graph()` — aggregates sender/recipient patterns into contact nodes with relationship edges
6. `build_topic_clusters()` — groups threads by topic, links to entities
7. All extracted knowledge → `knowledge_graph.add_entity()` / `add_relationship()`
8. Action items with deadlines → working memory for proactive notifications
9. `get_email_insights()` generates initial briefing stored in working memory

### On Each Incremental Sync Tick

1. `SyncEngine.incremental_sync()` picks up new/changed messages
2. Classify → organize (existing)
3. `EmailKnowledgeExtractor.process_message(msg)` for each new message — extracts entities/actions/topics, updates contact graph, checks for deadline updates
4. If high-priority action item detected → push to working memory → notification

### On User Query

- "What's happening with Project X?" → `email_topic_summary` → knowledge graph + thread summaries
- "Any pending tasks?" → `email_pending_actions` → action items table filtered by status
- "Tell me about alice@corp.com" → `email_contact_insights` → contact graph + recent threads
- "Brief me on my emails" → `email_briefing` → LLM digest combining knowledge graph + recent activity

## 7. Files to Modify

| File | Change |
|------|--------|
| `src/homie_core/email/provider.py` | Add ~20 abstract methods (threads, drafts, send, attachments, misc) |
| `src/homie_core/email/gmail_provider.py` | Implement all new methods + thread-level batch ops |
| `src/homie_core/email/models.py` | Add EmailDraft, EmailAttachment, ContactInsight, ActionItem; add `messages` field to EmailThread |
| `src/homie_core/email/__init__.py` | Add ~25 facade methods to EmailService |
| `src/homie_core/email/tools.py` | Register ~16 new brain tools |
| `src/homie_core/email/sync_engine.py` | Wire knowledge extractor into sync pipeline |
| `src/homie_core/email/knowledge_extractor.py` | **New file** — LLM extraction + knowledge graph integration (supersedes EmailIndexer) |
| `src/homie_core/config.py` | Add EmailConfig Pydantic model |
| `src/homie_core/knowledge/models.py` | Add `organization` to entity_type docstring; add `works_with`, `reports_to`, `client_of`, `contacted` to relation docstring |
| `src/homie_core/knowledge/email_indexer.py` | Deprecate — redirect callers to EmailKnowledgeExtractor |
| `tests/unit/email/` | Tests for all new provider, service, tool, and extractor methods |
