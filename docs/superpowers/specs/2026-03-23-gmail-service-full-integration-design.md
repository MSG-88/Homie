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
- `list_threads(query, max_results) -> list[EmailThread]` — search/list threads
- `get_inbox_threads(start, max) -> list[EmailThread]` — threads labeled INBOX
- `get_starred_threads(start, max) -> list[EmailThread]` — threads with STARRED
- `get_spam_threads(start, max) -> list[EmailThread]` — threads in SPAM
- `get_trash_threads(start, max) -> list[EmailThread]` — threads in TRASH
- `get_unread_count(label) -> int` — unread count for any label

### Drafts

- `list_drafts(max_results) -> list[EmailDraft]` — all drafts
- `get_draft(draft_id) -> EmailDraft` — single draft with message content
- `update_draft(draft_id, to, subject, body, cc, bcc) -> str` — update existing draft, returns draft ID
- `delete_draft(draft_id) -> None` — permanently delete a draft

### Send / Reply / Forward

- `send_email(to, subject, body, cc, bcc, attachments, reply_to_message_id) -> str` — send directly, returns message ID
- `send_draft(draft_id) -> str` — send an existing draft, returns message ID
- `reply(message_id, body, send=False) -> str` — reply; draft by default, returns draft/message ID
- `reply_all(message_id, body, send=False) -> str` — reply-all; draft by default
- `forward(message_id, to, body, send=False) -> str` — forward; draft by default

### Attachments

- `get_attachments(message_id) -> list[EmailAttachment]` — list attachment metadata
- `download_attachment(message_id, attachment_id, save_path) -> str` — download to local storage, returns file path
- `_build_mime_with_attachments(to, subject, body, cc, bcc, file_paths, reply_headers) -> MIMEMultipart` — helper to build multipart MIME with attachments

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

Feeds into existing `homie_core/knowledge/`:

- Entities → `knowledge_graph.add_entity(type, name, attributes)`
- Relationships → `knowledge_graph.add_relationship(entity_a, entity_b, relation_type)`
- Events/deadlines → stored as time-bound entities with reminder hooks
- Action items → surfaced proactively via working memory + notifications
- Thread context → stored as topic clusters for "what's happening with X" queries

### Processing Modes

- **Initial sync:** Batch-process all 7 days of emails through LLM extraction (chunked by `extraction_batch_size`, default 20)
- **Incremental sync:** Process new/changed messages as they arrive
- **On-demand:** Deep extraction when user asks about a specific email/thread/contact

### Knowledge Graph Entity Types for Email Domain

- `person` — contacts with email, name, org
- `organization` — companies/domains
- `project` — inferred from thread topics
- `event` — meetings, deadlines
- `action_item` — tasks extracted from emails
- `document` — attachments linked to threads

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

- All send tools route through existing HITL middleware — brain cannot send without user approval
- `email_send` and `email_send_draft` require explicit `send=True` flag; default creates draft
- Attachment downloads respect size limit from config
- Tools never expose raw tokens or credentials

## 5. Models & Config

### New Dataclasses in `models.py`

```python
@dataclass
class EmailThread:  # Extend existing
    messages: list[EmailMessage]  # full message objects

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

- `email_threads` — thread metadata + last fetch time
- `email_drafts` — cached drafts
- `email_attachments` — attachment metadata (not binary data)
- `email_action_items` — extracted actions with status tracking
- `email_contacts` — contact graph cache
- `email_topics` — topic clusters

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
| `src/homie_core/email/provider.py` | Add ~15 abstract methods |
| `src/homie_core/email/gmail_provider.py` | Implement all new methods |
| `src/homie_core/email/models.py` | Add EmailDraft, EmailAttachment, ContactInsight, ActionItem; extend EmailThread |
| `src/homie_core/email/__init__.py` | Add ~20 facade methods to EmailService |
| `src/homie_core/email/tools.py` | Register ~16 new brain tools |
| `src/homie_core/email/sync_engine.py` | Wire knowledge extractor into sync pipeline |
| `src/homie_core/email/knowledge_extractor.py` | **New file** — extraction + graph integration |
| `src/homie_core/config.py` | Add EmailConfig model |
| `tests/unit/email/` | Tests for all new methods |
