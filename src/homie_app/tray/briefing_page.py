"""Morning briefing HTML page generator.

Produces a self-contained dark-themed HTML page for the morning email
briefing.  No external CSS/JS dependencies — everything is inlined.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any


def _greeting(user_name: str) -> str:
    """Return a time-aware greeting."""
    hour = datetime.now().hour
    if hour < 12:
        period = "morning"
    elif hour < 17:
        period = "afternoon"
    else:
        period = "evening"
    return f"Good {period}, {escape(user_name)}"


def _section(title: str, emails: list[dict[str, Any]], style: str) -> str:
    """Render a group of email cards under *title*.

    *style* is ``"urgent"`` (red left-border) or ``"normal"``.
    """
    if not emails:
        return ""

    border_color = "#f85149" if style == "urgent" else "#30363d"
    cards = []
    for em in emails:
        subject = escape(str(em.get("subject", "(no subject)")))
        sender = escape(str(em.get("sender", "")))
        snippet = escape(str(em.get("snippet", "")))
        email_id = escape(str(em.get("id", "")))

        cards.append(
            f'<div class="email-card" data-id="{email_id}" '
            f'style="border-left:3px solid {border_color};">'
            f'<div class="email-subject">{subject}</div>'
            f'<div class="email-sender">{sender}</div>'
            f'<div class="email-snippet">{snippet}</div>'
            f'<button class="btn-mark" onclick="markRead(\'{email_id}\')">Mark read</button>'
            f"</div>"
        )

    return (
        f'<div class="section">'
        f'<h2 class="section-title">{escape(title)}</h2>'
        f'{"".join(cards)}'
        f"</div>"
    )


def render_briefing_page(
    *,
    user_name: str,
    summary: dict[str, Any],
    unread: dict[str, list[dict[str, Any]]],
    digest: str,
    session_token: str,
    api_port: int,
) -> str:
    """Return a complete HTML page for the morning briefing.

    Parameters
    ----------
    user_name:
        Display name shown in the greeting.
    summary:
        Dict with ``total``, ``unread``, and ``high_priority`` keys.
    unread:
        Dict with ``high``, ``medium``, ``low`` email lists.
    digest:
        AI-generated plain-text summary of the inbox.
    session_token:
        Token set as a cookie so JS calls are authenticated.
    api_port:
        Local API port for triage / mark-read calls.
    """
    total = summary.get("total", 0)
    unread_count = summary.get("unread", 0)
    high_priority = summary.get("high_priority", [])
    high_count = len(high_priority)

    now = datetime.now()
    date_str = now.strftime("%A, %B {}, %Y").format(now.day)

    greeting = _greeting(user_name)

    # Build email sections
    high_section = _section("Needs Attention", unread.get("high", []), "urgent")
    medium_section = _section("Updates", unread.get("medium", []), "normal")
    low_section = _section("Low Priority", unread.get("low", []), "normal")

    has_emails = bool(unread.get("high") or unread.get("medium") or unread.get("low"))

    inbox_zero_html = ""
    if not has_emails:
        inbox_zero_html = (
            '<div class="inbox-zero">'
            "<p>Inbox zero &mdash; nothing needs your attention right now.</p>"
            "</div>"
        )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Morning Briefing — Homie AI</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{
  background:#0d1117;color:#c9d1d9;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  line-height:1.6;padding:2rem;max-width:720px;margin:0 auto;
}}
h1{{font-size:1.5rem;font-weight:600;margin-bottom:.25rem}}
.date{{color:#8b949e;margin-bottom:1.5rem}}
.stats{{display:flex;gap:1rem;margin-bottom:2rem;flex-wrap:wrap}}
.stat-card{{
  flex:1;min-width:140px;
  background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:1rem;text-align:center;
}}
.stat-card .value{{font-size:2rem;font-weight:700}}
.stat-card .label{{color:#8b949e;font-size:.85rem}}
.stat-card.urgent .value{{color:#f85149}}
.section{{margin-bottom:2rem}}
.section-title{{font-size:1.1rem;font-weight:600;margin-bottom:.75rem;color:#c9d1d9}}
.email-card{{
  background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:.85rem 1rem;margin-bottom:.6rem;
}}
.email-subject{{font-weight:600;margin-bottom:.15rem}}
.email-sender{{color:#8b949e;font-size:.85rem}}
.email-snippet{{color:#8b949e;font-size:.85rem;margin-top:.25rem}}
.btn-mark{{
  margin-top:.5rem;background:transparent;border:1px solid #30363d;
  color:#58a6ff;border-radius:4px;padding:.2rem .6rem;cursor:pointer;
  font-size:.8rem;
}}
.btn-mark:hover{{background:#21262d}}
.digest{{
  background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:1rem;margin-bottom:2rem;
}}
.digest h2{{font-size:1.1rem;margin-bottom:.5rem}}
.digest p{{color:#c9d1d9}}
.inbox-zero{{
  text-align:center;padding:3rem 1rem;color:#8b949e;font-size:1.1rem;
}}
footer{{
  text-align:center;color:#484f58;font-size:.8rem;
  margin-top:3rem;padding-top:1rem;border-top:1px solid #21262d;
}}
</style>
</head>
<body>
<h1>{greeting}</h1>
<p class="date">{escape(date_str)}</p>

<div class="stats">
  <div class="stat-card">
    <div class="value">{unread_count}</div>
    <div class="label">Unread</div>
  </div>
  <div class="stat-card urgent">
    <div class="value">{high_count}</div>
    <div class="label">High Priority</div>
  </div>
  <div class="stat-card">
    <div class="value">{total}</div>
    <div class="label">Total (24h)</div>
  </div>
</div>

{high_section}
{medium_section}
{low_section}
{inbox_zero_html}

<div class="digest">
  <h2>AI Summary</h2>
  <p>{escape(digest)}</p>
</div>

<footer>Homie AI &mdash; all data stays local</footer>

<script>
(function() {{
  document.cookie = "session=" + {_js_string(session_token)} + "; path=/; SameSite=Strict";
  var API = "http://127.0.0.1:{int(api_port)}";

  window.markRead = function(emailId) {{
    fetch(API + "/api/email/" + encodeURIComponent(emailId) + "/read", {{
      method: "POST",
      credentials: "include",
    }}).then(function(r) {{
      if (r.ok) {{ location.reload(); }}
    }});
  }};

  window.triageRefresh = function() {{
    fetch(API + "/api/email/triage", {{
      method: "POST",
      credentials: "include",
    }}).then(function() {{ location.reload(); }});
  }};
}})();
</script>
</body>
</html>"""
    return html


def _js_string(value: str) -> str:
    """Encode *value* as a safe JavaScript string literal."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("<", "\\x3c")
        .replace(">", "\\x3e")
    )
    return f'"{escaped}"'
