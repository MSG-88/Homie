"""Settings page generator for Homie desktop companion."""
from __future__ import annotations


def render_settings_page(session_token: str, api_port: int) -> str:
    """Render a self-contained settings HTML page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Homie — Settings</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }}

  .nav {{ display: flex; gap: 8px; padding: 12px 24px; background: #161b22; border-bottom: 1px solid #30363d; }}
  .nav a {{ color: #8b949e; text-decoration: none; font-size: 13px; padding: 6px 12px; border-radius: 6px; }}
  .nav a:hover {{ color: #c9d1d9; background: #21262d; }}
  .nav a.active {{ color: #f0f6fc; background: #21262d; }}

  .container {{ max-width: 720px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; font-weight: 600; color: #f0f6fc; margin-bottom: 24px; }}

  .section {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 16px; }}
  .section-title {{ font-size: 14px; font-weight: 600; color: #f0f6fc; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }}

  .row {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #21262d; }}
  .row:last-child {{ border-bottom: none; }}
  .row-label {{ font-size: 13px; color: #8b949e; }}
  .row-value {{ font-size: 13px; color: #c9d1d9; font-weight: 500; }}

  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
  .badge-green {{ background: #1b4332; color: #4ade80; }}
  .badge-yellow {{ background: #422006; color: #fbbf24; }}
  .badge-red {{ background: #450a0a; color: #f87171; }}
  .badge-gray {{ background: #21262d; color: #8b949e; }}

  .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  .dot-green {{ background: #4ade80; }}
  .dot-yellow {{ background: #fbbf24; }}
  .dot-red {{ background: #f87171; }}

  .privacy-ok {{ color: #4ade80; font-size: 13px; }}
  .privacy-warn {{ color: #fbbf24; font-size: 13px; }}

  .footer {{ text-align: center; margin-top: 32px; font-size: 12px; color: #484f58; }}

  #loading {{ text-align: center; padding: 40px; color: #8b949e; }}
</style>
</head>
<body>

<div class="nav">
  <a href="/briefing">Briefing</a>
  <a href="/chat">Chat</a>
  <a href="/settings" class="active">Settings</a>
</div>

<div class="container">
  <h1>Settings</h1>
  <div id="loading">Loading...</div>
  <div id="content" style="display:none">

    <div class="section" id="inferenceSection">
      <div class="section-title"><span class="dot" id="inferenceDot"></span> Inference</div>
      <div class="row"><span class="row-label">Active Source</span><span class="row-value" id="activeSource">—</span></div>
      <div class="row"><span class="row-label">Model</span><span class="row-value" id="modelName">—</span></div>
      <div class="row"><span class="row-label">Priority Chain</span><span class="row-value" id="priorityChain">—</span></div>
    </div>

    <div class="section" id="emailSection">
      <div class="section-title"><span class="dot" id="emailDot"></span> Email</div>
      <div id="emailAccounts"></div>
      <div class="row"><span class="row-label">Unread</span><span class="row-value" id="unreadCount">—</span></div>
      <div class="row"><span class="row-label">Total (24h)</span><span class="row-value" id="totalCount">—</span></div>
    </div>

    <div class="section" id="privacySection">
      <div class="section-title">Privacy</div>
      <div class="row"><span class="row-label">Data Location</span><span class="row-value" id="dataLocation">—</span></div>
      <div class="row"><span class="row-label">Status</span><span id="privacyStatus">—</span></div>
    </div>

  </div>

  <div class="footer">Homie AI &mdash; all data stays local</div>
</div>

<script>
const API = "http://127.0.0.1:{api_port}";
document.cookie = "homie_session={session_token}; path=/; SameSite=Strict";

async function loadSettings() {{
  try {{
    const resp = await fetch(API + "/api/settings", {{credentials: "include"}});
    const data = await resp.json();

    // Inference
    const src = data.inference.active_source || "Not configured";
    document.getElementById("activeSource").textContent = src;
    document.getElementById("modelName").textContent = data.inference.model || "None";
    document.getElementById("priorityChain").textContent = (data.inference.priority || []).join(" → ") || "—";

    const dot = document.getElementById("inferenceDot");
    if (src === "Local") {{ dot.className = "dot dot-green"; }}
    else if (src === "Not configured") {{ dot.className = "dot dot-red"; }}
    else {{ dot.className = "dot dot-yellow"; }}

    // Email
    const accounts = data.email.accounts || [];
    const emailDot = document.getElementById("emailDot");
    const emailDiv = document.getElementById("emailAccounts");
    if (accounts.length > 0) {{
      emailDot.className = "dot dot-green";
      accounts.forEach(a => {{
        const row = document.createElement("div");
        row.className = "row";
        row.innerHTML = '<span class="row-label">' + a + '</span><span class="badge badge-green">Connected</span>';
        emailDiv.appendChild(row);
      }});
    }} else {{
      emailDot.className = "dot dot-red";
      emailDiv.innerHTML = '<div class="row"><span class="row-label">No accounts</span><span class="badge badge-gray">Not connected</span></div>';
    }}
    document.getElementById("unreadCount").textContent = data.email.unread || 0;
    document.getElementById("totalCount").textContent = data.email.total_24h || 0;

    // Privacy
    document.getElementById("dataLocation").textContent = data.privacy.data_location || "~/.homie/";
    const status = document.getElementById("privacyStatus");
    if (data.privacy.all_local) {{
      status.innerHTML = '<span class="privacy-ok">All data stays on this device</span>';
    }} else if (data.privacy.cloud_inference) {{
      status.innerHTML = '<span class="privacy-warn">Cloud inference active — queries are sent externally</span>';
    }} else {{
      status.innerHTML = '<span class="privacy-ok">Local processing</span>';
    }}

    document.getElementById("loading").style.display = "none";
    document.getElementById("content").style.display = "block";
  }} catch(e) {{
    document.getElementById("loading").textContent = "Failed to load settings: " + e.message;
  }}
}}

loadSettings();
</script>

</body>
</html>"""
