// Homie Desktop — Chat logic & WebSocket connection
// Connects to Homie daemon at localhost:3141

const API_URL = "http://localhost:3141/api/chat";
const WS_URL = "ws://localhost:3141/ws";

const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const typingIndicator = document.getElementById("typing-indicator");
const connectionDot = document.getElementById("connection-dot");
const btnVoice = document.getElementById("btn-voice");

let ws = null;
let voiceEnabled = false;
let currentStreamEl = null;

// ---- Markdown rendering (lightweight) ----

function renderMarkdown(text) {
  let html = text
    // Code blocks
    .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
      const escaped = escapeHtml(code.trim());
      return `<pre><code class="lang-${lang || "text"}">${escaped}</code></pre>`;
    })
    // Inline code
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // Bold
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    // Italic
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    // Blockquote
    .replace(/^> (.+)$/gm, "<blockquote>$1</blockquote>")
    // Unordered lists
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    // Links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
    // Line breaks -> paragraphs
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br/>");

  // Wrap list items
  html = html.replace(/(<li>.*?<\/li>)/gs, "<ul>$1</ul>");
  // Remove duplicate nested <ul> tags
  html = html.replace(/<\/ul>\s*<ul>/g, "");

  return `<p>${html}</p>`;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ---- Messages ----

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (role === "assistant") {
    div.innerHTML = renderMarkdown(content);
  } else {
    div.textContent = content;
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function showWelcome() {
  const div = document.createElement("div");
  div.className = "welcome";
  div.innerHTML = `<h3>Hey there!</h3><p>I'm Homie, your local AI assistant.<br/>Type a message or press the mic to talk.</p>`;
  messagesEl.appendChild(div);
}

// ---- WebSocket ----

function connectWebSocket() {
  if (ws && ws.readyState <= 1) return;

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    connectionDot.className = "dot connected";
    connectionDot.title = "Connected";
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      if (data.type === "stream_start") {
        typingIndicator.classList.remove("hidden");
        currentStreamEl = addMessage("assistant", "");
        currentStreamEl._content = "";
      } else if (data.type === "stream_token") {
        if (currentStreamEl) {
          currentStreamEl._content += data.token;
          currentStreamEl.innerHTML = renderMarkdown(currentStreamEl._content);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      } else if (data.type === "stream_end") {
        typingIndicator.classList.add("hidden");
        currentStreamEl = null;
      } else if (data.type === "response") {
        typingIndicator.classList.add("hidden");
        addMessage("assistant", data.content || data.message || "");
        currentStreamEl = null;
      } else if (data.type === "error") {
        typingIndicator.classList.add("hidden");
        addMessage("assistant", "Error: " + (data.message || "Something went wrong."));
        currentStreamEl = null;
      }
    } catch {
      // Plain text response fallback
      typingIndicator.classList.add("hidden");
      addMessage("assistant", event.data);
    }
  };

  ws.onclose = () => {
    connectionDot.className = "dot disconnected";
    connectionDot.title = "Disconnected — retrying...";
    setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = () => {
    connectionDot.className = "dot disconnected";
    connectionDot.title = "Connection error";
  };
}

// ---- Send message ----

async function sendMessage(text) {
  addMessage("user", text);
  typingIndicator.classList.remove("hidden");

  // Prefer WebSocket if connected
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: "chat",
      message: text,
      voice: voiceEnabled,
    }));
    return;
  }

  // Fallback to HTTP
  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, voice: voiceEnabled }),
    });
    const data = await res.json();
    typingIndicator.classList.add("hidden");
    addMessage("assistant", data.response || data.message || JSON.stringify(data));
  } catch (err) {
    typingIndicator.classList.add("hidden");
    addMessage("assistant", "Could not reach Homie daemon. Is it running on port 3141?");
  }
}

// ---- Voice toggle ----

btnVoice.addEventListener("click", () => {
  voiceEnabled = !voiceEnabled;
  btnVoice.classList.toggle("active", voiceEnabled);
  btnVoice.title = voiceEnabled ? "Voice enabled" : "Voice disabled";
});

// ---- Form submit ----

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = "";
  sendMessage(text);
});

// ---- Routing (hash-based for settings toggle) ----

function handleRoute() {
  const hash = window.location.hash;
  const chatView = document.getElementById("chat-view");
  const settingsView = document.getElementById("settings-view");

  if (hash === "#settings") {
    chatView.classList.add("hidden");
    settingsView.classList.remove("hidden");
  } else {
    settingsView.classList.add("hidden");
    chatView.classList.remove("hidden");
  }
}

window.addEventListener("hashchange", handleRoute);

// ---- Title bar buttons (Tauri IPC) ----

async function invokeTauri(cmd, args) {
  if (window.__TAURI_INTERNALS__) {
    return window.__TAURI_INTERNALS__.invoke(cmd, args);
  }
}

document.getElementById("btn-pin").addEventListener("click", async () => {
  const btn = document.getElementById("btn-pin");
  try {
    const isOnTop = await invokeTauri("toggle_always_on_top");
    btn.classList.toggle("active", isOnTop);
  } catch { /* dev mode — no Tauri runtime */ }
});

document.getElementById("btn-settings").addEventListener("click", () => {
  window.location.hash = "#settings";
});

document.getElementById("btn-minimize").addEventListener("click", async () => {
  try { await invokeTauri("minimize_to_tray"); } catch {}
});

document.getElementById("btn-close").addEventListener("click", async () => {
  try { await invokeTauri("minimize_to_tray"); } catch {}
});

// ---- Init ----

showWelcome();
handleRoute();
connectWebSocket();
