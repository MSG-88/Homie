// Homie Desktop — Settings panel logic
// Persists settings to localStorage and syncs with the UI.

const SETTINGS_KEY = "homie_settings";

const defaults = {
  model: "homie-default",
  voice: false,
  alwaysOnTop: false,
  theme: "dark",
  plugins: {
    calendar: true,
    knowledge: true,
    proactive: true,
  },
};

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? { ...defaults, ...JSON.parse(raw) } : { ...defaults };
  } catch {
    return { ...defaults };
  }
}

function saveSettings(settings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function applySettings(settings) {
  // Model
  document.getElementById("sel-model").value = settings.model;

  // Voice
  const togVoice = document.getElementById("tog-voice");
  togVoice.checked = settings.voice;
  // Sync global voice flag used by app.js
  if (typeof voiceEnabled !== "undefined") {
    window.voiceEnabled = settings.voice;
  }

  // Always on top
  document.getElementById("tog-aot").checked = settings.alwaysOnTop;

  // Theme
  document.getElementById("sel-theme").value = settings.theme;
  applyTheme(settings.theme);

  // Plugins
  document.getElementById("plug-calendar").checked = settings.plugins.calendar;
  document.getElementById("plug-knowledge").checked = settings.plugins.knowledge;
  document.getElementById("plug-proactive").checked = settings.plugins.proactive;
}

function applyTheme(theme) {
  const root = document.documentElement;
  switch (theme) {
    case "midnight":
      root.style.setProperty("--bg-base", "#05050d");
      root.style.setProperty("--bg-surface", "rgba(10, 10, 24, 0.9)");
      root.style.setProperty("--bg-glass", "rgba(15, 15, 35, 0.6)");
      break;
    case "light":
      root.style.setProperty("--bg-base", "#f0f0f5");
      root.style.setProperty("--bg-surface", "rgba(255, 255, 255, 0.85)");
      root.style.setProperty("--bg-glass", "rgba(240, 240, 250, 0.6)");
      root.style.setProperty("--text-primary", "#1a1a2e");
      root.style.setProperty("--text-secondary", "#6b6b8a");
      break;
    default: // dark
      root.style.setProperty("--bg-base", "#0c0c14");
      root.style.setProperty("--bg-surface", "rgba(20, 20, 36, 0.85)");
      root.style.setProperty("--bg-glass", "rgba(30, 30, 54, 0.55)");
      root.style.setProperty("--text-primary", "#e4e4ef");
      root.style.setProperty("--text-secondary", "#9494ac");
  }
}

// ---- Event wiring ----

document.addEventListener("DOMContentLoaded", () => {
  const settings = loadSettings();
  applySettings(settings);

  // Model
  document.getElementById("sel-model").addEventListener("change", (e) => {
    const s = loadSettings();
    s.model = e.target.value;
    saveSettings(s);
  });

  // Voice
  document.getElementById("tog-voice").addEventListener("change", (e) => {
    const s = loadSettings();
    s.voice = e.target.checked;
    saveSettings(s);
    // Sync
    const btn = document.getElementById("btn-voice");
    if (btn) btn.classList.toggle("active", s.voice);
  });

  // Always on top
  document.getElementById("tog-aot").addEventListener("change", async (e) => {
    const s = loadSettings();
    s.alwaysOnTop = e.target.checked;
    saveSettings(s);
    try {
      if (window.__TAURI_INTERNALS__) {
        await window.__TAURI_INTERNALS__.invoke("toggle_always_on_top");
      }
    } catch {}
  });

  // Theme
  document.getElementById("sel-theme").addEventListener("change", (e) => {
    const s = loadSettings();
    s.theme = e.target.value;
    saveSettings(s);
    applyTheme(s.theme);
  });

  // Plugins
  ["calendar", "knowledge", "proactive"].forEach((p) => {
    document.getElementById(`plug-${p}`).addEventListener("change", (e) => {
      const s = loadSettings();
      s.plugins[p] = e.target.checked;
      saveSettings(s);
    });
  });

  // Back button
  document.getElementById("btn-back").addEventListener("click", () => {
    window.location.hash = "";
  });
});
