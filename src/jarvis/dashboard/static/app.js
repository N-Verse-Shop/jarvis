/* Jarvis Control Room — vanilla ESM SPA.
   No bundler, no React: load this with <script type=module>.
   Targets recent Chromium / Safari only.

   Architecture:
   * Auth: token in localStorage.jarvisToken; the auth screen sets it.
     If a request returns 401 we clear the token + show auth again.
   * Routing: ?view=<name> hash; nav tabs swap classes only.
   * Polling: overview + state poll every 2 s. Live view uses /ws/events.
*/

const TOKEN_KEY = "jarvisToken";

function authHeaders() {
  const t = localStorage.getItem(TOKEN_KEY) || "";
  return t ? { Authorization: `Bearer ${t}` } : {};
}

async function api(path, opts = {}) {
  // R35-S32 FIX: callers pass `body` as a plain object (persona PUT,
  // settings PATCH). Without JSON.stringify, fetch coerces it to the
  // string "[object Object]" → the server's req.json() throws → HTTP 400
  // "invalid JSON body". Stringify any non-string body here so EVERY
  // object-body caller works. Callers that already pass a JSON string
  // (facts/chat) are left untouched (typeof === "string").
  const o = { ...opts };
  if (o.body != null && typeof o.body !== "string") {
    o.body = JSON.stringify(o.body);
  }
  const res = await fetch(path, {
    ...o,
    headers: { ...authHeaders(), "Content-Type": "application/json", ...(opts.headers || {}) },
  });
  if (res.status === 401) {
    localStorage.removeItem(TOKEN_KEY);
    showAuth("Token rejected — paste again.");
    throw new Error("401");
  }
  if (!res.ok) throw new Error(`HTTP ${res.status} ${path}`);
  return res.json();
}

// ─── auth ────────────────────────────────────────────────────────
function showAuth(errMsg) {
  document.getElementById("auth-screen").style.display = "flex";
  document.getElementById("app").hidden = true;
  if (errMsg) document.getElementById("auth-error").textContent = errMsg;
}
function hideAuth() {
  document.getElementById("auth-screen").style.display = "none";
  document.getElementById("app").hidden = false;
}

document.getElementById("auth-submit").addEventListener("click", () => {
  const tok = document.getElementById("auth-token").value.trim();
  if (!tok) { document.getElementById("auth-error").textContent = "Empty token."; return; }
  localStorage.setItem(TOKEN_KEY, tok);
  bootstrap();
});
document.getElementById("auth-token").addEventListener("keypress", (e) => {
  if (e.key === "Enter") document.getElementById("auth-submit").click();
});

// ─── routing ─────────────────────────────────────────────────────
const tabs = document.querySelectorAll(".nav-tab");
const views = document.querySelectorAll(".view");
function setView(name) {
  tabs.forEach(t => t.classList.toggle("active", t.dataset.view === name));
  views.forEach(v => v.classList.toggle("active", v.id === `view-${name}`));
  history.replaceState(null, "", `#${name}`);
  if (name === "skills") loadSkills();
  if (name === "audit") loadAudit();
  if (name === "persona") loadPersona();
  if (name === "capabilities") loadCapabilities();
  if (name === "facts") loadFacts();
  if (name === "settings") loadSettings();
  if (name === "live") startLiveStream();
  if (name === "chat") initChatTab();
  if (name === "brain") startBrainView();
  if (name !== "brain") stopBrainView();
  // R34-S57 (C-P2.1): close the live-events WebSocket when leaving
  // the Live tab. Previously the socket stayed open AND auto-
  // reconnected on close, so a backgrounded dashboard kept
  // receiving the 4 Hz event stream forever — combined with the
  // 1 s reconnect (C-P1.2), a crashed daemon meant infinite
  // hammering of the daemon's restart-loop.
  if (name !== "live") stopLiveStream();
}
tabs.forEach(t => t.addEventListener("click", () => setView(t.dataset.view)));

// ─── formatters ──────────────────────────────────────────────────
function fmtTs(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function fmtUptime(s) {
  if (!s) return "—";
  if (s < 60) return `${Math.floor(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}
function shorten(s, n = 80) {
  if (typeof s !== "string") s = JSON.stringify(s);
  return s.length > n ? s.slice(0, n) + "…" : s;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

// ─── overview ────────────────────────────────────────────────────
async function loadOverview() {
  try {
    const [h, state, stats, events] = await Promise.all([
      fetch("/api/health").then(r => r.json()),
      api("/api/state"),
      api("/api/audit/stats?since=" + (Date.now() / 1000 - 86400)),
      api("/api/audit?limit=10"),
    ]);
    document.getElementById("meta-pid").textContent = h.pid;
    document.getElementById("meta-uptime").textContent = fmtUptime(h.uptime_s);
    document.getElementById("ov-pid").textContent = h.pid;
    document.getElementById("ov-version").textContent = h.version;
    document.getElementById("ov-uptime").textContent = fmtUptime(h.uptime_s);
    document.getElementById("ov-state").textContent = state.state || "UNKNOWN";
    // footer
    const fv = document.getElementById("footer-version");
    if (fv) fv.textContent = h.version || "dev-local";

    // Activity stats
    const byStatus = stats.by_status || {};
    document.getElementById("ov-tools").textContent = stats.by_kind?.tool_call ?? 0;
    document.getElementById("ov-done").textContent = byStatus.completed ?? 0;
    document.getElementById("ov-failed").textContent = byStatus.failed ?? 0;
    document.getElementById("ov-turns").textContent = stats.by_kind?.fast_path ?? 0;

    // Top tools
    const topUl = document.getElementById("ov-top-tools");
    const max = Math.max(1, ...(stats.top_tools || []).map(t => t.n));
    topUl.innerHTML = (stats.top_tools || []).slice(0, 6).map(t => `
      <li style="--bar-width: ${(t.n / max * 100).toFixed(0)}%">
        <span>${escapeHtml(t.tool || "?")}</span><span class="muted">${t.n}</span>
      </li>
    `).join("");

    // Recent feed
    const feed = document.getElementById("ov-feed");
    feed.innerHTML = (events.events || []).slice(0, 8).map(e => renderFeedItem(e)).join("");

    // State orb
    setStateOrb(state.state);
    document.getElementById("brand-state").textContent = (state.state || "—").toLowerCase();
  } catch (e) {
    console.warn("overview load failed:", e);
  }
}

function setStateOrb(state) {
  // R34-S19: orb → EQ-pill. Kept the function name so existing
  // call-sites in overview poller + state poller don't need touching.
  const pill = document.getElementById("state-pill");
  if (!pill) return;
  const norm = (state || "OFFLINE").toLowerCase();
  pill.className = "status-pill state-" + norm;
}

// ─── skills ──────────────────────────────────────────────────────
async function loadSkills() {
  try {
    const { skills } = await api("/api/skills");
    const list = document.getElementById("skills-list");
    if (!skills.length) {
      list.innerHTML = '<p class="subtle">No skills yet. Create one under <code>~/.config/jarvis/skills/&lt;name&gt;/SKILL.md</code> and reload.</p>';
      return;
    }
    list.innerHTML = skills.map(s => `
      <div class="card glass skill-card" data-name="${escapeHtml(s.name)}">
        <div class="skill-name">${escapeHtml(s.name)}</div>
        <div class="skill-desc">${escapeHtml(s.description)}</div>
        <div class="skill-tags">
          ${s.tags.map(t => `<span class="skill-tag">${escapeHtml(t)}</span>`).join("")}
          <span class="skill-tag risk-${s.risk}">risk: ${escapeHtml(s.risk)}</span>
          ${s.references.length ? `<span class="skill-tag">${s.references.length} refs</span>` : ""}
        </div>
      </div>
    `).join("");
    list.querySelectorAll(".skill-card").forEach(card => {
      card.addEventListener("click", () => openSkillDetail(card.dataset.name));
    });
  } catch (e) {
    console.warn("skills load failed:", e);
  }
}

async function openSkillDetail(name) {
  try {
    const s = await api(`/api/skills/${encodeURIComponent(name)}`);
    document.getElementById("skill-detail-name").textContent = s.name;
    document.getElementById("skill-detail-tags").innerHTML = `
      ${s.tags.map(t => `<span class="skill-tag">${escapeHtml(t)}</span>`).join("")}
      <span class="skill-tag risk-${s.risk}">risk: ${escapeHtml(s.risk)}</span>
      <span class="skill-tag">v${escapeHtml(s.version)}</span>
      ${s.locale ? `<span class="skill-tag">${escapeHtml(s.locale)}</span>` : ""}
    `;
    document.getElementById("skill-detail-content").textContent = s.content;
    const refsDiv = document.getElementById("skill-refs-list");
    refsDiv.innerHTML = s.references.map(r => `
      <button class="ghost" data-ref="${escapeHtml(r)}">${escapeHtml(r)}.md</button>
    `).join("");
    refsDiv.querySelectorAll("button").forEach(b => {
      b.addEventListener("click", async () => {
        try {
          const ref = await api(`/api/skills/${encodeURIComponent(name)}/refs/${encodeURIComponent(b.dataset.ref)}`);
          document.getElementById("skill-detail-content").textContent = ref.content;
        } catch (e) { console.warn(e); }
      });
    });
    document.getElementById("skill-detail").hidden = false;
  } catch (e) { console.warn("skill detail failed:", e); }
}
document.getElementById("skill-close").addEventListener("click", () => {
  document.getElementById("skill-detail").hidden = true;
});
document.getElementById("skills-reload").addEventListener("click", async () => {
  await api("/api/skills/reload", { method: "POST" });
  loadSkills();
});

// ─── skill editor (R34-S23) ──────────────────────────────────────
// Tracks which skill we're editing; null = creating a new one.
let _editingSkillName = null;

function skillEditorToast(message, level = "ok") {
  const el = document.getElementById("skill-editor-toast");
  if (!el) return;
  el.textContent = message;
  el.className = `toast toast-${level}`;
  el.hidden = false;
  clearTimeout(window._skillEditorToastTimer);
  window._skillEditorToastTimer = setTimeout(() => { el.hidden = true; }, 4000);
}

function openSkillEditor({ name = null, content = "", description = "",
                          status = "active", version = "1.0.0",
                          tags = [], tools = [], risk = "low",
                          locale = "ru", isNew = false } = {}) {
  _editingSkillName = isNew ? null : name;
  document.getElementById("skill-detail").hidden = true;
  document.getElementById("skill-editor-title").textContent =
    isNew ? "New skill" : `Edit: ${name}`;
  const nameInput = document.getElementById("skill-edit-name");
  nameInput.value = name || "";
  nameInput.disabled = !isNew;       // immutable after create
  document.getElementById("skill-edit-desc").value = description;
  document.getElementById("skill-edit-status").value = status;
  document.getElementById("skill-edit-risk").value = risk;
  document.getElementById("skill-edit-locale").value = locale;
  document.getElementById("skill-edit-version").value = version;
  document.getElementById("skill-edit-tags").value =
    Array.isArray(tags) ? tags.join(", ") : (tags || "");
  document.getElementById("skill-edit-tools").value =
    Array.isArray(tools) ? tools.join(", ") : (tools || "");
  document.getElementById("skill-edit-content").value = content || "";
  document.getElementById("skill-editor-toast").hidden = true;
  document.getElementById("skill-editor").hidden = false;
}

async function saveSkillEditor() {
  const csv = id => (document.getElementById(id)?.value || "")
    .split(",").map(s => s.trim()).filter(Boolean);
  const body = {
    description: document.getElementById("skill-edit-desc").value.trim(),
    status: document.getElementById("skill-edit-status").value,
    version: document.getElementById("skill-edit-version").value.trim() || "1.0.0",
    tags: csv("skill-edit-tags"),
    tools: csv("skill-edit-tools"),
    risk: document.getElementById("skill-edit-risk").value,
    locale: document.getElementById("skill-edit-locale").value,
    content: document.getElementById("skill-edit-content").value,
  };
  if (!body.description) {
    skillEditorToast("Description обов'язковий", "error");
    return;
  }
  if (!body.content.trim()) {
    skillEditorToast("Body не може бути пустим", "error");
    return;
  }
  try {
    let res;
    if (_editingSkillName === null) {
      body.name = document.getElementById("skill-edit-name").value.trim();
      if (!body.name) {
        skillEditorToast("Name обов'язковий", "error");
        return;
      }
      res = await api("/api/skills", { method: "POST", body });
    } else {
      res = await api(`/api/skills/${encodeURIComponent(_editingSkillName)}`,
                      { method: "PUT", body });
    }
    if (res.ok) {
      skillEditorToast(`✓ Збережено: ${res.name}`, "ok");
      setTimeout(() => {
        document.getElementById("skill-editor").hidden = true;
        loadSkills();
      }, 600);
    } else {
      skillEditorToast(`✗ ${res.error || "невідома помилка"}`, "error");
    }
  } catch (exc) {
    skillEditorToast(`✗ ${exc.message || exc}`, "error");
  }
}

async function deleteCurrentSkill() {
  const name = document.getElementById("skill-detail-name").textContent.trim();
  if (!name) return;
  if (!confirm(`Видалити скіл «${name}»? Файл буде переміщено в __trash__/ — можна відновити вручну.`)) return;
  try {
    const res = await api(`/api/skills/${encodeURIComponent(name)}`, { method: "DELETE" });
    if (res.ok) {
      document.getElementById("skill-detail").hidden = true;
      loadSkills();
    } else {
      alert(`Не вдалось видалити: ${res.error || "невідома помилка"}`);
    }
  } catch (exc) {
    alert(`Помилка видалення: ${exc.message || exc}`);
  }
}

async function editCurrentSkill() {
  const name = document.getElementById("skill-detail-name").textContent.trim();
  if (!name) return;
  try {
    const s = await api(`/api/skills/${encodeURIComponent(name)}`);
    openSkillEditor({
      name: s.name,
      content: s.content,
      description: s.description,
      status: s.status || "active",
      version: s.version,
      tags: s.tags,
      tools: s.tools,
      risk: s.risk,
      locale: s.locale,
      isNew: false,
    });
  } catch (exc) {
    alert(`Не вдалось завантажити для редагування: ${exc.message || exc}`);
  }
}

document.getElementById("skills-new")?.addEventListener("click", () => {
  openSkillEditor({ isNew: true });
});
document.getElementById("skill-edit-save")?.addEventListener("click", saveSkillEditor);
document.getElementById("skill-edit-cancel")?.addEventListener("click", () => {
  document.getElementById("skill-editor").hidden = true;
});
document.getElementById("skill-editor-close")?.addEventListener("click", () => {
  document.getElementById("skill-editor").hidden = true;
});
document.getElementById("skill-edit-btn")?.addEventListener("click", editCurrentSkill);
document.getElementById("skill-delete-btn")?.addEventListener("click", deleteCurrentSkill);

// ─── audit ───────────────────────────────────────────────────────
async function loadAudit() {
  try {
    const params = new URLSearchParams({ limit: 200 });
    const k = document.getElementById("audit-kind").value;
    const s = document.getElementById("audit-status").value;
    const t = document.getElementById("audit-tool").value.trim();
    if (k) params.set("kind", k);
    if (s) params.set("status", s);
    if (t) params.set("tool", t);
    const { events } = await api(`/api/audit?${params}`);
    const tbody = document.getElementById("audit-tbody");
    if (!events.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty">No events match.</td></tr>';
      return;
    }
    tbody.innerHTML = events.map(e => `
      <tr>
        <td>${fmtTs(e.ts)}</td>
        <td>${escapeHtml(e.kind)}</td>
        <td>${escapeHtml(e.tool || "—")}</td>
        <td class="status-${escapeHtml(e.status || "")}">${escapeHtml(e.status || "—")}</td>
        <td>${escapeHtml(shorten(e.args || e.error || "", 60))}</td>
        <td>${e.duration_ms != null ? e.duration_ms + "ms" : "—"}</td>
      </tr>
    `).join("");
  } catch (e) { console.warn("audit load:", e); }
}
document.getElementById("audit-refresh").addEventListener("click", loadAudit);
["audit-kind", "audit-status"].forEach(id => {
  document.getElementById(id).addEventListener("change", loadAudit);
});
document.getElementById("audit-tool").addEventListener("keypress", (e) => {
  if (e.key === "Enter") loadAudit();
});

// ─── persona ─────────────────────────────────────────────────────
async function loadPersona() {
  try {
    const p = await api("/api/persona");
    document.getElementById("persona-name").textContent = p.name;
    document.getElementById("persona-owner").textContent = p.owner || "—";
    document.getElementById("persona-lang").textContent = p.language_preference || "auto";
    document.getElementById("persona-path").textContent = p.path || "—";
    document.getElementById("persona-identity").textContent = p.identity || "(empty)";
    document.getElementById("persona-boundaries").innerHTML =
      (p.boundaries || []).map(b => `<li>${escapeHtml(b)}</li>`).join("") || "<li class='subtle'>(none)</li>";
    document.getElementById("persona-shortcuts").innerHTML =
      (p.shortcuts || []).map(s => `<li>${escapeHtml(s)}</li>`).join("") || "<li class='subtle'>(none)</li>";
  } catch (e) { console.warn("persona load:", e); }
}
document.getElementById("persona-reload").addEventListener("click", async () => {
  await api("/api/persona/reload", { method: "POST" });
  loadPersona();
});

// ─── persona editor (R34-S23) ────────────────────────────────────
function personaToast(message, level = "ok") {
  const el = document.getElementById("persona-toast");
  if (!el) return;
  el.textContent = message;
  el.className = `toast toast-${level}`;
  el.hidden = false;
  clearTimeout(window._personaToastTimer);
  window._personaToastTimer = setTimeout(() => { el.hidden = true; }, 4000);
}

async function openPersonaEditor() {
  // Pull RAW file content (persona endpoint returns parsed, not raw).
  // The cleanest path is to extract identity/boundaries/shortcuts and
  // hand-roll a serialized form — but for full fidelity (incl. our
  // R34-S19 "Style" section), fetch /api/persona/raw if available, else
  // synthesize from parsed fields. We don't have a raw endpoint, so use
  // a one-off fetch helper.
  try {
    const r = await fetch("/api/persona/raw", { headers: authHeaders() });
    if (r.ok) {
      const j = await r.json();
      document.getElementById("persona-edit-content").value = j.content || "";
    } else {
      // Synthesize from parsed (rare — endpoint should exist post-S23).
      const p = await api("/api/persona");
      document.getElementById("persona-edit-content").value = _personaSynthesize(p);
    }
  } catch {
    const p = await api("/api/persona");
    document.getElementById("persona-edit-content").value = _personaSynthesize(p);
  }
  document.getElementById("persona-card").hidden = true;
  document.getElementById("persona-editor").hidden = false;
}

function _personaSynthesize(p) {
  // Best-effort serializer if raw isn't available.
  return [
    "---",
    `name: ${p.name || "Jarvis"}`,
    `owner: ${p.owner || ""}`,
    `language_preference: ${p.language_preference || "ru"}`,
    "---",
    "",
    "# Jarvis Persona",
    "",
    "## Identity",
    "",
    p.identity || "",
    "",
    "## Boundaries",
    "",
    ...(p.boundaries || []).map(b => `- ${b}`),
    "",
    "## Shortcuts",
    "",
    ...(p.shortcuts || []).map(s => `- ${s}`),
    "",
  ].join("\n");
}

async function savePersonaEditor() {
  const content = document.getElementById("persona-edit-content").value;
  if (!content.trim()) {
    personaToast("Файл не може бути пустим", "error");
    return;
  }
  if (!content.trimStart().startsWith("---")) {
    personaToast("persona.md повинна починатись з ---", "error");
    return;
  }
  try {
    const res = await api("/api/persona", { method: "PUT", body: { content } });
    if (res.ok) {
      personaToast("✓ Збережено + перезавантажено", "ok");
      document.getElementById("persona-editor").hidden = true;
      document.getElementById("persona-card").hidden = false;
      loadPersona();
    } else {
      personaToast(`✗ ${res.error || "невідома помилка"}`, "error");
    }
  } catch (exc) {
    personaToast(`✗ ${exc.message || exc}`, "error");
  }
}

document.getElementById("persona-edit-toggle")?.addEventListener("click", openPersonaEditor);
document.getElementById("persona-save")?.addEventListener("click", savePersonaEditor);
document.getElementById("persona-cancel")?.addEventListener("click", () => {
  document.getElementById("persona-editor").hidden = true;
  document.getElementById("persona-card").hidden = false;
});

// ─── capabilities ────────────────────────────────────────────────
async function loadCapabilities() {
  try {
    const { gates } = await api("/api/capabilities");
    const list = document.getElementById("capabilities-list");
    list.innerHTML = Object.entries(gates).map(([name, info]) => `
      <div class="card glass cap-card ${info.active ? "" : "closed"}">
        <div class="cap-head">
          <div class="cap-name">${escapeHtml(name)}</div>
          <div class="cap-badge ${info.active ? "open" : "closed"}">${info.active ? "OPEN" : "CLOSED"}</div>
        </div>
        <div class="cap-tools">${info.tools.length ? info.tools.join(", ") : "<em>(no tools registered)</em>"}</div>
        <div class="cap-env">
          ${info.active ? `disable: ${info.env_disable}=true` : `enable: ${info.env_enable}=true`}
        </div>
      </div>
    `).join("");
  } catch (e) { console.warn("capabilities load:", e); }
}

// ─── facts ───────────────────────────────────────────────────────
function fmtRelTime(ts) {
  const d = (Date.now() / 1000) - ts;
  if (d < 60) return `${Math.floor(d)}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

async function loadFacts() {
  const q = document.getElementById("facts-q")?.value.trim() || "";
  const prefix = document.getElementById("facts-prefix")?.value.trim() || "";
  try {
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (prefix) params.set("key_prefix", prefix);
    params.set("limit", "100");
    const { facts } = await api(`/api/facts?${params}`);
    const stats = await api("/api/facts/stats");
    document.getElementById("facts-count").textContent = stats.active;
    document.getElementById("facts-tomb").textContent = stats.tombstoned;
    const list = document.getElementById("facts-list");
    if (!facts.length) {
      list.innerHTML = `<div class="subtle" style="padding:1rem">No facts. Add one above ↑</div>`;
      return;
    }
    list.innerHTML = facts.map(f => `
      <div class="card glass fact-card" data-id="${f.id}">
        <div class="fact-head">
          <span class="fact-key">${escapeHtml(f.key)}</span>
          <span class="fact-score" title="decay-weighted score">${f.score.toFixed(2)}</span>
        </div>
        <div class="fact-value">${escapeHtml(f.value)}</div>
        <div class="fact-meta">
          <span>${escapeHtml(f.source)}</span>
          <span>conf ${f.confidence.toFixed(2)}</span>
          <span>hits ${f.hits}</span>
          <span>${fmtRelTime(f.last_used)}</span>
          <button class="fact-del" data-id="${f.id}">forget</button>
        </div>
      </div>
    `).join("");
    list.querySelectorAll(".fact-del").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const id = e.target.dataset.id;
        if (!confirm(`Forget fact #${id}?`)) return;
        await api(`/api/facts/${id}`, { method: "DELETE" });
        loadFacts();
      });
    });
  } catch (e) { console.warn("facts load:", e); }
}

document.getElementById("facts-add")?.addEventListener("click", async () => {
  const key = document.getElementById("facts-new-key").value.trim();
  const value = document.getElementById("facts-new-value").value.trim();
  if (!key || !value) {
    document.getElementById("facts-new-error").textContent = "Both key and value required.";
    return;
  }
  try {
    await api("/api/facts", {
      method: "POST",
      body: JSON.stringify({ key, value, source: "dashboard" }),
    });
    document.getElementById("facts-new-key").value = "";
    document.getElementById("facts-new-value").value = "";
    document.getElementById("facts-new-error").textContent = "";
    loadFacts();
  } catch (e) {
    document.getElementById("facts-new-error").textContent = e.message;
  }
});

document.getElementById("facts-refresh")?.addEventListener("click", loadFacts);
document.getElementById("facts-q")?.addEventListener("input", () => {
  clearTimeout(window._factsDebounce);
  window._factsDebounce = setTimeout(loadFacts, 200);
});
document.getElementById("facts-prefix")?.addEventListener("change", loadFacts);
document.getElementById("facts-prune")?.addEventListener("click", async () => {
  if (!confirm("Run decay-prune now? Old, low-score facts will be tombstoned.")) return;
  const r = await api("/api/facts/prune", { method: "POST" });
  alert(`Pruned ${r.pruned} fact(s).`);
  loadFacts();
});

// ─── live events stream ──────────────────────────────────────────
let liveSocket = null;
function renderFeedItem(e) {
  // ev is a parsed JSON object from events.jsonl. The structure is
  // open-ended; we extract a small summary line.
  const type = e.type || "event";
  let cls = "type";
  if (type.includes("error") || e.status === "failed") cls += " error";
  else if (e.status === "completed" || type === "tts_done") cls += " ok";
  else if (type === "wake_word" || type === "hot_window") cls += " warn";
  const body =
    e.text ? e.text :
    e.tool ? `${e.tool} ${e.status || ""}${e.duration_ms ? ` (${e.duration_ms}ms)` : ""}` :
    e.content ? e.content.slice(0, 80) :
    e.message ? e.message :
    "";
  return `
    <li>
      <span class="ts">${fmtTs(e.ts || Date.now() / 1000)}</span>
      <span class="${cls}">${escapeHtml(type)}</span>
      <span class="body">${escapeHtml(body)}</span>
    </li>
  `;
}

// R34-S57 (C-P1.2): exponential backoff for the WS reconnect.
// Previously the client unconditionally re-opened every 1s on
// close, so a crashed daemon meant the browser burned a WS handshake
// every second forever — and with CF Tunnel in front, also burned
// CF connection budget. 1/2/4/8/16/30s ceiling matches the typical
// daemon restart window.
let _liveReconnectMs = 1000;
const _LIVE_RECONNECT_MAX_MS = 30000;

function stopLiveStream() {
  // R34-S57 (C-P2.1): explicit close lets setView() drop the
  // socket when the user navigates away from Live.
  try { liveSocket?.close(); } catch (_) {}
  liveSocket = null;
  _liveReconnectMs = 1000; // reset backoff for next visit
}

function startLiveStream() {
  if (liveSocket && liveSocket.readyState !== WebSocket.CLOSED) return;
  const token = localStorage.getItem(TOKEN_KEY);
  const url = new URL("/ws/events", window.location.href);
  url.protocol = url.protocol.replace("http", "ws");
  // R34-S57 (C-P1.1) — known limitation:
  // The browser cannot set headers on a WebSocket handshake, so the
  // bearer token MUST ride in the URL. We keep this for backwards
  // compatibility with the existing dashboard. Defense-in-depth
  // mitigations live server-side (CSP + token rotation + 0600 file
  // perms on config.json). The right long-term fix is a one-shot
  // ticket exchange via POST /api/ws-ticket; tracked as a follow-up.
  url.searchParams.set("token", token);
  const feed = document.getElementById("live-feed");
  feed.innerHTML = "";
  liveSocket = new WebSocket(url.toString());
  liveSocket.addEventListener("open", () => {
    _liveReconnectMs = 1000; // reset on successful open
  });
  liveSocket.addEventListener("message", (ev) => {
    if (document.getElementById("live-pause").checked) return;
    try {
      const parsed = JSON.parse(ev.data);
      feed.insertAdjacentHTML("afterbegin", renderFeedItem(parsed));
      // Cap at 200 entries
      while (feed.children.length > 200) feed.removeChild(feed.lastChild);
    } catch (e) { console.warn("live parse:", e); }
  });
  liveSocket.addEventListener("close", () => {
    const delay = _liveReconnectMs;
    _liveReconnectMs = Math.min(_liveReconnectMs * 2, _LIVE_RECONNECT_MAX_MS);
    setTimeout(() => {
      // Only reconnect if the user is STILL on the Live tab AND
      // the page is visible (don't burn cycles in a background
      // tab).
      const active = document.getElementById("view-live").classList.contains("active");
      const visible = document.visibilityState !== "hidden";
      if (active && visible) {
        startLiveStream();
      }
    }, delay);
  });
}
document.getElementById("live-clear").addEventListener("click", () => {
  document.getElementById("live-feed").innerHTML = "";
});

// ─── bootstrap ───────────────────────────────────────────────────
async function bootstrap() {
  // Verify token quickly with /api/persona (any authed endpoint)
  try {
    await api("/api/persona");
  } catch (e) {
    showAuth("Token check failed — paste again.");
    return;
  }
  hideAuth();
  document.getElementById("auth-error").textContent = "";

  // Route by hash
  const hash = window.location.hash.replace("#", "") || "overview";
  setView(hash);

  // Poll overview every 2s when overview view is active
  loadOverview();
  setInterval(() => {
    if (document.getElementById("view-overview").classList.contains("active")) {
      loadOverview();
    } else {
      // Still refresh the header meta even when on other views
      fetch("/api/health").then(r => r.json()).then(h => {
        document.getElementById("meta-pid").textContent = h.pid;
        document.getElementById("meta-uptime").textContent = fmtUptime(h.uptime_s);
      }).catch(() => {});
      api("/api/state").then(s => {
        setStateOrb(s.state);
        document.getElementById("brand-state").textContent = (s.state || "—").toLowerCase();
      }).catch(() => {});
    }
  }, 2000);

  // Footer clock — updates every second, low-cost (no network).
  const tickClock = () => {
    const el = document.getElementById("footer-clock");
    if (!el) return;
    const now = new Date();
    el.textContent = now.toLocaleTimeString("uk-UA", {
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  };
  tickClock();
  setInterval(tickClock, 1000);
}

// ─── Settings (R34-S18) ──────────────────────────────────────────
const _settingsCurrent = { _writable_keys: [] };

async function loadSettings() {
  try {
    const data = await api("/api/settings");
    Object.assign(_settingsCurrent, data);
    // Populate Piper voice dropdown
    const sel = document.getElementById("set-piper-voice");
    if (sel) {
      sel.innerHTML = (data._piper_voice_catalog || [])
        .map(v => `<option value="${v.id}">${v.label}</option>`)
        .join("");
      sel.value = data.piper_voice || "ru_RU-ruslan-medium";
    }
    // Apply scalar values to inputs
    const setVal = (id, value, fallback) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.value = (value !== null && value !== undefined) ? value : fallback;
    };
    setVal("set-wake-word", data.wake_word, "jarvis");
    setVal("set-vad-aggr", data.vad_aggressiveness, 2);
    setVal("set-wake-fuzzy", data.wake_fuzzy_ratio, 0.68);
    setVal("set-hot-window", data.hot_window_seconds, 3);
    setVal("set-endpoint", data.endpoint_silence_ms, 700);
    setVal("set-active-lang", data.active_language || data.whisper_language, "ru");
    setVal("set-whisper-lang", data.whisper_language || data.active_language, "ru");
    setVal("set-chat-model", data.ollama_chat_model, "qwen3:8b");
    // Sync live readout spans
    document.getElementById("set-vad-aggr-val").textContent = data.vad_aggressiveness ?? 2;
    document.getElementById("set-wake-fuzzy-val").textContent = (data.wake_fuzzy_ratio ?? 0.68).toFixed(2);
    document.getElementById("set-hot-window-val").textContent = data.hot_window_seconds ?? 3;
    document.getElementById("set-endpoint-val").textContent = data.endpoint_silence_ms ?? 700;
    // R35-S31: auto-render the advanced model/privacy/Claude/behaviour
    // controls from the backend schema.
    renderExtraSettings(data._schema || [], data);
  } catch (exc) {
    settingsToast(`Завантажити не вдалося: ${exc.message || exc}`, "error");
  }
}

// R35-S31: schema-driven settings. The backend returns ``_schema`` (a
// list of {key,label,type,group,options,min,max}); we render grouped
// cards so EVERY exposed setting is editable without hand-writing HTML.
function _escAttr(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/"/g, "&quot;")
    .replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function renderField(f, val) {
  const id = `setx-${f.key}`;
  const common = `id="${id}" data-key="${f.key}" data-type="${f.type}" class="setting-input"`;
  let ctrl;
  if (f.type === "bool") {
    const checked = (val === true || val === "true" || val === 1) ? "checked" : "";
    ctrl = `<input type="checkbox" ${common} ${checked} />`;
  } else if (f.type === "enum") {
    ctrl = `<select ${common}>` + (f.options || [])
      .map(o => `<option value="${_escAttr(o)}" ${String(val) === o ? "selected" : ""}>${_escAttr(o)}</option>`)
      .join("") + `</select>`;
  } else if (f.type === "int") {
    ctrl = `<input type="number" ${common} value="${_escAttr(val)}" min="${f.min ?? ""}" max="${f.max ?? ""}" />`;
  } else if (f.type === "list") {
    const text = Array.isArray(val) ? val.join("\n") : (val || "");
    ctrl = `<textarea ${common} rows="6" placeholder="один термін на рядок">${_escAttr(text)}</textarea>`;
  } else {
    ctrl = `<input type="text" ${common} value="${_escAttr(val)}" />`;
  }
  return `<label class="setting"><span class="setting-label">${_escAttr(f.label)}</span>${ctrl}</label>`;
}
function renderExtraSettings(schema, data) {
  const host = document.getElementById("settings-extra");
  if (!host) return;
  const groups = {};
  schema.forEach(f => { (groups[f.group] ||= []).push(f); });
  host.innerHTML = Object.entries(groups).map(([g, fields]) =>
    `<div class="card settings-card"><h3>${_escAttr(g)}</h3>` +
    fields.map(f => renderField(f, data[f.key])).join("") +
    `</div>`
  ).join("");
}

function settingsToast(message, level = "ok") {
  const el = document.getElementById("settings-toast");
  if (!el) return;
  el.textContent = message;
  el.className = `toast toast-${level}`;
  el.hidden = false;
  clearTimeout(window._settingsToastTimer);
  window._settingsToastTimer = setTimeout(() => { el.hidden = true; }, 4000);
}

// R34-S57 (C-P2.2): button-disable helper. Prevents double-submit
// when the user mashes Save / Restart while a request is in flight
// — previously two PATCH /api/settings calls would race the file
// rewrite, or two POST /api/restart calls would queue two
// SIGTERMs (the second arriving as the new daemon spins up).
function _withButtonBusy(btnId, label, fn) {
  return async () => {
    const btn = document.getElementById(btnId);
    if (!btn) { await fn(); return; }
    if (btn.disabled) return; // already in flight
    const prev = btn.textContent;
    btn.disabled = true;
    btn.textContent = label;
    try {
      await fn();
    } finally {
      btn.disabled = false;
      btn.textContent = prev;
    }
  };
}

async function saveSettings() {
  // Collect form values into a patch object.
  const numOrUndef = (id, parser) => {
    const el = document.getElementById(id);
    if (!el || el.value === "" || el.value === null) return undefined;
    const v = parser(el.value);
    return Number.isFinite(v) ? v : undefined;
  };
  const strOrUndef = (id) => {
    const el = document.getElementById(id);
    const v = (el?.value || "").trim();
    return v ? v : undefined;
  };
  const patch = {
    piper_voice: strOrUndef("set-piper-voice"),
    wake_word: strOrUndef("set-wake-word"),
    vad_aggressiveness: numOrUndef("set-vad-aggr", parseInt),
    wake_fuzzy_ratio: numOrUndef("set-wake-fuzzy", parseFloat),
    hot_window_seconds: numOrUndef("set-hot-window", parseFloat),
    endpoint_silence_ms: numOrUndef("set-endpoint", parseInt),
    active_language: strOrUndef("set-active-lang"),
    whisper_language: strOrUndef("set-whisper-lang"),
    ollama_chat_model: strOrUndef("set-chat-model"),
  };
  // R35-S31: collect the auto-rendered advanced controls.
  document.querySelectorAll("#settings-extra [data-key]").forEach(el => {
    const key = el.getAttribute("data-key");
    const type = el.getAttribute("data-type");
    if (type === "bool") {
      patch[key] = !!el.checked;            // always send (true OR false)
    } else if (type === "list") {
      patch[key] = (el.value || "").split("\n").map(x => x.trim()).filter(Boolean);
    } else if (type === "int") {
      if (el.value === "" || el.value === null) return;
      const v = parseInt(el.value, 10);
      if (Number.isFinite(v)) patch[key] = v;
    } else {
      const v = (el.value || "").trim();
      if (v) patch[key] = v;
    }
  });
  // Strip undefined so the server doesn't reject "key: undefined".
  Object.keys(patch).forEach(k => patch[k] === undefined && delete patch[k]);
  if (Object.keys(patch).length === 0) {
    settingsToast("Нічого не змінено", "warn");
    return;
  }
  try {
    const res = await api("/api/settings", { method: "PATCH", body: patch });
    if (res.ok) {
      const n = Object.keys(res.updated || {}).length;
      settingsToast(`✓ Збережено ${n} налаштування. Натисни «Restart Jarvis» щоб застосувати.`, "ok");
    } else {
      settingsToast(`✗ ${res.error || "невідома помилка"}`, "error");
    }
  } catch (exc) {
    settingsToast(`✗ ${exc.message || exc}`, "error");
  }
}

async function restartDaemon() {
  if (!confirm("Перезапустити Jarvis daemon? Сесія дашборду залишиться відкритою, але новий PID підніметься за ~10-15s.")) return;
  // R34-S20: fetch() to /api/restart is RACING the daemon's own
  // SIGTERM — the response is written then the process dies. Even
  // with Connection: close + a 1.2s flush window, the browser
  // sometimes still surfaces `TypeError: Failed to fetch`. That's
  // EXPECTED during restart, not a real failure, so we treat the
  // network error the same as a 200 and poll /api/health until the
  // daemon comes back to confirm.
  settingsToast("⟳ Перезапуск… новий PID за ~10-15s.", "ok");
  try {
    await api("/api/restart", { method: "POST" });
  } catch (exc) {
    // Network failure mid-restart is fine — we expected it. Anything
    // else (e.g. 401 token rejection) bubbles back up to the user.
    const msg = String(exc.message || exc);
    if (msg.includes("Failed to fetch") || msg.includes("NetworkError") ||
        msg.includes("network") || msg.includes("aborted")) {
      // expected — keep going
    } else if (msg.startsWith("HTTP 5") || msg.startsWith("HTTP 4")) {
      settingsToast(`Restart відмова: ${msg}`, "error");
      return;
    }
    // else: treat as expected disconnect
  }
  // Poll /api/health until we see a NEW pid + uptime resetting.
  const startedAt = Date.now();
  let oldPid = null;
  try {
    const h0 = await fetch("/api/health").then(r => r.json()).catch(() => null);
    if (h0 && h0.pid) oldPid = h0.pid;
  } catch {}
  const poll = async () => {
    while (Date.now() - startedAt < 60_000) {
      await new Promise(r => setTimeout(r, 1500));
      try {
        const r = await fetch("/api/health", { cache: "no-store" });
        if (!r.ok) continue;
        const j = await r.json();
        if (j && j.pid && (!oldPid || j.pid !== oldPid) && j.uptime_s < 30) {
          settingsToast(`✓ Jarvis перезапущено (PID ${j.pid}).`, "ok");
          await loadSettings().catch(() => {});
          return;
        }
      } catch { /* daemon still down, keep polling */ }
    }
    settingsToast("⚠ Перезапуск зайняв більше 60s — перевір логи.", "warn");
  };
  poll();
}

// Wire up controls + live readouts
// R34-S57 (C-P2.2): wrap the click handlers so duplicate clicks
// while a request is in flight are dropped (and the user gets a
// visible "saving…" / "restarting…" state).
document.getElementById("settings-save")?.addEventListener(
  "click",
  _withButtonBusy("settings-save", "Saving…", saveSettings),
);
document.getElementById("settings-restart")?.addEventListener(
  "click",
  _withButtonBusy("settings-restart", "Restarting…", restartDaemon),
);
document.getElementById("settings-reload")?.addEventListener("click", loadSettings);
["set-vad-aggr", "set-wake-fuzzy", "set-hot-window", "set-endpoint"].forEach(id => {
  document.getElementById(id)?.addEventListener("input", e => {
    const valEl = document.getElementById(`${id}-val`);
    if (!valEl) return;
    const v = parseFloat(e.target.value);
    if (id === "set-wake-fuzzy") valEl.textContent = v.toFixed(2);
    else valEl.textContent = v;
  });
});

// ─── Chat tab (R34-S29) ──────────────────────────────────────────
// Text fallback for when voice path is wedged — sends user text
// straight to Ollama via /api/chat. Stop button hits /api/interrupt
// which the voice thread polls and translates to InterruptionFrame.
let _chatBusy = false;
let _chatAbort = null;
function _chatStatus(msg, level = "ok") {
  const el = document.getElementById("chat-status");
  if (!el) return;
  el.textContent = msg || "";
  el.className = `subtle chat-status chat-status-${level}`;
}
function _chatAppend(role, text) {
  const log = document.getElementById("chat-log");
  if (!log) return;
  const row = document.createElement("div");
  row.className = `chat-row chat-${role}`;
  row.innerHTML = `
    <div class="chat-role">${role === "user" ? "Ти" : "Jarvis"}</div>
    <div class="chat-text">${escapeHtml(text).replace(/\n/g, "<br>")}</div>
  `;
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}
function initChatTab() {
  // Lazy bind once; subsequent setView(chat) calls are no-ops.
  if (window._chatInit) return;
  window._chatInit = true;
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  const resetBtn = document.getElementById("chat-reset");
  const stopBtn = document.getElementById("chat-stop");
  if (!input || !sendBtn) return;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    } else if (e.key === "Escape") {
      e.preventDefault();
      input.value = "";
      _chatStatus("");
    }
  });
  sendBtn.addEventListener("click", sendChatMessage);
  resetBtn.addEventListener("click", () => {
    input.value = "";
    input.focus();
    _chatStatus("Очищено", "ok");
  });
  stopBtn.addEventListener("click", interruptJarvis);
  setTimeout(() => input.focus(), 50);
}
async function sendChatMessage() {
  if (_chatBusy) return;
  const input = document.getElementById("chat-input");
  const text = (input.value || "").trim();
  if (!text) {
    _chatStatus("Порожнє повідомлення", "warn");
    return;
  }
  _chatBusy = true;
  _chatStatus("Jarvis думає…", "ok");
  _chatAppend("user", text);
  input.value = "";
  _chatAbort = new AbortController();
  try {
    const r = await fetch(`/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders(),
      },
      body: JSON.stringify({ text }),
      signal: _chatAbort.signal,
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.error || `HTTP ${r.status}`);
    }
    const j = await r.json();
    _chatAppend("assistant", j.reply || "(порожня відповідь)");
    _chatStatus(`Готово · модель ${j.model || "?"}`, "ok");
  } catch (e) {
    if (e.name === "AbortError") {
      _chatStatus("Скасовано", "warn");
    } else {
      _chatAppend("assistant", `⚠ ${e.message || e}`);
      _chatStatus(`Помилка: ${e.message || e}`, "error");
    }
  } finally {
    _chatBusy = false;
    _chatAbort = null;
  }
}
async function interruptJarvis() {
  // 1. Abort in-flight /api/chat fetch (if any)
  if (_chatAbort) {
    try { _chatAbort.abort(); } catch {}
  }
  // 2. Tell voice pipeline to drop current TTS / LLM stream
  try {
    const r = await fetch(`/api/interrupt`, {
      method: "POST",
      headers: authHeaders(),
    });
    if (r.ok) {
      _chatStatus("Зупинено", "warn");
    } else {
      const j = await r.json().catch(() => ({}));
      _chatStatus(`Не вдалось: ${j.error || r.status}`, "error");
    }
  } catch (e) {
    _chatStatus(`Помилка: ${e.message || e}`, "error");
  }
}

// ─── Brain Graph (R34-S27 · 3D rebuild R35-S34) ───────────────────
// A real 3D force-directed knowledge graph (Obsidian-style) of
// everything Jarvis knows / can do / uses: skills, tools, memory
// facts and the Nexus Brain vault (wikilink edges). Built on the
// vendored 3d-force-graph (three.js). Live activity from /ws/events
// shoots particles along the links of whatever Jarvis is using.
const BRAIN_COLORS = {
  core: "#ffffff", hub: "#6C63FF", skill: "#00D4FF",
  tool: "#FFD166", fact: "#06D6A0", note: "#FF6B9D",
};
const BRAIN_GROUP_LABELS = {
  core: "Core", hub: "Clusters", skill: "Skills",
  tool: "Tools", fact: "Memory", note: "Vault",
};
const _brain = {
  graph: null, socket: null, reconnectMs: 1000,
  linksByNode: {}, nodesById: {}, spinTimer: null, built: false, _chipT: null,
};

function _hex2rgb(h) {
  const m = h.replace("#", "");
  return [parseInt(m.slice(0, 2), 16), parseInt(m.slice(2, 4), 16), parseInt(m.slice(4, 6), 16)];
}
function _mixHex(a, b, t) {
  const ca = _hex2rgb(a), cb = _hex2rgb(b);
  const c = ca.map((v, i) => Math.round(v + (cb[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function _brainNodeColor(n) {
  const base = BRAIN_COLORS[n.group] || "#8a90b8";
  return (n.heat && n.heat > 0.05) ? _mixHex(base, "#ff8a00", Math.min(1, n.heat)) : base;
}
function _brainLinkColor(l) {
  switch (l.kind) {
    case "wikilink": return "#FF6B9D";
    case "uses": return "#FFD166";
    case "hub": return "#6C63FF";
    default: return "#8a90b8";
  }
}
function _brainSize() {
  const el = document.getElementById("brain-graph");
  if (!el) return [800, 520];
  return [el.clientWidth || 800, el.clientHeight || 520];
}

// Deterministic camera framing — zoomToFit() proved unreliable here
// (the force layout overshoots, and fit picked wildly wrong distances).
// We compute the node bounding sphere and place the camera so it fits
// with a small margin, looking at the centre of mass.
function _brainFitCamera(ms) {
  const G = _brain.graph;
  if (!G) return;
  const ns = G.graphData().nodes;
  if (!ns.length) return;
  let cx = 0, cy = 0, cz = 0;
  for (const n of ns) { cx += n.x || 0; cy += n.y || 0; cz += n.z || 0; }
  cx /= ns.length; cy /= ns.length; cz /= ns.length;
  let maxR = 1;
  for (const n of ns) {
    const r = Math.hypot((n.x || 0) - cx, (n.y || 0) - cy, (n.z || 0) - cz);
    if (r > maxR) maxR = r;
  }
  const fov = ((G.camera() && G.camera().fov) || 50) * Math.PI / 180;
  const dist = (maxR / Math.tan(fov / 2)) * 1.25 + 30;
  try {
    G.cameraPosition({ x: cx, y: cy, z: cz + dist }, { x: cx, y: cy, z: cz }, ms || 0);
  } catch (_) {}
}

function _brainEnsureGraph() {
  if (_brain.graph) return _brain.graph;
  const el = document.getElementById("brain-graph");
  if (!el || typeof window.ForceGraph3D !== "function") return null;
  const [w, h] = _brainSize();
  const G = window.ForceGraph3D()(el)
    .width(w).height(h)
    .backgroundColor("#05060f")
    .showNavInfo(false)
    .nodeId("id")
    .nodeVal((n) => (n.group === "core" ? 5 : n.group === "hub" ? 3 : Math.min(n.val || 1, 2.2)))
    .nodeRelSize(2.0)
    .nodeOpacity(0.9)
    .nodeResolution(18)
    .nodeColor(_brainNodeColor)
    .nodeLabel((n) => `<div class="g-tip"><b>${escapeHtml(n.label)}</b><span>${escapeHtml(n.kind || n.group)}</span></div>`)
    .linkColor(_brainLinkColor)
    .linkOpacity(0.32)
    .linkWidth((l) => (l.kind === "hub" ? 1.4 : 0.5))
    .linkDirectionalParticles((l) => (l.kind === "hub" ? 2 : 0))
    .linkDirectionalParticleWidth(1.8)
    .linkDirectionalParticleSpeed(0.006)
    .linkDirectionalParticleColor(() => "#00D4FF")
    .cooldownTicks(220)
    .d3VelocityDecay(0.55)
    .onEngineStop(() => _brainFitCamera(400))
    .onNodeClick(_brainOpenNode)
    .onNodeDragEnd((n) => { n.fx = n.x; n.fy = n.y; n.fz = n.z; })
    .onBackgroundClick(_brainCloseDrawer);
  // Spread the graph out: strong repulsion + long hub spokes so the
  // clusters separate into readable lobes instead of one dense ball.
  try {
    G.d3Force("charge").strength(-55);
    G.d3Force("link").distance((l) => (l.kind === "hub" ? 45 : 22)).strength(0.85);
  } catch (_) {}
  _brain.graph = G;
  return G;
}

async function _brainFetch() {
  try {
    const data = await api("/api/brain");
    _brainBuild(data);
  } catch (e) {
    console.warn("brain fetch:", e);
    _brainShowEmpty("Не вдалося завантажити граф.");
  }
}

function _brainShowEmpty(msg) {
  const empty = document.getElementById("brain-empty");
  const gel = document.getElementById("brain-graph");
  if (empty) { empty.hidden = !msg; empty.textContent = msg || ""; }
  if (gel) gel.style.visibility = msg ? "hidden" : "visible";
}

function _brainBuild(data) {
  const G = _brainEnsureGraph();
  if (!G) { _brainShowEmpty("3D-рушій не завантажився."); return; }
  const nodes = (data.nodes || []).map((n) => ({ ...n }));
  const links = (data.links || []).map((l) => ({ ...l }));
  if (!nodes.length) { _brainShowEmpty("Поки немає даних для графа."); return; }
  _brainShowEmpty("");
  // Index for live particle emission + lookups.
  const byNode = {}, byId = {};
  for (const n of nodes) byId[n.id] = n;
  for (const l of links) {
    (byNode[l.source] = byNode[l.source] || []).push(l);
    (byNode[l.target] = byNode[l.target] || []).push(l);
  }
  _brain.linksByNode = byNode;
  _brain.nodesById = byId;
  G.graphData({ nodes, links });
  _brain.built = true;
  const tot = data.totals || {};
  const meta = data.meta || {};
  const trunc = meta.vault_truncated ? " · vault обрізано" : "";
  document.getElementById("brain-totals").textContent =
    `${tot.nodes || nodes.length} вузлів · ${tot.links || links.length} звʼязків${trunc}`;
  _brainRenderLegend(data);
  // Track the settling layout with a self-stopping fit loop. Robust
  // regardless of whether the engine emits tick/stop events (it often
  // doesn't here), and recomputes the bounding sphere each pass so the
  // camera ends framed on the final, contracted graph.
  if (_brain._fitTimer) clearInterval(_brain._fitTimer);
  let fits = 0;
  _brain._fitTimer = setInterval(() => {
    _brainFitCamera(0);
    if (++fits >= 22) { clearInterval(_brain._fitTimer); _brain._fitTimer = null; }
  }, 400);
}

function _brainRenderLegend(data) {
  const el = document.getElementById("brain-legend");
  if (!el) return;
  const groups = (data.meta && data.meta.groups) || {};
  const order = ["core", "hub", "skill", "tool", "fact", "note"];
  el.innerHTML = order
    .filter((g) => groups[g])
    .map((g) => `<span class="brain-legend-item">
      <span class="brain-dot" style="background:${BRAIN_COLORS[g]}"></span>
      ${BRAIN_GROUP_LABELS[g]} · ${groups[g]}
    </span>`)
    .join("");
}

function _brainFmtField(v) {
  if (Array.isArray(v)) return v.join(", ");
  if (v === true) return "так";
  if (v === false) return "ні";
  return String(v);
}

async function _brainOpenNode(node) {
  if (!node) return;
  const d = document.getElementById("brain-drawer");
  if (d) d.hidden = false;
  document.getElementById("brain-drawer-kind").textContent = node.kind || node.group || "";
  document.getElementById("brain-drawer-title").textContent = node.label || node.id;
  document.getElementById("brain-drawer-sub").textContent = "Завантаження…";
  document.getElementById("brain-drawer-tags").innerHTML = "";
  document.getElementById("brain-drawer-fields").innerHTML = "";
  document.getElementById("brain-drawer-content").textContent = "";
  // Fly the camera to the clicked node.
  try {
    const G = _brain.graph;
    const dist = 90;
    const r = 1 + dist / Math.hypot(node.x || 1, node.y || 1, node.z || 1);
    G.cameraPosition({ x: (node.x || 0) * r, y: (node.y || 0) * r, z: (node.z || 0) * r }, node, 800);
  } catch (_) {}
  try {
    const info = await api("/api/brain/node?id=" + encodeURIComponent(node.id));
    document.getElementById("brain-drawer-kind").textContent = info.kind || "";
    document.getElementById("brain-drawer-title").textContent = info.title || node.label;
    document.getElementById("brain-drawer-sub").textContent = info.subtitle || "";
    document.getElementById("brain-drawer-tags").innerHTML = (info.tags || [])
      .map((t) => `<span class="chip">${escapeHtml(t)}</span>`).join("");
    document.getElementById("brain-drawer-fields").innerHTML = Object.entries(info.fields || {})
      .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(_brainFmtField(v))}</dd>`)
      .join("");
    document.getElementById("brain-drawer-content").textContent = info.content || "";
  } catch (e) {
    document.getElementById("brain-drawer-sub").textContent = "";
    document.getElementById("brain-drawer-content").textContent = "Не вдалося завантажити вміст.";
  }
}
function _brainCloseDrawer() {
  const d = document.getElementById("brain-drawer");
  if (d) d.hidden = true;
}

// ── live activity: particles along the links of whatever Jarvis uses
const _BRAIN_CORE_TYPES = new Set([
  "state", "wake_word", "stt_final", "sentence", "tts_start",
  "bot_started_speaking", "dashboard_chat_reply", "direct_chat_reply",
  "voice_inject", "hot_window",
]);
const _BRAIN_IGNORE_TYPES = new Set([
  "pipecat_audio_rms", "heartbeat", "log", "vad",
]);

function _brainEventNodeIds(ev) {
  const ids = [];
  const t = ev.type || "";
  if (ev.tool) ids.push("tool:" + ev.tool);
  if (ev.skill) ids.push("skill:" + ev.skill);
  if (ev.key) ids.push("fact:" + ev.key);
  if (_BRAIN_CORE_TYPES.has(t)) ids.push("core");
  return ids;
}

function _brainOnEvent(ev) {
  const t = ev.type || "";
  if (_BRAIN_IGNORE_TYPES.has(t)) return;
  const chip = document.getElementById("brain-live");
  if (chip) {
    chip.textContent = ev.state ? `${t}: ${ev.state}` : t;
    chip.classList.add("hot");
    clearTimeout(_brain._chipT);
    _brain._chipT = setTimeout(() => chip.classList.remove("hot"), 1200);
  }
  const G = _brain.graph;
  if (!G || !_brain.built) return;
  for (const id of _brainEventNodeIds(ev)) {
    const ls = _brain.linksByNode[id];
    if (!ls) continue;
    let n = 0;
    for (const l of ls) {
      try { G.emitParticle(l); } catch (_) {}
      if (++n >= 8) break;
    }
  }
}

function _brainConnectWS() {
  if (_brain.socket && _brain.socket.readyState !== WebSocket.CLOSED) return;
  const token = localStorage.getItem(TOKEN_KEY);
  const url = new URL("/ws/events", window.location.href);
  url.protocol = url.protocol.replace("http", "ws");
  url.searchParams.set("token", token);
  const sock = new WebSocket(url.toString());
  _brain.socket = sock;
  sock.addEventListener("open", () => { _brain.reconnectMs = 1000; });
  sock.addEventListener("message", (e) => {
    try { _brainOnEvent(JSON.parse(e.data)); } catch (_) {}
  });
  sock.addEventListener("close", () => {
    const delay = _brain.reconnectMs;
    _brain.reconnectMs = Math.min(_brain.reconnectMs * 2, 30000);
    setTimeout(() => {
      const active = document.getElementById("view-brain").classList.contains("active");
      if (active && document.visibilityState !== "hidden") _brainConnectWS();
    }, delay);
  });
}
function _brainDisconnectWS() {
  try { _brain.socket?.close(); } catch (_) {}
  _brain.socket = null;
  _brain.reconnectMs = 1000;
}

function _brainResize() {
  if (!_brain.graph) return;
  if (!document.getElementById("view-brain").classList.contains("active")) return;
  const [w, h] = _brainSize();
  try { _brain.graph.width(w).height(h); } catch (_) {}
}
window.addEventListener("resize", _brainResize);

function _brainToggleSpin() {
  const btn = document.getElementById("brain-spin");
  if (_brain.spinTimer) {
    clearInterval(_brain.spinTimer);
    _brain.spinTimer = null;
    btn?.classList.remove("active");
    return;
  }
  btn?.classList.add("active");
  _brain.spinTimer = setInterval(() => {
    const G = _brain.graph;
    if (!G) return;
    try {
      const p = G.cameraPosition();
      const r = Math.hypot(p.x, p.z) || 1;
      const a = Math.atan2(p.x, p.z) + 0.0035;
      G.cameraPosition({ x: r * Math.sin(a), z: r * Math.cos(a) });
    } catch (_) {}
  }, 33);
}
function startBrainView() {
  const G = _brainEnsureGraph();
  if (!G) { _brainShowEmpty("3D-рушій не завантажився."); return; }
  try { G.resumeAnimation && G.resumeAnimation(); } catch (_) {}
  requestAnimationFrame(_brainResize);
  if (!_brain.built) _brainFetch();
  _brainConnectWS();
  const refresh = document.getElementById("brain-refresh");
  if (refresh && !refresh._wired) {
    refresh._wired = true;
    refresh.addEventListener("click", _brainFetch);
  }
  const spin = document.getElementById("brain-spin");
  if (spin && !spin._wired) {
    spin._wired = true;
    spin.addEventListener("click", _brainToggleSpin);
  }
  const close = document.getElementById("brain-drawer-close");
  if (close && !close._wired) {
    close._wired = true;
    close.addEventListener("click", _brainCloseDrawer);
  }
}
function stopBrainView() {
  _brainDisconnectWS();
  if (_brain.spinTimer) {
    clearInterval(_brain.spinTimer);
    _brain.spinTimer = null;
    document.getElementById("brain-spin")?.classList.remove("active");
  }
  if (_brain._fitTimer) { clearInterval(_brain._fitTimer); _brain._fitTimer = null; }
  try { _brain.graph?.pauseAnimation?.(); } catch (_) {}
}

// Boot
if (localStorage.getItem(TOKEN_KEY)) {
  bootstrap();
} else {
  showAuth();
}
