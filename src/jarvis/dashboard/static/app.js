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

// ─── Brain View (R34-S27) ─────────────────────────────────────────
// Canvas2D animated visualization of skills + memory + tool firing.
// Polls /api/brain at 5s TTL, renders nodes per region as glowing
// orbs that pulse when their region has recent activity. Designed
// to look like the @KZZY47 "Kronos" UI without the marketing
// dishonesty: every neuron is a real piece of state Jarvis tracks.
const _brainState = {
  raf: null,
  timer: null,
  data: null,
  t0: 0,
};
async function _fetchBrain() {
  try {
    const data = await api("/api/brain");
    if (data && data.regions) {
      _brainState.data = data;
      const tot = data.totals || {};
      document.getElementById("brain-totals").textContent =
        `${tot.neurons || 0} neurons · ${tot.regions || 0} regions`;
      _renderBrainLegend(data);
    }
  } catch (e) {
    console.warn("brain fetch:", e);
  }
}
function _renderBrainLegend(data) {
  const el = document.getElementById("brain-legend");
  if (!el) return;
  const palette = _brainPalette();
  el.innerHTML = (data.regions || [])
    .map((r, idx) => {
      const c = palette(idx);
      return `<span class="brain-legend-item">
        <span class="brain-dot" style="background:${c}"></span>
        ${escapeHtml(r.label)} · ${r.neurons.length}
      </span>`;
    })
    .join("");
}
function _brainPalette() {
  const colors = [
    "#6C63FF", "#00D4FF", "#FF6B9D", "#FFD166",
    "#06D6A0", "#EF476F", "#118AB2", "#073B4C",
  ];
  return (i) => colors[i % colors.length];
}
function _brainDraw() {
  const canvas = document.getElementById("brain-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  // R34-S57 (C-P2.3): the context is already scaled by DPR, so all
  // draw math must use LOGICAL pixels (rect.width / rect.height),
  // not the device-pixel buffer size (canvas.width / canvas.height).
  // Otherwise everything renders at 2× position on Retina and orbs
  // overshoot the visible canvas.
  const rect = canvas.getBoundingClientRect();
  const W = rect.width;
  const H = rect.height;
  // Background fade for trailing motion blur
  ctx.fillStyle = "rgba(5, 6, 16, 0.18)";
  ctx.fillRect(0, 0, W, H);
  if (!_brainState.data) {
    _brainState.raf = requestAnimationFrame(_brainDraw);
    return;
  }
  const t = (performance.now() - _brainState.t0) / 1000;
  const palette = _brainPalette();
  const regions = _brainState.data.regions || [];
  // Polar layout: regions evenly distributed around the centre,
  // neurons stretched radially out from each region anchor.
  const cx = W / 2, cy = H / 2;
  regions.forEach((region, ri) => {
    const angle = (ri / Math.max(1, regions.length)) * Math.PI * 2 - Math.PI / 2;
    const r0 = Math.min(W, H) * 0.18;
    const ax = cx + Math.cos(angle) * r0;
    const ay = cy + Math.sin(angle) * r0;
    const color = palette(ri);
    const firing = Math.max(0, Math.min(1, region.firing_rate || 0));
    const pulse = 0.55 + 0.45 * Math.sin(t * (1.5 + firing * 4) + ri);
    // Region anchor: bigger glow on higher firing
    ctx.beginPath();
    const ar = 8 + firing * 16 + pulse * 6;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.85;
    ctx.arc(ax, ay, ar, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 0.18;
    ctx.arc(ax, ay, ar * 2.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    // Region label
    ctx.fillStyle = "#aab0d4";
    ctx.font = "12px ui-sans-serif, system-ui, -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(region.label, ax, ay - ar - 8);
    // Neurons radiating outward
    const neurons = region.neurons || [];
    const ringRadius = Math.min(W, H) * 0.28;
    neurons.forEach((n, ni) => {
      const spread = Math.PI * 0.7;
      const localAngle = angle + ((ni - neurons.length / 2) / Math.max(1, neurons.length)) * spread;
      const d = ringRadius * (0.6 + 0.4 * ((ni + ri) % 7) / 7);
      const nx = ax + Math.cos(localAngle) * d * 0.6;
      const ny = ay + Math.sin(localAngle) * d * 0.6;
      // Connection line region→neuron
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.18;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(nx, ny);
      ctx.stroke();
      // Neuron dot
      const w = Math.max(0.5, Math.min(3, n.weight || 1));
      const flicker = 0.7 + 0.3 * Math.sin(t * 3 + ni * 0.6 + ri);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.6 + 0.4 * flicker;
      ctx.beginPath();
      ctx.arc(nx, ny, 2 + w * 0.8, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
  });
  _brainState.raf = requestAnimationFrame(_brainDraw);
}
function startBrainView() {
  // Resize canvas to actual pixel size so we get crisp rendering.
  const canvas = document.getElementById("brain-canvas");
  if (canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    // R34-S57 (C-P2.3): canvas DPR fix.
    // The original code set ``canvas.width = rect.width * dpr``,
    // applied ``ctx.scale(dpr, dpr)``, then IMMEDIATELY reset
    // ``canvas.width = canvas.width / dpr`` — assigning to
    // ``canvas.width`` RESETS the 2D context state (including the
    // scale transform) and snaps the buffer back to logical
    // pixels. Net effect: every byte of DPR work was thrown away
    // and the canvas rendered at logical resolution, so orbs
    // looked blurry on Retina + Hi-DPI displays.
    //
    // Correct pattern: set the backing-store size in device px,
    // pin the CSS box to the logical px, scale the context ONCE.
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    canvas.getContext("2d").scale(dpr, dpr);
  }
  _brainState.t0 = performance.now();
  _fetchBrain();
  if (_brainState.timer) clearInterval(_brainState.timer);
  _brainState.timer = setInterval(_fetchBrain, 5000);
  if (_brainState.raf) cancelAnimationFrame(_brainState.raf);
  _brainState.raf = requestAnimationFrame(_brainDraw);
  const btn = document.getElementById("brain-refresh");
  if (btn && !btn._wired) {
    btn._wired = true;
    btn.addEventListener("click", _fetchBrain);
  }
}
function stopBrainView() {
  if (_brainState.timer) {
    clearInterval(_brainState.timer);
    _brainState.timer = null;
  }
  if (_brainState.raf) {
    cancelAnimationFrame(_brainState.raf);
    _brainState.raf = null;
  }
}

// Boot
if (localStorage.getItem(TOKEN_KEY)) {
  bootstrap();
} else {
  showAuth();
}
