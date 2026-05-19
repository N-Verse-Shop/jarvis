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
  const res = await fetch(path, {
    ...opts,
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
  if (name === "live") startLiveStream();
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
  const orb = document.getElementById("state-orb");
  orb.className = "orb state-" + (state || "OFFLINE").toUpperCase();
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

function startLiveStream() {
  if (liveSocket && liveSocket.readyState !== WebSocket.CLOSED) return;
  const token = localStorage.getItem(TOKEN_KEY);
  const url = new URL("/ws/events", window.location.href);
  url.protocol = url.protocol.replace("http", "ws");
  url.searchParams.set("token", token);
  const feed = document.getElementById("live-feed");
  feed.innerHTML = "";
  liveSocket = new WebSocket(url.toString());
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
    setTimeout(() => {
      if (document.getElementById("view-live").classList.contains("active")) {
        startLiveStream();
      }
    }, 1000);
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
}

// Boot
if (localStorage.getItem(TOKEN_KEY)) {
  bootstrap();
} else {
  showAuth();
}
